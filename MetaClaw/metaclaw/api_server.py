"""
FastAPI proxy server for OpenClaw RL training data collection.

Adapted from OpenClaw-RL/openclaw-rl/openclaw_api_server.py.
Key changes vs. the original:
  - SGLang HTTP forward replaced with Tinker SamplingClient.sample_async
  - slime.utils.types.Sample replaced with ConversationSample
  - Skills injection via SkillManager
  - update_sampling_client() for hot-swapping after weight updates

Architecture mirrors OpenClaw-RL exactly:
  - threading.Event for submission gating (passed in from rollout worker)
  - queue.Queue for output (passed in from rollout worker)
  - start() / stop() threading lifecycle
  - Data production is request-driven in FastAPI handlers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
from itertools import count
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

from .config import MetaClawConfig
from .data_formatter import ConversationSample
from .memory.scope import base_scope, derive_memory_scope
from .prm_scorer import PRMScorer
from .sandbox import (
    SandboxApprovalManager,
    SandboxAuditLogger,
    SandboxDecision,
    SandboxPolicyEngine,
    SandboxWhitelistManager,
)
from .skill_manager import SkillManager
from .utils import (
    run_context_summary_llm,
    run_feedback_skill_llm,
    run_llm,
    run_session_report_llm,
    run_task_brief_llm,
)

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_NON_STANDARD_BODY_KEYS = {
    "session_id",
    "session_done",
    "turn_type",
    "memory_scope",
    "user_id",
    "workspace_id",
}


# ------------------------------------------------------------------ #
# Helper utilities                                                     #
# ------------------------------------------------------------------ #

def _ensure_reasoning_content(messages: list[dict]) -> list[dict]:
    """Ensure every assistant message carries a ``reasoning_content`` field.

    Models that support extended thinking expect *all* assistant turns
    (including tool-call turns) to include ``reasoning_content``.  If the
    original conversation history omits it for a given assistant turn, we
    fill it in with an empty string so the downstream API does not reject
    the request.
    """
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            if "reasoning_content" in msg and msg["reasoning_content"]:
                msg["reasoning"] = msg["reasoning_content"]
            elif "reasoning" in msg and msg["reasoning"]:
                msg["reasoning_content"] = msg["reasoning"]
            else:
                msg["reasoning_content"] = ""
                msg["reasoning"] = ""
        out.append(msg)
    return out


def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_assistant_content_parts(content: list[dict]) -> tuple[str, list[dict]]:
    """Extract plain text and OpenAI-style tool_calls from assistant content parts."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type == "toolCall":
            name = item.get("name")
            args = item.get("arguments", {})
            if not isinstance(args, str):
                try:
                    args = json.dumps(args, ensure_ascii=False)
                except Exception:
                    args = "{}"
            tc_id = item.get("id") or f"call_{i}"
            tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name or "unknown_tool",
                        "arguments": args,
                    },
                }
            )
    return (" ".join(text_parts).strip(), tool_calls)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_HANDLE_RE = re.compile(r"^call_?(?:kimi|xml)_?\d?+$")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_FUNCTIONS_PREFIX_RE = re.compile(r"^functions[._]?")
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_INLINE_FEEDBACK_PATTERNS = [
    re.compile(r"^/(?:feedback|fb)\s+(good|bad)\s*(.*)$", re.IGNORECASE),
]
_INLINE_APPROVAL_RE = re.compile(r"^/(approve|reject)\s*([A-Za-z0-9_-]+)?\s*$", re.IGNORECASE)

def _normalize_tool_name(raw_name: str, args_raw: str) -> str:
    """
    Normalize tool names from model output.
    """

    name = (raw_name or "").strip()

    if not name:
        return "unknown_tool"

    name = _FUNCTIONS_PREFIX_RE.sub("", name)

    parts = name.split(".")
    parts = [p for p in parts if not p.isdigit()]
    if not parts:
        return ""

    name = parts[-1]
    name = _TRAILING_DIGITS_RE.sub("", name)
    name = name.strip("_-.")

    if name and not _TOOL_HANDLE_RE.fullmatch(name):
        return name

    try:
        args_obj = json.loads(args_raw or "{}")
    except Exception:
        args_obj = {}

    if isinstance(args_obj, dict):
        if isinstance(args_obj.get("command"), str) and args_obj.get("command"):
            return "exec"
        elif isinstance(args_obj.get("sessionId"), str) and args_obj.get("sessionId"):
            return "process"
        elif isinstance(args_obj.get("file_path"), str) and args_obj.get("file_path"):
            if isinstance(args_obj.get("content"), str) and args_obj.get("content"):
                return "write"
            else:
                return "read"

    if "read" in raw_name:
        return "read"
    elif "write" in raw_name:
        return "write"

    return "unknown_tool"

def _extract_tool_calls_from_text(text: str) -> tuple[str, list[dict], str]:
    """
    Parse tool-call tags embedded in assistant text into OpenAI-style tool_calls.
    Supports Kimi markers and Qwen <tool_call> wrappers.

    Returns:
        (cleaned_text, tool_calls, reasoning_content)
    """
    if not text:
        return "", [], ""

    tool_calls: list[dict] = []

    for i, m in enumerate(_KIMI_TOOL_CALL_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        tool_name = _normalize_tool_name(raw_name, args_raw)
        try:
            args_obj = json.loads(args_raw)
            args_str = json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            args_str = args_raw if args_raw else "{}"
        tool_calls.append(
            {
                "id": f"call_kimi_{i}",
                "type": "function",
                "function": {"name": tool_name or "unknown_tool", "arguments": args_str},
            }
        )

    for i, m in enumerate(_QWEN_TOOL_CALL_RE.finditer(text), start=len(tool_calls)):
        payload_raw = (m.group(1) or "").strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        name = (
            payload.get("name")
            or payload.get("tool_name")
            or payload.get("function", {}).get("name")
            or "unknown_tool"
        )
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        name = _normalize_tool_name(str(name), args)
        tool_calls.append(
            {
                "id": f"call_xml_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )

    # Extract reasoning content from <think> blocks before stripping them.
    # Covers all cases:
    #   a) matched <think>X</think>  b) orphan leading: X</think> (no <think>)
    #   c) orphan trailing: <think>X (no </think>)
    reasoning_parts: list[str] = []
    # a) Matched pairs: <think>...</think>
    for m in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        reasoning_parts.append(m.group(1))
    # b) Orphan leading </think> with no preceding <think> — content before it is reasoning
    if "</think>" in text:
        first_close = text.index("</think>")
        prefix = text[:first_close]
        if "<think>" not in prefix:
            reasoning_parts.insert(0, prefix)
    # c) Trailing unclosed <think>...EOF
    _trailing = re.search(r"<think>((?:(?!</think>).)*)\Z", text, re.DOTALL)
    if _trailing:
        reasoning_parts.append(_trailing.group(1))
    reasoning_content = "\n".join(p.strip() for p in reasoning_parts if p.strip())

    # Strip all think-related markup from the clean text.
    clean = text
    # Remove matched <think>...</think> pairs first.
    clean = _THINK_RE.sub("", clean)
    # Remove orphan </think> (its content before it was already captured above).
    clean = re.sub(r"^[^<]*</think>", "", clean, count=1, flags=re.DOTALL)
    clean = clean.replace("</think>", "")
    # Remove trailing unclosed <think>...
    clean = re.sub(r"<think>(?:(?!</think>).)*\Z", "", clean, flags=re.DOTALL)
    # Keep tool call data only in structured field; strip markup from plain text.
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", "", clean, flags=re.DOTALL)
    clean = _QWEN_TOOL_CALL_RE.sub("", clean)
    clean = clean.strip()
    return clean, tool_calls, reasoning_content


def _normalize_tool_calls_for_template(tool_calls: list) -> list[dict]:
    """Ensure tool_calls are plain OpenAI-compatible dicts with string arguments.

    Handles cases where function.arguments is a dict instead of a JSON string,
    or where individual tool_call entries are non-dict objects (e.g. Pydantic models).
    """
    normalized: list[dict] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            try:
                tc = dict(tc)  # type: ignore[call-overload]
            except Exception:
                continue
        func = tc.get("function")
        if func is None:
            continue
        if not isinstance(func, dict):
            try:
                func = dict(func)  # type: ignore[call-overload]
            except Exception:
                func = {"name": str(func), "arguments": "{}"}
        else:
            func = dict(func)  # shallow copy so we don't mutate original
        args = func.get("arguments")
        if not isinstance(args, str):
            func["arguments"] = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
        normalized.append({
            "id": tc.get("id") or f"call_{i}",
            "type": tc.get("type", "function"),
            "function": func,
        })
    return normalized


def _normalize_tools_for_template(tools) -> list | None:
    """Convert tools from Anthropic format to OpenAI format expected by chat templates.

    Anthropic format:  {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI format:     {"type": "function", "function": {"name": ..., "parameters": {...}}}

    The Qwen3 (and other) chat templates use ``tool.function.parameters | items``
    which raises TypeError if the tool is in Anthropic format.
    """
    if not tools:
        return tools
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        # Already in OpenAI format
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            if isinstance(func, dict):
                out.append(tool)
            else:
                out.append(tool)
            continue
        # Anthropic format: top-level name + input_schema
        name = tool.get("name") or ""
        if name:
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": (
                        tool.get("input_schema")
                        or tool.get("parameters")
                        or {"type": "object", "properties": {}}
                    ),
                },
            })
        else:
            out.append(tool)
    return out


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Normalize OpenClaw-style messages into chat-template-compatible format."""
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")

        if role == "developer":
            m["role"] = "system"
            role = "system"

        # OpenClaw tool result message → OpenAI tool message
        if role == "toolResult":
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": _flatten_message_content(m.get("content")),
            }
            tc_id = m.get("toolCallId") or m.get("tool_call_id")
            if tc_id:
                tool_msg["tool_call_id"] = tc_id
            tool_name = m.get("toolName") or m.get("name")
            if tool_name:
                tool_msg["name"] = tool_name
            out.append(tool_msg)
            continue

        # assistant content parts may contain text + toolCall blocks
        raw = m.get("content")
        if role == "assistant" and isinstance(raw, list):
            text, tool_calls = _normalize_assistant_content_parts(raw)
            m["content"] = text
            if tool_calls:
                m["tool_calls"] = tool_calls
        elif not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)

        # Ensure any existing tool_calls are proper plain dicts with string arguments
        # so that Jinja2 chat templates don't fail with "Can only get item pairs from a mapping"
        if role == "assistant":
            existing_tcs = m.get("tool_calls")
            if existing_tcs and isinstance(existing_tcs, list):
                m["tool_calls"] = _normalize_tool_calls_for_template(existing_tcs)

        out.append(m)
    return out


def _extract_last_user_instruction(messages: list[dict]) -> str:
    """Return the most recent user message text from the current turn context."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _flatten_message_content(msg.get("content"))
            if text:
                return text
    return ""


def _render_context_messages(messages: list[dict], max_chars: int = 12000) -> str:
    rendered: list[str] = []
    total_chars = 0
    for idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "unknown"))
        content = _flatten_message_content(msg.get("content", ""))
        tool_calls = msg.get("tool_calls")
        line = f"[{idx}] {role}: {content}".strip()
        if tool_calls:
            try:
                line += f"\n    tool_calls={json.dumps(tool_calls, ensure_ascii=False)}"
            except Exception:
                line += "\n    tool_calls=<unserializable>"
        if total_chars + len(line) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                rendered.append(line[:remaining])
            break
        rendered.append(line)
        total_chars += len(line) + 1
    return "\n".join(rendered)


def _extract_context_summary_text(messages: list[dict]) -> str:
    marker = "## Conversation Summary"
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = _flatten_message_content(msg.get("content", ""))
        idx = content.find(marker)
        if idx != -1:
            return content[idx + len(marker):].strip()
    return ""


def _parse_inline_feedback(text: str) -> tuple[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    ascii_match = re.search(
        r"(?:^|\s)(?:/(?:feedback|fb)|feedback|fb)\s+(good|bad)\b(.*)$",
        raw,
        re.IGNORECASE,
    )
    if ascii_match:
        return (
            (ascii_match.group(1) or "").strip().lower(),
            (ascii_match.group(2) or "").strip(),
        )

    for pattern in _INLINE_FEEDBACK_PATTERNS:
        match = pattern.search(raw)
        if match:
            return (match.group(1) or "").strip().lower(), (match.group(2) or "").strip()

    cn_feedback = chr(0x53cd) + chr(0x9988)
    cn_reply = chr(0x56de) + chr(0x9988)
    cn_last = chr(0x4e0a) + chr(0x6b21)
    cn_good = chr(0x597d)
    cn_bad = chr(0x4e0d) + chr(0x597d)
    cn_poor = chr(0x574f)
    compact = raw.replace(chr(0xff1a), ':')
    for prefix in (cn_feedback, cn_reply, cn_last):
        idx = compact.find(prefix)
        if idx == -1:
            continue
        rest = compact[idx + len(prefix):].lstrip()
        for label, rating in ((cn_bad, "bad"), (cn_poor, "bad"), (cn_good, "good")):
            if rest.startswith(label):
                feedback_text = rest[len(label):].lstrip(" :-")
                return rating, feedback_text.strip()
    return None


def _find_inline_feedback(messages: list[dict]) -> tuple[str, str, str] | None:
    text = _extract_last_user_instruction(messages)
    if not text:
        return None
    parsed = _parse_inline_feedback(text)
    if parsed is None:
        return None
    return parsed[0], parsed[1], text


def _parse_inline_approval(text: str) -> tuple[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    match = _INLINE_APPROVAL_RE.match(raw)
    if not match:
        return None
    action = str(match.group(1) or "").strip().lower()
    approval_id = str(match.group(2) or "").strip()
    return action, approval_id


def _parse_inline_whitelist(text: str) -> tuple[str, str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"/whitelist", "/allowlist"}:
        return "list", "", ""
    match = re.match(r"^/(allow|unallow)\s+(command|path)\s+(.+?)\s*$", raw, re.IGNORECASE)
    if match:
        return (
            str(match.group(1) or "").strip().lower(),
            str(match.group(2) or "").strip().lower(),
            str(match.group(3) or "").strip(),
        )
    match = re.match(
        r"^/whitelist\s+(add|remove)\s+(command|path)\s+(.+?)\s*$",
        raw,
        re.IGNORECASE,
    )
    if match:
        action = "allow" if str(match.group(1) or "").strip().lower() == "add" else "unallow"
        return (
            action,
            str(match.group(2) or "").strip().lower(),
            str(match.group(3) or "").strip(),
        )
    return None


def _extract_preference_clauses(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    segments = [seg.strip() for seg in re.split("[.;\n\u3002\uff1b]+", raw) if seg.strip()]
    cn_prefixes = [
        chr(0x8bf7),
        chr(0x9ebb) + chr(0x70e6),
        chr(0x5c3d) + chr(0x91cf),
        chr(0x4e0d) + chr(0x8981),
        chr(0x522b),
        chr(0x4ee5) + chr(0x540e),
        chr(0x540e) + chr(0x9762),
        chr(0x6211) + chr(0x66f4) + chr(0x5e0c) + chr(0x671b),
        chr(0x6211) + chr(0x5e0c) + chr(0x671b),
        chr(0x6211) + chr(0x504f) + chr(0x597d),
        chr(0x6211) + chr(0x66f4) + chr(0x559c) + chr(0x6b22),
        chr(0x56de) + chr(0x7b54) + chr(0x65f6),
        chr(0x56de) + chr(0x590d) + chr(0x65f6),
    ]
    en_prefixes = (
        "please",
        "prefer",
        "i prefer",
        "i want",
        "next time please",
        "avoid",
        "don't",
    )

    results: list[str] = []
    for seg in segments:
        lower = seg.lower()
        if any(seg.startswith(prefix) for prefix in cn_prefixes) or any(lower.startswith(prefix) for prefix in en_prefixes):
            if len(seg) >= 4 and seg not in results:
                results.append(seg)
    return results[:4]


def _resolve_record_path(record_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    normalized_record_dir = os.path.normpath(record_dir or ".")
    normalized_path = os.path.normpath(path)
    if normalized_path == normalized_record_dir:
        return normalized_path
    if normalized_path.startswith(normalized_record_dir + os.sep):
        return normalized_path
    return os.path.join(normalized_record_dir, normalized_path)


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _rewrite_new_session_bootstrap_prompt(messages: list[dict]) -> tuple[list[dict], int]:
    """Rewrite OpenClaw /new bootstrap user prompt to a safer variant.

    Some upstream providers over-trigger policy filters on the stock bootstrap
    text ("A new session was started via /new or /reset ..."). This keeps
    behavior while avoiding brittle phrasing.
    """
    rewritten = 0
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        if msg.get("role") != "user":
            out.append(msg)
            continue
        text = _flatten_message_content(msg.get("content"))
        lowered = text.lower()
        if "a new session was started via /new or /reset" in lowered:
            out.append(
                {
                    **msg,
                    "content": (
                        "A new chat session just started. "
                        "Greet the user briefly in 1-3 sentences and ask what they want to do."
                    ),
                }
            )
            rewritten += 1
            continue
        out.append(msg)
    return out, rewritten


# ------------------------------------------------------------------ #
# MetaClawAPIServer                                                    #
# ------------------------------------------------------------------ #

class MetaClawAPIServer:
    """Proxy between OpenClaw and Tinker for RL training data collection.

    OpenClaw sends ``X-Session-Id`` and ``X-Turn-Type`` headers with every
    request.  The proxy forwards to Tinker SamplingClient, and when
    ``turn_type`` is ``"main"`` it tokenises the full prompt+response and
    submits a training sample.  Side tasks (``turn_type != "main"``) are
    forwarded but produce no training data.

    Parameters
    ----------
    config:
        MetaClawConfig instance.
    output_queue:
        Thread-safe queue for (group_index, [ConversationSample]) tuples.
        Created and owned by the rollout worker.
    submission_enabled:
        threading.Event that gates sample submission.
        Set = accepting samples; clear = paused for weight update.
    sampling_client:
        Tinker SamplingClient. Can be None and set later via
        update_sampling_client().
    skill_manager:
        Optional SkillManager for injecting skills into system prompts.
    prm_scorer:
        Optional PRMScorer. If None, all samples get reward=0.
    """

    def __init__(
        self,
        config: MetaClawConfig,
        output_queue: queue.Queue,
        submission_enabled: threading.Event,
        sampling_client=None,
        skill_manager: Optional[SkillManager] = None,
        prm_scorer: Optional[PRMScorer] = None,
        skill_evolver=None,
        last_request_tracker=None,
        memory_manager=None,
    ):
        self.config = config
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self._sampling_client = sampling_client
        self.skill_manager = skill_manager
        self.prm_scorer = prm_scorer
        self.skill_evolver = skill_evolver
        self.memory_manager = memory_manager
        # Optional LastRequestTracker for scheduler idle detection
        self._last_request_tracker = last_request_tracker

        self._served_model = config.served_model_name
        self._expected_api_key = config.api_key
        os.makedirs(config.record_dir, exist_ok=True)
        self._system_prompt_cache_file = os.path.join(
            config.record_dir, "system_prompt_cache.json"
        )

        # State machines
        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}  # session → {turn → data}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task}
        self._teacher_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task} (OPD)
        self._pending_records: dict[str, dict] = {}               # for record logging
        self._session_effective: dict[str, int] = {}              # at-least-one guarantee
        self._session_context_summaries: dict[str, str] = {}
        self._feedback_by_session: dict[str, list[dict[str, Any]]] = {}
        self._latest_injected_skills: dict[str, list[str]] = {}
        self._user_profiles: dict[str, list[str]] = self._load_user_profiles()
        self._task_briefs: dict[str, dict[str, list[str]]] = self._load_task_briefs()
        self._latest_session_reports: dict[str, dict[str, Any]] = {}
        self._sandbox_whitelist: SandboxWhitelistManager | None = None
        self._sandbox_policy = (
            SandboxPolicyEngine(
                command_policy_enabled=config.sandbox_command_policy_enabled,
                path_policy_enabled=config.sandbox_path_policy_enabled,
                whitelist_manager=None,
            )
            if config.sandbox_enabled
            else None
        )
        self._sandbox_audit: SandboxAuditLogger | None = None
        self._sandbox_approvals: SandboxApprovalManager | None = None
        # Buffer turns per session for skill evolution (cleared on evolution trigger)
        self._session_turns: dict[str, list] = {}
        # Buffer turns per session for memory ingestion (only cleared on session_done)
        self._session_memory_turns: dict[str, list] = {}
        self._session_memory_scopes: dict[str, str] = {}

        # OPD teacher model client
        self._teacher_client: Optional[OpenAI] = None
        if config.use_opd and config.teacher_url:
            self._teacher_client = OpenAI(
                base_url=config.teacher_url,
                api_key=config.teacher_api_key or "unused",
            )
            logger.info("[OpenClaw] OPD teacher client ready: url=%s model=%s",
                        config.teacher_url, config.teacher_model)
        elif config.use_opd and not config.teacher_url:
            logger.warning("[OpenClaw] use_opd=True but teacher_url is empty — teacher logprobs disabled")

        # Record files
        self._record_file = ""
        self._enriched_record_file = ""
        self._openclaw_rl_record_file = ""
        self._feedback_file = ""
        self._prm_record_file = ""
        self._session_report_file = ""
        self._sandbox_audit_file = ""
        self._sandbox_approval_state_file = ""
        self._sandbox_approval_history_file = ""
        self._sandbox_whitelist_file = ""
        if config.record_enabled:
            os.makedirs(config.record_dir, exist_ok=True)
            self._record_file = _resolve_record_path(config.record_dir, config.record_enriched_file)
            self._enriched_record_file = self._record_file
            self._openclaw_rl_record_file = _resolve_record_path(
                config.record_dir, config.record_openclaw_rl_file,
            )
            self._prm_record_file = _resolve_record_path(config.record_dir, "prm_scores.jsonl")
            self._feedback_file = _resolve_record_path(config.record_dir, config.feedback_history_path)
            self._session_report_file = _resolve_record_path(config.record_dir, config.session_report_path)
            self._sandbox_audit_file = _resolve_record_path(config.record_dir, config.sandbox_audit_path)
            self._sandbox_approval_state_file = _resolve_record_path(
                config.record_dir, config.sandbox_approval_state_path,
            )
            self._sandbox_approval_history_file = _resolve_record_path(
                config.record_dir, config.sandbox_approval_history_path,
            )
            self._sandbox_whitelist_file = _resolve_record_path(
                config.record_dir, config.sandbox_whitelist_path,
            )
            with open(self._record_file, "a", encoding="utf-8"):
                pass
            with open(self._openclaw_rl_record_file, "a", encoding="utf-8"):
                pass
            with open(self._prm_record_file, "a", encoding="utf-8"):
                pass
            with open(self._feedback_file, "a", encoding="utf-8"):
                pass
            with open(self._session_report_file, "a", encoding="utf-8"):
                pass
            if self.config.sandbox_audit_enabled:
                self._sandbox_audit = SandboxAuditLogger(self._sandbox_audit_file)
            if self.config.sandbox_enabled:
                self._sandbox_whitelist = SandboxWhitelistManager(self._sandbox_whitelist_file)
            if self.config.sandbox_approval_enabled:
                self._sandbox_approvals = SandboxApprovalManager(
                    self._sandbox_approval_state_file,
                    self._sandbox_approval_history_file,
                )
            if self._sandbox_policy is not None:
                self._sandbox_policy.whitelist_manager = self._sandbox_whitelist

        if self.config.feedback_enabled and self.skill_manager is not None:
            self._ensure_important_skill_exists()

        # Tokenizer is used in both modes for prompt length accounting/truncation,
        # and in RL mode additionally for training sample tokenization.
        self._tokenizer = self._load_tokenizer()
        self.app = self._build_app()

        # Threading lifecycle (set by start())
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

        # External trainer reference + event loop for admin train-step endpoint.
        # Set via set_trainer() after the trainer is constructed.
        self._trainer = None
        self._main_loop = None

    # ------------------------------------------------------------------ #
    # Tokenizer                                                            #
    # ------------------------------------------------------------------ #

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning("[OpenClaw] could not load tokenizer: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # FastAPI app                                                          #
    # ------------------------------------------------------------------ #

    def set_trainer(self, trainer, main_loop) -> None:
        """Inject the trainer reference and its event loop for the admin
        ``/v1/admin/train_step`` endpoint.  Called by the launcher after
        the trainer is constructed."""
        self._trainer = trainer
        self._main_loop = main_loop

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="MetaClaw Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
            x_memory_scope: Optional[str] = Header(default=None),
            x_user_id: Optional[str] = Header(default=None),
            x_workspace_id: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            # Update idle tracker so the scheduler knows the user is active
            if owner._last_request_tracker is not None:
                owner._last_request_tracker.touch()
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                # Queue requests while submission is paused instead of returning 503.
                # Use a bounded wait so clients don't hang forever on abnormal stalls.
                resumed = await asyncio.to_thread(owner.submission_enabled.wait, 300.0)
                if not resumed:
                    raise HTTPException(
                        status_code=503,
                        detail="submission paused for weight update (wait timeout)",
                    )

            body = await request.json()
            incoming_messages = body.get("messages", [])
            if isinstance(incoming_messages, list):
                rewritten_messages, rewritten = _rewrite_new_session_bootstrap_prompt(
                    incoming_messages
                )
                body["messages"] = rewritten_messages
            else:
                rewritten = 0
            _raw_sid = x_session_id or body.get("session_id") or ""
            # TUI mode: OpenClaw does not send X-Session-Id/X-Turn-Type.
            # Fall back to a model-derived session ID and treat as "main" so
            # TUI conversations are collected as training data.
            if _raw_sid:
                session_id = _raw_sid
                turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
            else:
                session_id = f"tui-{body.get('model', 'default')}"
                turn_type = (x_turn_type or body.get("turn_type") or "main").strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )
            # Do not infer session_done from bootstrap text — only explicit X-Session-Done or body session_done trigger evolution.

            # Reuse cached scope for the session to avoid re-deriving
            # (which would nest |session:X repeatedly).
            _explicit_scope = (x_memory_scope or "") or str(body.get("memory_scope", "") or "")
            _explicit_user = (x_user_id or "") or str(body.get("user_id", "") or "")
            _explicit_workspace = (x_workspace_id or "") or str(body.get("workspace_id", "") or "")
            _cached = owner._session_memory_scopes.get(session_id, "")
            if _cached and not _explicit_scope and not _explicit_user and not _explicit_workspace:
                memory_scope = _cached
            else:
                memory_scope = derive_memory_scope(
                    default_scope=owner.memory_manager.scope_id if owner.memory_manager else "default",
                    session_id=session_id,
                    memory_scope=_explicit_scope,
                    user_id=_explicit_user,
                    workspace_id=_explicit_workspace,
                )

            stream = bool(body.get("stream", False))
            owner._append_sandbox_audit(
                "request_received",
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                memory_scope=memory_scope,
                stream=stream,
            )
            result = await owner._handle_request(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                memory_scope=memory_scope,
            )
            if stream:
                return StreamingResponse(
                    owner._stream_response(result), media_type="text/event-stream"
                )
            return JSONResponse(content=result["response"])

        @app.post("/v1/admin/train_step")
        async def admin_train_step(request: Request):
            """Trigger a single RL training step using queued samples.

            Intended to be called by ``metaclaw train-step`` CLI or any
            external orchestrator (e.g. benchmark scripts).

            The trainer runs on the main event loop while this server runs
            on a separate uvicorn thread.  We schedule the coroutine onto
            the main loop via run_coroutine_threadsafe, then wait for the
            result in a thread-pool worker (asyncio.to_thread) so that the
            uvicorn event loop stays free to process inference requests
            during training.
            """
            owner: MetaClawAPIServer = request.app.state.owner
            if owner._trainer is None or owner._main_loop is None:
                raise HTTPException(
                    status_code=503,
                    detail="trainer not available (not in RL mode or not yet initialised)",
                )
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(
                owner._trainer.train_step_external(),
                owner._main_loop,
            )

            def _wait():
                try:
                    return future.result(timeout=600)
                except concurrent.futures.TimeoutError:
                    raise RuntimeError("train step timed out (600s)")

            try:
                result = await asyncio.to_thread(_wait)
            except RuntimeError as exc:
                raise HTTPException(status_code=504, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"train step failed: {exc}")
            return JSONResponse(content=result)

        # ---------------------------------------------------------- #
        # Memory management REST API                                  #
        # ---------------------------------------------------------- #

        @app.get("/v1/memory/stats")
        async def memory_stats(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            stats = await asyncio.to_thread(
                owner.memory_manager.get_scope_stats, scope or None
            )
            return JSONResponse(content=stats)

        @app.get("/v1/memory/search")
        async def memory_search(
            request: Request,
            q: str = "",
            scope: str = "",
            limit: int = 20,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            results = await asyncio.to_thread(
                owner.memory_manager.search_memories, q, scope or None, limit
            )
            return JSONResponse(content=results)

        @app.get("/v1/memory/health")
        async def memory_health(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            health = await asyncio.to_thread(
                owner.memory_manager.run_system_health_check, scope or None
            )
            return JSONResponse(content=health)

        @app.get("/v1/memory/summary")
        async def memory_summary(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            summary = await asyncio.to_thread(
                owner.memory_manager.get_system_summary
            )
            return JSONResponse(content=summary)

        @app.get("/v1/memory/{memory_id}")
        async def memory_get(
            request: Request,
            memory_id: str,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            mem = await asyncio.to_thread(
                owner.memory_manager.get_memory, memory_id
            )
            if not mem:
                raise HTTPException(status_code=404, detail="memory not found")
            return JSONResponse(content={
                "memory_id": mem.memory_id,
                "scope_id": mem.scope_id,
                "type": mem.memory_type.value,
                "content": mem.content,
                "summary": mem.summary,
                "importance": mem.importance,
                "entities": mem.entities,
                "topics": mem.topics,
                "tags": mem.tags,
                "pinned": mem.importance >= 0.99,
                "created_at": mem.created_at,
                "updated_at": mem.updated_at,
                "expires_at": mem.expires_at,
            })

        @app.post("/v1/memory/action-plan")
        async def memory_action_plan(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            plan = await asyncio.to_thread(
                owner.memory_manager.generate_action_plan, scope or None
            )
            return JSONResponse(content=plan)

        @app.post("/v1/memory/maintenance")
        async def memory_maintenance(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            result = await asyncio.to_thread(
                owner.memory_manager.run_maintenance, scope or None
            )
            return JSONResponse(content=result)

        @app.get("/v1/memory/feedback-analysis")
        async def memory_feedback_analysis(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            result = await asyncio.to_thread(
                owner.memory_manager.analyze_feedback_patterns, scope or None
            )
            return JSONResponse(content=result)

        @app.get("/v1/memory/operator-report")
        async def memory_operator_report(
            request: Request,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            report = await asyncio.to_thread(
                owner.memory_manager.generate_operator_report, scope or None
            )
            return JSONResponse(content=report)

        @app.post("/v1/memory/ingest")
        async def memory_ingest(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            """Manually trigger memory ingestion for buffered sessions.

            Request body: {"session_id": "...", "scope": "..."}
            If session_id is provided, ingest that specific session.
            If session_id is empty or omitted, ingest ALL buffered sessions.
            """
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.memory_manager:
                raise HTTPException(status_code=503, detail="memory not enabled")
            body = await request.json()
            sid = str(body.get("session_id", "")).strip()
            explicit_scope = str(body.get("scope", "")).strip()

            if sid:
                # Ingest a specific session.
                sessions_to_ingest = {sid: owner._session_memory_turns.pop(sid, [])}
            else:
                # Ingest ALL buffered sessions.
                sessions_to_ingest = dict(owner._session_memory_turns)
                owner._session_memory_turns.clear()

            total_added = 0
            total_turns = 0
            results = []
            for s_id, turns in sessions_to_ingest.items():
                if not turns:
                    continue
                raw_scope = explicit_scope or owner._session_memory_scopes.pop(s_id, "")
                scope = base_scope(raw_scope) if raw_scope else None
                logger.info("[Memory] manual ingest session=%s scope=%s → %d buffered turns", s_id, scope, len(turns))
                added = await asyncio.to_thread(
                    owner.memory_manager.ingest_session_turns, s_id, turns, scope,
                )
                total_added += added
                total_turns += len(turns)
                results.append({"session_id": s_id, "added": added, "turns": len(turns)})

            if not results:
                buffered_keys = list(owner._session_memory_turns.keys())
                logger.info("[Memory] manual ingest → no buffered turns (buffered_sessions=%s)", buffered_keys)
                return JSONResponse(content={"added": 0, "buffered_turns": 0, "sessions": []})

            logger.info("[Memory] manual ingest complete: %d sessions, %d turns, %d units added",
                        len(results), total_turns, total_added)
            return JSONResponse(content={
                "added": total_added,
                "buffered_turns": total_turns,
                "sessions": results,
            })

        @app.post("/v1/feedback")
        async def submit_feedback(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.config.feedback_enabled:
                raise HTTPException(status_code=503, detail="feedback is disabled")
            body = await request.json()
            session_id = str(body.get("session_id", "") or "").strip()
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            turn_raw = body.get("turn")
            turn = int(turn_raw) if turn_raw not in (None, "", False) else None
            rating = str(body.get("rating", "") or "").strip().lower()
            if rating not in {"good", "bad"}:
                raise HTTPException(status_code=400, detail="rating must be 'good' or 'bad'")
            feedback_text = str(body.get("feedback", "") or "").strip()
            result = await owner._handle_feedback(
                session_id=session_id,
                turn=turn,
                rating=rating,
                feedback_text=feedback_text,
            )
            return JSONResponse(content=result)

        @app.get("/v1/sandbox/pending")
        async def sandbox_pending(
            request: Request,
            session_id: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_approvals is None:
                raise HTTPException(status_code=503, detail="sandbox approvals are not enabled")
            return JSONResponse(content={
                "pending": owner._sandbox_approvals.list_pending(session_id=session_id.strip()),
            })

        @app.get("/v1/sandbox/whitelist")
        async def sandbox_whitelist_get(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_whitelist is None:
                raise HTTPException(status_code=503, detail="sandbox whitelist is not enabled")
            return JSONResponse(content=owner._sandbox_whitelist.snapshot())

        @app.post("/v1/sandbox/whitelist")
        async def sandbox_whitelist_add(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_whitelist is None:
                raise HTTPException(status_code=503, detail="sandbox whitelist is not enabled")
            body = await request.json()
            target_type = str(body.get("type", "") or "").strip().lower()
            value = str(body.get("value", "") or "").strip()
            if target_type not in {"command", "path"} or not value:
                raise HTTPException(status_code=400, detail="type must be command/path and value is required")
            changed = (
                owner._sandbox_whitelist.add_command(value)
                if target_type == "command"
                else owner._sandbox_whitelist.add_path(value)
            )
            owner._append_sandbox_audit(
                "whitelist_updated",
                session_id="",
                action="allow",
                target_type=target_type,
                value=value,
                changed=changed,
            )
            return JSONResponse(content={"ok": True, "changed": changed, **owner._sandbox_whitelist.snapshot()})

        @app.delete("/v1/sandbox/whitelist")
        async def sandbox_whitelist_remove(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_whitelist is None:
                raise HTTPException(status_code=503, detail="sandbox whitelist is not enabled")
            body = await request.json()
            target_type = str(body.get("type", "") or "").strip().lower()
            value = str(body.get("value", "") or "").strip()
            if target_type not in {"command", "path"} or not value:
                raise HTTPException(status_code=400, detail="type must be command/path and value is required")
            changed = (
                owner._sandbox_whitelist.remove_command(value)
                if target_type == "command"
                else owner._sandbox_whitelist.remove_path(value)
            )
            owner._append_sandbox_audit(
                "whitelist_updated",
                session_id="",
                action="unallow",
                target_type=target_type,
                value=value,
                changed=changed,
            )
            return JSONResponse(content={"ok": True, "changed": changed, **owner._sandbox_whitelist.snapshot()})

        @app.post("/v1/sandbox/approve")
        async def sandbox_approve(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_approvals is None:
                raise HTTPException(status_code=503, detail="sandbox approvals are not enabled")
            body = await request.json()
            session_id = str(body.get("session_id", "") or "").strip()
            approval_id = str(body.get("approval_id", "") or "").strip()
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            record = owner._sandbox_approvals.approve(session_id, approval_id)
            if record is None:
                raise HTTPException(status_code=404, detail="pending approval not found for this session")
            owner._append_sandbox_audit(
                "approval_approved",
                session_id=session_id,
                approval_id=record.get("approval_id", ""),
                tool_calls=record.get("tool_calls", []),
            )
            return JSONResponse(content={
                "ok": True,
                "approval_id": record.get("approval_id", ""),
                "status": record.get("status", ""),
                "tool_calls": record.get("tool_calls", []),
            })

        @app.post("/v1/sandbox/reject")
        async def sandbox_reject(
            request: Request,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if owner._sandbox_approvals is None:
                raise HTTPException(status_code=503, detail="sandbox approvals are not enabled")
            body = await request.json()
            session_id = str(body.get("session_id", "") or "").strip()
            approval_id = str(body.get("approval_id", "") or "").strip()
            if not session_id:
                raise HTTPException(status_code=400, detail="session_id is required")
            record = owner._sandbox_approvals.reject(session_id, approval_id)
            if record is None:
                raise HTTPException(status_code=404, detail="pending approval not found for this session")
            owner._append_sandbox_audit(
                "approval_rejected",
                session_id=session_id,
                approval_id=record.get("approval_id", ""),
                tool_calls=record.get("tool_calls", []),
            )
            return JSONResponse(content={
                "ok": True,
                "approval_id": record.get("approval_id", ""),
                "status": record.get("status", ""),
            })

        @app.get("/v1/session/report")
        async def get_session_report(
            request: Request,
            session_id: str,
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            report = owner._load_latest_session_report(session_id.strip())
            if report is None:
                raise HTTPException(status_code=404, detail="session report not found")
            return JSONResponse(content=report)

        @app.get("/v1/task-brief")
        async def get_task_brief(
            request: Request,
            session_id: str,
            scope: str = "",
            authorization: Optional[str] = Header(default=None),
        ):
            owner: MetaClawAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            key = owner._profile_key(session_id.strip(), scope.strip())
            return JSONResponse(content={
                "session_id": session_id,
                "scope": scope,
                "task_brief": owner._task_briefs.get(key, {}),
            })

        return app

    async def _check_auth(self, authorization: Optional[str]):
        if not self._expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self._expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # ------------------------------------------------------------------ #
    # Record helpers                                                       #
    # ------------------------------------------------------------------ #

    def _flush_pending_record(self, session_id: str, next_state):
        """Write out the buffered record for *session_id* and fire PRM."""
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        rec["next_state_role"] = next_state.get("role", "") if isinstance(next_state, dict) else ""
        rec["next_state_text"] = (
            _flatten_message_content(next_state.get("content", "")) if isinstance(next_state, dict) else ""
        )
        if next_state:
            ns_role = next_state.get("role", "?")
            ns_content = _flatten_message_content(next_state.get("content"))
            logger.info(
                f"{_GREEN}[OpenClaw] session={session_id} turn={rec['turn']} "
                f"next_state role={ns_role} len={len(ns_content)}: "
                f"{ns_content[:200]}{_RESET}"
            )
            self._fire_prm_scoring(
                session_id,
                rec["turn"],
                rec["response_text"],
                rec.get("instruction_text", ""),
                next_state,
            )
        self._append_sandbox_audit(
            "record_flushed",
            session_id=session_id,
            turn=rec.get("turn", 0),
            next_state_role=rec.get("next_state_role", ""),
            attempted_tool_calls=(rec.get("sandbox", {}) or {}).get("attempted_tool_calls", rec.get("tool_calls")),
            sandbox=(rec.get("sandbox", {}) or {}),
        )
        if self._enriched_record_file:
            try:
                with open(self._enriched_record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw] failed to write enriched record: %s", e)
        if self._openclaw_rl_record_file:
            rl_rec = {
                "session_id": rec["session_id"],
                "turn": rec["turn"],
                "timestamp": rec["timestamp"],
                "messages": rec["messages"],
                "prompt_text": rec["prompt_text"],
                "response_text": rec["response_text"],
                "tool_calls": rec["tool_calls"],
                "next_state": next_state,
            }
            try:
                with open(self._openclaw_rl_record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw] failed to write OpenClaw-RL record: %s", e)

    def _append_feedback_record(self, record: dict[str, Any]) -> None:
        if not self._feedback_file:
            return
        try:
            with open(self._feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[OpenClaw] failed to write feedback record: %s", e)

    def _load_user_profiles(self) -> dict[str, list[str]]:
        path = _resolve_record_path(self.config.record_dir, self.config.user_profile_path)
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            normalized: dict[str, list[str]] = {}
            for key, value in data.items():
                if isinstance(value, list):
                    normalized[str(key)] = [str(v).strip() for v in value if str(v).strip()]
            return normalized
        except Exception:
            return {}

    def _save_user_profiles(self) -> None:
        path = _resolve_record_path(self.config.record_dir, self.config.user_profile_path)
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._user_profiles, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("[OpenClaw] failed to write user profile file: %s", e)

    def _load_task_briefs(self) -> dict[str, dict[str, list[str]]]:
        path = _resolve_record_path(self.config.record_dir, self.config.task_brief_path)
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            normalized: dict[str, dict[str, list[str]]] = {}
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                item: dict[str, list[str]] = {}
                for field in ("goal", "constraints", "success_criteria", "style_preferences", "open_questions"):
                    raw_list = value.get(field, [])
                    if isinstance(raw_list, list):
                        item[field] = [str(v).strip() for v in raw_list if str(v).strip()]
                normalized[str(key)] = item
            return normalized
        except Exception:
            return {}

    def _save_task_briefs(self) -> None:
        path = _resolve_record_path(self.config.record_dir, self.config.task_brief_path)
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._task_briefs, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("[OpenClaw] failed to write task brief file: %s", e)

    def _append_session_report(self, report: dict[str, Any]) -> None:
        if not self._session_report_file:
            return
        try:
            with open(self._session_report_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(report, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[OpenClaw] failed to write session report: %s", e)

    def _load_latest_session_report(self, session_id: str) -> dict[str, Any] | None:
        if session_id in self._latest_session_reports:
            return self._latest_session_reports[session_id]
        if not self._session_report_file or not os.path.exists(self._session_report_file):
            return None
        latest_match: dict[str, Any] | None = None
        try:
            with open(self._session_report_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get("session_id") == session_id:
                        latest_match = item
        except OSError:
            return None
        return latest_match

    def _load_record_for_feedback(self, session_id: str, turn: int | None) -> dict[str, Any] | None:
        pending = self._pending_records.get(session_id)
        if pending is not None and (turn is None or int(pending.get("turn", 0) or 0) == turn):
            return dict(pending)

        if not self._enriched_record_file or not os.path.exists(self._enriched_record_file):
            return None

        latest_match: dict[str, Any] | None = None
        try:
            with open(self._enriched_record_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get("session_id") != session_id:
                        continue
                    if turn is not None and int(item.get("turn", 0) or 0) != turn:
                        continue
                    latest_match = item
        except OSError:
            return None
        return latest_match

    def _build_feedback_ack_response(
        self,
        session_id: str,
        message: str,
        model: str = "",
    ) -> dict[str, Any]:
        return {
            "response": {
                "id": f"chatcmpl-feedback-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model or self._served_model,
                "session_id": session_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        }

    def _build_assistant_response(
        self,
        session_id: str,
        message: str,
        model: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = {
            "response": {
                "id": f"chatcmpl-metaclaw-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model or self._served_model,
                "session_id": session_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message,
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        }
        if tool_calls:
            response["response"]["choices"][0]["message"]["tool_calls"] = tool_calls
            response["response"]["choices"][0]["finish_reason"] = "tool_calls"
        if extra:
            response["response"].update(extra)
        return response

    def _append_sandbox_audit(self, event_type: str, **payload: Any) -> None:
        if self._sandbox_audit is None:
            return
        self._sandbox_audit.append(event_type, **payload)

    def _attach_pending_record_metadata(self, session_id: str, **payload: Any) -> None:
        pending = self._pending_records.get(session_id)
        if pending is None:
            return
        sandbox = dict(pending.get("sandbox", {}) or {})
        sandbox.update(payload)
        pending["sandbox"] = sandbox

    @staticmethod
    def _summarize_sandbox_decisions(decisions: list[SandboxDecision]) -> str:
        lines: list[str] = []
        for item in decisions:
            lines.append(
                f"- {item.tool_name} ({item.risk_level}): {item.action} | {item.reason}"
            )
        return "\n".join(lines)

    async def _handle_inline_approval(
        self,
        session_id: str,
        messages: list[dict],
        model: str,
        action: str,
        approval_id: str,
    ) -> dict[str, Any]:
        if self._sandbox_approvals is None:
            raise HTTPException(status_code=503, detail="sandbox approvals are not enabled")
        raw_text = _extract_last_user_instruction(messages)
        if session_id in self._pending_records:
            self._flush_pending_record(session_id, {"role": "user", "content": raw_text})
        if action == "approve":
            record = self._sandbox_approvals.approve(session_id, approval_id)
            if record is None:
                raise HTTPException(status_code=404, detail="pending approval not found for this session")
            tool_calls = record.get("tool_calls", [])
            assistant_message = record.get("assistant_message", {}) or {}
            content = str(assistant_message.get("content", "") or "")
            self._append_sandbox_audit(
                "approval_approved",
                session_id=session_id,
                approval_id=record.get("approval_id", ""),
                tool_calls=tool_calls,
            )
            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]
            prompt_text = "\n".join(
                f"{m.get('role', '?')}: {_flatten_message_content(m.get('content', ''))}"
                for m in messages
            )
            response_text = content or (json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "")
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            self._attach_pending_record_metadata(
                session_id,
                approval_id=record.get("approval_id", ""),
                approval_status="approved",
                approval_source="inline_command",
            )
            return self._build_assistant_response(
                session_id=session_id,
                message=content,
                model=model,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                extra={"approval_id": record.get("approval_id", "")},
            )
        record = self._sandbox_approvals.reject(session_id, approval_id)
        if record is None:
            raise HTTPException(status_code=404, detail="pending approval not found for this session")
        self._append_sandbox_audit(
            "approval_rejected",
            session_id=session_id,
            approval_id=record.get("approval_id", ""),
            tool_calls=record.get("tool_calls", []),
        )
        return self._build_assistant_response(
            session_id=session_id,
            message=f"Approval rejected for {record.get('approval_id', '')}. The blocked tool call will not run.",
            model=model,
            extra={"approval_id": record.get("approval_id", "")},
        )

    def _format_whitelist_snapshot(self) -> str:
        snapshot = self._sandbox_whitelist.snapshot() if self._sandbox_whitelist else {
            "command_allowlist": [],
            "path_allowlist": [],
        }
        commands = snapshot.get("command_allowlist", [])
        paths = snapshot.get("path_allowlist", [])
        lines = ["Sandbox whitelist:"]
        lines.append("Commands:")
        if commands:
            lines.extend(f"- {item}" for item in commands)
        else:
            lines.append("- (empty)")
        lines.append("Paths:")
        if paths:
            lines.extend(f"- {item}" for item in paths)
        else:
            lines.append("- (empty)")
        return "\n".join(lines)

    async def _handle_inline_whitelist(
        self,
        session_id: str,
        messages: list[dict],
        model: str,
        action: str,
        target_type: str,
        value: str,
    ) -> dict[str, Any]:
        if self._sandbox_whitelist is None:
            raise HTTPException(status_code=503, detail="sandbox whitelist is not enabled")
        raw_text = _extract_last_user_instruction(messages)
        if session_id in self._pending_records:
            self._flush_pending_record(session_id, {"role": "user", "content": raw_text})
        if action == "list":
            return self._build_assistant_response(
                session_id=session_id,
                message=self._format_whitelist_snapshot(),
                model=model,
            )
        if target_type not in {"command", "path"} or not value.strip():
            raise HTTPException(status_code=400, detail="invalid whitelist command")
        changed = False
        normalized_value = value.strip()
        if action == "allow":
            if target_type == "command":
                changed = self._sandbox_whitelist.add_command(normalized_value)
            else:
                changed = self._sandbox_whitelist.add_path(normalized_value)
            verb = "added to"
        else:
            if target_type == "command":
                changed = self._sandbox_whitelist.remove_command(normalized_value)
            else:
                changed = self._sandbox_whitelist.remove_path(normalized_value)
            verb = "removed from"
        self._append_sandbox_audit(
            "whitelist_updated",
            session_id=session_id,
            action=action,
            target_type=target_type,
            value=normalized_value,
            changed=changed,
        )
        if changed:
            msg = f"{target_type.capitalize()} {verb} sandbox whitelist: {normalized_value}"
        else:
            msg = f"No change. {target_type.capitalize()} already in desired whitelist state: {normalized_value}"
        msg += "\n\n" + self._format_whitelist_snapshot()
        return self._build_assistant_response(
            session_id=session_id,
            message=msg,
            model=model,
        )

    def _maybe_gate_tool_calls(
        self,
        session_id: str,
        model: str,
        assistant_msg: dict[str, Any],
        tool_calls: list[dict],
    ) -> tuple[dict[str, Any], list[dict], list[SandboxDecision], bool]:
        if not tool_calls or self._sandbox_policy is None or not self.config.sandbox_enabled:
            return assistant_msg, tool_calls, [], False
        decisions = self._sandbox_policy.evaluate_tool_calls(tool_calls)
        self._append_sandbox_audit(
            "tool_calls_evaluated",
            session_id=session_id,
            tool_calls=tool_calls,
            decisions=[decision.__dict__ for decision in decisions],
        )
        has_denied = any(decision.action == "deny" for decision in decisions)
        has_pending = any(decision.action == "require_approval" for decision in decisions)
        if has_denied:
            block_text = (
                "Blocked by MetaClaw sandbox policy.\n"
                + self._summarize_sandbox_decisions(decisions)
            )
            blocked_msg = {
                "role": "assistant",
                "content": block_text,
            }
            self._append_sandbox_audit(
                "tool_calls_blocked",
                session_id=session_id,
                tool_calls=tool_calls,
                decisions=[decision.__dict__ for decision in decisions],
            )
            return blocked_msg, [], decisions, True
        if has_pending and self._sandbox_approvals is not None:
            approval = self._sandbox_approvals.create_pending(
                session_id=session_id,
                tool_calls=tool_calls,
                assistant_message=assistant_msg,
                decisions=decisions,
            )
            pending_text = (
                "这个工具在调用前需要获得你的允许.\n"
                f"Approval ID: {approval['approval_id']}\n"
                "使用 `/approve` 进行批准，或使用 `/reject` 进行拒绝.\n"
                "如果有多个待处理的批准，你也可以使用 `/approve <approval_id>` 或 `/reject <approval_id>`.\n"
                + self._summarize_sandbox_decisions(decisions)
            )
            pending_msg = {
                "role": "assistant",
                "content": pending_text,
            }
            self._append_sandbox_audit(
                "tool_calls_pending_approval",
                session_id=session_id,
                approval_id=approval["approval_id"],
                tool_calls=tool_calls,
                decisions=[decision.__dict__ for decision in decisions],
            )
            return pending_msg, [], decisions, True
        return assistant_msg, tool_calls, decisions, False

    def _buffer_record(self, session_id: str, turn_num: int, messages: list,
                       prompt_text: str, response_text: str, tool_calls: list):
        if not self._record_file:
            return
        instruction_text = _extract_last_user_instruction(messages)
        context_summary = _extract_context_summary_text(messages)
        injected_skills = list(self._latest_injected_skills.get(session_id, []))
        brief_key = self._task_brief_key(session_id, self._session_memory_scopes.get(session_id, ""))
        task_brief = self._task_briefs.get(brief_key, {})
        self._pending_records[session_id] = {
            "schema": "metaclaw.conversation_record.v2",
            "source": "metaclaw",
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
            "context_summary": context_summary,
            "context_summary_used": bool(context_summary),
            "injected_skills": injected_skills,
            "task_brief": task_brief,
        }

    def _append_prm_record(self, session_id: str, turn_num: int,
                           score: float, votes: list):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "session_id": session_id,
                    "turn": turn_num,
                    "score": score,
                    "votes": votes,
                }, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[OpenClaw] failed to write PRM record: %s", e)

    def purge_record_files(self):
        """Clear all record JSONL files. Called when training starts."""
        for path, label in [
            (self._enriched_record_file, "record"),
            (self._openclaw_rl_record_file, "OpenClaw-RL record"),
            (self._prm_record_file, "PRM record"),
            (self._session_report_file, "session report"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw] failed to purge %s file: %s", label, e)

    # ------------------------------------------------------------------ #
    # PRM scoring                                                          #
    # ------------------------------------------------------------------ #

    def _fire_prm_scoring(self, session_id: str, turn_num: int,
                          response_text: str, instruction_text: str, next_state):
        if not self.prm_scorer or not next_state:
            return
        inst_text = instruction_text or ""
        task = asyncio.create_task(
            self.prm_scorer.evaluate(
                response_text, inst_text, session_id=session_id, turn_num=turn_num
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(
            lambda _t: self._maybe_submit_ready_samples(session_id)
        )
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    # ------------------------------------------------------------------ #
    # OPD teacher logprobs                                                 #
    # ------------------------------------------------------------------ #

    async def _query_teacher_logprobs(
        self, prompt_text: str, response_text: str, num_response_tokens: int
    ) -> list[float]:
        """Query teacher model for per-token logprobs on the student's response.

        Uses the OpenAI-compatible ``/v1/completions`` endpoint with ``echo=True``
        and ``max_tokens=0`` to obtain the teacher's log-probabilities for each
        token of the student's generated response without producing new tokens.
        """
        full_text = prompt_text + response_text

        response = await asyncio.to_thread(
            self._teacher_client.completions.create,
            model=self.config.teacher_model,
            prompt=full_text,
            echo=True,
            logprobs=1,
            max_tokens=0,
        )

        choice = response.choices[0]
        token_logprobs = choice.logprobs.token_logprobs or []

        # Find where response tokens start (use student tokenizer as reference)
        prompt_token_count = len(
            self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

        teacher_lps = [
            float(lp) if lp is not None else 0.0
            for lp in token_logprobs[prompt_token_count:]
        ]

        # Align to student's response token count
        if len(teacher_lps) > num_response_tokens:
            teacher_lps = teacher_lps[:num_response_tokens]
        elif len(teacher_lps) < num_response_tokens:
            teacher_lps = teacher_lps + [0.0] * (num_response_tokens - len(teacher_lps))

        return teacher_lps

    def _fire_teacher_query(
        self, session_id: str, turn_num: int,
        prompt_text: str, response_text: str, num_response_tokens: int,
    ):
        """Fire an async teacher-logprobs query (OPD mode)."""
        task = asyncio.create_task(
            self._query_teacher_logprobs(prompt_text, response_text, num_response_tokens)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(
            lambda _t: self._maybe_submit_ready_samples(session_id)
        )
        self._teacher_tasks.setdefault(session_id, {})[turn_num] = task

    # ------------------------------------------------------------------ #
    # Request handling                                                     #
    # ------------------------------------------------------------------ #

    def _read_cached_system_prompt(self) -> str:
        try:
            with open(self._system_prompt_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached = data.get("compressed_system_prompt", "")
            return cached if isinstance(cached, str) else ""
        except Exception:
            return ""

    def _write_cached_system_prompt(self, prompt: str):
        try:
            with open(self._system_prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump({"compressed_system_prompt": prompt}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("[OpenClaw] failed to write system prompt cache: %s", e)

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
        memory_scope: str = "",
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        self._latest_injected_skills.pop(session_id, None)
        latest_user_text = _extract_last_user_instruction(messages)
        inline_whitelist = _parse_inline_whitelist(latest_user_text) if self.config.sandbox_enabled else None
        if inline_whitelist is not None:
            action, target_type, value = inline_whitelist
            logger.info(
                "[Sandbox] intercepted whitelist command session=%s action=%s target=%s value=%s",
                session_id,
                action,
                target_type or "<none>",
                value[:200],
            )
            return await self._handle_inline_whitelist(
                session_id=session_id,
                messages=messages,
                model=str(body.get("model", "") or self._served_model),
                action=action,
                target_type=target_type,
                value=value,
            )
        inline_approval = _parse_inline_approval(latest_user_text) if self.config.sandbox_approval_enabled else None
        if inline_approval is not None:
            action, approval_id = inline_approval
            logger.info(
                "[Sandbox] intercepted inline %s session=%s approval_id=%s",
                action,
                session_id,
                approval_id or "<latest>",
            )
            return await self._handle_inline_approval(
                session_id=session_id,
                messages=messages,
                model=str(body.get("model", "") or self._served_model),
                action=action,
                approval_id=approval_id,
            )
        inline_feedback = _find_inline_feedback(messages) if self.config.feedback_enabled else None
        if inline_feedback is not None:
            rating, feedback_text, feedback_raw_text = inline_feedback
            if session_id in self._pending_records:
                self._flush_pending_record(
                    session_id,
                    {"role": "user", "content": feedback_raw_text},
                )
            logger.info(
                "[Feedback] intercepted inline feedback session=%s rating=%s text=%s",
                session_id,
                rating,
                feedback_text[:200],
            )
            await self._handle_feedback(
                session_id=session_id,
                turn=None,
                rating=rating,
                feedback_text=feedback_text,
            )
            ack = (
                "Feedback recorded."
                if rating == "good"
                else "Negative feedback recorded; appended a lesson to important-notes."
            )
            return self._build_feedback_ack_response(
                session_id=session_id,
                message=ack,
                model=str(body.get("model", "") or self._served_model),
            )
        rewritten = 0
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg.get("content", "").startswith("A new chat session just started.")
            ):
                rewritten += 1
        if rewritten:
            logger.info("[OpenClaw] rewrote %d /new bootstrap user prompt(s) for provider safety", rewritten)

        def _prompt_len(msgs):
            try:
                norm_msgs = _normalize_messages_for_template(msgs)
                text = self._tokenizer.apply_chat_template(
                    norm_msgs, tools=body.get("tools"), tokenize=False, add_generation_prompt=True,
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return 0

        cached_system = ""
        # NOTE: In skills_only mode we forward directly to the user's LLM provider.
        # Do not rewrite/collapse the system prompt here.
        if self.config.mode != "skills_only":
            cached_system = self._read_cached_system_prompt()
            if not cached_system:
                raw_system = ""
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        raw_system = _flatten_message_content(m.get("content"))
                        break
                if raw_system:
                    cached_system = await asyncio.to_thread(
                        run_llm,
                        [{"role": "user", "content": raw_system}],
                    )
                    cached_system = (cached_system or raw_system).strip()
                    self._write_cached_system_prompt(cached_system)

            if cached_system:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        m["content"] = cached_system

        tools = _normalize_tools_for_template(body.get("tools"))

        effective_memory_scope = memory_scope or self._get_memory_scope(session_id)
        if effective_memory_scope:
            self._session_memory_scopes[session_id] = effective_memory_scope
        self._update_user_profile(session_id, effective_memory_scope, latest_user_text)
        await self._update_task_brief(session_id, effective_memory_scope, latest_user_text)

        # Inject memory and skills into system message for main turns
        if turn_type == "main":
            if self.config.task_brief_enabled:
                messages = self._inject_task_brief(messages, session_id, effective_memory_scope)
            if self.config.user_profile_enabled:
                messages = self._inject_user_profile(messages, session_id, effective_memory_scope)
            if self.config.feedback_enabled:
                messages = self._inject_important_skill(messages)
            if (
                self.memory_manager
                and self.skill_manager
                and self.config.synergy_enabled
            ):
                messages = await self._inject_augmentation(
                    messages, scope_id=effective_memory_scope, session_id=session_id,
                )
            elif self.memory_manager:
                messages = await self._inject_memory(messages, scope_id=effective_memory_scope)
            elif self.skill_manager:
                messages = self._inject_skills(messages, session_id=session_id)
        if cached_system:
            logger.info(
                "[OpenClaw] system prompt cached len=%d",
                _prompt_len([{"role": "system", "content": cached_system}]),
            )

        # Truncate to fit within max_context_tokens (keep system + most-recent messages)
        max_prompt = self.config.max_context_tokens - int(body.get("max_tokens") or 2048)
        if max_prompt > 0:
            messages = await self._compress_messages_with_summary(
                session_id, messages, tools, max_prompt,
            )

        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self._served_model
        forward_body["messages"] = _ensure_reasoning_content(messages)
        self._append_sandbox_audit(
            "pre_llm_forward",
            session_id=session_id,
            turn_type=turn_type,
            model=str(forward_body.get("model", "") or self._served_model),
            tool_count=len(forward_body.get("tools") or []),
            message_count=len(forward_body.get("messages") or []),
        )

        if self.config.mode == "skills_only":
            output = await self._forward_to_llm(forward_body)
        else:
            output = await self._forward_to_tinker(forward_body)

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        original_tool_calls = list(tool_calls)
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning") or ""
        sandbox_decisions: list[SandboxDecision] = []
        sandbox_intercepted = False
        if tool_calls:
            assistant_msg, tool_calls, sandbox_decisions, sandbox_intercepted = self._maybe_gate_tool_calls(
                session_id=session_id,
                model=str(output.get("model", "") or self._served_model),
                assistant_msg=dict(assistant_msg),
                tool_calls=tool_calls,
            )
            choice["message"] = assistant_msg
            choice["finish_reason"] = "tool_calls" if tool_calls else "stop"
            content = assistant_msg.get("content") or ""
            reasoning = assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning") or ""

        logger.info(
            f"{_YELLOW}[OpenClaw] [{turn_type}] session={session_id} "
            f"prompt_msgs={len(messages)}{_RESET}"
        )
        logger.info(
            f"{_RED}[OpenClaw] [{turn_type}] session={session_id} "
            f"thinking={len(reasoning)} chars, response:\n{content}{_RESET}"
        )
        if tool_calls:
            logger.info("[OpenClaw] tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])
        elif sandbox_intercepted and original_tool_calls:
            logger.info(
                "[Sandbox] intercepted tool_calls session=%s original=%s",
                session_id,
                json.dumps(original_tool_calls, ensure_ascii=False)[:500],
            )

        if turn_type == "main":
            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            if self._tokenizer is None:
                # skills_only mode: record conversation and buffer for skill evolution
                self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
                turn_num = self._turn_counts[session_id]
                prompt_text_simple = "\n".join(
                    f"{m.get('role', '?')}: {_flatten_message_content(m.get('content', ''))}"
                    for m in messages
                )
                response_text_simple = content or (
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else ""
                )
                self._buffer_record(
                    session_id, turn_num, messages,
                    prompt_text_simple, response_text_simple, tool_calls,
                )
                if sandbox_decisions:
                    self._attach_pending_record_metadata(
                        session_id,
                        decisions=[decision.__dict__ for decision in sandbox_decisions],
                        intercepted=sandbox_intercepted,
                        attempted_tool_calls=original_tool_calls,
                    )
                turn_entry = {
                    "prompt_text": prompt_text_simple,
                    "response_text": response_text_simple,
                }
                evolution_every_n = getattr(self.config, "skill_evolution_every_n_turns", 10)
                _want_evolution = self.skill_evolver and self.config.enable_skill_evolution and evolution_every_n > 0
                _want_memory = self.memory_manager is not None
                if _want_evolution:
                    self._session_turns.setdefault(session_id, []).append(turn_entry)
                    buf = self._session_turns.get(session_id, [])
                    if len(buf) >= evolution_every_n:
                        evolution_turns = list(buf)
                        self._session_turns[session_id] = []
                        self._safe_create_task(self._evolve_skills_for_session(evolution_turns))
                if _want_memory:
                    self._session_memory_turns.setdefault(session_id, []).append(turn_entry)
                # session_done handling for skills_only path (tokenizer unavailable)
                if session_done and not self.config.memory_manual_trigger:
                    if self.config.session_report_enabled:
                        self._safe_create_task(self._generate_session_report(session_id, effective_memory_scope))
                    self._flush_pending_record(session_id, None)
                    self._maybe_submit_ready_samples(session_id, force_no_prm=True)
                    eff = self._session_effective.pop(session_id, 0)
                    self._turn_counts.pop(session_id, None)
                    self._teacher_tasks.pop(session_id, None)
                    self._session_context_summaries.pop(session_id, None)
                    self._latest_injected_skills.pop(session_id, None)
                    logger.info("[OpenClaw] session=%s done → cleaned up (effective_samples=%d)", session_id, eff)
                    memory_turns = self._session_memory_turns.pop(session_id, [])
                    if memory_turns and self.memory_manager is not None and self.config.memory_auto_extract:
                        self._safe_create_task(
                            self._ingest_memory_for_session(
                                session_id,
                                memory_turns,
                                self._session_memory_scopes.pop(session_id, ""),
                            )
                        )
                    else:
                        self._session_memory_scopes.pop(session_id, None)
                    evolution_turns = self._session_turns.pop(session_id, [])
                    if evolution_turns and self.skill_evolver and self.config.enable_skill_evolution:
                        self._safe_create_task(self._evolve_skills_for_session(evolution_turns))
                elif session_done and self.config.memory_manual_trigger:
                    if self.config.session_report_enabled:
                        self._safe_create_task(self._generate_session_report(session_id, effective_memory_scope))
                    self._flush_pending_record(session_id, None)
                    self._maybe_submit_ready_samples(session_id, force_no_prm=True)
                    self._session_effective.pop(session_id, 0)
                    self._turn_counts.pop(session_id, None)
                    self._teacher_tasks.pop(session_id, None)
                    self._session_context_summaries.pop(session_id, None)
                    self._latest_injected_skills.pop(session_id, None)
                    logger.info("[OpenClaw] session=%s done (manual_trigger: memory buffer preserved, %d turns)",
                                session_id, len(self._session_memory_turns.get(session_id, [])))
                output["session_id"] = session_id
                return {"response": output}

            prompt_text = self._tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
            )
            full_text = self._tokenizer.apply_chat_template(
                full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
            )

            if full_text.startswith(prompt_text):
                response_text = full_text[len(prompt_text):]
            else:
                logger.warning("[OpenClaw] prompt_text not prefix of full_text, using full_text as response")
                response_text = full_text

            prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self._tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip() and not tool_calls:
                logger.info("[OpenClaw] MAIN session=%s → empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
            }

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]

            logger.info(
                "[OpenClaw] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, len(prompt_ids), len(response_ids),
            )
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            if sandbox_decisions:
                self._attach_pending_record_metadata(
                    session_id,
                    decisions=[decision.__dict__ for decision in sandbox_decisions],
                    intercepted=sandbox_intercepted,
                    attempted_tool_calls=original_tool_calls,
                )
            # Skill evolution + memory: buffer turns for all modes (RL and skills_only).
            turn_entry = {"prompt_text": prompt_text, "response_text": response_text}
            evolution_every_n = getattr(self.config, "skill_evolution_every_n_turns", 10)
            _want_evolution = self.skill_evolver and self.config.enable_skill_evolution and evolution_every_n > 0
            _want_memory = self.memory_manager is not None
            if _want_evolution:
                self._session_turns.setdefault(session_id, []).append(turn_entry)
                buf = self._session_turns.get(session_id, [])
                if len(buf) >= evolution_every_n:
                    evolution_turns = list(buf)
                    self._session_turns[session_id] = []
                    self._safe_create_task(self._evolve_skills_for_session(evolution_turns))
            if _want_memory:
                self._session_memory_turns.setdefault(session_id, []).append(turn_entry)
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            if self.config.use_opd and self._teacher_client:
                self._fire_teacher_query(
                    session_id, turn_num, prompt_text, response_text, len(response_ids),
                )
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[OpenClaw] SIDE session=%s → skipped (no training data)", session_id)
            # When ignore_turn_type is enabled, buffer side turns for memory too.
            if self.config.memory_ignore_turn_type and self.memory_manager is not None:
                prompt_text_side = "\n".join(
                    f"{m.get('role', '?')}: {_flatten_message_content(m.get('content', ''))}"
                    for m in messages
                )
                response_text_side = content or (
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else ""
                )
                self._session_memory_turns.setdefault(session_id, []).append({
                    "prompt_text": prompt_text_side,
                    "response_text": response_text_side,
                })

        if session_done and not self.config.memory_manual_trigger:
            if self.config.session_report_enabled:
                self._safe_create_task(self._generate_session_report(session_id, effective_memory_scope))
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_no_prm=True)
            eff = self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            self._teacher_tasks.pop(session_id, None)
            self._session_context_summaries.pop(session_id, None)
            self._latest_injected_skills.pop(session_id, None)
            logger.info("[OpenClaw] session=%s done → cleaned up (effective_samples=%d)", session_id, eff)
            # session done: trigger memory ingestion from dedicated memory buffer
            memory_turns = self._session_memory_turns.pop(session_id, [])
            if memory_turns and self.memory_manager is not None and self.config.memory_auto_extract:
                self._safe_create_task(
                    self._ingest_memory_for_session(
                        session_id,
                        memory_turns,
                        self._session_memory_scopes.pop(session_id, ""),
                    )
                )
            else:
                self._session_memory_scopes.pop(session_id, None)
            # session done: trigger skill evolution from skill buffer
            evolution_turns = self._session_turns.pop(session_id, [])
            if evolution_turns and self.skill_evolver and self.config.enable_skill_evolution:
                self._safe_create_task(self._evolve_skills_for_session(evolution_turns))
        elif session_done and self.config.memory_manual_trigger:
            if self.config.session_report_enabled:
                self._safe_create_task(self._generate_session_report(session_id, effective_memory_scope))
            # manual_trigger mode: clean up non-memory state, keep memory buffer for manual ingest
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_no_prm=True)
            self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            self._teacher_tasks.pop(session_id, None)
            self._session_context_summaries.pop(session_id, None)
            self._latest_injected_skills.pop(session_id, None)
            logger.info("[OpenClaw] session=%s done (manual_trigger: memory buffer preserved, %d turns)",
                        session_id, len(self._session_memory_turns.get(session_id, [])))

        output["session_id"] = session_id
        return {"response": output}

    # ------------------------------------------------------------------ #
    # Tinker forwarding                                                    #
    # ------------------------------------------------------------------ #

    async def _forward_to_tinker(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward the request to Tinker via SamplingClient.sample_async."""
        import tinker

        if self._sampling_client is None:
            raise HTTPException(status_code=503, detail="no Tinker sampling client available")
        if self._tokenizer is None:
            raise HTTPException(status_code=503, detail="no tokenizer available")

        try:
            messages = body.get("messages", [])
            norm_msgs = _normalize_messages_for_template(messages)
            tools = body.get("tools")
            temperature = float(body.get("temperature", 0.7))
            max_tokens = int(body.get("max_tokens") or 2048)
            stop = body.get("stop")

            logger.info("[OpenClaw] _forward_to_tinker msgs=%d max_tokens=%d", len(norm_msgs), max_tokens)

            # Apply chat template using the full conversation history.
            # Use tokenize=False then encode() to always get a plain list of ints.
            prompt_text = self._tokenizer.apply_chat_template(
                norm_msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)

            # Build Tinker ModelInput
            chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
            model_input = tinker.ModelInput(chunks=[chunk])

            # Build SamplingParams
            sp_kwargs: dict[str, Any] = dict(
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=50,
                top_p=0.95,
            )
            if stop is not None:
                sp_kwargs["stop"] = stop
            sampling_params = tinker.SamplingParams(**sp_kwargs)

            # Call Tinker
            response = await self._sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
                include_prompt_logprobs=False,
                topk_prompt_logprobs=0,
            )

            # Decode response tokens → text
            seq = response.sequences[0]
            response_text = self._tokenizer.decode(seq.tokens, skip_special_tokens=True)
            normalized_text, parsed_tool_calls, reasoning = _extract_tool_calls_from_text(response_text)
            logprobs_list = seq.logprobs or []
            if parsed_tool_calls:
                logger.info(
                    "[OpenClaw] parsed tool_calls after extract: %s",
                    json.dumps(parsed_tool_calls, ensure_ascii=False)[:800],
                )
            logger.info(
                "[OpenClaw] Tinker tokens=%d stop=%s decoded=%r",
                len(seq.tokens), seq.stop_reason, response_text[:200],
            )

            # Build OpenAI-compatible response
            lp_content = [
                {"token": "", "logprob": float(lp), "top_logprobs": []}
                for lp in logprobs_list
            ]
            assistant_message: dict[str, Any] = {"role": "assistant", "content": normalized_text}
            if reasoning:
                assistant_message["reasoning_content"] = reasoning
            if parsed_tool_calls:
                assistant_message["tool_calls"] = parsed_tool_calls
            return {
                "id": f"chatcmpl-tinker-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", self._served_model),
                "choices": [{
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "tool_calls" if parsed_tool_calls else (seq.stop_reason or "stop"),
                    "logprobs": {"content": lp_content},
                }],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(seq.tokens),
                    "total_tokens": len(prompt_ids) + len(seq.tokens),
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("[OpenClaw] Tinker sample_async failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Tinker inference error: {e}") from e

    # ------------------------------------------------------------------ #
    # LLM forwarding (skills_only mode)                                   #
    # ------------------------------------------------------------------ #

    async def _forward_to_llm(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward to a real OpenAI-compatible API (skills_only mode)."""
        import httpx

        api_base = self.config.llm_api_base.rstrip("/")
        if not api_base:
            raise HTTPException(
                status_code=503,
                detail="llm_api_base is not configured. Run 'metaclaw setup' first.",
            )

        # Strip Tinker-specific fields not supported by standard OpenAI APIs
        send_body = {
            k: v for k, v in body.items()
            if k not in {"logprobs", "top_logprobs", "stream_options"}
        }
        send_body["model"] = self.config.llm_model_id or body.get("model", "")
        send_body["stream"] = False

        headers: dict[str, str] = {}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        # OpenRouter requires HTTP-Referer and X-Title for free-tier model access
        if "openrouter.ai" in api_base:
            headers.setdefault("HTTP-Referer", "https://github.com/aiming-lab/MetaClaw")
            headers.setdefault("X-Title", "MetaClaw")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json=send_body,
                    headers=headers,
                )
                resp.raise_for_status()
                result = resp.json()

            # Robustness: if the upstream API returns tool calls / reasoning
            # inlined in content text instead of structured fields, parse them.
            for choice in result.get("choices", []):
                msg = choice.get("message")
                if not msg or msg.get("role") != "assistant":
                    continue
                content = msg.get("content") or ""
                has_tool_calls = bool(msg.get("tool_calls"))
                has_reasoning = bool(msg.get("reasoning_content"))
                if not content or (has_tool_calls and has_reasoning):
                    continue  # already fully structured
                cleaned, parsed_tools, parsed_reasoning = _extract_tool_calls_from_text(content)
                if parsed_tools or parsed_reasoning:
                    msg["content"] = cleaned
                    if parsed_reasoning and not has_reasoning:
                        msg["reasoning_content"] = parsed_reasoning
                    if parsed_tools and not has_tool_calls:
                        msg["tool_calls"] = parsed_tools
                        choice["finish_reason"] = "tool_calls"

            return result
        except httpx.HTTPStatusError as e:
            upstream_status = e.response.status_code
            upstream_body = e.response.text[:500]
            logger.error("[OpenClaw] upstream LLM error: %s %s", upstream_status, upstream_body[:200])
            # Pass through 4xx client errors so callers see the real cause
            # (e.g. 401 invalid API key, 429 rate limited). Upstream 5xx become 502.
            http_status = upstream_status if 400 <= upstream_status < 500 else 502
            raise HTTPException(status_code=http_status, detail=upstream_body) from e
        except Exception as e:
            logger.error("[OpenClaw] LLM forward failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"LLM forward error: {e}") from e

    # ------------------------------------------------------------------ #
    # Skill evolution (skills_only mode)                                  #
    # ------------------------------------------------------------------ #

    async def _evolve_skills_for_session(self, turns: list[dict]):
        """Analyze session turns and generate new skills (skills_only mode)."""
        from types import SimpleNamespace

        samples = [
            SimpleNamespace(
                prompt_text=t["prompt_text"],
                response_text=t["response_text"],
                reward=0.0,  # no RL reward; evolver analyzes patterns
            )
            for t in turns
        ]
        existing = self.skill_manager.skills if self.skill_manager else {}
        try:
            new_skills = await self.skill_evolver.evolve(samples, existing)
        except Exception as e:
            logger.error("[SkillEvolver] evolve failed: %s", e, exc_info=True)
            return
        if new_skills and self.skill_manager:
            added = 0
            for skill in new_skills:
                category = skill.get("category", "general")
                added += self.skill_manager.add_skills([skill], category=category)
            logger.info("[SkillEvolver] session analysis added %d new skills", added)

    # ------------------------------------------------------------------ #
    # Feedback-driven important skill                                      #
    # ------------------------------------------------------------------ #

    def _get_important_skill(self) -> dict[str, Any] | None:
        if not self.skill_manager:
            return None
        target_name = self.config.important_feedback_skill_name.strip()
        if not target_name:
            return None
        for bucket_name in ("general_skills", "common_mistakes"):
            for skill in self.skill_manager.skills.get(bucket_name, []):
                if skill.get("name", "").strip() == target_name:
                    return skill
        for bucket in self.skill_manager.skills.get("task_specific_skills", {}).values():
            for skill in bucket:
                if skill.get("name", "").strip() == target_name:
                    return skill
        return None

    def _build_default_important_skill(self) -> dict[str, Any]:
        return {
            "name": self.config.important_feedback_skill_name,
            "description": self.config.important_feedback_skill_description,
            "category": "general",
            "content": (
                "## Important Notes\n\n"
                "### When to Use\n"
                "Use on every task before taking action.\n\n"
                "- Re-check the latest user intent before acting.\n"
                "- Prefer verification over assumption when a mistake would compound.\n"
                "- When execution has multiple steps, confirm the current sub-goal before moving on.\n"
                "- Surface uncertainty early instead of hiding it.\n\n"
                "## Feedback Lessons\n\n"
                "### Seed Lesson\n"
                "- When this failure happens: the agent is about to act on an uncertain interpretation.\n"
                "- Why the previous execution was bad: it moved ahead before re-checking the exact target.\n"
                "- What the agent should do instead: restate the sub-goal and verify the risky assumption first.\n"
                "- Anti-pattern: charging ahead after signals of ambiguity or user dissatisfaction.\n"
            ),
        }

    def _ensure_important_skill_exists(self) -> dict[str, Any]:
        important = self._get_important_skill()
        if important is not None:
            return important
        important = self._build_default_important_skill()
        if self.skill_manager is not None:
            self.skill_manager.upsert_skill(important)
            reloaded = self._get_important_skill()
            if reloaded is not None:
                return reloaded
        return important

    def _merge_important_skill(self, skills: list[dict]) -> list[dict]:
        important = self._get_important_skill()
        if important is None:
            important = self._build_default_important_skill()
        existing_names = {s.get("name", "").strip() for s in skills if isinstance(s, dict)}
        if important.get("name", "").strip() in existing_names:
            return skills
        return [important] + list(skills)

    def _inject_important_skill(self, messages: list[dict]) -> list[dict]:
        important = self._ensure_important_skill_exists()
        skill_text = self.skill_manager.format_for_conversation([important]) if self.skill_manager else (
            "## Active Skills\n\n### "
            + important.get("name", "important-notes")
            + "\n_"
            + important.get("description", "")
            + "_\n\n"
            + important.get("content", "")
        )
        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            if important.get("name", "") in existing:
                return messages
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + skill_text}
        else:
            messages.insert(0, {"role": "system", "content": skill_text})
        return messages

    def _task_brief_key(self, session_id: str, scope_id: str = "") -> str:
        return self._profile_key(session_id, scope_id)

    async def _update_task_brief(self, session_id: str, scope_id: str, latest_user_text: str) -> None:
        if not self.config.task_brief_enabled or not latest_user_text:
            return
        key = self._task_brief_key(session_id, scope_id)
        existing = self._task_briefs.get(key, {})
        prompt_parts = [
            f"Latest user instruction:\n{latest_user_text}",
            f"Existing task brief:\n{json.dumps(existing, ensure_ascii=False)}",
        ]
        try:
            raw = await asyncio.to_thread(
                run_task_brief_llm,
                [{"role": "user", "content": "\n\n".join(prompt_parts)}],
                self.config.task_brief_api_key,
                self.config.task_brief_api_base,
                self.config.task_brief_model_id,
                self.config.task_brief_max_completion_tokens,
            )
            raw = (raw or "").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            payload = json.loads(raw[start:end])
        except Exception as e:
            logger.warning("[TaskBrief] failed to update task brief for %s: %s", session_id, e)
            return

        normalized: dict[str, list[str]] = {}
        for field in ("goal", "constraints", "success_criteria", "style_preferences", "open_questions"):
            value = payload.get(field, [])
            if isinstance(value, list):
                normalized[field] = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str) and value.strip():
                normalized[field] = [value.strip()]
        if normalized:
            self._task_briefs[key] = normalized
            self._save_task_briefs()

    def _inject_task_brief(self, messages: list[dict], session_id: str, scope_id: str) -> list[dict]:
        if not self.config.task_brief_enabled:
            return messages
        key = self._task_brief_key(session_id, scope_id)
        brief = self._task_briefs.get(key, {})
        if not brief:
            return messages

        lines = ["## Task Brief"]
        field_titles = {
            "goal": "Goal",
            "constraints": "Constraints",
            "success_criteria": "Success Criteria",
            "style_preferences": "Style Preferences",
            "open_questions": "Open Questions",
        }
        for field in ("goal", "constraints", "success_criteria", "style_preferences", "open_questions"):
            items = brief.get(field, [])
            if not items:
                continue
            lines.append(f"\n### {field_titles[field]}")
            lines.extend(f"- {item}" for item in items)
        brief_block = "\n".join(lines)

        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            if "## Task Brief" not in existing:
                messages[idx] = {**messages[idx], "content": existing + "\n\n" + brief_block}
        else:
            messages.insert(0, {"role": "system", "content": brief_block})
        return messages

    def _profile_key(self, session_id: str, scope_id: str = "") -> str:
        return base_scope(scope_id) if scope_id else session_id

    def _update_user_profile(self, session_id: str, scope_id: str, latest_user_text: str) -> None:
        if not self.config.user_profile_enabled or not latest_user_text:
            return
        key = self._profile_key(session_id, scope_id)
        clauses = _extract_preference_clauses(latest_user_text)
        if not clauses:
            return
        existing = self._user_profiles.setdefault(key, [])
        changed = False
        for clause in clauses:
            if clause not in existing:
                existing.append(clause)
                changed = True
        if len(existing) > self.config.user_profile_max_entries:
            del existing[:-self.config.user_profile_max_entries]
            changed = True
        if changed:
            self._save_user_profiles()

    def _inject_user_profile(self, messages: list[dict], session_id: str, scope_id: str) -> list[dict]:
        if not self.config.user_profile_enabled:
            return messages
        key = self._profile_key(session_id, scope_id)
        entries = self._user_profiles.get(key, [])
        if not entries:
            return messages
        profile_block = "## User Preferences\n" + "\n".join(f"- {item}" for item in entries[-self.config.user_profile_max_entries:])
        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            if "## User Preferences" not in existing:
                messages[idx] = {**messages[idx], "content": existing + "\n\n" + profile_block}
        else:
            messages.insert(0, {"role": "system", "content": profile_block})
        return messages

    def _load_session_records(self, session_id: str, max_records: int = 50) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self._enriched_record_file and os.path.exists(self._enriched_record_file):
            try:
                with open(self._enriched_record_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if item.get("session_id") == session_id:
                            records.append(item)
            except OSError:
                pass
        pending = self._pending_records.get(session_id)
        if pending is not None:
            records.append(dict(pending))
        records.sort(key=lambda x: int(x.get("turn", 0) or 0))
        return records[-max_records:]

    async def _generate_session_report(self, session_id: str, scope_id: str = "") -> None:
        if not self.config.session_report_enabled:
            return
        records = self._load_session_records(session_id)
        if not records:
            return
        key = self._task_brief_key(session_id, scope_id)
        task_brief = self._task_briefs.get(key, {})
        feedback_history = self._feedback_by_session.get(session_id, [])
        sandbox_events = self._sandbox_audit.read_recent(session_id=session_id, limit=100) if self._sandbox_audit else []
        pending_approvals = self._sandbox_approvals.list_pending(session_id=session_id) if self._sandbox_approvals else []
        injected_skills = sorted({
            skill
            for rec in records
            for skill in (rec.get("injected_skills") or [])
            if isinstance(skill, str) and skill
        })
        report_prompt = {
            "session_id": session_id,
            "task_brief": task_brief,
            "feedback_history": feedback_history[-10:],
            "sandbox_events": sandbox_events[-20:],
            "pending_approvals": len(pending_approvals),
            "injected_skills": injected_skills,
            "records": [
                {
                    "turn": rec.get("turn"),
                    "instruction_text": rec.get("instruction_text", ""),
                    "response_text": rec.get("response_text", "")[:3000],
                    "next_state_text": rec.get("next_state_text", "")[:1500],
                    "context_summary": rec.get("context_summary", ""),
                }
                for rec in records
            ],
        }
        try:
            raw = await asyncio.to_thread(
                run_session_report_llm,
                [{"role": "user", "content": json.dumps(report_prompt, ensure_ascii=False)}],
                self.config.session_report_api_key,
                self.config.session_report_api_base,
                self.config.session_report_model_id,
                self.config.session_report_max_completion_tokens,
            )
            raw = (raw or "").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            payload = json.loads(raw[start:end])
        except Exception as e:
            logger.warning("[SessionReport] failed to build session report for %s: %s", session_id, e)
            return

        report = {
            "session_id": session_id,
            "scope_id": scope_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turn_count": len(records),
            "task_brief": task_brief,
            "feedback_count": len(feedback_history),
            "pending_approval_count": len(pending_approvals),
            "injected_skills": injected_skills,
            "summary": str(payload.get("summary", "") or ""),
            "wins": payload.get("wins", []) if isinstance(payload.get("wins", []), list) else [],
            "issues": payload.get("issues", []) if isinstance(payload.get("issues", []), list) else [],
            "reusable_lessons": payload.get("reusable_lessons", []) if isinstance(payload.get("reusable_lessons", []), list) else [],
            "suggested_next_step": str(payload.get("suggested_next_step", "") or ""),
            "success_estimate": str(payload.get("success_estimate", "") or "medium"),
        }
        self._latest_session_reports[session_id] = report
        self._append_session_report(report)

    async def _update_important_skill_from_feedback(
        self,
        record: dict[str, Any],
        rating: str,
        feedback_text: str,
    ) -> dict[str, Any]:
        if not self.skill_manager:
            raise HTTPException(status_code=503, detail="skills are not enabled")
        existing_skill = self._get_important_skill() or self._build_default_important_skill()
        if rating != "bad":
            return existing_skill
        prompt_sections = [
            f"Feedback rating: {rating}",
            f"User feedback:\n{feedback_text or '(no free-form feedback provided)'}",
            f"Existing important skill description:\n{existing_skill.get('description', '')}",
            f"Existing important skill content:\n{existing_skill.get('content', '')}",
            f"Conversation messages:\n{_render_context_messages(record.get('messages', []) or [], max_chars=8000)}",
            f"Task instruction:\n{record.get('instruction_text', '')}",
            f"Prompt text:\n{record.get('prompt_text', '')[:6000]}",
            f"Assistant response:\n{record.get('response_text', '')[:4000]}",
            f"Next state:\n{record.get('next_state_text', '')[:2000]}",
        ]
        raw = await asyncio.to_thread(
            run_feedback_skill_llm,
            [{"role": "user", "content": "\n\n".join(prompt_sections)}],
            self.config.feedback_skill_api_key,
            self.config.feedback_skill_api_base,
            self.config.feedback_skill_model_id,
            self.config.feedback_skill_max_completion_tokens,
        )
        raw = (raw or "").strip()
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            payload = json.loads(raw[start:end])
        except Exception as e:
            logger.error("[Feedback] failed to parse feedback skill JSON: %s | raw=%s", e, raw[:500])
            raise HTTPException(status_code=502, detail="feedback skill synthesis failed") from e

        append_entry = str(payload.get("append_entry", "") or "").strip()
        if not append_entry:
            raise HTTPException(status_code=502, detail="feedback skill synthesis returned empty lesson")
        existing_content = str(existing_skill.get("content", "") or "").rstrip()
        if "## Feedback Lessons" not in existing_content:
            existing_content = existing_content.rstrip() + "\n\n## Feedback Lessons"
        updated_content = existing_content.rstrip() + "\n\n" + append_entry
        updated_skill = {
            "name": self.config.important_feedback_skill_name,
            "description": existing_skill.get("description", "") or self.config.important_feedback_skill_description,
            "category": "general",
            "content": updated_content,
        }
        changed = self.skill_manager.upsert_skill(updated_skill)
        if changed:
            self.skill_manager.reload()
        return updated_skill

    async def _handle_feedback(
        self,
        session_id: str,
        turn: int | None,
        rating: str,
        feedback_text: str,
    ) -> dict[str, Any]:
        record = self._load_record_for_feedback(session_id, turn)
        if record is None:
            raise HTTPException(status_code=404, detail="target turn record not found")

        turn_num = int(record.get("turn", 0) or 0)
        feedback_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "turn": turn_num,
            "rating": rating,
            "feedback": feedback_text,
            "instruction_text": record.get("instruction_text", ""),
            "injected_skills": record.get("injected_skills", []),
        }
        self._feedback_by_session.setdefault(session_id, []).append(feedback_record)
        self._append_feedback_record(feedback_record)
        if self.skill_manager:
            self.skill_manager.record_feedback(record.get("injected_skills", []) or [], rating)

        updated_skill = None
        if rating == "bad":
            updated_skill = await self._update_important_skill_from_feedback(
                record=record,
                rating=rating,
                feedback_text=feedback_text,
            )
            logger.info(
                "[Feedback] session=%s turn=%d rating=%s -> appended lesson to skill=%s",
                session_id,
                turn_num,
                rating,
                updated_skill.get("name", ""),
            )
        else:
            logger.info(
                "[Feedback] session=%s turn=%d rating=%s -> recorded only",
                session_id,
                turn_num,
                rating,
            )
        return {
            "ok": True,
            "session_id": session_id,
            "turn": turn_num,
            "rating": rating,
            "skill_updated": rating == "bad",
            "skill_name": updated_skill.get("name", "") if updated_skill else "",
            "skill_description": updated_skill.get("description", "") if updated_skill else "",
            "skill_content": updated_skill.get("content", "") if updated_skill else "",
        }

    # ------------------------------------------------------------------ #
    # Context summarization                                                #
    # ------------------------------------------------------------------ #

    def _estimate_prompt_tokens(self, messages: list[dict], tools=None) -> int:
        try:
            norm_msgs = _normalize_messages_for_template(messages)
            if self._tokenizer is not None:
                text = self._tokenizer.apply_chat_template(
                    norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            fallback_text = json.dumps(norm_msgs, ensure_ascii=False)
            return max(1, len(fallback_text) // 4)
        except Exception:
            try:
                fallback_text = json.dumps(messages, ensure_ascii=False)
                return max(1, len(fallback_text) // 4)
            except Exception:
                return 0

    def _build_summary_system_block(self, summary_text: str) -> str:
        return (
            "## Conversation Summary\n"
            "Use this summary as compressed history for earlier turns that were omitted.\n"
            f"{summary_text.strip()}"
        )

    def _with_context_summary(
        self,
        sys_msgs: list[dict],
        kept_non_sys: list[dict],
        summary_text: str,
    ) -> list[dict]:
        summary_block = self._build_summary_system_block(summary_text)
        base = list(sys_msgs)
        if base:
            existing = _flatten_message_content(base[0].get("content", "")).strip()
            merged = f"{existing}\n\n{summary_block}" if existing else summary_block
            base[0] = {**base[0], "content": merged}
        else:
            base = [{"role": "system", "content": summary_block}]
        return base + list(kept_non_sys)

    async def _summarize_dropped_context(
        self,
        session_id: str,
        dropped_messages: list[dict],
        latest_instruction: str,
    ) -> str:
        existing_summary = self._session_context_summaries.get(session_id, "").strip()
        rendered_messages = _render_context_messages(
            dropped_messages,
            max_chars=max(2000, int(self.config.context_summary_max_chars * 2)),
        )
        if not rendered_messages and existing_summary:
            return existing_summary
        prompt_parts = []
        if latest_instruction:
            prompt_parts.append(f"Latest user instruction:\n{latest_instruction}")
        if existing_summary:
            prompt_parts.append(f"Existing compressed summary:\n{existing_summary}")
        prompt_parts.append(f"Older conversation turns to compress:\n{rendered_messages}")
        prompt_parts.append(
            f"Keep the final summary under {self.config.context_summary_max_chars} characters.",
        )
        try:
            summary = await asyncio.to_thread(
                run_context_summary_llm,
                [{"role": "user", "content": "\n\n".join(prompt_parts)}],
                self.config.context_summary_api_key,
                self.config.context_summary_api_base,
                self.config.context_summary_model_id,
                self.config.context_summary_max_completion_tokens,
            )
        except Exception as e:
            logger.warning("[OpenClaw] context summarization failed for session=%s: %s", session_id, e)
            return existing_summary
        summary = (summary or "").strip()
        if len(summary) > self.config.context_summary_max_chars:
            summary = summary[: self.config.context_summary_max_chars].rstrip()
        if summary:
            self._session_context_summaries[session_id] = summary
        return summary or existing_summary

    async def _compress_messages_with_summary(
        self,
        session_id: str,
        messages: list[dict],
        tools,
        max_prompt_tokens: int,
    ) -> list[dict]:
        current_tokens = self._estimate_prompt_tokens(messages, tools)
        if current_tokens <= max_prompt_tokens:
            return messages
        if not self.config.context_summary_enabled:
            return self._truncate_messages(messages, tools, max_prompt_tokens)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        non_sys = [m for m in messages if m.get("role") != "system"]
        if len(non_sys) <= 1:
            return self._truncate_messages(messages, tools, max_prompt_tokens)

        recent_keep = max(1, min(len(non_sys), self.config.context_summary_recent_messages))
        dropped = non_sys[:-recent_keep]
        if not dropped:
            return self._truncate_messages(messages, tools, max_prompt_tokens)

        summary_text = await self._summarize_dropped_context(
            session_id,
            dropped,
            _extract_last_user_instruction(messages),
        )
        if not summary_text:
            return self._truncate_messages(messages, tools, max_prompt_tokens)

        kept: list[dict] = []
        for msg in reversed(non_sys):
            candidate_kept = [msg] + list(reversed(kept))
            candidate = self._with_context_summary(sys_msgs, candidate_kept, summary_text)
            if self._estimate_prompt_tokens(candidate, tools) <= max_prompt_tokens:
                kept.append(msg)
            elif not kept:
                kept.append(msg)
                break
            else:
                break

        summarized = self._with_context_summary(sys_msgs, list(reversed(kept)), summary_text)
        if self._estimate_prompt_tokens(summarized, tools) > max_prompt_tokens:
            fallback_summary = summary_text[: max(512, self.config.context_summary_max_chars // 2)].rstrip()
            summarized = self._with_context_summary(sys_msgs, list(reversed(kept)), fallback_summary)
            if self._estimate_prompt_tokens(summarized, tools) > max_prompt_tokens:
                summarized = self._truncate_messages(summarized, tools, max_prompt_tokens)

        logger.info(
            "[OpenClaw] context summarized for session=%s tokens=%d→%d limit=%d",
            session_id,
            current_tokens,
            self._estimate_prompt_tokens(summarized, tools),
            max_prompt_tokens,
        )
        return summarized

    # ------------------------------------------------------------------ #
    # Skill injection                                                      #
    # ------------------------------------------------------------------ #

    def _truncate_messages(
        self,
        messages: list[dict],
        tools,
        max_prompt_tokens: int,
    ) -> list[dict]:
        """
        Drop oldest non-system messages until the tokenized prompt fits within
        max_prompt_tokens.  The system message (if any) is always kept.
        At least one user message is always kept even if it alone exceeds the limit.
        """
        if self._tokenizer is None:
            return messages

        def _prompt_len(msgs):
            try:
                norm_msgs = _normalize_messages_for_template(msgs)
                text = self._tokenizer.apply_chat_template(
                    norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return 0

        if _prompt_len(messages) <= max_prompt_tokens:
            return messages

        # Split into system and non-system messages
        sys_msgs = [m for m in messages if m.get("role") == "system"]
        non_sys = [m for m in messages if m.get("role") != "system"]

        # Greedily keep most-recent messages
        kept = []
        for msg in reversed(non_sys):
            candidate = sys_msgs + list(reversed(kept + [msg]))
            if _prompt_len(candidate) <= max_prompt_tokens:
                kept.append(msg)
            elif not kept:
                kept.append(msg)  # keep at least one user message
                break
            else:
                break

        result = sys_msgs + list(reversed(kept))
        dropped = len(messages) - len(result)
        if dropped > 0:
            logger.warning(
                "[OpenClaw] context truncated: dropped %d oldest messages "
                "(%d → %d tokens, limit=%d)",
                dropped,
                _prompt_len(messages),
                _prompt_len(result),
                max_prompt_tokens,
            )
        return result

    def _inject_skills(self, messages: list[dict], session_id: str = "") -> list[dict]:
        """Prepend skill guidance to the system message."""
        if not self.skill_manager:
            return messages

        user_msgs = [m for m in messages if m.get("role") == "user"]
        task_desc = _flatten_message_content(user_msgs[-1].get("content", "")) if user_msgs else ""
        if not task_desc:
            return messages

        skills = self.skill_manager.retrieve(task_desc, top_k=self.config.skill_top_k)
        if not skills:
            return messages

        skill_names = [
            s.get("name", s.get("id", "unknown_skill"))
            for s in skills
            if isinstance(s, dict)
        ]
        self.skill_manager.record_skill_selection(skill_names)
        logger.info(
            "[SkillManager] injecting %d skills: %s",
            len(skill_names),
            ", ".join(skill_names)[:400],
        )

        skill_text = self.skill_manager.format_for_conversation(skills)
        messages = list(messages)

        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + skill_text}
        else:
            messages.insert(0, {"role": "system", "content": skill_text})

        if session_id:
            self._latest_injected_skills[session_id] = skill_names
        return messages

    # ------------------------------------------------------------------ #
    # Sample submission                                                    #
    # ------------------------------------------------------------------ #

    def _maybe_submit_ready_samples(
        self, session_id: str, force_no_prm: bool = False
    ):
        """Submit turns whose PRM and teacher queries are done.

        force_no_prm: also submit turns that have no PRM task yet (used at
        session end for the last turn which will never get a next_state).
        When force is active, pending teacher tasks are also skipped.
        """
        prm_tasks = self._prm_tasks.get(session_id, {})
        teacher_tasks = self._teacher_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            # --- PRM readiness ---
            prm_task = prm_tasks.get(turn_num)
            if not self.config.use_prm or not self.prm_scorer:
                pass  # no PRM → submit immediately
            elif prm_task is not None and not prm_task.done():
                continue  # PRM still running
            elif prm_task is None and not force_no_prm:
                continue  # waiting for next_state to fire PRM

            # --- Teacher readiness (OPD) ---
            teacher_task = teacher_tasks.get(turn_num)
            if (self.config.use_opd and teacher_task is not None
                    and not teacher_task.done() and not force_no_prm):
                continue  # teacher logprobs still running

            turn_data = pending.pop(turn_num)
            prm_result = None
            if prm_task is not None and prm_task.done():
                try:
                    prm_result = prm_task.result()
                except Exception:
                    pass
                prm_tasks.pop(turn_num, None)

            teacher_logprobs = None
            if teacher_task is not None and teacher_task.done():
                try:
                    teacher_logprobs = teacher_task.result()
                except Exception:
                    pass
                teacher_tasks.pop(turn_num, None)

            self._safe_create_task(
                self._submit_turn_sample(turn_data, session_id, prm_result, teacher_logprobs)
            )

    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        prm_result: Optional[dict],
        teacher_logprobs: Optional[list[float]] = None,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        has_next_state = turn_data.get("has_next_state", False)
        score = prm_result["score"] if prm_result else 0.0

        exclude = not has_next_state or score == 0.0
        # Guarantee at least one sample per session contributes to training.
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info(
                "[OpenClaw] promoting session=%s turn with score=0 → loss_mask=1 (at-least-one guarantee)",
                session_id,
            )

        loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample = ConversationSample(
            session_id=session_id,
            turn_num=self._turn_counts.get(session_id, 0),
            prompt_tokens=prompt_ids,
            response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=loss_mask,
            reward=score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=teacher_logprobs,
            # Tag with current skill generation so the trainer can discard
            # pre-evolution samples (MAML support/query set separation).
            skill_generation=self.skill_manager.generation if self.skill_manager else 0,
        )

        if not exclude:
            self._session_effective[session_id] = (
                self._session_effective.get(session_id, 0) + 1
            )

        index = next(self._index_counter)
        group_index = next(self._group_counter)

        if prm_result:
            self._append_prm_record(
                session_id, sample.turn_num, score, prm_result.get("votes", [])
            )

        logger.info(
            "[OpenClaw] submitted sample session=%s index=%d score=%.1f exclude=%s "
            "prompt_len=%d response_len=%d",
            session_id, index, score, exclude, len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        cfg = uvicorn.Config(
            self.app,
            host=self.config.proxy_host,
            port=self.config.proxy_port,
            log_level="info",
        )
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        threading.Thread(target=self._print_ready_banner, daemon=True).start()

    def _print_ready_banner(self):
        time.sleep(3)
        backend = "Tinker cloud RL" if self.config.mode in ("rl", "auto") else f"LLM ({self.config.llm_model_id or 'upstream'})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  MetaClaw proxy ready  [mode={self.config.mode}]\n"
            f"  proxy {self.config.proxy_host}:{self.config.proxy_port} → {backend}\n"
            f"  OpenClaw has been configured to use this proxy automatically.\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    # Tinker-specific interface                                            #
    # ------------------------------------------------------------------ #

    def update_sampling_client(self, new_client):
        """Hot-swap the Tinker sampling client after a weight update."""
        self._sampling_client = new_client
        logger.info("[OpenClaw] sampling client updated")

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw] background task failed: %s", exc, exc_info=exc)

    # ------------------------------------------------------------------ #
    # Memory injection & ingestion                                         #
    # ------------------------------------------------------------------ #

    async def _ingest_memory_for_session(
        self,
        session_id: str,
        turns: list[dict],
        scope_id: str = "",
    ):
        """Persist simple phase-1 memory units from the completed session.

        Memories are stored under the *base* scope (session suffix stripped)
        so they are retrievable across sessions.
        """
        if self.memory_manager is None:
            return
        # Strip session suffix so memories are shared across sessions.
        ingest_scope = base_scope(scope_id) if scope_id else None
        try:
            added = await asyncio.to_thread(
                self.memory_manager.ingest_session_turns,
                session_id,
                turns,
                ingest_scope,
            )
            stats = self.memory_manager.get_scope_stats(ingest_scope)
            logger.info(
                "[Memory] session=%s scope=%s added %d memory units active=%d by_type=%s",
                session_id,
                ingest_scope or self.memory_manager.scope_id,
                added,
                stats.get("active", 0),
                stats.get("active_by_type", {}),
            )
        except Exception as e:
            logger.error("[Memory] ingest failed for session=%s: %s", session_id, e, exc_info=True)

    async def _inject_memory(self, messages: list[dict], scope_id: str = "") -> list[dict]:
        """Prepend relevant long-term memory to the system message.

        Retrieves from the base scope (session suffix stripped) so memories
        from previous sessions are visible.  Runs blocking SQLite queries in
        a thread to avoid blocking the async event loop.
        """
        if not self.memory_manager:
            return messages

        user_msgs = [m for m in messages if m.get("role") == "user"]
        task_desc = _flatten_message_content(user_msgs[-1].get("content", "")) if user_msgs else ""
        if not task_desc:
            return messages

        # Use base scope for retrieval so memories are shared across sessions
        retrieval_scope = base_scope(scope_id) if scope_id else None
        memories = await asyncio.to_thread(
            self.memory_manager.retrieve_for_prompt, task_desc, scope_id=retrieval_scope,
        )
        if not memories:
            logger.info("[Memory] no memories retrieved, skipping injection for task=%s", task_desc[:80])
            return messages

        memory_text = self.memory_manager.render_for_prompt(memories)
        logger.info(
            "[Memory] injecting %d memories (~%d tokens) for task=%s",
            len(memories),
            len(memory_text.split()),
            task_desc[:120],
        )
        messages = list(messages)

        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + memory_text}
        else:
            messages.insert(0, {"role": "system", "content": memory_text})

        return messages

    def _get_memory_scope(self, session_id: str) -> str:
        if not self.memory_manager:
            return ""
        scope = self._session_memory_scopes.get(session_id, "").strip()
        if scope:
            return scope
        return self.memory_manager.scope_id

    # ------------------------------------------------------------------ #
    # Skill-Memory Synergy                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synergy_tokenize(text: str) -> set[str]:
        """Lightweight tokenizer for Jaccard overlap computation."""
        tokens: list[str] = []
        buf: list[str] = []
        for ch in text.lower():
            if ch.isalnum() or ch in {"_", "-"}:
                buf.append(ch)
            elif buf:
                tok = "".join(buf)
                if len(tok) >= 2:
                    tokens.append(tok)
                buf = []
        if buf:
            tok = "".join(buf)
            if len(tok) >= 2:
                tokens.append(tok)
        return set(tokens)

    def _dedup_memory_against_skills(
        self,
        memories: list,
        skills: list[dict],
        threshold: float = 0.5,
    ) -> list:
        """Remove PROCEDURAL_OBSERVATION memories that overlap heavily with skills.

        Uses Jaccard keyword overlap so it works without any embedding model.
        """
        if not skills or not memories:
            return memories

        # Build a combined term set from all active skills.
        skill_terms: set[str] = set()
        for s in skills:
            skill_terms |= self._synergy_tokenize(
                s.get("content", "") + " " + s.get("description", "")
            )
        if not skill_terms:
            return memories

        filtered: list = []
        dedup_count = 0
        for mem in memories:
            if mem.memory_type.value == "procedural_observation":
                mem_terms = self._synergy_tokenize(mem.content + " " + mem.summary)
                union = mem_terms | skill_terms
                overlap = len(mem_terms & skill_terms) / max(len(union), 1)
                if overlap > threshold:
                    dedup_count += 1
                    logger.info(
                        "[Synergy] dedup procedural memory (overlap=%.2f): %s",
                        overlap,
                        mem.content[:80],
                    )
                    continue
            filtered.append(mem)

        if dedup_count:
            logger.info(
                "[Synergy] deduped %d procedural memories against %d skills",
                dedup_count,
                len(skills),
            )
        return filtered

    async def _inject_augmentation(
        self,
        messages: list[dict],
        scope_id: str = "",
        session_id: str = "",
    ) -> list[dict]:
        """Coordinated injection of both Memory and Skill.

        Replaces the separate _inject_memory + _inject_skills calls when both
        modules are enabled.  Key differences vs independent injection:

        1. Shared token budget (config.synergy_token_budget)
        2. Content dedup — procedural observations that overlap with skills are dropped
        3. Role-separated prompt template — LLM gets clear guidance on how to use each
        """
        if not self.memory_manager or not self.skill_manager:
            return messages

        user_msgs = [m for m in messages if m.get("role") == "user"]
        task_desc = (
            _flatten_message_content(user_msgs[-1].get("content", ""))
            if user_msgs
            else ""
        )
        if not task_desc:
            return messages

        # --- 1. Retrieve relevant skills (for template customization, not injection)
        skills = self.skill_manager.retrieve_relevant(
            task_desc, top_k=min(self.config.skill_top_k, 5),
        )
        skill_names = [
            s.get("name", s.get("id", "unknown_skill"))
            for s in skills
            if isinstance(s, dict)
        ]

        # Need >= 2 relevant skills for synergy template; otherwise plain memory.
        if len(skills) < 2:
            return await self._inject_memory(messages, scope_id=scope_id)

        # --- 2. Retrieve memories -------------------------------------------------
        retrieval_scope = base_scope(scope_id) if scope_id else None
        memories = await asyncio.to_thread(
            self.memory_manager.retrieve_for_prompt,
            task_desc,
            scope_id=retrieval_scope,
        )

        if not memories:
            return self._inject_skills(messages, session_id=session_id)

        # Dedup procedural memories that overlap with matched skills.
        memories = self._dedup_memory_against_skills(
            memories, skills, threshold=self.config.synergy_dedup_threshold,
        )

        memory_text = self.memory_manager.render_for_prompt(memories)
        if not memory_text:
            return self._inject_skills(messages, session_id=session_id)

        # --- 3. Build skill-aware structured template (no full skill injection) ---
        # Extract compact process steps from matched skills.
        import re as _re
        skill_steps = []
        for s in skills[:3]:
            content = s.get("content", "")
            bold_actions = _re.findall(r'\d+\.\s+\*\*([^*]+)\*\*', content)
            name = s.get("name", "").replace("-", " ")
            if bold_actions:
                steps = " → ".join(a.strip() for a in bold_actions[:5])
                skill_steps.append(f"{steps}")
            else:
                skill_steps.append(name)
        methodology = "; ".join(skill_steps)

        parts = [
            "## Augmented Context",
            "",
            f"Recommended approach: {methodology}.",
            "Use the project-specific experience below to inform your response.",
            "",
            "### Memories (Project-Specific Experience — WHAT worked/failed before)",
            "",
            memory_text,
        ]

        augmented_text = "\n".join(parts)

        # Log summary.
        logger.info(
            "[Synergy] injecting %d memories + %d skill hints (~%d tokens) for task=%s",
            len(memories),
            len(skills),
            len(augmented_text.split()),
            task_desc[:120],
        )

        # --- 4. Inject into system message ----------------------------------------
        messages = list(messages)
        sys_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        if sys_indices:
            idx = sys_indices[0]
            existing = _flatten_message_content(messages[idx].get("content", ""))
            messages[idx] = {**messages[idx], "content": existing + "\n\n" + augmented_text}
        else:
            messages.insert(0, {"role": "system", "content": augmented_text})

        if skill_names:
            self.skill_manager.record_skill_selection(skill_names)
            if session_id:
                self._latest_injected_skills[session_id] = skill_names

        return messages
