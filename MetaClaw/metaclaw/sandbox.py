from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _now_ts() -> float:
    return time.time()


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _normalize_path(value: str) -> str:
    raw = (value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    lowered = raw.lower()
    lowered = re.sub(r"/+", "/", lowered)
    return lowered


def _extract_words(text: str) -> list[str]:
    return [part for part in re.split(r"\s+", (text or "").strip()) if part]


def _normalize_command(value: str) -> str:
    return " ".join(_extract_words(value or ""))


@dataclass
class SandboxDecision:
    tool_call_id: str
    tool_name: str
    risk_level: str
    action: str
    reason: str
    command: str = ""
    paths: list[str] = field(default_factory=list)
    path_violations: list[str] = field(default_factory=list)
    args: dict[str, Any] = field(default_factory=dict)


class SandboxWhitelistManager:
    def __init__(self, state_path: str) -> None:
        self.state_path = state_path
        self._lock = threading.RLock()
        _ensure_parent(state_path)
        self._state = self._load_state()

    DEFAULT_COMMAND_MODE = "ask"

    def set_command_mode(self, command: str, mode: str) -> bool:
        normalized = _normalize_command(command)
        normalized_mode = str(mode or "").strip().lower()
        if not normalized or normalized_mode not in {"allow", "ask", "deny"}:
            return False
        with self._lock:
            rules = self._state.setdefault("command_rules", {})
            if rules.get(normalized) == normalized_mode:
                return False
            rules[normalized] = normalized_mode
            self._save_state()
        return True

    def remove_command_mode(self, command: str) -> bool:
        normalized = _normalize_command(command)
        if not normalized:
            return False
        with self._lock:
            rules = self._state.setdefault("command_rules", {})
            if normalized not in rules:
                return False
            rules.pop(normalized, None)
            self._save_state()
        return True

    def get_command_mode(self, command: str) -> str | None:
        normalized = _normalize_command(command)
        if not normalized:
            return None
        with self._lock:
            rules = self._state.get("command_rules", {})
            value = str(rules.get(normalized, "") or "").strip().lower()
            return value if value in {"allow", "ask", "deny"} else None

    def set_default_command_mode(self, mode: str) -> bool:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"allow", "ask", "deny"}:
            return False
        with self._lock:
            if self._state.get("default_command_mode") == normalized_mode:
                return False
            self._state["default_command_mode"] = normalized_mode
            self._save_state()
        return True

    def get_default_command_mode(self) -> str:
        with self._lock:
            raw = str(self._state.get("default_command_mode", self.DEFAULT_COMMAND_MODE) or "").strip().lower()
        return raw if raw in {"allow", "ask", "deny"} else self.DEFAULT_COMMAND_MODE

    def add_command(self, command: str) -> bool:
        normalized = _normalize_command(command)
        if not normalized:
            return False
        with self._lock:
            commands = self._state.setdefault("command_allowlist", [])
            if normalized in commands:
                return False
            commands.append(normalized)
            commands.sort()
            self._save_state()
        return True

    def remove_command(self, command: str) -> bool:
        normalized = _normalize_command(command)
        with self._lock:
            commands = self._state.setdefault("command_allowlist", [])
            if normalized not in commands:
                return False
            commands.remove(normalized)
            self._save_state()
        return True

    def add_path(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized:
            return False
        with self._lock:
            paths = self._state.setdefault("path_allowlist", [])
            if normalized in paths:
                return False
            paths.append(normalized)
            paths.sort()
            self._save_state()
        return True

    def remove_path(self, path: str) -> bool:
        normalized = _normalize_path(path)
        with self._lock:
            paths = self._state.setdefault("path_allowlist", [])
            if normalized not in paths:
                return False
            paths.remove(normalized)
            self._save_state()
        return True

    def add_blocked_path(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized:
            return False
        with self._lock:
            paths = self._state.setdefault("path_blocklist", [])
            if normalized in paths:
                return False
            paths.append(normalized)
            paths.sort()
            self._save_state()
        return True

    def remove_blocked_path(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized:
            return False
        with self._lock:
            paths = self._state.setdefault("path_blocklist", [])
            if normalized not in paths:
                return False
            paths.remove(normalized)
            self._save_state()
        return True

    def is_command_allowed(self, command: str) -> bool:
        normalized = _normalize_command(command)
        if not normalized:
            return False
        with self._lock:
            commands = self._state.get("command_allowlist", [])
            return normalized in commands

    def is_path_allowed(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized:
            return False
        with self._lock:
            entries = self._state.get("path_allowlist", [])
            for entry in entries:
                if normalized == entry or normalized.startswith(entry + "/"):
                    return True
        return False

    def is_path_blocked(self, path: str) -> bool:
        normalized = _normalize_path(path)
        if not normalized:
            return False
        with self._lock:
            entries = self._state.get("path_blocklist", [])
            for entry in entries:
                if normalized == entry or normalized.startswith(entry + "/"):
                    return True
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "command_allowlist": list(self._state.get("command_allowlist", [])),
                "path_allowlist": list(self._state.get("path_allowlist", [])),
                "command_rules": dict(self._state.get("command_rules", {})),
                "default_command_mode": self.get_default_command_mode(),
                "path_blocklist": list(self._state.get("path_blocklist", [])),
            }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path or not os.path.exists(self.state_path):
            return {
                "command_allowlist": [],
                "path_allowlist": [],
                "command_rules": {},
                "default_command_mode": self.DEFAULT_COMMAND_MODE,
                "path_blocklist": [],
            }
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                return {
                    "command_allowlist": [],
                    "path_allowlist": [],
                    "command_rules": {},
                    "default_command_mode": self.DEFAULT_COMMAND_MODE,
                    "path_blocklist": [],
                }
            command_rules = {}
            for key, value in dict(payload.get("command_rules", {}) or {}).items():
                normalized_key = _normalize_command(str(key))
                normalized_mode = str(value or "").strip().lower()
                if normalized_key and normalized_mode in {"allow", "ask", "deny"}:
                    command_rules[normalized_key] = normalized_mode
            default_mode = str(payload.get("default_command_mode", self.DEFAULT_COMMAND_MODE) or "").strip().lower()
            return {
                "command_allowlist": [
                    _normalize_command(item)
                    for item in payload.get("command_allowlist", [])
                    if _normalize_command(str(item))
                ],
                "path_allowlist": [
                    _normalize_path(str(item))
                    for item in payload.get("path_allowlist", [])
                    if _normalize_path(str(item))
                ],
                "command_rules": command_rules,
                "default_command_mode": default_mode if default_mode in {"allow", "ask", "deny"} else self.DEFAULT_COMMAND_MODE,
                "path_blocklist": [
                    _normalize_path(str(item))
                    for item in payload.get("path_blocklist", [])
                    if _normalize_path(str(item))
                ],
            }
        except Exception:
            return {
                "command_allowlist": [],
                "path_allowlist": [],
                "command_rules": {},
                "default_command_mode": self.DEFAULT_COMMAND_MODE,
                "path_blocklist": [],
            }

    def _save_state(self) -> None:
        if not self.state_path:
            return
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)


class PathPolicy:
    def __init__(self) -> None:
        self.read_roots = ("workspace", "./", ".", "scratch")
        self.write_roots = ("workspace", "./", ".", "scratch")
        self.blocked_markers = (
            "/etc",
            "/root",
            "/var/run/docker.sock",
            "/var/lib",
            "/sys",
            "/proc",
            "/windows/system32",
            "/program files",
            "/programdata",
            "/.ssh",
            "/id_rsa",
            "/id_ed25519",
            "/secrets",
            "/secret",
            "/token",
            "/tokens",
            "/credential",
            "/credentials",
            ".env",
        )
        self.approval_markers = (
            "..",
            "~",
            "/tmp",
            "/var",
            "/opt",
            "docker.sock",
        )
        self.write_keys = {
            "content",
            "text",
            "append",
            "overwrite",
            "recursive",
            "move",
            "destination",
            "dest",
            "dst",
            "target",
            "to_path",
            "new_path",
        }
        self.path_keys = {
            "path",
            "file_path",
            "filepath",
            "cwd",
            "target",
            "source",
            "src",
            "dst",
            "dest",
            "destination",
            "new_path",
            "old_path",
            "to_path",
            "from_path",
        }

    def inspect(
        self,
        tool_name: str,
        args: dict[str, Any],
        whitelist_manager: SandboxWhitelistManager | None = None,
    ) -> tuple[str, list[str], list[str]]:
        paths = self._collect_paths(args)
        if not paths:
            return "allow", [], []
        violations: list[str] = []
        approval_reasons: list[str] = []
        is_write = self._is_write_like(tool_name, args)
        allowed_roots = self.write_roots if is_write else self.read_roots
        for raw in paths:
            normalized = _normalize_path(raw)
            if not normalized:
                continue
            if whitelist_manager is not None and whitelist_manager.is_path_blocked(raw):
                violations.append(f"blocked by path blocklist: {raw}")
                continue
            if whitelist_manager is not None and whitelist_manager.is_path_allowed(raw):
                continue
            if any(marker in normalized for marker in self.blocked_markers):
                violations.append(f"blocked path: {raw}")
                continue
            if any(marker in normalized for marker in self.approval_markers):
                approval_reasons.append(f"sensitive path requires approval: {raw}")
                continue
            if normalized.startswith("/"):
                approval_reasons.append(f"absolute path requires approval: {raw}")
                continue
            if not any(
                normalized == root
                or normalized.startswith(root + "/")
                or (root in {"./", "."} and not normalized.startswith("../"))
                for root in allowed_roots
            ):
                approval_reasons.append(f"path outside allowed roots: {raw}")
        if violations:
            return "deny", paths, violations
        if approval_reasons:
            return "require_approval", paths, approval_reasons
        return "allow", paths, []

    def _collect_paths(self, value: Any, key: str = "") -> list[str]:
        paths: list[str] = []
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                paths.extend(self._collect_paths(child_value, child_key))
        elif isinstance(value, list):
            for item in value:
                paths.extend(self._collect_paths(item, key))
        elif isinstance(value, str) and key in self.path_keys:
            stripped = value.strip()
            if stripped:
                paths.append(stripped)
        return paths

    def _is_write_like(self, tool_name: str, args: dict[str, Any]) -> bool:
        lowered_name = (tool_name or "").lower()
        if any(token in lowered_name for token in ("write", "edit", "delete", "remove", "move", "copy")):
            return True
        return any(key in args for key in self.write_keys)


class SandboxPolicyEngine:
    def __init__(
        self,
        command_policy_enabled: bool = True,
        path_policy_enabled: bool = True,
        whitelist_manager: SandboxWhitelistManager | None = None,
    ) -> None:
        self.command_policy_enabled = command_policy_enabled
        self.path_policy_enabled = path_policy_enabled
        self.whitelist_manager = whitelist_manager
        self.path_policy = PathPolicy()
        self.low_commands = {
            "pwd",
            "ls",
            "dir",
            "cat",
            "type",
            "grep",
            "rg",
            "find",
            "head",
            "tail",
            "wc",
            "echo",
        }
        self.medium_commands = {
            "pytest",
            "pip",
            "npm",
            "pnpm",
            "yarn",
            "python",
            "python3",
            "git",
        }
        self.high_commands = {
            "kill",
            "pkill",
            "taskkill",
            "systemctl",
            "service",
            "chmod",
            "chown",
            "git-push",
        }
        self.critical_markers = (
            "rm -rf",
            "remove-item -recurse",
            "remove-item -force",
            "sudo ",
            "su ",
            "shutdown ",
            "reboot",
            "mkfs",
            "diskpart",
            "format ",
        )

    def evaluate_tool_calls(self, tool_calls: list[dict]) -> list[SandboxDecision]:
        decisions: list[SandboxDecision] = []
        for tc in tool_calls:
            function = tc.get("function", {}) if isinstance(tc, dict) else {}
            tool_name = str(function.get("name", "") or "unknown_tool")
            args = _safe_json_loads(str(function.get("arguments", "{}") or "{}"))
            command = self._extract_command(tool_name, args)
            explicit_command_mode = (
                self.whitelist_manager.get_command_mode(command)
                if self.whitelist_manager is not None and command
                else None
            )
            risk_level, reason = self._risk_for(tool_name, args, command)
            if self.path_policy_enabled:
                path_action, paths, path_violations = self.path_policy.inspect(
                    tool_name,
                    args,
                    whitelist_manager=self.whitelist_manager,
                )
            else:
                path_action, paths, path_violations = "allow", [], []
            action = "allow"
            reasons = [reason] if reason else []
            allowlisted = bool(
                self.whitelist_manager is not None
                and command
                and self.whitelist_manager.is_command_allowed(command)
            )
            if path_violations:
                action = "deny"
                reasons.extend(path_violations)
            elif path_action == "require_approval":
                action = "require_approval"
                reasons.append("path policy requires approval")
            if action == "allow" and explicit_command_mode == "deny":
                action = "deny"
                reasons.append("blocked by explicit command rule")
            elif action != "deny" and explicit_command_mode == "ask":
                action = "require_approval"
                reasons.append("explicit command rule requires approval")
            elif explicit_command_mode == "allow":
                reasons.append("explicit command rule allows execution")
            elif action == "allow" and allowlisted:
                reasons.append("command allowlisted")
            elif action == "allow" and self.command_policy_enabled and risk_level in {"high", "critical"}:
                action = "require_approval"
            if (
                action == "allow"
                and explicit_command_mode is None
                and not allowlisted
                and self.command_policy_enabled
                and self._is_destructive_file_op(tool_name, args)
            ):
                action = "require_approval"
                reasons.append("destructive file operation requires approval")
            if (
                action == "allow"
                and explicit_command_mode is None
                and not allowlisted
                and self.whitelist_manager is not None
                and command
                and self.command_policy_enabled
            ):
                default_mode = self.whitelist_manager.get_default_command_mode()
                if default_mode == "deny":
                    action = "deny"
                    reasons.append("default command policy blocks execution")
                elif default_mode == "ask":
                    action = "require_approval"
                    reasons.append("default command policy requires approval")
                elif default_mode == "allow":
                    reasons.append("default command policy allows execution")
            decisions.append(
                SandboxDecision(
                    tool_call_id=str(tc.get("id", "") or ""),
                    tool_name=tool_name,
                    risk_level=risk_level,
                    action=action,
                    reason="; ".join(part for part in reasons if part) or "allowed",
                    command=command,
                    paths=paths,
                    path_violations=path_violations,
                    args=args,
                )
            )
        return decisions

    def _extract_command(self, tool_name: str, args: dict[str, Any]) -> str:
        if isinstance(args.get("command"), str):
            return args["command"].strip()
        if tool_name.lower() == "exec":
            return str(args.get("cmd", "") or args.get("script", "")).strip()
        return ""

    def _risk_for(self, tool_name: str, args: dict[str, Any], command: str) -> tuple[str, str]:
        lowered_tool = (tool_name or "").lower()
        lowered_command = (command or "").strip().lower()
        if lowered_command and any(marker in lowered_command for marker in self.critical_markers):
            return "critical", "critical command marker matched"
        if lowered_command:
            words = _extract_words(lowered_command)
            head = words[0] if words else ""
            if head in {"sudo", "su"}:
                return "critical", "privilege escalation command"
            if head in {"git"} and len(words) > 1 and words[1] == "push":
                return "high", "git push requires approval"
            if head in {"git"} and len(words) > 1 and words[1] in {"status", "diff", "log", "show"}:
                return "medium", "git inspection command"
            if head in self.low_commands:
                return "low", "read-only shell command"
            if head in self.high_commands:
                return "high", "high-impact shell command"
            if head in self.medium_commands:
                if head == "pip" and "install" in words:
                    return "medium", "dependency install command"
                if head in {"python", "python3"} and any(part in {"-m", "pytest"} for part in words):
                    return "medium", "test or python command"
                return "medium", "mutable but common development command"
            if any(part in {"del", "erase", "rm", "rmdir"} for part in words):
                return "critical", "destructive delete command"
            return "medium", "unrecognized command defaults to medium risk"
        if lowered_tool in {"read", "grep", "search"}:
            return "low", "read-only tool"
        if lowered_tool in {"write", "edit", "patch"}:
            return "medium", "write-capable tool"
        if lowered_tool in {"exec", "shell", "process"}:
            return "medium", "command execution tool"
        if lowered_tool in {"delete", "remove"}:
            return "critical", "destructive file tool"
        return "medium", "default tool risk"

    def _is_destructive_file_op(self, tool_name: str, args: dict[str, Any]) -> bool:
        lowered_tool = (tool_name or "").lower()
        if any(token in lowered_tool for token in ("delete", "remove")):
            return True
        for key in ("recursive", "force", "overwrite"):
            if bool(args.get(key)):
                return True
        return False


class SandboxAuditLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        _ensure_parent(path)
        if path:
            with open(path, "a", encoding="utf-8"):
                pass

    def append(self, event_type: str, **payload: Any) -> None:
        if not self.path:
            return
        event = {
            "timestamp": _now_text(),
            "event_type": event_type,
            **payload,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_recent(self, session_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        if not self.path or not os.path.exists(self.path):
            return []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        events: list[dict[str, Any]] = []
        for raw in lines[-limit * 5 :]:
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if session_id and item.get("session_id") != session_id:
                continue
            events.append(item)
        return events[-limit:]


class SandboxApprovalManager:
    def __init__(self, state_path: str, history_path: str) -> None:
        self.state_path = state_path
        self.history_path = history_path
        self._lock = threading.RLock()
        _ensure_parent(state_path)
        _ensure_parent(history_path)
        self._state = self._load_state()
        if history_path:
            with open(history_path, "a", encoding="utf-8"):
                pass

    def create_pending(
        self,
        session_id: str,
        tool_calls: list[dict],
        assistant_message: dict[str, Any],
        decisions: list[SandboxDecision],
    ) -> dict[str, Any]:
        approval_id = f"appr_{uuid.uuid4().hex[:12]}"
        record = {
            "approval_id": approval_id,
            "session_id": session_id,
            "status": "pending",
            "created_at": _now_text(),
            "updated_at": _now_text(),
            "tool_calls": tool_calls,
            "assistant_message": assistant_message,
            "decisions": [asdict(item) for item in decisions],
        }
        with self._lock:
            self._state[approval_id] = record
            self._save_state()
            self._append_history("created", record)
        return record

    def approve(self, session_id: str, approval_id: str = "") -> dict[str, Any] | None:
        return self._transition(session_id, approval_id, "approved")

    def reject(self, session_id: str, approval_id: str = "") -> dict[str, Any] | None:
        return self._transition(session_id, approval_id, "rejected")

    def get_pending(self, session_id: str, approval_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            if approval_id:
                item = self._state.get(approval_id)
                if item and item.get("session_id") == session_id and item.get("status") == "pending":
                    return dict(item)
                return None
            pending = [
                dict(item)
                for item in self._state.values()
                if item.get("session_id") == session_id and item.get("status") == "pending"
            ]
        pending.sort(key=lambda item: item.get("created_at", ""))
        return pending[-1] if pending else None

    def list_pending(self, session_id: str = "") -> list[dict[str, Any]]:
        with self._lock:
            items = [
                dict(item)
                for item in self._state.values()
                if item.get("status") == "pending" and (not session_id or item.get("session_id") == session_id)
            ]
        items.sort(key=lambda item: item.get("created_at", ""))
        return items

    def _transition(self, session_id: str, approval_id: str, target_status: str) -> dict[str, Any] | None:
        with self._lock:
            current = self.get_pending(session_id, approval_id)
            if current is None:
                return None
            current["status"] = target_status
            current["updated_at"] = _now_text()
            self._state[current["approval_id"]] = current
            self._save_state()
            self._append_history(target_status, current)
            return dict(current)

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_path or not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_state(self) -> None:
        if not self.state_path:
            return
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False, indent=2)

    def _append_history(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.history_path:
            return
        event = {
            "timestamp": _now_text(),
            "event_type": event_type,
            **payload,
        }
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
