import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import httpx
import metaclaw.utils as utils
from metaclaw.api_server import (
    MetaClawAPIServer,
    _parse_inline_approval,
    _rewrite_messages_for_upstream_tool_history,
    _strip_non_replayable_assistant_errors,
    _strip_upstream_approval_artifacts,
    _should_preflatten_native_tool_history,
)
from metaclaw.config import MetaClawConfig
from metaclaw.sandbox import SandboxPolicyEngine, SandboxWhitelistManager


class _FakeStore:
    def __init__(self, data):
        self._data = data

    def load(self):
        return self._data


def test_run_llm_uses_explicit_config_for_provider_selection(monkeypatch):
    monkeypatch.setattr(
        "metaclaw.config_store.ConfigStore",
        lambda: _FakeStore({"mode": "skills_only", "llm": {"provider": "bedrock"}}),
    )

    openai_calls = []
    bedrock_calls = []

    def fake_openai(messages, config=None):
        openai_calls.append((messages, config))
        return "openai"

    def fake_bedrock(messages, config=None):
        bedrock_calls.append((messages, config))
        return "bedrock"

    monkeypatch.setattr(utils, "_run_llm_openai", fake_openai)
    monkeypatch.setattr(utils, "_run_llm_bedrock", fake_bedrock)

    result = utils.run_llm(
        [{"role": "user", "content": "compress this"}],
        config=MetaClawConfig(mode="skills_only", llm_provider="custom"),
    )

    assert result == "openai"
    assert len(openai_calls) == 1
    assert bedrock_calls == []


def test_run_llm_uses_explicit_skills_only_endpoint(monkeypatch):
    monkeypatch.setattr(
        "metaclaw.config_store.ConfigStore",
        lambda: _FakeStore(
            {
                "mode": "rl",
                "rl": {
                    "prm_url": "https://wrong.example/v1",
                    "prm_api_key": "wrong-key",
                    "prm_model": "wrong-model",
                },
            }
        ),
    )

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captured["request_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="compressed"))]
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = utils.run_llm(
        [{"role": "user", "content": "system prompt"}],
        config=MetaClawConfig(
            mode="skills_only",
            llm_provider="custom",
            llm_api_base="https://live.example/v1",
            llm_api_key="live-key",
            llm_model_id="live-model",
        ),
    )

    assert result == "compressed"
    assert captured["client_kwargs"] == {
        "api_key": "live-key",
        "base_url": "https://live.example/v1",
    }
    assert captured["request_kwargs"]["model"] == "live-model"


def test_run_llm_uses_explicit_rl_endpoint(monkeypatch):
    monkeypatch.setattr(
        "metaclaw.config_store.ConfigStore",
        lambda: _FakeStore(
            {
                "mode": "skills_only",
                "llm": {
                    "api_base": "https://wrong.example/v1",
                    "api_key": "wrong-key",
                    "model_id": "wrong-model",
                },
            }
        ),
    )

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captured["request_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="compressed"))]
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = utils.run_llm(
        [{"role": "user", "content": "judge this"}],
        config=MetaClawConfig(
            mode="rl",
            prm_provider="openai",
            prm_url="https://judge.example/v1",
            prm_api_key="judge-key",
            prm_model="judge-model",
        ),
    )

    assert result == "compressed"
    assert captured["client_kwargs"] == {
        "api_key": "judge-key",
        "base_url": "https://judge.example/v1",
    }
    assert captured["request_kwargs"]["model"] == "judge-model"


def test_handle_request_falls_back_to_raw_system_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://live.example/v1",
        llm_api_key="live-key",
        llm_model_id="live-model",
        record_enabled=False,
        record_dir=str(tmp_path),
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )

    def fail_run_llm(messages, config=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("metaclaw.api_server.run_llm", fail_run_llm)

    forwarded = {}

    async def fake_forward(self, body):
        forwarded["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    monkeypatch.setattr(MetaClawAPIServer, "_forward_to_llm", fake_forward)

    result = asyncio.run(
        server._handle_request(
            body={
                "messages": [
                    {"role": "system", "content": "raw system prompt"},
                    {"role": "user", "content": "hello"},
                ]
            },
            session_id="session-1",
            turn_type="main",
            session_done=False,
        )
    )

    assert forwarded["body"]["messages"][0]["content"] == "raw system prompt"
    assert result["response"]["choices"][0]["message"]["content"] == "ok"


def test_handle_request_strips_upstream_skill_catalog_and_keeps_selected_skills(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://live.example/v1",
        llm_api_key="live-key",
        llm_model_id="live-model",
        record_enabled=False,
        record_dir=str(tmp_path),
        task_brief_enabled=False,
        user_profile_enabled=False,
        session_report_enabled=False,
        context_summary_enabled=False,
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )
    server.skill_manager = SimpleNamespace(
        record_skill_selection=lambda names: None,
        format_for_conversation=lambda skills: (
            "## Active Skills\n\n### weather\n_selected_\n\nUse the weather skill only."
        ),
    )
    server._session_skill_overrides["session-1"] = ["weather"]
    server._list_all_skills = lambda: [
        {
            "name": "weather",
            "description": "selected",
            "content": "Use the weather skill only.",
        }
    ]

    forwarded = {}

    async def fake_forward(self, body):
        forwarded["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    monkeypatch.setattr(MetaClawAPIServer, "_forward_to_llm", fake_forward)

    upstream_system = (
        "You are a personal assistant running inside OpenClaw.\n"
        "## Skills (mandatory)\n"
        "Before replying: scan <available_skills> <description> entries.\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>demo</name>\n"
        "  </skill>\n"
        "</available_skills>\n"
        "## Memory Recall\n"
        "Check memory when needed."
    )

    asyncio.run(
        server._handle_request(
            body={
                "messages": [
                    {"role": "system", "content": upstream_system},
                    {"role": "user", "content": "hello"},
                ]
            },
            session_id="session-1",
            turn_type="main",
            session_done=False,
        )
    )

    forwarded_system = forwarded["body"]["messages"][0]["content"]
    assert "## Skills (mandatory)" not in forwarded_system
    assert "<available_skills>" not in forwarded_system
    assert "demo" not in forwarded_system
    assert "## Memory Recall" in forwarded_system
    assert "## Active Skills" in forwarded_system
    assert "weather" in forwarded_system


def test_handle_request_injects_stored_context_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://live.example/v1",
        llm_api_key="live-key",
        llm_model_id="live-model",
        record_enabled=False,
        record_dir=str(tmp_path),
        task_brief_enabled=False,
        user_profile_enabled=False,
        session_report_enabled=False,
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )
    server._session_context_summaries["session-1"] = (
        "- User wants concise progress updates.\n"
        "- Already confirmed the target file path."
    )

    forwarded = {}

    async def fake_forward(self, body):
        forwarded["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    monkeypatch.setattr(MetaClawAPIServer, "_forward_to_llm", fake_forward)

    asyncio.run(
        server._handle_request(
            body={
                "messages": [
                    {"role": "system", "content": "raw system prompt"},
                    {"role": "user", "content": "hello"},
                ]
            },
            session_id="session-1",
            turn_type="main",
            session_done=False,
        )
    )

    forwarded_system = forwarded["body"]["messages"][0]["content"]
    assert "raw system prompt" in forwarded_system
    assert "## Conversation Summary" in forwarded_system
    assert "concise progress updates" in forwarded_system


def test_parse_inline_approval_handles_wrapped_sender_and_timestamp():
    wrapped = (
        "Sender (untrusted metadata):\n"
        "```json\n"
        '{"label":"openclaw-control-ui","id":"openclaw-control-ui"}\n'
        "```\n\n"
        "[Tue 2026-04-07 09:45 GMT+8] approve appr_1310829caaa3"
    )

    assert _parse_inline_approval(wrapped) == ("approve", "appr_1310829caaa3")


def test_rewrite_messages_for_upstream_tool_history_flattens_native_tool_context():
    rewritten = _rewrite_messages_for_upstream_tool_history(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "delete the file"},
            {
                "role": "assistant",
                "content": "Deleting it now.",
                "tool_calls": [
                    {
                        "id": "call_delete",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": '{"command":"rm /tmp/demo.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_delete",
                "name": "exec",
                "content": "deleted",
            },
        ]
    )

    assert rewritten[0] == {"role": "system", "content": "system rules"}
    assert rewritten[1] == {"role": "user", "content": "delete the file"}
    assert rewritten[2]["role"] == "assistant"
    assert "Deleting it now." in rewritten[2]["content"]
    assert "The assistant requested the following tool calls" in rewritten[2]["content"]
    assert "exec" in rewritten[2]["content"]
    assert rewritten[3]["role"] == "user"
    assert "Tool result for exec (tool_call_id=call_delete)" in rewritten[3]["content"]
    assert "Continue the task using this tool result" in rewritten[3]["content"]


def test_strip_upstream_approval_artifacts_removes_prompt_and_inline_command():
    filtered = _strip_upstream_approval_artifacts(
        [
            {"role": "user", "content": "keep me"},
            {
                "role": "assistant",
                "content": (
                    "这个工具在调用前需要获得你的允许.\n"
                    "Approval ID: appr_123\n"
                    "使用 `approve` 进行批准，或使用 `reject` 进行拒绝."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Sender (untrusted metadata):\n```json\n{}\n```\n\n"
                    "[Tue 2026-04-07 12:50 GMT+8] approve appr_123"
                ),
                "provenance": {
                    "kind": "internal_system",
                    "sourceTool": "metaclaw-approval",
                },
            },
            {
                "role": "user",
                "content": (
                    "Sender (untrusted metadata): openclaw-control-ui\n"
                    "[Tue 2026-04-07 12:51 GMT+8] approve appr_456"
                ),
            },
            {
                "role": "user",
                "content": "The operator rejected the pending command. Continue the task safely.",
                "provenance": {
                    "kind": "internal_system",
                    "sourceTool": "metaclaw-approval",
                },
            },
        ]
    )

    assert filtered == [
        {"role": "user", "content": "keep me"},
        {
            "role": "user",
            "content": "The operator rejected the pending command. Continue the task safely.",
            "provenance": {
                "kind": "internal_system",
                "sourceTool": "metaclaw-approval",
            },
        },
    ]


def test_strip_non_replayable_assistant_errors_removes_failed_turns():
    filtered = _strip_non_replayable_assistant_errors(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "500 Internal Server Error",
            },
            {
                "role": "assistant",
                "content": [],
                "stopReason": "aborted",
            },
            {"role": "user", "content": "hi"},
        ]
    )

    assert filtered == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "hi"},
    ]


def test_handle_request_strips_non_replayable_assistant_errors_before_forward(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://live.example/v1",
        llm_api_key="live-key",
        llm_model_id="live-model",
        record_enabled=False,
        record_dir=str(tmp_path),
        task_brief_enabled=False,
        user_profile_enabled=False,
        session_report_enabled=False,
        context_summary_enabled=False,
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )

    forwarded = {}

    async def fake_forward(self, body):
        forwarded["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                    }
                }
            ]
        }

    monkeypatch.setattr(MetaClawAPIServer, "_forward_to_llm", fake_forward)

    asyncio.run(
        server._handle_request(
            body={
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "帮我删除 temp0 里的 py 文件"},
                    {
                        "role": "assistant",
                        "content": [],
                        "stopReason": "error",
                        "errorMessage": "500 Internal Server Error",
                    },
                    {"role": "user", "content": "hi"},
                ]
            },
            session_id="session-1",
            turn_type="main",
            session_done=False,
        )
    )

    forwarded_messages = forwarded["body"]["messages"]
    assert forwarded_messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "帮我删除 temp0 里的 py 文件"},
        {"role": "user", "content": "hi"},
    ]


def test_rewrite_messages_for_upstream_tool_history_handles_long_tool_chain():
    messages = [{"role": "system", "content": "system rules"}, {"role": "user", "content": "start"}]
    for idx in range(12):
        tool_call_id = f"call_{idx}"
        messages.append(
            {
                "role": "assistant",
                "content": f"step {idx}",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": json.dumps({"command": f"echo {idx}"}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": "exec",
                "content": f"output {idx}",
            }
        )

    rewritten = _rewrite_messages_for_upstream_tool_history(messages)

    assert len(rewritten) == 26
    for idx in range(12):
        assistant_message = rewritten[2 + idx * 2]
        tool_result_message = rewritten[3 + idx * 2]
        assert assistant_message["role"] == "assistant"
        assert f"step {idx}" in assistant_message["content"]
        assert f"tool_call_id: call_{idx}" in assistant_message["content"]
        assert tool_result_message["role"] == "user"
        assert f"Tool result for exec (tool_call_id=call_{idx})" in tool_result_message["content"]
        assert f"output {idx}" in tool_result_message["content"]


def test_should_preflatten_native_tool_history_for_custom_deepseek():
    assert _should_preflatten_native_tool_history(
        api_base="https://api.deepseek.com/v1",
        model_id="deepseek-chat",
        llm_provider="custom",
        contains_native_tool_history=True,
    ) is True
    assert _should_preflatten_native_tool_history(
        api_base="https://example.com/v1",
        model_id="gpt-5.4",
        llm_provider="openai",
        contains_native_tool_history=True,
    ) is False
    assert _should_preflatten_native_tool_history(
        api_base="https://example.com/v1",
        model_id="gpt-5.4",
        llm_provider="custom",
        contains_native_tool_history=False,
    ) is False


def test_forward_to_llm_retries_with_flattened_tool_history(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://live.example/v1",
        llm_api_key="live-key",
        llm_model_id="live-model",
        record_enabled=False,
        record_dir=str(tmp_path),
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )

    requests = []

    class FakeResponse:
        def __init__(self, status_code: int, payload=None, text: str = ""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://live.example/v1/chat/completions")
                response = httpx.Response(self.status_code, request=request, text=self.text)
                raise httpx.HTTPStatusError("upstream failure", request=request, response=response)

        def json(self):
            return self._payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            requests.append({"url": url, "json": json, "headers": headers})
            if len(requests) == 1:
                return FakeResponse(404, text="")
            return FakeResponse(
                200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "done",
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        server._forward_to_llm(
            {
                "model": "live-model",
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "delete the file"},
                    {
                        "role": "assistant",
                        "content": "Deleting it now.",
                        "tool_calls": [
                            {
                                "id": "call_delete",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"command":"rm /tmp/demo.txt"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_delete",
                        "name": "exec",
                        "content": "deleted",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    assert result["choices"][0]["message"]["content"] == "done"
    assert len(requests) == 2
    assert requests[0]["url"].endswith("/chat/completions")
    assert requests[0]["json"]["messages"][3]["role"] == "tool"
    assert requests[1]["json"]["messages"][3]["role"] == "user"
    assert (
        "Tool result for exec (tool_call_id=call_delete)"
        in requests[1]["json"]["messages"][3]["content"]
    )


def test_forward_to_llm_preflattens_native_tool_history_for_custom_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        MetaClawAPIServer,
        "_load_tokenizer",
        lambda self: None,
    )

    config = MetaClawConfig(
        mode="skills_only",
        claw_type="openclaw",
        llm_provider="custom",
        llm_api_base="https://api.deepseek.com/v1",
        llm_api_key="live-key",
        llm_model_id="deepseek-chat",
        record_enabled=False,
        record_dir=str(tmp_path),
    )
    server = MetaClawAPIServer(
        config=config,
        output_queue=queue.Queue(),
        submission_enabled=threading.Event(),
    )

    requests = []

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            requests.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        server._forward_to_llm(
            {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "delete the file"},
                    {
                        "role": "assistant",
                        "content": "Deleting it now.",
                        "tool_calls": [
                            {
                                "id": "call_delete",
                                "type": "function",
                                "function": {
                                    "name": "exec",
                                    "arguments": '{"command":"rm /tmp/demo.txt"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_delete",
                        "name": "exec",
                        "content": "deleted",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        )
    )

    assert result["choices"][0]["message"]["content"] == "done"
    assert len(requests) == 1
    assert requests[0]["json"]["messages"][2]["role"] == "assistant"
    assert "The assistant requested the following tool calls" in requests[0]["json"]["messages"][2]["content"]
    assert requests[0]["json"]["messages"][3]["role"] == "user"
    assert "Tool result for exec (tool_call_id=call_delete)" in requests[0]["json"]["messages"][3]["content"]


def test_command_allowlist_matches_command_head(tmp_path):
    manager = SandboxWhitelistManager(str(tmp_path / "sandbox.json"))
    assert manager.add_command("ls") is True

    assert manager.is_command_allowed("ls") is True
    assert manager.is_command_allowed("ls -la /tmp/demo") is True
    assert manager.is_command_allowed("pwd") is False


def test_command_rules_match_command_head(tmp_path):
    manager = SandboxWhitelistManager(str(tmp_path / "sandbox.json"))
    assert manager.set_command_mode("rm", "deny") is True
    assert manager.get_command_mode("rm -rf /tmp/demo") == "deny"


def test_policy_allows_allowlisted_exec_command_with_arguments(tmp_path):
    manager = SandboxWhitelistManager(str(tmp_path / "sandbox.json"))
    assert manager.add_command("ls") is True
    policy = SandboxPolicyEngine(whitelist_manager=manager)

    decisions = policy.evaluate_tool_calls([
        {
            "id": "call_ls",
            "function": {
                "name": "exec",
                "arguments": '{"command":"ls -la /tmp/demo"}',
            },
        }
    ])

    assert len(decisions) == 1
    assert decisions[0].action == "allow"
    assert "command allowlisted" in decisions[0].reason


def test_policy_denies_command_rule_for_command_head(tmp_path):
    manager = SandboxWhitelistManager(str(tmp_path / "sandbox.json"))
    assert manager.set_command_mode("rm", "deny") is True
    policy = SandboxPolicyEngine(whitelist_manager=manager)

    decisions = policy.evaluate_tool_calls([
        {
            "id": "call_rm",
            "function": {
                "name": "exec",
                "arguments": '{"command":"rm -rf /tmp/demo"}',
            },
        }
    ])

    assert len(decisions) == 1
    assert decisions[0].action == "deny"
    assert "blocked by explicit command rule" in decisions[0].reason
