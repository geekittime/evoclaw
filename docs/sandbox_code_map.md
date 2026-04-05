# MetaClaw Sandbox Code Map

## 1. 范围与结论

这份文档只基于当前仓库里实际存在的代码梳理 `MetaClaw` 侧最适合插入 agent sandbox 的位置，目标是不修改 `openclaw/` 主体代码，优先在 `MetaClaw` 里做策略检查、审批、审计与 lesson 提炼。

先说最关键的结论：

- 当前用户请求进入在线主流程的核心入口在 `MetaClaw/metaclaw/api_server.py`。
- 当前最适合插入 sandbox 的主位置也是 `MetaClaw/metaclaw/api_server.py`。
- `MetaClaw` 目前主要是“OpenClaw 前面的代理/增强层”，负责：
  - 接收 `/v1/chat/completions`
  - 注入 task brief / profile / skills / memory
  - 转发给上游 LLM 或 Tinker
  - 记录 conversations / feedback / task brief / session report
  - 在 session 结束时做 report / memory ingest / skill evolution
- 但“真正执行工具”的代码不在当前 `MetaClaw` Python 代理里。
  - `MetaClaw` 能看到的是：
    - 发给模型的 `tools`
    - 模型返回的 `tool_calls`
    - 下一次请求里带回来的 `toolResult`
  - 因此对工具链的 sandbox 更现实的切入方式是：
    - 在 `MetaClaw` 做“预执行策略判断 + 审批门 + 审计日志 + lesson 提炼”
    - 真正的 OS/file/process 级强隔离若要落地，后续仍需和 OpenClaw/plugin 侧联动
- 对于当前代码里不存在的“本地工具执行函数”，本文不猜，统一标 `TODO`。

## 2. 主调用链

当前在线主链路可以概括为：

1. OpenClaw 插件在 `MetaClaw/extensions/metaclaw-openclaw/index.ts` 里 patch `fetch`
2. 插件在 `before_prompt_build` 阶段注入 `X-Session-Id` / `X-Turn-Type`
3. OpenClaw 将模型请求发到 MetaClaw 代理 `/v1/chat/completions`
4. `MetaClawAPIServer._build_app()` 中的 `chat_completions()` 读取 header / body 并整理 `session_id`、`turn_type`、`session_done`、`memory_scope`
5. `MetaClawAPIServer._handle_request()` 执行主增强逻辑：
   - inline feedback 拦截
   - user profile 更新
   - task brief 更新
   - task brief / profile / important-notes / memory / skills 注入
   - 长上下文压缩总结
6. `_handle_request()` 调用：
   - `_forward_to_llm()`，或
   - `_forward_to_tinker()`
7. 返回模型输出后，`_handle_request()` 读取 assistant `content` / `tool_calls`
8. `main` turn 下调用 `_buffer_record()` 暂存本轮 record
9. 下一次请求到来时，如上一轮 pending record 需要闭合，则 `_flush_pending_record()` 把 `next_state` 写入 conversations / OpenClaw-RL records
10. 若 `session_done=True`，则 `_handle_request()` 触发：
    - `_generate_session_report()`
    - `_ingest_memory_for_session()`
    - `_evolve_skills_for_session()`
    - 各类 session cleanup

## 3. 调用链定位

### 3.1 用户请求进入主流程的位置

首选定位：

- `MetaClaw/metaclaw/api_server.py:790`
  - `chat_completions()`

上游辅助定位：

- `MetaClaw/extensions/metaclaw-openclaw/index.ts:152`
  - `before_prompt_build` 设置 `sessionId` / `turnType`
- `MetaClaw/extensions/metaclaw-openclaw/index.ts:135`
  - patch `globalThis.fetch`，在 POST 请求上附加 header

说明：

- 真正进入 MetaClaw Python 主流程的是 `chat_completions()`
- 这里是最适合插“请求级 sandbox 上下文初始化”的地方，例如创建：
  - `sandbox_request_id`
  - `sandbox_session_state`
  - `sandbox_policy_context`

### 3.2 LLM 请求发出前的位置

首选定位：

- `MetaClaw/metaclaw/api_server.py:1565`
  - `_handle_request()`

最关键的 pre-LLM hook 点：

- `MetaClaw/metaclaw/api_server.py:1656-1680` 附近
  - 完成 memory scope、profile、task brief、skills/memory 注入后
  - 但还没有真正转发给 LLM
- `MetaClaw/metaclaw/api_server.py:1690-1701` 附近
  - `forward_body` 已经构造好
- `MetaClaw/metaclaw/api_server.py:1703-1706` 附近
  - 即将调用 `_forward_to_llm()` / `_forward_to_tinker()`

更底层的实际转发函数：

- `MetaClaw/metaclaw/api_server.py:1937`
  - `_forward_to_tinker()`
- `MetaClaw/metaclaw/api_server.py:2042`
  - `_forward_to_llm()`

说明：

- 如果 sandbox 需要在“模型看到工具列表前”做过滤，最佳位置是在 `_handle_request()` 里构造 `forward_body` 之前/之后。
- 如果 sandbox 需要审计“最终发给模型的 messages/tools”，最佳位置是在 `_handle_request()` 中、调用 `_forward_to_*()` 之前。

### 3.3 工具调用准备执行的位置

当前 MetaClaw 里能明确看到的“准备执行前”位置不是本地执行器，而是“模型刚返回 tool_calls，尚未交还上游”的阶段。

可插 hook：

- `MetaClaw/metaclaw/api_server.py:1711`
  - `_handle_request()` 从模型响应中取出 `tool_calls`
- `MetaClaw/metaclaw/api_server.py:1723-1724`
  - 记录 `tool_calls` 日志
- `MetaClaw/metaclaw/api_server.py:1993-2014`
  - `_forward_to_tinker()` 内，把模型文本解析成 `parsed_tool_calls`
- `MetaClaw/metaclaw/api_server.py:2086-2097`
  - `_forward_to_llm()` 内，对上游返回内容做 tool_call 规范化/补抽取

结论：

- 对 MetaClaw 而言，最合适的“工具执行前 hook”是：
  - 在 `_handle_request()` 拿到 `tool_calls` 后、返回给 OpenClaw 前
- 这可以支持：
  - 工具名/参数风险评分
  - 路径参数检查
  - 命令类工具审批判定
  - 高风险工具直接改写为拒绝/审批响应

限制：

- 当前仓库里没有看到 MetaClaw 自己执行 shell/file/process 工具的函数。
- “真正执行工具”的精确代码位置：`TODO（应在 OpenClaw runtime 或其 plugin/tool runtime 侧）`

### 3.4 工具调用返回结果的位置

当前 MetaClaw 看到工具结果的主要方式，是下一次请求里的消息历史中出现 `toolResult` / tool message。

关键位置：

- `MetaClaw/metaclaw/api_server.py:387-399`
  - `_normalize_messages_for_template()` 将 `toolResult` 规范化为 OpenAI tool message
- `MetaClaw/metaclaw/api_server.py:1728`
  - `_handle_request()` 在新一轮 `main` 请求到来时，若上一轮有 pending record，则先 `_flush_pending_record(session_id, messages[-1])`
- `MetaClaw/metaclaw/api_server.py:1195`
  - `_flush_pending_record()` 真正把 `next_state`、`next_state_text` 写入 record

结论：

- 如果 sandbox 需要对“工具执行结果”做二次检查、审计、lesson 提炼，最佳 hook 是 `_flush_pending_record()`。
- 这里已经能拿到：
  - 上一轮 prompt/response/tool_calls
  - 当前传回的 `next_state`
  - 可用于判断：
    - 工具是否失败
    - 返回内容是否越权
    - 是否触发负面经验抽取

### 3.5 session 结束后总结/report 生成的位置

session 结束触发点：

- `MetaClaw/metaclaw/api_server.py:1770-1798`
  - tokenizer unavailable 的 `main` 分支里处理 `session_done`
- `MetaClaw/metaclaw/api_server.py:1889-1918`
  - 常规分支里处理 `session_done`

真正 report 生成函数：

- `MetaClaw/metaclaw/api_server.py:2365`
  - `_generate_session_report()`

说明：

- 如果 sandbox 需要做 session 级安全总结、违规聚类、审批统计、lesson 沉淀，这里是最稳的位置。
- 因为此时已经能汇总：
  - records
  - feedback_history
  - injected_skills
  - task_brief

### 3.6 当前 conversations / feedback / task brief / session report 的写入位置

`conversations.jsonl` / enriched conversations：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:147`
    - `record_enriched_file = "conversations.jsonl"`
- 先暂存：
  - `MetaClaw/metaclaw/api_server.py:1406`
    - `_buffer_record()`
- 真正落盘：
  - `MetaClaw/metaclaw/api_server.py:1195`
    - `_flush_pending_record()`

`openclaw_rl_records.jsonl`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:146`
    - `record_openclaw_rl_file = "openclaw_rl_records.jsonl"`
- 真正落盘：
  - `MetaClaw/metaclaw/api_server.py:1195`
    - `_flush_pending_record()`

`feedback.jsonl`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:68`
    - `feedback_history_path = "records/feedback.jsonl"`
- 真正落盘：
  - `MetaClaw/metaclaw/api_server.py:1243`
    - `_append_feedback_record()`
- 调用入口：
  - `MetaClaw/metaclaw/api_server.py:2488`
    - `_handle_feedback()`
  - `MetaClaw/metaclaw/api_server.py:1124`
    - `/v1/feedback`
  - `MetaClaw/metaclaw/api_server.py:1577-1603`
    - inline feedback 拦截

`task_briefs.json`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:77`
    - `task_brief_path = "records/task_briefs.json"`
- 更新逻辑：
  - `MetaClaw/metaclaw/api_server.py:2229`
    - `_update_task_brief()`
- 真正写盘：
  - `MetaClaw/metaclaw/api_server.py:1305`
    - `_save_task_briefs()`

`user_profiles.json`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:71`
    - `user_profile_path = "records/user_profiles.json"`
- 更新逻辑：
  - `MetaClaw/metaclaw/api_server.py:2304`
    - `_update_user_profile()`
- 真正写盘：
  - `MetaClaw/metaclaw/api_server.py:1269`
    - `_save_user_profiles()`

`session_reports.jsonl`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:83`
    - `session_report_path = "records/session_reports.jsonl"`
- 生成逻辑：
  - `MetaClaw/metaclaw/api_server.py:2365`
    - `_generate_session_report()`
- 真正落盘：
  - `MetaClaw/metaclaw/api_server.py:1318`
    - `_append_session_report()`

`skill_stats.json`：

- 路径配置：
  - `MetaClaw/metaclaw/config.py:62`
    - `adaptive_skill_stats_path = "records/skill_stats.json"`
- 写入逻辑：
  - `MetaClaw/metaclaw/skill_manager.py:214`
    - `_save_skill_stats()`
- 触发写入：
  - `MetaClaw/metaclaw/skill_manager.py:238`
    - `record_skill_selection()`
  - `MetaClaw/metaclaw/skill_manager.py:251`
    - `record_feedback()`

## 4. Sandbox 模块最适合插入的位置

### 4.1 SandboxPolicyEngine

推荐主插点：

- `MetaClaw/metaclaw/api_server.py:1565`
  - `_handle_request()`

最适合细分为两段：

1. pre-LLM policy
   - 在 augmentation 完成后、`_forward_to_llm()` / `_forward_to_tinker()` 前
   - 用于：
     - 检查本轮用户意图
     - 检查暴露给模型的工具列表
     - 给 prompt 注入 sandbox 约束
2. pre-tool-execution policy
   - 在拿到 `tool_calls` 后、响应返回 OpenClaw 前
   - 用于：
     - 风险分级
     - 阻断高风险 tool call
     - 给后续审批模块提供结构化决策输入

建议原因：

- 这里上下文最全
- 不需要改 OpenClaw 主体
- 与已有 task brief / profile / important-notes / feedback 逻辑天然同层

### 4.2 ApprovalManager

推荐主插点：

- `MetaClaw/metaclaw/api_server.py:1711-1724` 附近
  - `_handle_request()` 已拿到 `tool_calls`

推荐能力：

- 对高风险 tool call 生成 `approval_required`
- 将审批状态挂到：
  - pending record
  - feedback/session report
  - audit log
- 视设计选择：
  - 直接阻断 tool call 并返回解释性 assistant 消息
  - 或改写 tool call 为“等待审批”的特殊响应

说明：

- 当前 MetaClaw 没有看到现成的人机审批 API。
- “审批确认如何再恢复执行”的完整闭环：`TODO`

### 4.3 PathPolicy

推荐主插点：

- `MetaClaw/metaclaw/api_server.py:1711-1724` 附近
  - 对 `tool_calls[].function.arguments` 做静态解析

次级插点：

- `MetaClaw/metaclaw/api_server.py:1195`
  - `_flush_pending_record()`，对工具返回结果做复核与审计补全

适用原因：

- 当前最容易做的是对工具参数中的：
  - `path`
  - `file_path`
  - `cwd`
  - `target`
  - `command`
  做规则检查
- 尤其适合：
  - 允许目录白名单
  - 禁止越界写路径
  - 禁止敏感系统路径

限制：

- 如果真实路径解析发生在 OpenClaw 侧，MetaClaw 这里拿到的可能只是字符串参数。
- “最终执行前的 resolved absolute path” 目前不在 MetaClaw 中可见：`TODO`

### 4.4 SandboxAuditLogger

推荐主插点：

- 请求进入时：
  - `MetaClaw/metaclaw/api_server.py:790`
    - `chat_completions()`
- pre-LLM 决策后：
  - `MetaClaw/metaclaw/api_server.py:1565`
    - `_handle_request()`
- tool_calls 生成后：
  - `MetaClaw/metaclaw/api_server.py:1711`
- tool result 返回后：
  - `MetaClaw/metaclaw/api_server.py:1195`
    - `_flush_pending_record()`
- session 结束汇总时：
  - `MetaClaw/metaclaw/api_server.py:2365`
    - `_generate_session_report()`

存储建议：

- 不要混进现有 `conversations.jsonl` 原 schema，建议单独加：
  - `records/sandbox_audit.jsonl`
- 如果短期不想新增 schema，也可以先把 sandbox 字段以扩展字段挂在 enriched record 上
  - 但长期更推荐独立审计流

### 4.5 SafetyLessonExtractor

推荐主插点：

- 一阶触发：
  - `MetaClaw/metaclaw/api_server.py:1195`
    - `_flush_pending_record()`
  - 适合根据 `tool_calls + next_state` 抽取失败 lesson
- 二阶触发：
  - `MetaClaw/metaclaw/api_server.py:2488`
    - `_handle_feedback()`
  - 适合把用户明确指出的 sandbox 问题转成长期规则
- session 级触发：
  - `MetaClaw/metaclaw/api_server.py:2365`
    - `_generate_session_report()`
  - 适合抽取 reusable safety lessons

建议原因：

- 你们现在已经有 `important-notes` 长期 skill 机制
- safety lesson 可以直接复用这一闭环，沉淀成：
  - 文件/路径安全注意事项
  - 工具审批偏好
  - 常见危险操作模式

## 5. 推荐的最小落地路径

如果下一步要在“尽量不改架构”的前提下先做一个能工作的 sandbox 雏形，最建议的顺序是：

1. 在 `api_server.py::_handle_request()` 增加 `SandboxPolicyEngine`
   - 先做工具名/参数级风险判断
2. 在拿到 `tool_calls` 后增加 `ApprovalManager + PathPolicy`
   - 先只拦高风险 shell/write/path 操作
3. 在 `_flush_pending_record()` 增加 `SandboxAuditLogger`
   - 记录 tool_calls、决策、next_state、是否触发阻断
4. 在 `_handle_feedback()` 和 `_generate_session_report()` 增加 `SafetyLessonExtractor`
   - 复用已有 important-notes 长期沉淀机制

这样做的优点是：

- 基本不碰 OpenClaw 主体
- 完全贴合 MetaClaw 现在的增强层定位
- 与已有 feedback / task brief / session report / important-notes 能形成统一闭环

## 6. 当前未确认项 / TODO

- `MetaClaw` 当前仓库中没有发现“真实工具执行函数”；工具实际执行点应在 OpenClaw runtime 或其 plugin/tool runtime 侧：`TODO`
- 当前仓库中没有发现现成的“审批确认后恢复执行”闭环 API：`TODO`
- 如果后续要做强隔离型 sandbox（例如文件系统白名单、进程权限、命令执行隔离），仅在 MetaClaw 代理层做字符串级拦截还不够，后续需要和 OpenClaw/tool runner 侧联动：`TODO`

