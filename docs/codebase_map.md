# OpenClaw / MetaClaw 代码地图

## 1. 目标与范围

这份文档只基于当前仓库里的实际代码，梳理后续接入 training-free 自进化模块最相关的路径：

- agent 主入口
- LLM 请求代理 / 转发路径
- memory 读写 / 检索路径
- skill 注入与演化逻辑
- session 结束后的总结 / 后处理逻辑
- 日志记录与状态记录逻辑
- RL 入口

说明：

- 当前仓库同时包含 `openclaw/`、`MetaClaw/`、`OpenClaw-RL/`
- 你现在的运行方式是“浏览器使用 OpenClaw，后端模型请求经过 MetaClaw proxy”，所以最重要的在线链路在 `MetaClaw/`
- 不存在的逻辑不会猜；不确定的地方会标 `TODO`

## 2. 仓库分层总览

### 2.1 `openclaw/`

职责：

- TypeScript 主体工程
- CLI / gateway / agent runtime / plugin hook 框架
- 负责浏览器侧和网关侧的原生 agent 执行

对本项目的意义：

- 如果后续要做“原生 OpenClaw 插件式自进化”，这里是底层插点
- 当前部署里，它主要负责发起 agent 运行，并把模型请求导向 MetaClaw

### 2.2 `MetaClaw/`

职责：

- Python proxy 层
- 兼容 OpenAI `/v1/chat/completions`
- 在请求进入真实 LLM 之前做 memory / skill 注入
- 在 session 结束后做 memory ingest / skill evolution / RL sample 产出

对本项目的意义：

- 这是目前最适合接 training-free 自进化模块的位置
- 它已经具备“观测请求、持久化轨迹、提炼 memory、生成 skill、触发 RL”的闭环雏形

### 2.3 `OpenClaw-RL/`

职责：

- RL 研究代码与参考实现
- 包含原版 `openclaw_api_server.py` / `openclaw_rollout.py` 等

对本项目的意义：

- 是 MetaClaw RL 路径的重要来源
- 当前在线主链路不是这里，但这里能帮助理解 RL 设计来源

## 3. 当前部署下的主调用链

按当前仓库和插件逻辑，浏览器侧的主链路是：

1. OpenClaw gateway / embedded agent runtime 启动 agent
2. OpenClaw 在构造 prompt 时触发插件 hook
3. `MetaClaw/extensions/metaclaw-openclaw/index.ts` 在 `before_prompt_build` 阶段准备 `X-Session-Id` / `X-Turn-Type`
4. 插件 patch `globalThis.fetch`，把这些 header 注入后续模型请求
5. OpenClaw 把模型请求发到 MetaClaw proxy
6. `MetaClaw/metaclaw/api_server.py` 接收 `/v1/chat/completions`
7. MetaClaw 做 memory / skill augmentation
8. MetaClaw 转发到真实 LLM 或 Tinker
9. 响应返回 OpenClaw
10. 如果显式收到 `session_done`，MetaClaw 执行 session 级后处理

这个链路意味着：

- request-time 自进化插点：`MetaClaw/metaclaw/api_server.py`
- session-end 自进化插点：`MetaClaw/metaclaw/api_server.py`
- 更底层原生插点：`openclaw/src/agents/pi-embedded-runner/run/setup.ts`、`openclaw/src/agents/pi-embedded-runner/run/attempt.ts`

## 4. Agent 主入口

### 4.1 OpenClaw CLI / Gateway 入口

关键文件：

- `openclaw/src/entry.ts`
- `openclaw/src/cli/run-main.ts`
- `openclaw/src/cli/gateway-cli/run.ts`
- `openclaw/src/gateway/server.ts`

职责：

- `openclaw/src/entry.ts`：Node CLI 入口
- `openclaw/src/cli/run-main.ts`：加载环境、构建 CLI、注册命令、开启 console capture
- `openclaw/src/cli/gateway-cli/run.ts`：gateway 子命令入口
- `openclaw/src/gateway/server.ts`：导出 `startGatewayServer`

主要调用链：

`openclaw` CLI
-> `openclaw/src/entry.ts`
-> `runCli()` in `openclaw/src/cli/run-main.ts`
-> gateway 子命令
-> `runGatewayCommand()` in `openclaw/src/cli/gateway-cli/run.ts`
-> `startGatewayServer()` in `openclaw/src/gateway/server.ts`

适合插 hook / adapter 的位置：

- gateway 启动前后的 plugin 初始化阶段
- plugin loader 完成后，全局 hook runner 初始化完成的位置

### 4.2 OpenClaw embedded agent runtime 入口

关键文件：

- `openclaw/src/agents/pi-embedded-runner/run.ts`
- `openclaw/src/agents/pi-embedded-runner/run/setup.ts`
- `openclaw/src/agents/pi-embedded-runner/run/attempt.ts`

职责：

- `runEmbeddedPiAgent()` 是 embedded agent 真正入口
- `run/setup.ts` 负责 model/provider 选择与 `before_model_resolve`
- `run/attempt.ts` 负责 prompt 组装、LLM 调用、tool loop、hook 触发

主要调用链：

上游消息处理
-> `runEmbeddedPiAgent()`
-> `resolveHookModelSelection()`
-> runtime / model / session 初始化
-> `runEmbeddedAttempt()`
-> `before_prompt_build`
-> `llm_input`
-> LLM 调用
-> `llm_output`
-> `agent_end`

适合插 hook / adapter 的位置：

- `openclaw/src/agents/pi-embedded-runner/run/setup.ts`
  - `before_model_resolve`
- `openclaw/src/agents/pi-embedded-runner/run/attempt.ts`
  - `before_prompt_build`
  - `llm_input`
  - `llm_output`
  - `agent_end`
  - `before_tool_call`
  - `after_tool_call`

### 4.3 MetaClaw 服务入口

关键文件：

- `MetaClaw/pyproject.toml`
- `MetaClaw/metaclaw/cli.py`
- `MetaClaw/metaclaw/launcher.py`
- `MetaClaw/metaclaw/rollout.py`

职责：

- `MetaClaw/pyproject.toml` 将 `metaclaw` CLI 指向 `metaclaw.cli:metaclaw`
- `MetaClaw/metaclaw/cli.py` 提供 `metaclaw start --mode skills_only|rl`
- `MetaClaw/metaclaw/launcher.py` 根据模式装配 SkillManager / MemoryManager / Trainer / Proxy
- `MetaClaw/metaclaw/rollout.py` 创建 `AsyncRolloutWorker`，其内部持有 `MetaClawAPIServer`

主要调用链：

`metaclaw start --mode skills_only`
-> `MetaClaw/metaclaw/cli.py`
-> `MetaClawLauncher.start()`
-> `MetaClawLauncher._start_skills_only()`
-> `AsyncRolloutWorker(...)`
-> `MetaClawAPIServer.start()`

`metaclaw start --mode rl`
-> `MetaClawLauncher.start()`
-> `MetaClawLauncher._start_rl()`
-> `MetaClawTrainer.setup()` / `MetaClawTrainer.run()`
-> `AsyncRolloutWorker(...)`
-> `MetaClawAPIServer.start()`

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/launcher.py`
  - 模块装配层，适合挂一个新的 evolution coordinator
- `MetaClaw/metaclaw/rollout.py`
  - 如果要观察 proxy worker 生命周期
- `MetaClaw/metaclaw/api_server.py`
  - 在线请求级最关键插点

## 5. LLM 请求代理 / 转发路径

### 5.1 OpenClaw 侧：把请求导向 MetaClaw

关键文件：

- `MetaClaw/metaclaw/launcher.py`
- `MetaClaw/metaclaw/claw_adapter.py`
- `MetaClaw/extensions/metaclaw-openclaw/index.ts`

职责：

- `launcher.py` 和 `claw_adapter.py` 都有 OpenClaw 配置写入逻辑
- 它们通过 `openclaw config set models.providers.metaclaw ...` 把 OpenClaw 默认模型 provider 指向 `http://127.0.0.1:{proxy_port}/v1`
- `MetaClaw/extensions/metaclaw-openclaw/index.ts` 额外在 OpenClaw 端注入 `X-Session-Id` / `X-Turn-Type`

主要调用链：

`MetaClawLauncher._configure_openclaw()`
-> `openclaw config set models.providers.metaclaw`
-> `openclaw config set agents.defaults.model.primary`
-> `openclaw gateway restart`

插件辅助链：

OpenClaw `before_prompt_build`
-> `MetaClaw/extensions/metaclaw-openclaw/index.ts`
-> patch `globalThis.fetch`
-> 后续 LLM POST 自动带上 `X-Session-Id` / `X-Turn-Type`

适合插 hook / adapter 的位置：

- `MetaClaw/extensions/metaclaw-openclaw/index.ts`
  - 若要在 OpenClaw 侧附加额外 trace header，这是最直接的位置

### 5.2 MetaClaw 侧：统一代理入口

关键文件：

- `MetaClaw/metaclaw/api_server.py`

职责：

- 提供 `/v1/chat/completions`
- 解析 `session_id`、`turn_type`、`session_done`、`memory_scope`
- 在转发前做 memory / skill augmentation
- 在转发后做记录、训练样本缓冲、session-end 后处理

主要调用链：

OpenClaw model request
-> `MetaClawAPIServer._build_app()`
-> `/v1/chat/completions`
-> `MetaClawAPIServer._handle_request(...)`
-> `_inject_memory()` / `_inject_skills()` / `_inject_augmentation()`
-> `_forward_to_llm()` 或 `_forward_to_tinker()`
-> 返回 OpenAI-compatible response

两个真实转发出口：

- `MetaClaw/metaclaw/api_server.py::_forward_to_llm`
  - `skills_only` 模式
  - 转发到真实 OpenAI-compatible LLM
- `MetaClaw/metaclaw/api_server.py::_forward_to_tinker`
  - `rl` 模式
  - 转发到 Tinker `SamplingClient.sample_async`

最适合插自定义 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_handle_request`
  - 全局 choke point，最适合接 request-time 自进化
- `MetaClaw/metaclaw/api_server.py::_forward_to_llm`
  - 最适合做上游 LLM 适配、response wrapper、埋点
- `MetaClaw/metaclaw/api_server.py::_forward_to_tinker`
  - RL 模式等价插点

## 6. Memory 读写 / 检索路径

### 6.1 Memory 初始化

关键文件：

- `MetaClaw/metaclaw/launcher.py`
- `MetaClaw/metaclaw/memory/manager.py`

职责：

- `launcher.py` 在启动时通过 `MemoryManager.from_config(cfg)` 初始化 memory 子系统
- `MemoryManager` 是统一 facade，持有：
  - `MemoryStore`
  - `MemoryRetriever`
  - `MemoryPolicyStore`
  - `MemoryTelemetryStore`
  - `MemoryConsolidator`

主要调用链：

`MetaClawLauncher._start_skills_only()` / `_start_rl()`
-> `MemoryManager.from_config(cfg)`
-> 构造 `MemoryRetriever` / `MemoryStore` / `MemoryTelemetryStore`
-> 注入 `AsyncRolloutWorker`
-> 注入 `MetaClawAPIServer`

### 6.2 Memory 读路径

关键文件：

- `MetaClaw/metaclaw/api_server.py`
- `MetaClaw/metaclaw/memory/manager.py`
- `MetaClaw/metaclaw/memory/retriever.py`

职责：

- `api_server.py::_inject_memory()` 在 prompt 构造阶段检索 memory
- `MemoryManager.retrieve_for_prompt()` 做 query 构造、cache、token budget 裁剪、render
- `MemoryRetriever.retrieve()` 执行 keyword / embedding / hybrid / auto 检索

主要调用链：

`MetaClawAPIServer._handle_request()`
-> `_inject_memory(messages, scope_id)`
-> `MemoryManager.retrieve_for_prompt(task_desc, scope_id)`
-> `MemoryRetriever.retrieve(query)`
-> `MemoryStore.search_keyword(...)` 或 embedding / hybrid 路径
-> `MemoryManager.render_for_prompt(units)`
-> 注入 system prompt

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_inject_memory`
  - 适合插入 retrieval policy、reranker、prompt template
- `MetaClaw/metaclaw/memory/manager.py::retrieve_for_prompt`
  - 适合插 cache、query rewrite、token budgeting
- `MetaClaw/metaclaw/memory/retriever.py::retrieve`
  - 适合研究 training-free 检索增强

### 6.3 Memory 写路径

关键文件：

- `MetaClaw/metaclaw/api_server.py`
- `MetaClaw/metaclaw/memory/manager.py`
- `MetaClaw/metaclaw/memory/telemetry.py`

职责：

- `api_server.py` 先把 turn 缓存在 `_session_memory_turns`
- session 结束后调用 `_ingest_memory_for_session()`
- `MemoryManager.ingest_session_turns()` 从 session turns 里抽取 memory units、去重、冲突检测、入库、consolidation、telemetry 记录
- `MemoryTelemetryStore.record()` 把事件写成 JSONL

主要调用链：

每个 turn
-> `MetaClawAPIServer._handle_request()`
-> `_session_memory_turns[session_id].append(turn_entry)`

session 结束
-> `MetaClawAPIServer._ingest_memory_for_session(...)`
-> `MemoryManager.ingest_session_turns(session_id, turns, scope_id)`
-> `_extract_memory_units_for_turn(...)`
-> `MemoryStore.add_memories(units)`
-> `MemoryTelemetryStore.record(...)`

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_ingest_memory_for_session`
  - session 级反思、摘要、过滤
- `MetaClaw/metaclaw/memory/manager.py::ingest_session_turns`
  - 写侧主入口，最适合插自定义抽取 / 去重 / merge policy
- `MetaClaw/metaclaw/memory/manager.py::register_event_callback`
  - 非侵入式 telemetry / evolution signal 接入点

### 6.4 Memory 相关状态 / API

关键文件：

- `MetaClaw/metaclaw/api_server.py`

已确认存在的 memory API：

- `/v1/memory/stats`
- `/v1/memory/search`
- `/v1/memory/health`
- `/v1/memory/summary`
- `/v1/memory/{memory_id}`
- `/v1/memory/action-plan`
- `/v1/memory/maintenance`
- `/v1/memory/feedback-analysis`
- `/v1/memory/operator-report`
- `/v1/memory/ingest`

说明：

- 这些接口很适合做离线调试与自进化效果观测

## 7. Skill 注入与演化逻辑

### 7.1 SkillManager：加载与检索

关键文件：

- `MetaClaw/metaclaw/skill_manager.py`

职责：

- 扫描 `skills_dir/*/SKILL.md`
- 解析 frontmatter
- 支持 `template` 与 `embedding` 两种检索
- 提供 `retrieve()` / `retrieve_relevant()` / `format_for_conversation()`
- 提供 `add_skills()` 并维护 `generation`

主要调用链：

启动时
-> `SkillManager(...)`
-> `_load_skills()`

请求时
-> `retrieve()` 或 `retrieve_relevant()`
-> `format_for_conversation()`

演化后
-> `add_skills(new_skills)`
-> 写入 `skills_dir/<skill>/SKILL.md`
-> `generation += 1`

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/skill_manager.py::_load_skills`
  - 自定义 skill 元数据加载
- `MetaClaw/metaclaw/skill_manager.py::retrieve`
  - request-time skill 检索策略
- `MetaClaw/metaclaw/skill_manager.py::retrieve_relevant`
  - synergy 模式下更稳妥的相关性过滤
- `MetaClaw/metaclaw/skill_manager.py::add_skills`
  - skill 审核 / 去重 / 打分

### 7.2 在线 skill 注入

关键文件：

- `MetaClaw/metaclaw/api_server.py`

职责：

- `_inject_skills()`：纯 skill 模式下注入技能文本
- `_inject_augmentation()`：memory + skill synergy 模式下联合注入

主要调用链：

`MetaClawAPIServer._handle_request()`
-> `_inject_skills(messages)` 或 `_inject_augmentation(messages, scope_id)`
-> `SkillManager.retrieve(...)` / `retrieve_relevant(...)`
-> `SkillManager.format_for_conversation(...)`
-> 写回 system message

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_inject_skills`
- `MetaClaw/metaclaw/api_server.py::_inject_augmentation`

### 7.3 Session 内 / Session 末 skill evolution

关键文件：

- `MetaClaw/metaclaw/api_server.py`
- `MetaClaw/metaclaw/skill_evolver.py`
- `MetaClaw/metaclaw/trainer.py`

职责：

- `api_server.py` 在在线对话中把 turn 缓存在 `_session_turns`
- 达到 `skill_evolution_every_n_turns` 时可提前触发 `_evolve_skills_for_session()`
- session 结束时也会对剩余 turns 做一次 evolution
- `skill_evolver.py` 使用 LLM 基于失败样本生成新技能
- `trainer.py` 的 `_maybe_evolve_skills()` 是 RL 训练阶段的另一条 evolution 路

主要调用链：

在线路径：

每个 main turn
-> `_session_turns[session_id].append(turn_entry)`
-> 达到阈值后 `_evolve_skills_for_session(turns)`
-> `SkillEvolver.evolve(...)`
-> `SkillManager.add_skills(...)`

RL 路径：

`MetaClawTrainer._maybe_evolve_skills(batch)`
-> `SkillEvolver.should_evolve(batch, threshold)`
-> `SkillEvolver.evolve(failed, current_skills)`
-> `SkillManager.add_skills(...)`
-> 若 `generation` 增加，则丢弃旧样本

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_evolve_skills_for_session`
  - session 级 training-free 反思最合适的位置之一
- `MetaClaw/metaclaw/skill_evolver.py::evolve`
  - skill proposal / critique / validation
- `MetaClaw/metaclaw/trainer.py::_maybe_evolve_skills`
  - hybrid “training-free + RL” 路径

## 8. Session 结束后的总结 / 后处理逻辑

### 8.1 MetaClaw：当前最完整的 session-end 后处理

关键文件：

- `MetaClaw/metaclaw/api_server.py`

职责：

- `session_done` 由 header 或 body 显式触发
- session 结束后会：
  - flush pending record
  - 尝试提交剩余 sample
  - 清理 turn 级状态
  - ingest memory
  - evolve skills

主要调用链：

`/v1/chat/completions`
-> 解析 `X-Session-Done` / `session_done`
-> `MetaClawAPIServer._handle_request(..., session_done=True, ...)`
-> `_flush_pending_record()`
-> `_maybe_submit_ready_samples(...)`
-> `_ingest_memory_for_session(...)`
-> `_evolve_skills_for_session(...)`
-> 清理 `_turn_counts` / `_teacher_tasks` / `_session_turns` / `_session_memory_turns`

说明：

- 这是当前仓库里最适合接 session-level reflection / summarization / trajectory distillation 的位置

最适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_handle_request` 内 `session_done` 分支
- `MetaClaw/metaclaw/api_server.py::_ingest_memory_for_session`
- `MetaClaw/metaclaw/api_server.py::_evolve_skills_for_session`

### 8.2 OpenClaw 原生 session lifecycle

关键文件：

- `openclaw/src/auto-reply/reply/session.ts`
- `openclaw/src/auto-reply/reply/session-hooks.ts`
- `openclaw/src/plugins/hooks.ts`

职责：

- `session.ts` 在新旧 session 切换时构造 start/end hook payload
- `session-hooks.ts` 只负责 payload 构造
- `plugins/hooks.ts` 提供 `runSessionStart()` / `runSessionEnd()`

主要调用链：

`initSessionState()`
-> `buildSessionEndHookPayload(...)`
-> `hookRunner.runSessionEnd(...)`
-> `buildSessionStartHookPayload(...)`
-> `hookRunner.runSessionStart(...)`

结论：

- OpenClaw 原生有 session 生命周期 hook
- 但没有在当前代码里看到独立的“session 总结 / 总结写回”主流程
- 如果要做原生总结，应该挂在 `session_end` hook 上

TODO：

- 若后续要做 OpenClaw 原生总结落盘，需要继续追具体哪个插件消费了 `session_end`

## 9. 日志记录与状态记录逻辑

### 9.1 MetaClaw：在线记录、训练记录、内存状态

关键文件：

- `MetaClaw/metaclaw/api_server.py`
- `MetaClaw/metaclaw/memory/telemetry.py`
- `MetaClaw/metaclaw/runtime_state.py`
- `MetaClaw/metaclaw/log_color.py`

职责：

- `api_server.py` 维护在线会话状态：
  - `_turn_counts`
  - `_pending_turn_data`
  - `_prm_tasks`
  - `_teacher_tasks`
  - `_pending_records`
  - `_session_effective`
  - `_session_turns`
  - `_session_memory_turns`
  - `_session_memory_scopes`
- `_buffer_record()` / `_flush_pending_record()` 负责 conversation JSONL 落盘
- `_append_prm_record()` 负责 PRM 结果落盘
- `_write_cached_system_prompt()` 负责 system prompt cache 落盘
- `MemoryTelemetryStore.record()` 负责 memory telemetry JSONL
- `runtime_state.py` 负责 PID / daemon start lock 等运行时状态

已确认的落盘文件：

- `conversations.jsonl`
- `prm_scores.jsonl`
- `system_prompt_cache.json`
- memory telemetry JSONL
- PID / lock 文件位于 `~/.metaclaw/`

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/api_server.py::_buffer_record`
  - trajectory 缓冲
- `MetaClaw/metaclaw/api_server.py::_flush_pending_record`
  - session / turn 级落盘前插点
- `MetaClaw/metaclaw/api_server.py::_append_prm_record`
  - reward / scoring 记录
- `MetaClaw/metaclaw/memory/telemetry.py::record`
  - memory 事件总线式埋点

### 9.2 OpenClaw：原生日志与插件状态

关键文件：

- `openclaw/src/cli/run-main.ts`
- `openclaw/src/plugins/loader.ts`
- `openclaw/src/plugins/hook-runner-global.ts`
- `openclaw/src/plugins/memory-state.ts`

职责：

- `run-main.ts` 通过 `enableConsoleCapture()` 把 CLI 输出接入结构化日志
- `loader.ts` 负责插件发现、加载、registry 构建、全局 hook runner 初始化
- `hook-runner-global.ts` 持有全局 hook runner 单例
- `memory-state.ts` 持有 memory 插件注册状态

说明：

- 如果未来想在 OpenClaw 原生层做统一事件采集，`plugins/loader.ts` 和 `hook-runner-global.ts` 是最稳的入口

## 10. RL 入口

### 10.1 MetaClaw 内部 RL 入口

关键文件：

- `MetaClaw/metaclaw/launcher.py`
- `MetaClaw/metaclaw/trainer.py`
- `MetaClaw/metaclaw/rollout.py`
- `MetaClaw/metaclaw/openclaw_env_rollout.py`
- `MetaClaw/metaclaw/api_server.py`

职责：

- `launcher.py::_start_rl()`：选择 RL 模式并装配 trainer
- `trainer.py::setup()`：创建 Tinker training client / sampling client / SkillManager / MemoryManager / rollout worker
- `trainer.py::run()`：主训练循环
- `rollout.py`：桥接在线 proxy 与训练端 queue
- `openclaw_env_rollout.py::rollout_loop()`：可选程序化任务 rollout
- `api_server.py::_forward_to_tinker()`：在线采样出口

主要调用链：

`metaclaw start --mode rl`
-> `MetaClawLauncher._start_rl()`
-> `MetaClawTrainer.setup()`
-> `AsyncRolloutWorker.start()`
-> 在线流量经 `MetaClawAPIServer`
-> 样本入 `output_queue`
-> `MetaClawTrainer.run()`
-> `_drain_with_pause_check()`
-> `_train_on_batch()`
-> `rollout_worker.update_sampling_client(...)`
-> `_maybe_evolve_skills(...)`

手动触发训练路径：

`MetaClawTrainer.train_step_external()`
-> 从 `output_queue` 取样本
-> `_train_on_batch()`
-> `_maybe_evolve_skills()`

适合插 hook / adapter 的位置：

- `MetaClaw/metaclaw/trainer.py::_train_on_batch`
  - hybrid 学习策略
- `MetaClaw/metaclaw/trainer.py::train_step_external`
  - bench-driven / offline-controlled 训练触发
- `MetaClaw/metaclaw/openclaw_env_rollout.py::rollout_loop`
  - 自动任务评测 / 采样

### 10.2 `OpenClaw-RL/` 中已存在的 RL 入口

关键文件：

- `OpenClaw-RL/openclaw-rl/openclaw_api_server.py`
- `OpenClaw-RL/openclaw-rl/openclaw_rollout.py`
- `OpenClaw-RL/openclaw-tinker/run.py`
- `OpenClaw-RL/openclaw-tinker/trainer.py`

职责：

- `openclaw-rl/` 是较原始的 OpenClaw RL proxy + rollout 实现
- `openclaw-tinker/run.py` 是 Tinker 版训练入口

说明：

- MetaClaw 明确继承了 `openclaw_api_server.py` / `openclaw_rollout.py` 的结构
- 如果后续要对照 RL 设计来源，可以从这两个目录继续深挖

## 11. OpenClaw 原生 hook / memory 扩展位

### 11.1 Hook 框架

关键文件：

- `openclaw/src/plugins/loader.ts`
- `openclaw/src/plugins/hook-runner-global.ts`
- `openclaw/src/plugins/hooks.ts`

职责：

- `loader.ts` 加载插件后调用 `initializeGlobalHookRunner(registry)`
- `hook-runner-global.ts` 暴露全局 hook runner
- `hooks.ts` 提供各类 hook 的运行接口

当前对自进化最相关的 hook：

- `before_model_resolve`
- `before_prompt_build`
- `llm_input`
- `llm_output`
- `agent_end`
- `session_start`
- `session_end`
- `before_tool_call`
- `after_tool_call`

### 11.2 Memory 插件状态

关键文件：

- `openclaw/src/plugins/memory-state.ts`

职责：

- 提供原生 memory 扩展注册位：
  - `registerMemoryPromptSection`
  - `registerMemoryFlushPlanResolver`
  - `registerMemoryRuntime`

结论：

- 如果未来希望去掉 MetaClaw proxy、直接在 OpenClaw 内部做长期记忆 / 自进化，可以优先围绕这三个接口设计

TODO：

- 当前仓库中没有继续追到“哪个具体插件在你当前部署里注册了这些 memory runtime 回调”

## 12. 最适合插入 training-free 自进化模块的位置

按优先级推荐：

### 12.1 第一优先级：MetaClaw session-end 后处理

位置：

- `MetaClaw/metaclaw/api_server.py::_handle_request` 的 `session_done` 分支
- `MetaClaw/metaclaw/api_server.py::_ingest_memory_for_session`
- `MetaClaw/metaclaw/api_server.py::_evolve_skills_for_session`

原因：

- 已经有完整 session trajectory
- 已经有 memory buffer / skill buffer / record buffer
- 最容易实现“总结 -> 提炼 -> 写回 -> 下一轮生效”

### 12.2 第二优先级：MetaClaw request-time augmentation

位置：

- `MetaClaw/metaclaw/api_server.py::_handle_request`
- `MetaClaw/metaclaw/api_server.py::_inject_memory`
- `MetaClaw/metaclaw/api_server.py::_inject_skills`
- `MetaClaw/metaclaw/api_server.py::_inject_augmentation`

原因：

- 这是“Agent 变强是否在推理时可见”的直接位置
- 适合做 training-free retrieval / rerank / prompt assembly 改进

### 12.3 第三优先级：Memory 写侧主入口

位置：

- `MetaClaw/metaclaw/memory/manager.py::ingest_session_turns`

原因：

- 是所有 session 经验进入长期记忆的总入口
- 适合做反思摘要、经验压缩、冲突消解、重要性估计

### 12.4 第四优先级：OpenClaw 原生 hooks

位置：

- `openclaw/src/agents/pi-embedded-runner/run/setup.ts`
- `openclaw/src/agents/pi-embedded-runner/run/attempt.ts`
- `openclaw/src/auto-reply/reply/session.ts`

原因：

- 如果未来想做“无 proxy 的原生集成”，这是正确方向
- 但相较 MetaClaw，当前这里缺少现成的 session-level memory / skill 写回闭环

## 13. 当前结论

对你现在这套部署来说，最关键的不是 OpenClaw core，而是 MetaClaw proxy 层。

如果目标是在 1 个月内尽快做出 training-free 自进化原型，最推荐的实现切入点是：

1. `MetaClaw/metaclaw/api_server.py::_handle_request`
2. `MetaClaw/metaclaw/api_server.py` 的 `session_done` 后处理分支
3. `MetaClaw/metaclaw/memory/manager.py::ingest_session_turns`
4. `MetaClaw/metaclaw/api_server.py::_inject_augmentation`

如果目标是后续做更“原生、可上游化”的 OpenClaw 插件实现，再把同样的逻辑逐步迁移到：

- `before_model_resolve`
- `before_prompt_build`
- `llm_input`
- `llm_output`
- `session_end`

## 14. TODO

- TODO: 继续追你当前部署里是否还有其他 OpenClaw memory 插件同时参与 prompt 注入
- TODO: 若后续要实现原生 OpenClaw memory 版本，需要继续定位谁调用了 `registerMemoryPromptSection` / `registerMemoryRuntime`
- TODO: 若后续要做更细的 hook matrix，可单独再产出一份“hook payload 对照表”
