"""
Unified configuration for MetaClaw training.

Dataclass-based config compatible with command-line overrides.
"""

from dataclasses import dataclass, field


@dataclass
class MetaClawConfig:
    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    model_name: str = "Qwen/Qwen3-4B"
    lora_rank: int = 32
    renderer_name: str = "qwen3"  # Tinker renderer: "qwen3", "llama3", "role_colon"

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    learning_rate: float = 1e-4
    batch_size: int = 4           # Number of ConversationSamples per training step
    max_steps: int = 1000
    loss_fn: str = "importance_sampling"  # "ppo" | "importance_sampling" | "cispo"
    save_weights_timeout_s: float = 200.0  # timeout for sampling-client refresh
    resume_from_ckpt: str = ""    # optional Tinker resume path, e.g. tinker://.../weights/step_0003

    # ------------------------------------------------------------------ #
    # Reward / PRM                                                        #
    # ------------------------------------------------------------------ #
    use_prm: bool = True
    # Provider: "openai" (any OpenAI-compatible URL) | "bedrock" (AWS Bedrock)
    prm_provider: str = "openai"
    # Any OpenAI-compatible base URL (ignored when prm_provider="bedrock"):
    prm_url: str = "https://api.openai.com/v1"
    prm_model: str = "gpt-5.2"  # judge model
    prm_api_key: str = ""                    # set via env var or directly (ignored for bedrock)
    prm_m: int = 3                           # majority-vote count
    prm_temperature: float = 0.6
    prm_max_new_tokens: int = 1024
    use_opd: bool = False                    # OPD (teacher logprobs) mode
    teacher_url: str = ""                    # Teacher model base URL (OpenAI-compatible /v1/completions)
    teacher_model: str = ""                  # Teacher model name
    teacher_api_key: str = ""                # Teacher model API key
    kl_penalty_coef: float = 1.0             # KL penalty coefficient for OPD

    # ------------------------------------------------------------------ #
    # Skills                                                              #
    # ------------------------------------------------------------------ #
    use_skills: bool = False
    skills_dir: str = "memory_data/skills"    # directory of individual *.md skill files
    retrieval_mode: str = "template"          # "template" | "embedding"
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    skill_top_k: int = 6                      # General skills to inject
    task_specific_top_k: int = 10    # Task-specific skills cap; None means no cap
    enable_skill_evolution: bool = False
    skill_evolution_every_n_turns: int = 10  # Every N conversation turns (main turns), run skill evolution on those turns (RL and skills_only)
    skill_update_threshold: float = 0.4       # Evolve when success rate < threshold (trainer batch evolution)
    max_new_skills: int = 3
    adaptive_skill_routing_enabled: bool = True
    adaptive_skill_stats_path: str = "records/skill_stats.json"
    adaptive_skill_feedback_weight: float = 0.35
    adaptive_skill_relevance_weight: float = 1.0
    important_feedback_skill_name: str = "important-notes"
    important_feedback_skill_description: str = "Use on every task. Contains persistent cautionary notes distilled from real user feedback on previous agent behavior."
    feedback_enabled: bool = True
    feedback_history_path: str = "records/feedback.jsonl"
    feedback_skill_model_id: str = "deepseek-chat"
    feedback_skill_api_base: str = "https://api.deepseek.com/v1"
    feedback_skill_api_key: str = "sk-33021c0bec434de4b877c3142cc409c9"
    feedback_skill_max_completion_tokens: int = 1200
    user_profile_enabled: bool = True
    user_profile_path: str = "records/user_profiles.json"
    user_profile_max_entries: int = 12
    task_brief_enabled: bool = True
    task_brief_path: str = "records/task_briefs.json"
    task_brief_model_id: str = "deepseek-chat"
    task_brief_api_base: str = "https://api.deepseek.com/v1"
    task_brief_api_key: str = "sk-33021c0bec434de4b877c3142cc409c9"
    task_brief_max_completion_tokens: int = 700
    session_report_enabled: bool = True
    session_report_path: str = "records/session_reports.jsonl"
    session_report_model_id: str = "deepseek-chat"
    session_report_api_base: str = "https://api.deepseek.com/v1"
    session_report_api_key: str = "sk-33021c0bec434de4b877c3142cc409c9"
    session_report_max_completion_tokens: int = 1000

    # ------------------------------------------------------------------ #
    # Memory                                                              #
    # ------------------------------------------------------------------ #
    memory_enabled: bool = False
    memory_dir: str = "memory_data/store"
    memory_store_path: str = "memory_data/store/memory.db"
    memory_scope: str = "default"
    memory_retrieval_mode: str = "keyword"   # "keyword" | "hybrid" | "embedding"
    memory_use_embeddings: bool = False
    memory_embedding_mode: str = "hashing"  # "hashing" | "semantic"
    memory_embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model name
    memory_embedding_model_path: str = "Qwen/Qwen3-Embedding-0.6B"
    memory_policy_path: str = "memory_data/store/policy.json"
    memory_telemetry_path: str = "memory_data/store/telemetry.jsonl"
    memory_auto_upgrade_enabled: bool = False
    memory_auto_upgrade_interval_seconds: int = 900
    memory_auto_upgrade_require_review: bool = True
    memory_review_stale_after_hours: int = 72
    memory_max_injected_units: int = 6
    memory_max_injected_tokens: int = 800
    memory_auto_extract: bool = True
    memory_auto_consolidate: bool = True
    memory_ignore_turn_type: bool = False   # buffer all turns (incl. side) for memory
    memory_manual_trigger: bool = False     # disable auto-ingest on session_done; use POST /v1/memory/ingest instead

    # ------------------------------------------------------------------ #
    # Skill-Memory Synergy (only active when both are enabled)            #
    # ------------------------------------------------------------------ #
    synergy_enabled: bool = True              # enable coordinated injection
    synergy_token_budget: int = 1200          # combined token cap for skill + memory
    synergy_skill_ratio: float = 0.35         # initial skill share (memory gets 1 - ratio)
    synergy_dedup_threshold: float = 0.5      # Jaccard overlap to drop a procedural memory

    # ------------------------------------------------------------------ #
    # Context window                                                       #
    # ------------------------------------------------------------------ #
    max_context_tokens: int = 200000            # hard cap on prompt token count; must match
                                              # Tinker's max_seq_len minus headroom for response
    context_summary_enabled: bool = True
    context_summary_trigger_ratio: float = 0.85
    context_summary_recent_messages: int = 8
    context_summary_max_chars: int = 4000
    context_summary_max_completion_tokens: int = 900
    context_summary_model_id: str = "deepseek-chat"
    context_summary_api_base: str = "https://api.deepseek.com/v1"
    context_summary_api_key: str = "sk-33021c0bec434de4b877c3142cc409c9"
    context_summary_store_path: str = "records/context_summaries.json"
    context_summary_compact_retries: int = 2

    # ------------------------------------------------------------------ #
    # API Server                                                          #
    # ------------------------------------------------------------------ #
    proxy_port: int = 30000
    proxy_host: str = "0.0.0.0"
    tinker_sampling_url: str = "http://localhost:8080"  # Tinker sampling endpoint
    served_model_name: str = "qwen3-4b"
    api_key: str = ""                         # Optional bearer token check
    record_enabled: bool = True
    record_dir: str = "records/"
    record_openclaw_rl_file: str = "openclaw_rl_records.jsonl"
    record_enriched_file: str = "conversations.jsonl"
    sandbox_enabled: bool = True
    sandbox_command_policy_enabled: bool = True
    sandbox_path_policy_enabled: bool = True
    sandbox_approval_enabled: bool = True
    sandbox_audit_enabled: bool = True
    sandbox_audit_path: str = "records/sandbox_audit.jsonl"
    sandbox_approval_state_path: str = "records/sandbox_approvals.json"
    sandbox_approval_history_path: str = "records/sandbox_approval_history.jsonl"
    sandbox_whitelist_path: str = "records/sandbox_whitelist.json"
    skill_selection_state_path: str = "records/skill_selection_state.json"

    # ------------------------------------------------------------------ #
    # Programmatic task rollout (Qwen3-native, no OpenClaw TUI needed)  #
    # ------------------------------------------------------------------ #
    # Directory containing task JSONL files in slime-compatible format:
    #   <openclaw_env_data_dir>/<split>.jsonl
    # Each line: {"task_id": "...", "instruction": "..."}
    # Leave empty ("") to skip programmatic rollout (passive proxy mode,
    # consistent with OpenClaw-RL's --disable-rollout-global-dataset).
    openclaw_env_data_dir: str = ""           # e.g. "/path/to/tasks"
    openclaw_env_split: str = "train"         # jsonl split name
    openclaw_env_concurrency: int = 4         # parallel episodes
    openclaw_env_max_steps: int = 15          # max turns per episode
    openclaw_env_python_path: str = ""        # unused (kept for compatibility)

    # ------------------------------------------------------------------ #
    # Operating mode                                                      #
    # ------------------------------------------------------------------ #
    # "auto"        — v0.3: RL + scheduler (trains during idle/sleep windows)
    # "rl"          — v0.2: RL without scheduler (trains immediately on full batch)
    # "skills_only" — proxy + skill injection only (no Tinker, no RL)
    mode: str = "auto"
    # When True (RL/auto mode only), the trainer does NOT run its own
    # collection loop.  Instead it waits for ``metaclaw train-step`` CLI
    # invocations (or admin API calls) to trigger individual RL updates.
    manual_train_trigger: bool = False

    # ------------------------------------------------------------------ #
    # Scheduler (meta-learning: gate slow RL updates to idle windows)     #
    # ------------------------------------------------------------------ #
    
    scheduler_enabled: bool = True
    scheduler_idle_threshold_minutes: int = 30
    scheduler_sleep_start: str = "23:00"   # HH:MM 24h local time
    scheduler_sleep_end: str = "07:00"
    scheduler_min_window_minutes: int = 15  # minimum window needed for one RL step
    scheduler_calendar_enabled: bool = False
    scheduler_calendar_credentials_path: str = ""
    scheduler_calendar_token_path: str = ""  # default set in config_store

    # ------------------------------------------------------------------ #
    # LLM for skills_only forwarding (OpenAI-compatible)                 #
    # ------------------------------------------------------------------ #
    llm_api_base: str = ""      # e.g. https://api.moonshot.cn/v1
    llm_api_key: str = ""       # bearer token for upstream LLM
    llm_model_id: str = ""      # model name to forward to

    # ------------------------------------------------------------------ #
    # LLM for skill evolution                                             #
    # ------------------------------------------------------------------ #
    # Provider: "openai" | "bedrock"
    evolver_provider: str = "openai"
    azure_openai_deployment: str = "o3"  # kept for backward compat
    evolver_api_base: str = ""           # leave empty to reuse llm_api_base
    evolver_api_key: str = ""            # leave empty to reuse llm_api_key
    evolver_model_id: str = ""
    # AWS Bedrock region (used when prm_provider or evolver_provider = "bedrock")
    bedrock_region: str = "us-east-1"
    skill_evolution_history_path: str = "memory_data/skills/evolution_history.jsonl"

    # ------------------------------------------------------------------ #
    # WeChat (official openclaw-weixin plugin, auto-installed)           #
    # ------------------------------------------------------------------ #
    wechat_enabled: bool = False
