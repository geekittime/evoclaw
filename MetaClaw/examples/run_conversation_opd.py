"""
End-to-end example: conversation RL in OPD (teacher logprobs) mode.

OPD uses teacher model logprobs alongside the student's rollout logprobs.
Configure with loss_fn="cispo" (or "importance_sampling") and use_opd=True.

Run:
    python examples/metaclaw/run_conversation_opd.py
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metaclaw.config import MetaClawConfig
from metaclaw.trainer import MetaClawTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


async def main():
    config = MetaClawConfig(
        model_name="Qwen/Qwen3-8B",
        lora_rank=32,
        renderer_name="qwen3",

        # OPD mode — teacher model provides reference logprobs
        use_opd=True,
        loss_fn="cispo",
        teacher_url="http://localhost:8082/v1",   # OpenAI-compatible completions endpoint
        teacher_model="Qwen/Qwen3-32B",           # teacher (larger) model
        teacher_api_key="",                        # set via env or directly
        kl_penalty_coef=1.0,                       # KL penalty strength

        learning_rate=1e-4,
        batch_size=32,
        max_steps=300,

        use_prm=True,
        prm_url="http://localhost:8081",
        prm_m=3,

        use_skills=True,
        skills_json_path="memory_data/conversation/conversation_skills.json",
        retrieval_mode="template",

        enable_skill_evolution=False,

        proxy_port=30000,
    )

    trainer = MetaClawTrainer(config)
    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
