"""
End-to-end example: auto training + auto replayed conversation input.

Run:
    python examples/metaclaw/run_conversation_replay.py

Behavior:
  - Start MetaClawTrainer for RL training.
  - Read user messages from a source JSONL file.
  - Replay each message via run_turn(message), one by one.
  - Optionally loop replay continuously until training ends.
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

# Make sure SkillRL root is importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metaclaw.config import MetaClawConfig
from metaclaw.trainer import MetaClawTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = PROJECT_ROOT / "metaclaw" / "utils.py"


def _load_run_turn():
    spec = importlib.util.spec_from_file_location("metaclaw_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load utils module from: {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_turn


run_turn = _load_run_turn()


def _extract_user_text(record: dict) -> str | None:
    if record.get("type") != "message":
        return None
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None

    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        return "\n".join(parts).strip() or None
    return None


def _iter_user_messages(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_text = _extract_user_text(record)
            if user_text:
                yield line_num, user_text


def _load_user_messages(jsonl_path: Path) -> list[tuple[int, str]]:
    return list(_iter_user_messages(jsonl_path))


async def _replay_loop(
    items: list[tuple[int, str]],
    sleep_seconds: float,
    replay_epochs: int,
    trainer_task: asyncio.Task,
):
    if not items:
        logger.warning("[Replay] no user messages found; replay loop exits")
        return

    if replay_epochs < 0:
        raise ValueError("replay_epochs must be >= 0")

    epoch = 1
    # replay_epochs == 0 means replay forever until trainer exits.
    while replay_epochs == 0 or epoch <= replay_epochs:
        if trainer_task.done():
            logger.info("[Replay] trainer finished, stopping replay loop")
            return

        logger.info("[Replay] starting epoch %d", epoch)
        for turn_idx, (line_num, user_text) in enumerate(items, start=1):
            if trainer_task.done():
                logger.info("[Replay] trainer finished, stopping replay loop")
                return

            logger.info(
                "[Replay] epoch=%d turn=%d/%d source_line=%d",
                epoch,
                turn_idx,
                len(items),
                line_num,
            )
            logger.info("[Replay] user=%s", user_text[:160].replace("\n", " "))
            output = await asyncio.to_thread(run_turn, user_text)
            logger.info("[Replay] output=%s", (output or "").strip()[:300])

            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

        epoch += 1

    logger.info("[Replay] replay finished after %d epochs", replay_epochs)


async def _wait_for_proxy_ready(host: str, port: int, timeout_s: float = 180.0):
    """Wait until FastAPI proxy is reachable before replay starts."""
    url = f"http://{host}:{port}/docs"
    if host == "0.0.0.0":
        url = f"http://127.0.0.1:{port}/docs"

    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
            if resp.status_code < 500:
                logger.info("[Replay] proxy ready at %s (status=%d)", url, resp.status_code)
                return
        except Exception:
            pass

        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Proxy did not become ready within {timeout_s:.1f}s: {url}")
        await asyncio.sleep(1.0)


async def main():
    # Replay config
    input_path = Path("data/e28edc95-e830-4ac9-afd9-27502870d645.jsonl")
    replay_limit = 0            # 0 means use all user messages
    replay_epochs = 5           # 0 means replay forever until training completes
    sleep_seconds = 0.5         # pacing between auto inputs
    warmup_seconds = 0.0        # optional extra delay after proxy-ready check

    # Trainer config (same style as run_conversation_rl.py)
    config = MetaClawConfig(
        # Model
        model_name="moonshotai/Kimi-K2.5",
        served_model_name="Kimi-K2.5",
        lora_rank=32,
        renderer_name="kimi",

        # Training
        learning_rate=1e-4,
        batch_size=12,
        max_steps=300,
        loss_fn="importance_sampling",

        # PRM reward
        use_prm=True,
        prm_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        prm_model=os.environ.get("PRM_MODEL", ""),
        prm_api_key=os.environ.get("OPENAI_API_KEY", ""),
        prm_m=3,

        # Skills
        use_skills=True,
        skills_dir="memory_data/skills",
        retrieval_mode="template",
        skill_top_k=6,

        # Skill evolution
        enable_skill_evolution=True,
        skill_update_threshold=0.4,
        max_new_skills=3,

        # Proxy server
        proxy_port=30000,
        proxy_host="0.0.0.0",

        # Keep passive mode; replay loop injects "manual-like" inputs.
        openclaw_env_data_dir="",
    )

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    items = _load_user_messages(input_path)
    if replay_limit > 0:
        items = items[:replay_limit]
    logger.info("[Replay] loaded %d user messages from %s", len(items), input_path)

    trainer = MetaClawTrainer(config)
    trainer_task = asyncio.create_task(trainer.run())
    await _wait_for_proxy_ready(config.proxy_host, config.proxy_port)
    if warmup_seconds > 0:
        await asyncio.sleep(warmup_seconds)

    replay_task = asyncio.create_task(
        _replay_loop(items, sleep_seconds, replay_epochs, trainer_task)
    )

    done, pending = await asyncio.wait(
        {trainer_task, replay_task},
        return_when=asyncio.FIRST_EXCEPTION,
    )
    for task in pending:
        task.cancel()
    for task in done:
        exc = task.exception()
        if exc is not None:
            raise exc


if __name__ == "__main__":
    asyncio.run(main())
