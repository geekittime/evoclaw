#!/usr/bin/env python3
"""Normalize MetaClaw trajectory records into OpenClaw-RL-compatible JSONL.

Usage:
    python import_metaclaw_records.py \
        --input /path/to/openclaw_rl_records.jsonl \
        --output /path/to/results/normalized_records.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize MetaClaw records for OpenClaw-RL.")
    parser.add_argument("--input", required=True, help="Path to MetaClaw OpenClaw-RL JSONL records.")
    parser.add_argument("--output", required=True, help="Path to normalized JSONL output.")
    return parser.parse_args()



def normalize_record(raw: dict) -> dict:
    return {
        "session_id": raw.get("session_id", ""),
        "turn": int(raw.get("turn", 0) or 0),
        "timestamp": raw.get("timestamp", ""),
        "messages": raw.get("messages", []) or [],
        "prompt_text": raw.get("prompt_text", "") or "",
        "response_text": raw.get("response_text", "") or "",
        "tool_calls": raw.get("tool_calls"),
        "next_state": raw.get("next_state"),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            dst.write(json.dumps(normalize_record(raw), ensure_ascii=False) + "\n")
            count += 1

    print(f"normalized {count} records -> {output_path}")


if __name__ == "__main__":
    main()
