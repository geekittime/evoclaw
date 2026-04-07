#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export METACLAW_PID_DIR="${METACLAW_PID_DIR:-/tmp/metaclaw}"
METACLAW_BIN="${METACLAW_BIN:-metaclaw}"

cd "${ROOT_DIR}/MetaClaw"
exec "${METACLAW_BIN}" start --mode skills_only --port 30001
