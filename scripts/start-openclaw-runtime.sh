#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/runtime/openclaw-state"
SOURCE_CONFIG="${OPENCLAW_SOURCE_CONFIG_PATH:-${HOME}/.openclaw/openclaw.json}"
TARGET_CONFIG="${RUNTIME_DIR}/openclaw.json"
METACLAW_UPSTREAM="${OPENCLAW_METACLAW_UPSTREAM:-http://127.0.0.1:30001}"

mkdir -p "${RUNTIME_DIR}"

node - "${SOURCE_CONFIG}" "${TARGET_CONFIG}" "${METACLAW_UPSTREAM}" <<'NODE'
const fs = require("node:fs");

const [sourceConfig, targetConfig, metaclawUpstream] = process.argv.slice(2);
const cfg = JSON.parse(fs.readFileSync(sourceConfig, "utf8"));

cfg.models ??= {};
cfg.models.providers ??= {};
cfg.models.providers.metaclaw ??= {};
cfg.models.providers.metaclaw.baseUrl = `${metaclawUpstream.replace(/\/+$/, "")}/v1`;

cfg.gateway ??= {};
cfg.gateway.port = 18789;
cfg.gateway.mode = "local";
cfg.gateway.bind = "loopback";
cfg.gateway.controlUi ??= {};
cfg.gateway.controlUi.allowInsecureAuth = true;

fs.writeFileSync(targetConfig, JSON.stringify(cfg, null, 2));
NODE

export OPENCLAW_CONFIG_PATH="${TARGET_CONFIG}"
export OPENCLAW_STATE_DIR="${RUNTIME_DIR}"
export OPENCLAW_METACLAW_UPSTREAM="${METACLAW_UPSTREAM}"

cd "${ROOT_DIR}/openclaw"
exec node openclaw.mjs gateway run --force --bind loopback --port 18789
