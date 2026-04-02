## MetaClaw Plugin for OpenClaw v0.4.0

One-click installer for [MetaClaw](https://github.com/aiming-lab/MetaClaw) as an OpenClaw extension. No `git clone` required — download the zip, enable, and go.

### One-Click Install

#### macOS / Linux

```bash
curl -LO https://github.com/aiming-lab/MetaClaw/releases/download/v0.4.0/metaclaw-plugin.zip
unzip metaclaw-plugin.zip -d ~/.openclaw/extensions
openclaw plugins enable metaclaw-openclaw
openclaw gateway restart
```

> **China users**: If GitHub downloads are slow or timeout, use a mirror:
> ```bash
> curl -LO https://ghfast.top/https://github.com/aiming-lab/MetaClaw/releases/download/v0.4.0/metaclaw-plugin.zip
> ```

#### Windows (PowerShell)

```powershell
Invoke-WebRequest -Uri https://github.com/aiming-lab/MetaClaw/releases/download/v0.4.0/metaclaw-plugin.zip -OutFile metaclaw-plugin.zip
Expand-Archive metaclaw-plugin.zip -DestinationPath $env:USERPROFILE\.openclaw\extensions
openclaw plugins enable metaclaw-openclaw
openclaw gateway restart
```

> **China users**: If GitHub downloads are slow or timeout, replace the download URL with:
> ```
> https://ghfast.top/https://github.com/aiming-lab/MetaClaw/releases/download/v0.4.0/metaclaw-plugin.zip
> ```

### Then run

```bash
metaclaw setup
metaclaw start
```

### What the plugin does automatically

- Creates an isolated Python virtual environment (`.venv`)
- Installs MetaClaw (`[rl,evolve,scheduler]`) via pip
- Auto-installs the official WeChat plugin (`@tencent-weixin/openclaw-weixin`) when enabled
- Installs `metaclaw` CLI wrapper and configures PATH (macOS / Linux / Windows)
- Patches outbound LLM `fetch` to inject `X-Session-Id` / `X-Turn-Type` headers

### Requirements

- **Python ≥ 3.11**
- **OpenClaw** (any version)
- macOS, Linux, or Windows
- **RAM ≥ 4 GB recommended.** On machines with ≤ 2 GB RAM, add swap before installing:
  ```bash
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  ```

### What's new in v0.4.0

- **Native OpenClaw extension** — MetaClaw plugs into OpenClaw’s plugin model instead of sitting beside it as a second, hand-wired stack.
- **Two-step install** — drop the folder into OpenClaw’s extensions tree, then one command to enable and restart the gateway.
- **Self-contained bundle** — everything ships in the extension package so you’re not gluing repos, paths, and runtimes together yourself.

### Full auto mode (optional)

```bash
openclaw config set plugins.entries.metaclaw-openclaw.config.oneClickMetaclaw true
```

Enables: venv + pip + default config + `metaclaw start` on gateway load.

### Configuration

See [README](https://github.com/aiming-lab/MetaClaw/blob/main/extensions/metaclaw-openclaw/README.md) for all config options.
