# MetaClaw UI Integration

## Overview

This repository now contains a mixed `MetaClaw + openclaw UI` implementation.

The MetaClaw backend is responsible for:

- feedback ingestion
- important-notes persistence
- session-level skill selection
- sandbox command and path policy
- pending command approvals
- REST APIs for the Web UI

The openclaw frontend is responsible for:

- rendering approval buttons instead of requiring `/approve`
- rendering per-answer Good / Bad feedback controls
- rendering command policy and path access controls
- rendering skill selection controls
- rendering the current `important-notes`

## Backend APIs

Implemented in [MetaClaw/metaclaw/api_server.py](/c:/Users/wby/Desktop/evoclaw/MetaClaw/metaclaw/api_server.py).

- `POST /v1/feedback`
  Sends `session_id`, `turn`, `rating`, and `feedback`.
  The related user question, assistant answer, and feedback are summarized by the feedback model and appended into `important-notes`.

- `GET /v1/skills`
  Returns all available skills, session selection state, latest injected skills, and current `important-notes`.

- `PUT /v1/skills/selection`
  Persists the enabled skill names for a session.

- `GET /v1/sandbox/pending`
  Returns pending sandbox approval requests for the current session.

- `GET /v1/sandbox/whitelist`
  Returns the sandbox state snapshot:
  command allowlist, path allowlist, command rules, default command mode, and path blocklist.

- `PUT /v1/sandbox/policy`
  Updates default command mode, command three-state rules, and blocked paths.

- `POST /v1/sandbox/whitelist`
  Adds a command or path allowlist entry.

- `DELETE /v1/sandbox/whitelist`
  Removes a command or path allowlist entry.

- `POST /v1/sandbox/approve`
  Approves a pending sandbox request.

- `POST /v1/sandbox/reject`
  Rejects a pending sandbox request.

## Sandbox State

Implemented in [MetaClaw/metaclaw/sandbox.py](/c:/Users/wby/Desktop/evoclaw/MetaClaw/metaclaw/sandbox.py).

Persistent state now includes:

- `default_command_mode`
- `command_rules`
- `command_allowlist`
- `path_allowlist`
- `path_blocklist`

Command resolution priority is:

1. path violations
2. explicit command rule: `deny`
3. explicit command rule: `ask`
4. explicit command rule: `allow`
5. command allowlist
6. built-in high-risk checks
7. default command mode

## Frontend Integration

Implemented in:

- [openclaw/ui/src/ui/controllers/metaclaw.ts](/c:/Users/wby/Desktop/evoclaw/openclaw/ui/src/ui/controllers/metaclaw.ts)
- [openclaw/ui/src/ui/views/chat.ts](/c:/Users/wby/Desktop/evoclaw/openclaw/ui/src/ui/views/chat.ts)
- [openclaw/ui/src/ui/app.ts](/c:/Users/wby/Desktop/evoclaw/openclaw/ui/src/ui/app.ts)
- [openclaw/ui/src/ui/app-render.ts](/c:/Users/wby/Desktop/evoclaw/openclaw/ui/src/ui/app-render.ts)
- [openclaw/ui/src/styles/chat/layout.css](/c:/Users/wby/Desktop/evoclaw/openclaw/ui/src/styles/chat/layout.css)

The UI now includes:

- MetaClaw Studio dashboard inside chat
- approval buttons for pending commands
- command allow / ask / deny rule editing
- allowlist and blocklist path editing
- command allowlist editing
- session skill selection with persisted overrides
- `important-notes` viewer
- feedback controls under each assistant answer

## Feedback Behavior

Feedback is attached to a specific assistant turn in the UI.

Both `good` and `bad` feedback now:

- target a specific turn
- include the related conversation context
- get summarized by the configured feedback model
- append a reusable lesson into `important-notes`

## Local Verification Notes

In this environment:

- Python source can be syntax-checked locally.
- Node is available.
- `pnpm` and `node_modules` are not present, so a full UI build cannot be completed without installing frontend dependencies first.
