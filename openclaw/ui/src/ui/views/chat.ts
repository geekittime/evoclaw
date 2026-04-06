import { html, nothing, type TemplateResult } from "lit";
import { ref } from "lit/directives/ref.js";
import { repeat } from "lit/directives/repeat.js";
import type {
  CompactionStatus as CompactionIndicatorStatus,
  FallbackStatus as FallbackIndicatorStatus,
} from "../app-tool-stream.ts";
import {
  CHAT_ATTACHMENT_ACCEPT,
  isSupportedChatAttachmentMimeType,
} from "../chat/attachment-support.ts";
import { DeletedMessages } from "../chat/deleted-messages.ts";
import { exportChatMarkdown } from "../chat/export.ts";
import {
  renderMessageGroup,
  renderReadingIndicatorGroup,
  renderStreamingGroup,
} from "../chat/grouped-render.ts";
import { InputHistory } from "../chat/input-history.ts";
import { normalizeMessage, normalizeRoleForGrouping } from "../chat/message-normalizer.ts";
import { PinnedMessages } from "../chat/pinned-messages.ts";
import { getPinnedMessageSummary } from "../chat/pinned-summary.ts";
import { messageMatchesSearchQuery } from "../chat/search-match.ts";
import { getOrCreateSessionCacheValue } from "../chat/session-cache.ts";
import {
  CATEGORY_LABELS,
  SLASH_COMMANDS,
  getSlashCommandCompletions,
  type SlashCommandCategory,
  type SlashCommandDef,
} from "../chat/slash-commands.ts";
import { isSttSupported, startStt, stopStt } from "../chat/speech.ts";
import { icons } from "../icons.ts";
import { detectTextDirection } from "../text-direction.ts";
import type { GatewaySessionRow, SessionsListResult } from "../types.ts";
import type { ChatItem, MessageGroup } from "../types/chat-types.ts";
import type { ChatAttachment, ChatQueueItem } from "../ui-types.ts";
import type { MetaclawFeedbackResponse } from "../controllers/metaclaw.ts";
import { agentLogoUrl, resolveAgentAvatarUrl } from "./agents-utils.ts";
import { renderMarkdownSidebar } from "./markdown-sidebar.ts";
import "../components/resizable-divider.ts";

export type ChatProps = {
  sessionKey: string;
  onSessionKeyChange: (next: string) => void;
  thinkingLevel: string | null;
  showThinking: boolean;
  showToolCalls: boolean;
  loading: boolean;
  sending: boolean;
  canAbort?: boolean;
  compactionStatus?: CompactionIndicatorStatus | null;
  fallbackStatus?: FallbackIndicatorStatus | null;
  messages: unknown[];
  toolMessages: unknown[];
  streamSegments: Array<{ text: string; ts: number }>;
  stream: string | null;
  streamStartedAt: number | null;
  assistantAvatarUrl?: string | null;
  draft: string;
  queue: ChatQueueItem[];
  connected: boolean;
  canSend: boolean;
  disabledReason: string | null;
  error: string | null;
  sessions: SessionsListResult | null;
  focusMode: boolean;
  sidebarOpen?: boolean;
  sidebarContent?: string | null;
  sidebarError?: string | null;
  splitRatio?: number;
  assistantName: string;
  assistantAvatar: string | null;
  attachments?: ChatAttachment[];
  onAttachmentsChange?: (attachments: ChatAttachment[]) => void;
  showNewMessages?: boolean;
  onScrollToBottom?: () => void;
  onRefresh: () => void;
  onToggleFocusMode: () => void;
  getDraft?: () => string;
  onDraftChange: (next: string) => void;
  onRequestUpdate?: () => void;
  onSend: () => void;
  onAbort?: () => void;
  onQueueRemove: (id: string) => void;
  onNewSession: () => void;
  onClearHistory?: () => void;
  agentsList: {
    agents: Array<{ id: string; name?: string; identity?: { name?: string; avatarUrl?: string } }>;
    defaultId?: string;
  } | null;
  currentAgentId: string;
  onAgentChange: (agentId: string) => void;
  onNavigateToAgent?: () => void;
  onSessionSelect?: (sessionKey: string) => void;
  onOpenSidebar?: (content: string) => void;
  onCloseSidebar?: () => void;
  onSplitRatioChange?: (ratio: number) => void;
  onChatScroll?: (event: Event) => void;
  basePath?: string;
  metaclaw?: {
    apiBase: string;
    token: string;
    loading: boolean;
    saving: boolean;
    connected: boolean;
    error: string | null;
    pendingApprovals: Array<{
      approval_id: string;
      created_at: string;
      decisions?: Array<{ tool_name?: string; command?: string; reason?: string; action?: string }>;
    }>;
    sandboxPolicy: {
      command_allowlist: string[];
      path_allowlist: string[];
      command_rules: Record<string, "allow" | "ask" | "deny">;
      default_command_mode: "allow" | "ask" | "deny";
      path_blocklist: string[];
    } | null;
    skills: Array<{ name: string; description: string; category: string }>;
    selectedSkillNames: string[];
    selectionCustomized: boolean;
    latestInjectedSkills: string[];
    importantNotes: { name: string; description: string; content: string } | null;
    onApiBaseChange: (value: string) => void;
    onTokenChange: (value: string) => void;
    onRefresh: () => void;
    onApprove: (approvalId: string) => void;
    onReject: (approvalId: string) => void;
    onSavePolicy: (policy: {
      command_allowlist: string[];
      path_allowlist: string[];
      command_rules: Record<string, "allow" | "ask" | "deny">;
      default_command_mode: "allow" | "ask" | "deny";
      path_blocklist: string[];
    }) => void;
    onAddWhitelistEntry: (type: "command" | "path", value: string) => void;
    onRemoveWhitelistEntry: (type: "command" | "path", value: string) => void;
    onSaveSkillSelection: (skillNames: string[] | null) => void;
    onSubmitFeedback: (
      turn: number | null,
      rating: "good" | "bad",
      feedback: string,
    ) => Promise<MetaclawFeedbackResponse>;
  };
};

const COMPACTION_TOAST_DURATION_MS = 5000;
const FALLBACK_TOAST_DURATION_MS = 8000;

// Persistent instances keyed by session
const inputHistories = new Map<string, InputHistory>();
const pinnedMessagesMap = new Map<string, PinnedMessages>();
const deletedMessagesMap = new Map<string, DeletedMessages>();

function getInputHistory(sessionKey: string): InputHistory {
  return getOrCreateSessionCacheValue(inputHistories, sessionKey, () => new InputHistory());
}

function getPinnedMessages(sessionKey: string): PinnedMessages {
  return getOrCreateSessionCacheValue(
    pinnedMessagesMap,
    sessionKey,
    () => new PinnedMessages(sessionKey),
  );
}

function getDeletedMessages(sessionKey: string): DeletedMessages {
  return getOrCreateSessionCacheValue(
    deletedMessagesMap,
    sessionKey,
    () => new DeletedMessages(sessionKey),
  );
}

interface ChatEphemeralState {
  sttRecording: boolean;
  sttInterimText: string;
  slashMenuOpen: boolean;
  slashMenuItems: SlashCommandDef[];
  slashMenuIndex: number;
  slashMenuMode: "command" | "args";
  slashMenuCommand: SlashCommandDef | null;
  slashMenuArgItems: string[];
  searchOpen: boolean;
  searchQuery: string;
  pinnedExpanded: boolean;
  feedbackTargetTurn: number | null;
  feedbackRating: "good" | "bad";
  feedbackText: string;
  feedbackSaving: boolean;
  feedbackMessage: string;
  metaclawRuleCommand: string;
  metaclawRuleMode: "allow" | "ask" | "deny";
  metaclawWhitelistCommand: string;
  metaclawWhitelistPath: string;
  metaclawBlockedPath: string;
}

function createChatEphemeralState(): ChatEphemeralState {
  return {
    sttRecording: false,
    sttInterimText: "",
    slashMenuOpen: false,
    slashMenuItems: [],
    slashMenuIndex: 0,
    slashMenuMode: "command",
    slashMenuCommand: null,
    slashMenuArgItems: [],
    searchOpen: false,
    searchQuery: "",
    pinnedExpanded: false,
    feedbackTargetTurn: null,
    feedbackRating: "good",
    feedbackText: "",
    feedbackSaving: false,
    feedbackMessage: "",
    metaclawRuleCommand: "",
    metaclawRuleMode: "ask",
    metaclawWhitelistCommand: "",
    metaclawWhitelistPath: "",
    metaclawBlockedPath: "",
  };
}

const vs = createChatEphemeralState();

/**
 * Reset chat view ephemeral state when navigating away.
 * Stops STT recording and clears search/slash UI that should not survive navigation.
 */
export function resetChatViewState() {
  if (vs.sttRecording) {
    stopStt();
  }
  Object.assign(vs, createChatEphemeralState());
}

export const cleanupChatModuleState = resetChatViewState;

function adjustTextareaHeight(el: HTMLTextAreaElement) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
}

function renderCompactionIndicator(status: CompactionIndicatorStatus | null | undefined) {
  if (!status) {
    return nothing;
  }
  if (status.phase === "active") {
    return html`
      <div
        class="compaction-indicator compaction-indicator--active"
        role="status"
        aria-live="polite"
      >
        ${icons.loader} Compacting context...
      </div>
    `;
  }
  if (status.phase === "retrying") {
    return html`
      <div
        class="compaction-indicator compaction-indicator--active"
        role="status"
        aria-live="polite"
      >
        ${icons.loader} Retrying after compaction...
      </div>
    `;
  }
  if (status.phase === "complete" && status.completedAt) {
    const elapsed = Date.now() - status.completedAt;
    if (elapsed < COMPACTION_TOAST_DURATION_MS) {
      return html`
        <div
          class="compaction-indicator compaction-indicator--complete"
          role="status"
          aria-live="polite"
        >
          ${icons.check} Context compacted
        </div>
      `;
    }
  }
  return nothing;
}

function renderFallbackIndicator(status: FallbackIndicatorStatus | null | undefined) {
  if (!status) {
    return nothing;
  }
  const phase = status.phase ?? "active";
  const elapsed = Date.now() - status.occurredAt;
  if (elapsed >= FALLBACK_TOAST_DURATION_MS) {
    return nothing;
  }
  const details = [
    `Selected: ${status.selected}`,
    phase === "cleared" ? `Active: ${status.selected}` : `Active: ${status.active}`,
    phase === "cleared" && status.previous ? `Previous fallback: ${status.previous}` : null,
    status.reason ? `Reason: ${status.reason}` : null,
    status.attempts.length > 0 ? `Attempts: ${status.attempts.slice(0, 3).join(" | ")}` : null,
  ]
    .filter(Boolean)
    .join(" • ");
  const message =
    phase === "cleared"
      ? `Fallback cleared: ${status.selected}`
      : `Fallback active: ${status.active}`;
  const className =
    phase === "cleared"
      ? "compaction-indicator compaction-indicator--fallback-cleared"
      : "compaction-indicator compaction-indicator--fallback";
  const icon = phase === "cleared" ? icons.check : icons.brain;
  return html`
    <div class=${className} role="status" aria-live="polite" title=${details}>
      ${icon} ${message}
    </div>
  `;
}

/**
 * Compact notice when context usage reaches 85%+.
 * Progressively shifts from amber (85%) to red (90%+).
 */
/** Parse a 6-digit CSS hex color string to [r, g, b] integer components. */
function parseHexRgb(hex: string): [number, number, number] | null {
  const h = hex.trim().replace(/^#/, "");
  if (!/^[0-9a-fA-F]{6}$/.test(h)) {
    return null;
  }
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

let cachedThemeNoticeColors: {
  warnHex: string;
  dangerHex: string;
  warnRgb: [number, number, number];
  dangerRgb: [number, number, number];
} | null = null;

function getThemeNoticeColors() {
  if (cachedThemeNoticeColors) {
    return cachedThemeNoticeColors;
  }
  const rootStyle = getComputedStyle(document.documentElement);
  const warnHex = rootStyle.getPropertyValue("--warn").trim() || "#f59e0b";
  const dangerHex = rootStyle.getPropertyValue("--danger").trim() || "#ef4444";
  cachedThemeNoticeColors = {
    warnHex,
    dangerHex,
    warnRgb: parseHexRgb(warnHex) ?? [245, 158, 11],
    dangerRgb: parseHexRgb(dangerHex) ?? [239, 68, 68],
  };
  return cachedThemeNoticeColors;
}

function renderContextNotice(
  session: GatewaySessionRow | undefined,
  defaultContextTokens: number | null,
) {
  if (session?.totalTokensFresh === false) {
    return nothing;
  }
  const used = session?.totalTokens ?? 0;
  const limit = session?.contextTokens ?? defaultContextTokens ?? 0;
  if (!used || !limit) {
    return nothing;
  }
  const ratio = used / limit;
  if (ratio < 0.85) {
    return nothing;
  }
  const pct = Math.min(Math.round(ratio * 100), 100);
  // Read theme semantic tokens so color tracks the active theme (Dash, dark, light …)
  const { warnRgb, dangerRgb } = getThemeNoticeColors();
  const [wr, wg, wb] = warnRgb;
  const [dr, dg, db] = dangerRgb;
  // Blend from --warn at 85% usage to --danger at 95%+ usage
  const t = Math.min(Math.max((ratio - 0.85) / 0.1, 0), 1);
  const r = Math.round(wr + (dr - wr) * t);
  const g = Math.round(wg + (dg - wg) * t);
  const b = Math.round(wb + (db - wb) * t);
  const color = `rgb(${r}, ${g}, ${b})`;
  const bgOpacity = 0.08 + 0.08 * t;
  const bg = `rgba(${r}, ${g}, ${b}, ${bgOpacity})`;
  return html`
    <div class="context-notice" role="status" style="--ctx-color:${color};--ctx-bg:${bg}">
      <svg
        class="context-notice__icon"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      <span>${pct}% context used</span>
      <span class="context-notice__detail"
        >${formatTokensCompact(used)} / ${formatTokensCompact(limit)}</span
      >
    </div>
  `;
}

/** Format token count compactly (e.g. 128000 → "128k"). */
function formatTokensCompact(n: number): string {
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  }
  if (n >= 1_000) {
    return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  }
  return String(n);
}

function generateAttachmentId(): string {
  return `att-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function handlePaste(e: ClipboardEvent, props: ChatProps) {
  const items = e.clipboardData?.items;
  if (!items || !props.onAttachmentsChange) {
    return;
  }
  const imageItems: DataTransferItem[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.startsWith("image/")) {
      imageItems.push(item);
    }
  }
  if (imageItems.length === 0) {
    return;
  }
  e.preventDefault();
  for (const item of imageItems) {
    const file = item.getAsFile();
    if (!file) {
      continue;
    }
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const dataUrl = reader.result as string;
      const newAttachment: ChatAttachment = {
        id: generateAttachmentId(),
        dataUrl,
        mimeType: file.type,
      };
      const current = props.attachments ?? [];
      props.onAttachmentsChange?.([...current, newAttachment]);
    });
    reader.readAsDataURL(file);
  }
}

function handleFileSelect(e: Event, props: ChatProps) {
  const input = e.target as HTMLInputElement;
  if (!input.files || !props.onAttachmentsChange) {
    return;
  }
  const current = props.attachments ?? [];
  const additions: ChatAttachment[] = [];
  let pending = 0;
  for (const file of input.files) {
    if (!isSupportedChatAttachmentMimeType(file.type)) {
      continue;
    }
    pending++;
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      additions.push({
        id: generateAttachmentId(),
        dataUrl: reader.result as string,
        mimeType: file.type,
      });
      pending--;
      if (pending === 0) {
        props.onAttachmentsChange?.([...current, ...additions]);
      }
    });
    reader.readAsDataURL(file);
  }
  input.value = "";
}

function handleDrop(e: DragEvent, props: ChatProps) {
  e.preventDefault();
  const files = e.dataTransfer?.files;
  if (!files || !props.onAttachmentsChange) {
    return;
  }
  const current = props.attachments ?? [];
  const additions: ChatAttachment[] = [];
  let pending = 0;
  for (const file of files) {
    if (!isSupportedChatAttachmentMimeType(file.type)) {
      continue;
    }
    pending++;
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      additions.push({
        id: generateAttachmentId(),
        dataUrl: reader.result as string,
        mimeType: file.type,
      });
      pending--;
      if (pending === 0) {
        props.onAttachmentsChange?.([...current, ...additions]);
      }
    });
    reader.readAsDataURL(file);
  }
}

function renderAttachmentPreview(props: ChatProps): TemplateResult | typeof nothing {
  const attachments = props.attachments ?? [];
  if (attachments.length === 0) {
    return nothing;
  }
  return html`
    <div class="chat-attachments-preview">
      ${attachments.map(
        (att) => html`
          <div class="chat-attachment-thumb">
            <img src=${att.dataUrl} alt="Attachment preview" />
            <button
              class="chat-attachment-remove"
              type="button"
              aria-label="Remove attachment"
              @click=${() => {
                const next = (props.attachments ?? []).filter((a) => a.id !== att.id);
                props.onAttachmentsChange?.(next);
              }}
            >
              &times;
            </button>
          </div>
        `,
      )}
    </div>
  `;
}

function resetSlashMenuState(): void {
  vs.slashMenuMode = "command";
  vs.slashMenuCommand = null;
  vs.slashMenuArgItems = [];
  vs.slashMenuItems = [];
}

function updateSlashMenu(value: string, requestUpdate: () => void): void {
  // Arg mode: /command <partial-arg>
  const argMatch = value.match(/^\/(\S+)\s(.*)$/);
  if (argMatch) {
    const cmdName = argMatch[1].toLowerCase();
    const argFilter = argMatch[2].toLowerCase();
    const cmd = SLASH_COMMANDS.find((c) => c.name === cmdName);
    if (cmd?.argOptions?.length) {
      const filtered = argFilter
        ? cmd.argOptions.filter((opt) => opt.toLowerCase().startsWith(argFilter))
        : cmd.argOptions;
      if (filtered.length > 0) {
        vs.slashMenuMode = "args";
        vs.slashMenuCommand = cmd;
        vs.slashMenuArgItems = filtered;
        vs.slashMenuOpen = true;
        vs.slashMenuIndex = 0;
        vs.slashMenuItems = [];
        requestUpdate();
        return;
      }
    }
    vs.slashMenuOpen = false;
    resetSlashMenuState();
    requestUpdate();
    return;
  }

  // Command mode: /partial-command
  const match = value.match(/^\/(\S*)$/);
  if (match) {
    const items = getSlashCommandCompletions(match[1]);
    vs.slashMenuItems = items;
    vs.slashMenuOpen = items.length > 0;
    vs.slashMenuIndex = 0;
    vs.slashMenuMode = "command";
    vs.slashMenuCommand = null;
    vs.slashMenuArgItems = [];
  } else {
    vs.slashMenuOpen = false;
    resetSlashMenuState();
  }
  requestUpdate();
}

function selectSlashCommand(
  cmd: SlashCommandDef,
  props: ChatProps,
  requestUpdate: () => void,
): void {
  // Transition to arg picker when the command has fixed options
  if (cmd.argOptions?.length) {
    props.onDraftChange(`/${cmd.name} `);
    vs.slashMenuMode = "args";
    vs.slashMenuCommand = cmd;
    vs.slashMenuArgItems = cmd.argOptions;
    vs.slashMenuOpen = true;
    vs.slashMenuIndex = 0;
    vs.slashMenuItems = [];
    requestUpdate();
    return;
  }

  vs.slashMenuOpen = false;
  resetSlashMenuState();

  if (cmd.executeLocal && !cmd.args) {
    props.onDraftChange(`/${cmd.name}`);
    requestUpdate();
    props.onSend();
  } else {
    props.onDraftChange(`/${cmd.name} `);
    requestUpdate();
  }
}

function tabCompleteSlashCommand(
  cmd: SlashCommandDef,
  props: ChatProps,
  requestUpdate: () => void,
): void {
  // Tab: fill in the command text without executing
  if (cmd.argOptions?.length) {
    props.onDraftChange(`/${cmd.name} `);
    vs.slashMenuMode = "args";
    vs.slashMenuCommand = cmd;
    vs.slashMenuArgItems = cmd.argOptions;
    vs.slashMenuOpen = true;
    vs.slashMenuIndex = 0;
    vs.slashMenuItems = [];
    requestUpdate();
    return;
  }

  vs.slashMenuOpen = false;
  resetSlashMenuState();
  props.onDraftChange(cmd.args ? `/${cmd.name} ` : `/${cmd.name}`);
  requestUpdate();
}

function selectSlashArg(
  arg: string,
  props: ChatProps,
  requestUpdate: () => void,
  execute: boolean,
): void {
  const cmdName = vs.slashMenuCommand?.name ?? "";
  vs.slashMenuOpen = false;
  resetSlashMenuState();
  props.onDraftChange(`/${cmdName} ${arg}`);
  requestUpdate();
  if (execute) {
    props.onSend();
  }
}

function tokenEstimate(draft: string): string | null {
  if (draft.length < 100) {
    return null;
  }
  return `~${Math.ceil(draft.length / 4)} tokens`;
}

/**
 * Export chat markdown - delegates to shared utility.
 */
function exportMarkdown(props: ChatProps): void {
  exportChatMarkdown(props.messages, props.assistantName);
}

const WELCOME_SUGGESTIONS = [
  "What can you do?",
  "Summarize my recent sessions",
  "Help me configure a channel",
  "Check system health",
];

function renderWelcomeState(props: ChatProps): TemplateResult {
  const name = props.assistantName || "Assistant";
  const avatar = resolveAgentAvatarUrl({
    identity: {
      avatar: props.assistantAvatar ?? undefined,
      avatarUrl: props.assistantAvatarUrl ?? undefined,
    },
  });
  const logoUrl = agentLogoUrl(props.basePath ?? "");

  return html`
    <div class="agent-chat__welcome" style="--agent-color: var(--accent)">
      <div class="agent-chat__welcome-glow"></div>
      ${
        avatar
          ? html`<img
            src=${avatar}
            alt=${name}
            style="width:56px; height:56px; border-radius:50%; object-fit:cover;"
          />`
          : html`<div class="agent-chat__avatar agent-chat__avatar--logo">
            <img src=${logoUrl} alt="OpenClaw" />
          </div>`
      }
      <h2>${name}</h2>
      <div class="agent-chat__badges">
        <span class="agent-chat__badge"><img src=${logoUrl} alt="" /> Ready to chat</span>
      </div>
      <p class="agent-chat__hint">Type a message below &middot; <kbd>/</kbd> for commands</p>
      <div class="agent-chat__suggestions">
        ${WELCOME_SUGGESTIONS.map(
          (text) => html`
            <button
              type="button"
              class="agent-chat__suggestion"
              @click=${() => {
                props.onDraftChange(text);
                props.onSend();
              }}
            >
              ${text}
            </button>
          `,
        )}
      </div>
    </div>
  `;
}

function renderMetaclawStudio(props: ChatProps, requestUpdate: () => void): TemplateResult | typeof nothing {
  const state = props.metaclaw;
  if (!state) {
    return nothing;
  }
  const policy = state.sandboxPolicy;
  const commandRules = policy ? Object.entries(policy.command_rules) : [];
  const selected = new Set(
    state.selectionCustomized ? state.selectedSkillNames : state.skills.map((skill) => skill.name),
  );
  const pendingCount = state.pendingApprovals.length;
  const allowCount = commandRules.filter(([, mode]) => mode === "allow").length;
  const askCount = commandRules.filter(([, mode]) => mode === "ask").length;
  const denyCount = commandRules.filter(([, mode]) => mode === "deny").length;
  return html`
    <section class="metaclaw-studio">
      <div class="metaclaw-studio__hero">
        <div>
          <div class="metaclaw-studio__eyebrow">MetaClaw Studio</div>
          <div class="metaclaw-studio__title">Feedback, sandbox approvals, skills, and notes</div>
        </div>
        <button class="btn btn--ghost" type="button" ?disabled=${state.loading || state.saving} @click=${state.onRefresh}>
          ${icons.refresh} Refresh
        </button>
      </div>
      <div class="metaclaw-summary">
        <div class="metaclaw-summary__card ${state.connected ? "is-ok" : "is-warn"}">
          <strong>${state.connected ? "API online" : "API offline"}</strong>
          <span>${state.connected ? "MetaClaw frontend features are live." : "Check the API URL, token, or CORS reachability."}</span>
        </div>
        <div class="metaclaw-summary__card ${pendingCount ? "is-warn" : "is-ok"}">
          <strong>${pendingCount}</strong>
          <span>${pendingCount ? "Command requests are waiting for your decision." : "No commands are waiting for approval."}</span>
        </div>
        <div class="metaclaw-summary__card">
          <strong>${state.selectionCustomized ? state.selectedSkillNames.length : state.skills.length}/${state.skills.length}</strong>
          <span>${state.selectionCustomized ? "Skills explicitly selected for this session." : "All available skills are active by default."}</span>
        </div>
      </div>
      <div class="metaclaw-studio__grid">
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Connection</div>
          <input class="input" .value=${state.apiBase} @input=${(e: Event) => state.onApiBaseChange((e.target as HTMLInputElement).value)} placeholder="http://localhost:30000" />
          <input class="input" .value=${state.token} @input=${(e: Event) => state.onTokenChange((e.target as HTMLInputElement).value)} placeholder="Bearer token (optional)" />
          <div class="metaclaw-status ${state.connected ? "ok" : "warn"}">${state.connected ? "Connected" : "Unavailable"}</div>
          ${state.error ? html`<div class="callout danger">${state.error}</div>` : nothing}
        </div>
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Pending Approvals</div>
          ${state.pendingApprovals.length === 0
            ? html`<div class="muted">No pending command approvals for this session.</div>`
            : state.pendingApprovals.map((item) => html`
                <div class="metaclaw-approval">
                  <div class="metaclaw-approval__head">
                    <span class="mono">${item.approval_id}</span>
                    <span class="muted">${item.created_at}</span>
                  </div>
                  ${(item.decisions ?? []).map((decision) => html`
                    <div class="metaclaw-approval__row">
                      <strong>${decision.tool_name ?? "tool"}</strong>
                      <span class="mono">${decision.command ?? ""}</span>
                      <span class="muted">${decision.reason ?? decision.action ?? ""}</span>
                    </div>
                  `)}
                  <div class="metaclaw-approval__actions">
                    <button class="btn primary" type="button" ?disabled=${state.saving} @click=${() => state.onApprove(item.approval_id)}>Approve</button>
                    <button class="btn danger" type="button" ?disabled=${state.saving} @click=${() => state.onReject(item.approval_id)}>Reject</button>
                  </div>
                </div>
              `)}
        </div>
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Command Policy</div>
          ${policy
            ? html`
                <label class="field">
                  <span>Default command mode</span>
                  <select
                    .value=${policy.default_command_mode}
                    @change=${(e: Event) =>
                      state.onSavePolicy({
                        ...policy,
                        default_command_mode: (e.target as HTMLSelectElement).value as "allow" | "ask" | "deny",
                      })}
                  >
                    <option value="allow">Allow</option>
                    <option value="ask">Ask</option>
                    <option value="deny">Deny</option>
                  </select>
                </label>
                <div class="metaclaw-chip-group">
                  <span class="chip">${allowCount} allow</span>
                  <span class="chip">${askCount} ask</span>
                  <span class="chip chip--danger">${denyCount} deny</span>
                </div>
                <div class="metaclaw-rule-list">
                  ${commandRules.map(([command, mode]) => html`
                    <div class="metaclaw-rule-row">
                      <span class="mono">${command}</span>
                      <div class="metaclaw-rule-row__actions">
                        <select
                          .value=${mode}
                          @change=${(e: Event) => {
                            const nextMode = (e.target as HTMLSelectElement).value as "allow" | "ask" | "deny";
                            state.onSavePolicy({ ...policy, command_rules: { ...policy.command_rules, [command]: nextMode } });
                          }}
                        >
                          <option value="allow">Allow</option>
                          <option value="ask">Ask</option>
                          <option value="deny">Deny</option>
                        </select>
                        <button
                          class="btn btn--ghost"
                          type="button"
                          ?disabled=${state.saving}
                          @click=${() => {
                            const nextRules = { ...policy.command_rules };
                            delete nextRules[command];
                            state.onSavePolicy({ ...policy, command_rules: nextRules });
                          }}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  `)}
                </div>
                <div class="metaclaw-inline-form">
                  <input class="input" .value=${vs.metaclawRuleCommand} @input=${(e: Event) => { vs.metaclawRuleCommand = (e.target as HTMLInputElement).value; requestUpdate(); }} placeholder="Add command rule, e.g. pwd" />
                  <select
                    .value=${vs.metaclawRuleMode}
                    @change=${(e: Event) => {
                      vs.metaclawRuleMode = (e.target as HTMLSelectElement).value as "allow" | "ask" | "deny";
                      requestUpdate();
                    }}
                  >
                    <option value="allow">Allow</option>
                    <option value="ask">Ask</option>
                    <option value="deny">Deny</option>
                  </select>
                  <button class="btn" type="button" ?disabled=${!vs.metaclawRuleCommand.trim() || state.saving} @click=${() => {
                    const command = vs.metaclawRuleCommand.trim();
                    vs.metaclawRuleCommand = "";
                    requestUpdate();
                    state.onSavePolicy({ ...policy, command_rules: { ...policy.command_rules, [command]: vs.metaclawRuleMode } });
                  }}>Save Rule</button>
                </div>
              `
            : html`<div class="muted">Sandbox policy unavailable.</div>`}
        </div>
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Path Access</div>
          ${policy
            ? html`
                <div class="metaclaw-panel__sub">Allowlisted paths stay accessible even when the default path policy is restrictive.</div>
                <div class="metaclaw-chip-group">
                  ${policy.path_allowlist.map((path) => html`<button class="chip" type="button" @click=${() => state.onRemoveWhitelistEntry("path", path)}>${path} ${icons.x}</button>`)}
                </div>
                <div class="metaclaw-inline-form">
                  <input class="input" .value=${vs.metaclawWhitelistPath} @input=${(e: Event) => { vs.metaclawWhitelistPath = (e.target as HTMLInputElement).value; requestUpdate(); }} placeholder="Allow path" />
                  <button class="btn" type="button" ?disabled=${!vs.metaclawWhitelistPath.trim() || state.saving} @click=${() => {
                    const value = vs.metaclawWhitelistPath.trim();
                    vs.metaclawWhitelistPath = "";
                    requestUpdate();
                    state.onAddWhitelistEntry("path", value);
                  }}>Allow</button>
                </div>
                <div class="metaclaw-chip-group">
                  ${policy.path_blocklist.map((path) => html`
                    <button class="chip chip--danger" type="button" @click=${() =>
                      state.onSavePolicy({
                        ...policy,
                        path_blocklist: policy.path_blocklist.filter((item) => item !== path),
                      })}
                    >
                      ${path} ${icons.x}
                    </button>
                  `)}
                </div>
                <div class="metaclaw-inline-form">
                  <input class="input" .value=${vs.metaclawBlockedPath} @input=${(e: Event) => { vs.metaclawBlockedPath = (e.target as HTMLInputElement).value; requestUpdate(); }} placeholder="Block path" />
                  <button class="btn danger" type="button" ?disabled=${!vs.metaclawBlockedPath.trim() || state.saving} @click=${() => {
                    const value = vs.metaclawBlockedPath.trim();
                    vs.metaclawBlockedPath = "";
                    requestUpdate();
                    state.onSavePolicy({ ...policy, path_blocklist: [...policy.path_blocklist, value] });
                  }}>Block</button>
                </div>
              `
            : nothing}
        </div>
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Command Allowlist</div>
          <div class="metaclaw-panel__sub">Use this for commands that should bypass the default ask/deny flow.</div>
          ${policy
            ? html`
                <div class="metaclaw-chip-group">
                  ${policy.command_allowlist.map((command) => html`<button class="chip" type="button" @click=${() => state.onRemoveWhitelistEntry("command", command)}>${command} ${icons.x}</button>`)}
                </div>
                <div class="metaclaw-inline-form">
                  <input class="input" .value=${vs.metaclawWhitelistCommand} @input=${(e: Event) => { vs.metaclawWhitelistCommand = (e.target as HTMLInputElement).value; requestUpdate(); }} placeholder="Always allow command" />
                  <button class="btn" type="button" ?disabled=${!vs.metaclawWhitelistCommand.trim() || state.saving} @click=${() => {
                    const value = vs.metaclawWhitelistCommand.trim();
                    vs.metaclawWhitelistCommand = "";
                    requestUpdate();
                    state.onAddWhitelistEntry("command", value);
                  }}>Allow</button>
                </div>
              `
            : nothing}
        </div>
        <div class="metaclaw-panel">
          <div class="metaclaw-panel__title">Skills</div>
          <div class="metaclaw-panel__sub">
            ${state.selectionCustomized ? "Custom selection" : "All skills enabled by default"}
            ${state.latestInjectedSkills.length ? html`<span> | Last injected: ${state.latestInjectedSkills.join(", ")}</span>` : nothing}
          </div>
          <div class="metaclaw-skill-list">
            ${state.skills.map((skill) => html`
              <label class="metaclaw-skill">
                <input
                  type="checkbox"
                  .checked=${selected.has(skill.name)}
                  @change=${(e: Event) => {
                    const next = new Set(selected);
                    if ((e.target as HTMLInputElement).checked) {
                      next.add(skill.name);
                    } else {
                      next.delete(skill.name);
                    }
                    state.onSaveSkillSelection([...next]);
                  }}
                />
                <span>
                  <strong>${skill.name}</strong>
                  <small>${skill.category}</small>
                  <span>${skill.description}</span>
                </span>
              </label>
            `)}
          </div>
          <div class="metaclaw-approval__actions">
            <button class="btn" type="button" ?disabled=${state.saving} @click=${() => state.onSaveSkillSelection(null)}>Use All Skills</button>
            <button class="btn" type="button" ?disabled=${state.saving} @click=${() => state.onSaveSkillSelection([])}>Disable All</button>
          </div>
        </div>
        <div class="metaclaw-panel metaclaw-panel--notes">
          <div class="metaclaw-panel__title">${state.importantNotes?.name ?? "important-notes"}</div>
          <div class="metaclaw-panel__sub">${state.importantNotes?.description ?? "Persistent notes appended from bad-answer feedback."}</div>
          <pre class="metaclaw-notes">${state.importantNotes?.content ?? "No important notes yet."}</pre>
        </div>
      </div>
    </section>
  `;
}

function extractAssistantTurn(group: MessageGroup): number | null {
  for (let index = group.messages.length - 1; index >= 0; index -= 1) {
    const message = group.messages[index]?.message as Record<string, unknown> | undefined;
    const rawTurn = message?.turn ?? message?.metaclaw_turn;
    if (typeof rawTurn === "number" && Number.isFinite(rawTurn)) {
      return rawTurn;
    }
    if (typeof rawTurn === "string" && rawTurn.trim()) {
      const parsed = Number(rawTurn);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return null;
}

function renderAssistantFeedback(
  group: MessageGroup,
  props: ChatProps,
  requestUpdate: () => void,
): TemplateResult | typeof nothing {
  if (!props.metaclaw || normalizeRoleForGrouping(group.role) !== "assistant") {
    return nothing;
  }
  const turn = extractAssistantTurn(group);
  if (turn == null) {
    return nothing;
  }
  const isOpen = vs.feedbackTargetTurn === turn;
  return html`
    <div class="metaclaw-feedback">
      <div class="metaclaw-feedback__title">Feedback for answer #${turn}</div>
      <div class="metaclaw-feedback__actions">
        <button class="btn ${isOpen && vs.feedbackRating === "good" ? "primary" : ""}" type="button" @click=${() => {
          vs.feedbackTargetTurn = turn;
          vs.feedbackRating = "good";
          vs.feedbackText = "";
          vs.feedbackMessage = "";
          requestUpdate();
        }}>
          ${icons.check} Good
        </button>
        <button class="btn ${isOpen && vs.feedbackRating === "bad" ? "danger" : ""}" type="button" @click=${() => {
          vs.feedbackTargetTurn = turn;
          vs.feedbackRating = "bad";
          vs.feedbackText = "";
          vs.feedbackMessage = "";
          requestUpdate();
        }}>
          ${icons.x} Bad
        </button>
      </div>
      ${isOpen
        ? html`
            <textarea
              class="metaclaw-feedback__input"
              .value=${vs.feedbackText}
              @input=${(e: Event) => {
                vs.feedbackText = (e.target as HTMLTextAreaElement).value;
                requestUpdate();
              }}
              placeholder="Tell MetaClaw what was good or what should have been done better."
            ></textarea>
            <div class="metaclaw-feedback__actions">
              <button class="btn primary" type="button" ?disabled=${vs.feedbackSaving} @click=${async () => {
                vs.feedbackSaving = true;
                vs.feedbackMessage = "";
                requestUpdate();
                try {
                  const result = await props.metaclaw!.onSubmitFeedback(turn, vs.feedbackRating, vs.feedbackText.trim());
                  vs.feedbackMessage = result.skill_updated
                    ? `Summarized into ${result.skill_name || "important-notes"}.`
                    : "Feedback recorded.";
                  vs.feedbackText = "";
                } finally {
                  vs.feedbackSaving = false;
                  requestUpdate();
                }
              }}>Submit</button>
              <button class="btn" type="button" ?disabled=${vs.feedbackSaving} @click=${() => {
                vs.feedbackTargetTurn = null;
                vs.feedbackText = "";
                vs.feedbackMessage = "";
                requestUpdate();
              }}>Cancel</button>
            </div>
            ${vs.feedbackMessage ? html`<div class="callout success">${vs.feedbackMessage}</div>` : nothing}
          `
        : nothing}
    </div>
  `;
}

function renderSearchBar(requestUpdate: () => void): TemplateResult | typeof nothing {
  if (!vs.searchOpen) {
    return nothing;
  }
  return html`
    <div class="agent-chat__search-bar">
      ${icons.search}
      <input
        type="text"
        placeholder="Search messages..."
        aria-label="Search messages"
        .value=${vs.searchQuery}
        @input=${(e: Event) => {
          vs.searchQuery = (e.target as HTMLInputElement).value;
          requestUpdate();
        }}
      />
      <button
        class="btn btn--ghost"
        aria-label="Close search"
        @click=${() => {
          vs.searchOpen = false;
          vs.searchQuery = "";
          requestUpdate();
        }}
      >
        ${icons.x}
      </button>
    </div>
  `;
}

function renderPinnedSection(
  props: ChatProps,
  pinned: PinnedMessages,
  requestUpdate: () => void,
): TemplateResult | typeof nothing {
  const messages = Array.isArray(props.messages) ? props.messages : [];
  const entries: Array<{ index: number; text: string; role: string }> = [];
  for (const idx of pinned.indices) {
    const msg = messages[idx] as Record<string, unknown> | undefined;
    if (!msg) {
      continue;
    }
    const text = getPinnedMessageSummary(msg);
    const role = typeof msg.role === "string" ? msg.role : "unknown";
    entries.push({ index: idx, text, role });
  }
  if (entries.length === 0) {
    return nothing;
  }
  return html`
    <div class="agent-chat__pinned">
      <button
        class="agent-chat__pinned-toggle"
        @click=${() => {
          vs.pinnedExpanded = !vs.pinnedExpanded;
          requestUpdate();
        }}
      >
        ${icons.bookmark} ${entries.length} pinned
        <span class="collapse-chevron ${vs.pinnedExpanded ? "" : "collapse-chevron--collapsed"}"
          >${icons.chevronDown}</span
        >
      </button>
      ${
        vs.pinnedExpanded
          ? html`
            <div class="agent-chat__pinned-list">
              ${entries.map(
                ({ index, text, role }) => html`
                  <div class="agent-chat__pinned-item">
                    <span class="agent-chat__pinned-role"
                      >${role === "user" ? "You" : "Assistant"}</span
                    >
                    <span class="agent-chat__pinned-text"
                      >${text.slice(0, 100)}${text.length > 100 ? "..." : ""}</span
                    >
                    <button
                      class="btn btn--ghost"
                      @click=${() => {
                        pinned.unpin(index);
                        requestUpdate();
                      }}
                      title="Unpin"
                    >
                      ${icons.x}
                    </button>
                  </div>
                `,
              )}
            </div>
          `
          : nothing
      }
    </div>
  `;
}

function renderSlashMenu(
  requestUpdate: () => void,
  props: ChatProps,
): TemplateResult | typeof nothing {
  if (!vs.slashMenuOpen) {
    return nothing;
  }

  // Arg-picker mode: show options for the selected command
  if (vs.slashMenuMode === "args" && vs.slashMenuCommand && vs.slashMenuArgItems.length > 0) {
    return html`
      <div class="slash-menu" role="listbox" aria-label="Command arguments">
        <div class="slash-menu-group">
          <div class="slash-menu-group__label">
            /${vs.slashMenuCommand.name} ${vs.slashMenuCommand.description}
          </div>
          ${vs.slashMenuArgItems.map(
            (arg, i) => html`
              <div
                class="slash-menu-item ${i === vs.slashMenuIndex ? "slash-menu-item--active" : ""}"
                role="option"
                aria-selected=${i === vs.slashMenuIndex}
                @click=${() => selectSlashArg(arg, props, requestUpdate, true)}
                @mouseenter=${() => {
                  vs.slashMenuIndex = i;
                  requestUpdate();
                }}
              >
                ${
                  vs.slashMenuCommand?.icon
                    ? html`<span class="slash-menu-icon">${icons[vs.slashMenuCommand.icon]}</span>`
                    : nothing
                }
                <span class="slash-menu-name">${arg}</span>
                <span class="slash-menu-desc">/${vs.slashMenuCommand?.name} ${arg}</span>
              </div>
            `,
          )}
        </div>
        <div class="slash-menu-footer">
          <kbd>↑↓</kbd> navigate <kbd>Tab</kbd> fill <kbd>Enter</kbd> run <kbd>Esc</kbd> close
        </div>
      </div>
    `;
  }

  // Command mode: show grouped commands
  if (vs.slashMenuItems.length === 0) {
    return nothing;
  }

  const grouped = new Map<
    SlashCommandCategory,
    Array<{ cmd: SlashCommandDef; globalIdx: number }>
  >();
  for (let i = 0; i < vs.slashMenuItems.length; i++) {
    const cmd = vs.slashMenuItems[i];
    const cat = cmd.category ?? "session";
    let list = grouped.get(cat);
    if (!list) {
      list = [];
      grouped.set(cat, list);
    }
    list.push({ cmd, globalIdx: i });
  }

  const sections: TemplateResult[] = [];
  for (const [cat, entries] of grouped) {
    sections.push(html`
      <div class="slash-menu-group">
        <div class="slash-menu-group__label">${CATEGORY_LABELS[cat]}</div>
        ${entries.map(
          ({ cmd, globalIdx }) => html`
            <div
              class="slash-menu-item ${
                globalIdx === vs.slashMenuIndex ? "slash-menu-item--active" : ""
              }"
              role="option"
              aria-selected=${globalIdx === vs.slashMenuIndex}
              @click=${() => selectSlashCommand(cmd, props, requestUpdate)}
              @mouseenter=${() => {
                vs.slashMenuIndex = globalIdx;
                requestUpdate();
              }}
            >
              ${cmd.icon ? html`<span class="slash-menu-icon">${icons[cmd.icon]}</span>` : nothing}
              <span class="slash-menu-name">/${cmd.name}</span>
              ${cmd.args ? html`<span class="slash-menu-args">${cmd.args}</span>` : nothing}
              <span class="slash-menu-desc">${cmd.description}</span>
              ${
                cmd.argOptions?.length
                  ? html`<span class="slash-menu-badge">${cmd.argOptions.length} options</span>`
                  : cmd.executeLocal && !cmd.args
                    ? html`
                        <span class="slash-menu-badge">instant</span>
                      `
                    : nothing
              }
            </div>
          `,
        )}
      </div>
    `);
  }

  return html`
    <div class="slash-menu" role="listbox" aria-label="Slash commands">
      ${sections}
      <div class="slash-menu-footer">
        <kbd>↑↓</kbd> navigate <kbd>Tab</kbd> fill <kbd>Enter</kbd> select <kbd>Esc</kbd> close
      </div>
    </div>
  `;
}

export function renderChat(props: ChatProps) {
  const canCompose = props.connected;
  const isBusy = props.sending || props.stream !== null;
  const canAbort = Boolean(props.canAbort && props.onAbort);
  const activeSession = props.sessions?.sessions?.find((row) => row.key === props.sessionKey);
  const reasoningLevel = activeSession?.reasoningLevel ?? "off";
  const showReasoning = props.showThinking && reasoningLevel !== "off";
  const assistantIdentity = {
    name: props.assistantName,
    avatar:
      resolveAgentAvatarUrl({
        identity: {
          avatar: props.assistantAvatar ?? undefined,
          avatarUrl: props.assistantAvatarUrl ?? undefined,
        },
      }) ?? null,
  };
  const pinned = getPinnedMessages(props.sessionKey);
  const deleted = getDeletedMessages(props.sessionKey);
  const inputHistory = getInputHistory(props.sessionKey);
  const hasAttachments = (props.attachments?.length ?? 0) > 0;
  const tokens = tokenEstimate(props.draft);

  const placeholder = props.connected
    ? hasAttachments
      ? "Add a message or paste more images..."
      : `Message ${props.assistantName || "agent"} (Enter to send)`
    : "Connect to the gateway to start chatting...";

  const requestUpdate = props.onRequestUpdate ?? (() => {});
  const getDraft = props.getDraft ?? (() => props.draft);

  const splitRatio = props.splitRatio ?? 0.6;
  const sidebarOpen = Boolean(props.sidebarOpen && props.onCloseSidebar);

  const handleCodeBlockCopy = (e: Event) => {
    const btn = (e.target as HTMLElement).closest(".code-block-copy");
    if (!btn) {
      return;
    }
    const code = (btn as HTMLElement).dataset.code ?? "";
    navigator.clipboard.writeText(code).then(
      () => {
        btn.classList.add("copied");
        setTimeout(() => btn.classList.remove("copied"), 1500);
      },
      () => {},
    );
  };

  const chatItems = buildChatItems(props);
  const isEmpty = chatItems.length === 0 && !props.loading;

  const thread = html`
    <div
      class="chat-thread"
      role="log"
      aria-live="polite"
      @scroll=${props.onChatScroll}
      @click=${handleCodeBlockCopy}
    >
      <div class="chat-thread-inner">
        ${
          props.loading
            ? html`
                <div class="chat-loading-skeleton" aria-label="Loading chat">
                  <div class="chat-line assistant">
                    <div class="chat-msg">
                      <div class="chat-bubble">
                        <div class="skeleton skeleton-line skeleton-line--long" style="margin-bottom: 8px"></div>
                        <div class="skeleton skeleton-line skeleton-line--medium" style="margin-bottom: 8px"></div>
                        <div class="skeleton skeleton-line skeleton-line--short"></div>
                      </div>
                    </div>
                  </div>
                  <div class="chat-line user" style="margin-top: 12px">
                    <div class="chat-msg">
                      <div class="chat-bubble">
                        <div class="skeleton skeleton-line skeleton-line--medium"></div>
                      </div>
                    </div>
                  </div>
                  <div class="chat-line assistant" style="margin-top: 12px">
                    <div class="chat-msg">
                      <div class="chat-bubble">
                        <div class="skeleton skeleton-line skeleton-line--long" style="margin-bottom: 8px"></div>
                        <div class="skeleton skeleton-line skeleton-line--short"></div>
                      </div>
                    </div>
                  </div>
                </div>
              `
            : nothing
        }
        ${isEmpty && !vs.searchOpen ? renderWelcomeState(props) : nothing}
        ${
          isEmpty && vs.searchOpen
            ? html`
                <div class="agent-chat__empty">No matching messages</div>
              `
            : nothing
        }
        ${repeat(
          chatItems,
          (item) => item.key,
          (item) => {
            if (item.kind === "divider") {
              return html`
                <div class="chat-divider" role="separator" data-ts=${String(item.timestamp)}>
                  <span class="chat-divider__line"></span>
                  <span class="chat-divider__label">${item.label}</span>
                  <span class="chat-divider__line"></span>
                </div>
              `;
            }
            if (item.kind === "reading-indicator") {
              return renderReadingIndicatorGroup(assistantIdentity, props.basePath);
            }
            if (item.kind === "stream") {
              return renderStreamingGroup(
                item.text,
                item.startedAt,
                props.onOpenSidebar,
                assistantIdentity,
                props.basePath,
              );
            }
            if (item.kind === "group") {
              if (deleted.has(item.key)) {
                return nothing;
              }
              const groupView = renderMessageGroup(item, {
                onOpenSidebar: props.onOpenSidebar,
                showReasoning,
                showToolCalls: props.showToolCalls,
                assistantName: props.assistantName,
                assistantAvatar: assistantIdentity.avatar,
                basePath: props.basePath,
                contextWindow:
                  activeSession?.contextTokens ?? props.sessions?.defaults?.contextTokens ?? null,
                onDelete: () => {
                  deleted.delete(item.key);
                  requestUpdate();
                },
              });
              return html`${groupView}${renderAssistantFeedback(item, props, requestUpdate)}`;
            }
            return nothing;
          },
        )}
      </div>
    </div>
  `;

  const handleKeyDown = (e: KeyboardEvent) => {
    // Slash menu navigation — arg mode
    if (vs.slashMenuOpen && vs.slashMenuMode === "args" && vs.slashMenuArgItems.length > 0) {
      const len = vs.slashMenuArgItems.length;
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          vs.slashMenuIndex = (vs.slashMenuIndex + 1) % len;
          requestUpdate();
          return;
        case "ArrowUp":
          e.preventDefault();
          vs.slashMenuIndex = (vs.slashMenuIndex - 1 + len) % len;
          requestUpdate();
          return;
        case "Tab":
          e.preventDefault();
          selectSlashArg(vs.slashMenuArgItems[vs.slashMenuIndex], props, requestUpdate, false);
          return;
        case "Enter":
          e.preventDefault();
          selectSlashArg(vs.slashMenuArgItems[vs.slashMenuIndex], props, requestUpdate, true);
          return;
        case "Escape":
          e.preventDefault();
          vs.slashMenuOpen = false;
          resetSlashMenuState();
          requestUpdate();
          return;
      }
    }

    // Slash menu navigation — command mode
    if (vs.slashMenuOpen && vs.slashMenuItems.length > 0) {
      const len = vs.slashMenuItems.length;
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          vs.slashMenuIndex = (vs.slashMenuIndex + 1) % len;
          requestUpdate();
          return;
        case "ArrowUp":
          e.preventDefault();
          vs.slashMenuIndex = (vs.slashMenuIndex - 1 + len) % len;
          requestUpdate();
          return;
        case "Tab":
          e.preventDefault();
          tabCompleteSlashCommand(vs.slashMenuItems[vs.slashMenuIndex], props, requestUpdate);
          return;
        case "Enter":
          e.preventDefault();
          selectSlashCommand(vs.slashMenuItems[vs.slashMenuIndex], props, requestUpdate);
          return;
        case "Escape":
          e.preventDefault();
          vs.slashMenuOpen = false;
          resetSlashMenuState();
          requestUpdate();
          return;
      }
    }

    // Input history (only when input is empty)
    if (!props.draft.trim()) {
      if (e.key === "ArrowUp") {
        const prev = inputHistory.up();
        if (prev !== null) {
          e.preventDefault();
          props.onDraftChange(prev);
        }
        return;
      }
      if (e.key === "ArrowDown") {
        const next = inputHistory.down();
        e.preventDefault();
        props.onDraftChange(next ?? "");
        return;
      }
    }

    // Cmd+F for search
    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === "f") {
      e.preventDefault();
      vs.searchOpen = !vs.searchOpen;
      if (!vs.searchOpen) {
        vs.searchQuery = "";
      }
      requestUpdate();
      return;
    }

    // Send on Enter (without shift)
    if (e.key === "Enter" && !e.shiftKey) {
      if (e.isComposing || e.keyCode === 229) {
        return;
      }
      if (!props.connected) {
        return;
      }
      e.preventDefault();
      if (canCompose) {
        if (props.draft.trim()) {
          inputHistory.push(props.draft);
        }
        props.onSend();
      }
    }
  };

  const handleInput = (e: Event) => {
    const target = e.target as HTMLTextAreaElement;
    adjustTextareaHeight(target);
    updateSlashMenu(target.value, requestUpdate);
    inputHistory.reset();
    props.onDraftChange(target.value);
  };

  return html`
    <section
      class="card chat"
      @drop=${(e: DragEvent) => handleDrop(e, props)}
      @dragover=${(e: DragEvent) => e.preventDefault()}
    >
      ${props.disabledReason ? html`<div class="callout">${props.disabledReason}</div>` : nothing}
      ${props.error ? html`<div class="callout danger">${props.error}</div>` : nothing}
      ${
        props.focusMode
          ? html`
            <button
              class="chat-focus-exit"
              type="button"
              @click=${props.onToggleFocusMode}
              aria-label="Exit focus mode"
              title="Exit focus mode"
            >
              ${icons.x}
            </button>
          `
          : nothing
      }
      ${renderSearchBar(requestUpdate)} ${renderPinnedSection(props, pinned, requestUpdate)}
      ${renderMetaclawStudio(props, requestUpdate)}

      <div class="chat-split-container ${sidebarOpen ? "chat-split-container--open" : ""}">
        <div
          class="chat-main"
          style="flex: ${sidebarOpen ? `0 0 ${splitRatio * 100}%` : "1 1 100%"}"
        >
          ${thread}
        </div>

        ${
          sidebarOpen
            ? html`
              <resizable-divider
                .splitRatio=${splitRatio}
                @resize=${(e: CustomEvent) => props.onSplitRatioChange?.(e.detail.splitRatio)}
              ></resizable-divider>
              <div class="chat-sidebar">
                ${renderMarkdownSidebar({
                  content: props.sidebarContent ?? null,
                  error: props.sidebarError ?? null,
                  onClose: props.onCloseSidebar!,
                  onViewRawText: () => {
                    if (!props.sidebarContent || !props.onOpenSidebar) {
                      return;
                    }
                    props.onOpenSidebar(`\`\`\`\n${props.sidebarContent}\n\`\`\``);
                  },
                })}
              </div>
            `
            : nothing
        }
      </div>

      ${
        props.queue.length
          ? html`
            <div class="chat-queue" role="status" aria-live="polite">
              <div class="chat-queue__title">Queued (${props.queue.length})</div>
              <div class="chat-queue__list">
                ${props.queue.map(
                  (item) => html`
                    <div class="chat-queue__item">
                      <div class="chat-queue__text">
                        ${
                          item.text ||
                          (item.attachments?.length ? `Image (${item.attachments.length})` : "")
                        }
                      </div>
                      <button
                        class="btn chat-queue__remove"
                        type="button"
                        aria-label="Remove queued message"
                        @click=${() => props.onQueueRemove(item.id)}
                      >
                        ${icons.x}
                      </button>
                    </div>
                  `,
                )}
              </div>
            </div>
          `
          : nothing
      }
      ${renderFallbackIndicator(props.fallbackStatus)}
      ${renderCompactionIndicator(props.compactionStatus)}
      ${renderContextNotice(activeSession, props.sessions?.defaults?.contextTokens ?? null)}
      ${
        props.showNewMessages
          ? html`
            <button class="chat-new-messages" type="button" @click=${props.onScrollToBottom}>
              ${icons.arrowDown} New messages
            </button>
          `
          : nothing
      }

      <!-- Input bar -->
      <div class="agent-chat__input">
        ${renderSlashMenu(requestUpdate, props)} ${renderAttachmentPreview(props)}

        <input
          type="file"
          accept=${CHAT_ATTACHMENT_ACCEPT}
          multiple
          class="agent-chat__file-input"
          @change=${(e: Event) => handleFileSelect(e, props)}
        />

        ${
          vs.sttRecording && vs.sttInterimText
            ? html`<div class="agent-chat__stt-interim">${vs.sttInterimText}</div>`
            : nothing
        }

        <textarea
          ${ref((el) => el && adjustTextareaHeight(el as HTMLTextAreaElement))}
          .value=${props.draft}
          dir=${detectTextDirection(props.draft)}
          ?disabled=${!props.connected}
          @keydown=${handleKeyDown}
          @input=${handleInput}
          @paste=${(e: ClipboardEvent) => handlePaste(e, props)}
          placeholder=${vs.sttRecording ? "Listening..." : placeholder}
          rows="1"
        ></textarea>

        <div class="agent-chat__toolbar">
          <div class="agent-chat__toolbar-left">
            <button
              class="agent-chat__input-btn"
              @click=${() => {
                document.querySelector<HTMLInputElement>(".agent-chat__file-input")?.click();
              }}
              title="Attach file"
              aria-label="Attach file"
              ?disabled=${!props.connected}
            >
              ${icons.paperclip}
            </button>

            ${
              isSttSupported()
                ? html`
                  <button
                    class="agent-chat__input-btn ${
                      vs.sttRecording ? "agent-chat__input-btn--recording" : ""
                    }"
                    @click=${() => {
                      if (vs.sttRecording) {
                        stopStt();
                        vs.sttRecording = false;
                        vs.sttInterimText = "";
                        requestUpdate();
                      } else {
                        const started = startStt({
                          onTranscript: (text, isFinal) => {
                            if (isFinal) {
                              const current = getDraft();
                              const sep = current && !current.endsWith(" ") ? " " : "";
                              props.onDraftChange(current + sep + text);
                              vs.sttInterimText = "";
                            } else {
                              vs.sttInterimText = text;
                            }
                            requestUpdate();
                          },
                          onStart: () => {
                            vs.sttRecording = true;
                            requestUpdate();
                          },
                          onEnd: () => {
                            vs.sttRecording = false;
                            vs.sttInterimText = "";
                            requestUpdate();
                          },
                          onError: () => {
                            vs.sttRecording = false;
                            vs.sttInterimText = "";
                            requestUpdate();
                          },
                        });
                        if (started) {
                          vs.sttRecording = true;
                          requestUpdate();
                        }
                      }
                    }}
                    title=${vs.sttRecording ? "Stop recording" : "Voice input"}
                    ?disabled=${!props.connected}
                  >
                    ${vs.sttRecording ? icons.micOff : icons.mic}
                  </button>
                `
                : nothing
            }
            ${tokens ? html`<span class="agent-chat__token-count">${tokens}</span>` : nothing}
          </div>

          <div class="agent-chat__toolbar-right">
            ${nothing /* search hidden for now */}
            ${
              canAbort
                ? nothing
                : html`
                  <button
                    class="btn btn--ghost"
                    @click=${props.onNewSession}
                    title="New session"
                    aria-label="New session"
                  >
                    ${icons.plus}
                  </button>
                `
            }
            <button
              class="btn btn--ghost"
              @click=${() => exportMarkdown(props)}
              title="Export"
              aria-label="Export chat"
              ?disabled=${props.messages.length === 0}
            >
              ${icons.download}
            </button>

            ${
              canAbort && (isBusy || props.sending)
                ? html`
                  <button
                    class="chat-send-btn chat-send-btn--stop"
                    @click=${props.onAbort}
                    title="Stop"
                    aria-label="Stop generating"
                  >
                    ${icons.stop}
                  </button>
                `
                : html`
                  <button
                    class="chat-send-btn"
                    @click=${() => {
                      if (props.draft.trim()) {
                        inputHistory.push(props.draft);
                      }
                      props.onSend();
                    }}
                    ?disabled=${!props.connected || props.sending}
                    title=${isBusy ? "Queue" : "Send"}
                    aria-label=${isBusy ? "Queue message" : "Send message"}
                  >
                    ${icons.send}
                  </button>
                `
            }
          </div>
        </div>
      </div>
    </section>
  `;
}

const CHAT_HISTORY_RENDER_LIMIT = 200;

function groupMessages(items: ChatItem[]): Array<ChatItem | MessageGroup> {
  const result: Array<ChatItem | MessageGroup> = [];
  let currentGroup: MessageGroup | null = null;

  for (const item of items) {
    if (item.kind !== "message") {
      if (currentGroup) {
        result.push(currentGroup);
        currentGroup = null;
      }
      result.push(item);
      continue;
    }

    const normalized = normalizeMessage(item.message);
    const role = normalizeRoleForGrouping(normalized.role);
    const senderLabel = role.toLowerCase() === "user" ? (normalized.senderLabel ?? null) : null;
    const timestamp = normalized.timestamp || Date.now();

    if (
      !currentGroup ||
      currentGroup.role !== role ||
      (role.toLowerCase() === "user" && currentGroup.senderLabel !== senderLabel)
    ) {
      if (currentGroup) {
        result.push(currentGroup);
      }
      currentGroup = {
        kind: "group",
        key: `group:${role}:${item.key}`,
        role,
        senderLabel,
        messages: [{ message: item.message, key: item.key }],
        timestamp,
        isStreaming: false,
      };
    } else {
      currentGroup.messages.push({ message: item.message, key: item.key });
    }
  }

  if (currentGroup) {
    result.push(currentGroup);
  }
  return result;
}

function buildChatItems(props: ChatProps): Array<ChatItem | MessageGroup> {
  const items: ChatItem[] = [];
  const history = Array.isArray(props.messages) ? props.messages : [];
  const tools = Array.isArray(props.toolMessages) ? props.toolMessages : [];
  const historyStart = Math.max(0, history.length - CHAT_HISTORY_RENDER_LIMIT);
  if (historyStart > 0) {
    items.push({
      kind: "message",
      key: "chat:history:notice",
      message: {
        role: "system",
        content: `Showing last ${CHAT_HISTORY_RENDER_LIMIT} messages (${historyStart} hidden).`,
        timestamp: Date.now(),
      },
    });
  }
  for (let i = historyStart; i < history.length; i++) {
    const msg = history[i];
    const normalized = normalizeMessage(msg);
    const raw = msg as Record<string, unknown>;
    const marker = raw.__openclaw as Record<string, unknown> | undefined;
    if (marker && marker.kind === "compaction") {
      items.push({
        kind: "divider",
        key:
          typeof marker.id === "string"
            ? `divider:compaction:${marker.id}`
            : `divider:compaction:${normalized.timestamp}:${i}`,
        label: "Compaction",
        timestamp: normalized.timestamp ?? Date.now(),
      });
      continue;
    }

    if (!props.showToolCalls && normalized.role.toLowerCase() === "toolresult") {
      continue;
    }

    // Apply search filter if active
    if (vs.searchOpen && vs.searchQuery.trim() && !messageMatchesSearchQuery(msg, vs.searchQuery)) {
      continue;
    }

    items.push({
      kind: "message",
      key: messageKey(msg, i),
      message: msg,
    });
  }
  // Interleave stream segments and tool cards in order. Each segment
  // contains text that was streaming before the corresponding tool started.
  // This ensures correct visual ordering: text → tool → text → tool → ...
  const segments = props.streamSegments ?? [];
  const maxLen = Math.max(segments.length, tools.length);
  for (let i = 0; i < maxLen; i++) {
    if (i < segments.length && segments[i].text.trim().length > 0) {
      items.push({
        kind: "stream" as const,
        key: `stream-seg:${props.sessionKey}:${i}`,
        text: segments[i].text,
        startedAt: segments[i].ts,
      });
    }
    if (i < tools.length && props.showToolCalls) {
      items.push({
        kind: "message",
        key: messageKey(tools[i], i + history.length),
        message: tools[i],
      });
    }
  }

  if (props.stream !== null) {
    const key = `stream:${props.sessionKey}:${props.streamStartedAt ?? "live"}`;
    if (props.stream.trim().length > 0) {
      items.push({
        kind: "stream",
        key,
        text: props.stream,
        startedAt: props.streamStartedAt ?? Date.now(),
      });
    } else {
      items.push({ kind: "reading-indicator", key });
    }
  }

  return groupMessages(items);
}

function messageKey(message: unknown, index: number): string {
  const m = message as Record<string, unknown>;
  const toolCallId = typeof m.toolCallId === "string" ? m.toolCallId : "";
  if (toolCallId) {
    return `tool:${toolCallId}`;
  }
  const id = typeof m.id === "string" ? m.id : "";
  if (id) {
    return `msg:${id}`;
  }
  const messageId = typeof m.messageId === "string" ? m.messageId : "";
  if (messageId) {
    return `msg:${messageId}`;
  }
  const timestamp = typeof m.timestamp === "number" ? m.timestamp : null;
  const role = typeof m.role === "string" ? m.role : "unknown";
  if (timestamp != null) {
    return `msg:${role}:${timestamp}:${index}`;
  }
  return `msg:${role}:${index}`;
}
