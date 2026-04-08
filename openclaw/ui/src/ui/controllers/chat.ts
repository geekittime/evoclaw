import { formatRawAssistantErrorForUi } from "../../../../src/shared/assistant-error-format.js";
import {
  normalizeInputProvenance,
  type InputProvenance,
} from "../../../../src/sessions/input-provenance.js";
import { resetToolStream } from "../app-tool-stream.ts";
import {
  buildAssistantApprovalFollowupMessage,
  extractCommandHead,
  isAssistantMetaclawApprovalPromptMessage,
  parseAssistantSuggestedExecApprovalMessage,
} from "../chat/metaclaw-approval.ts";
import { extractText } from "../chat/message-extract.ts";
import { formatConnectError } from "../connect-error.ts";
import { addExecApproval, type ExecApprovalRequest } from "./exec-approval.ts";
import type { ExecApprovalsFile, ExecApprovalsSnapshot } from "./exec-approvals.ts";
import type { GatewayBrowserClient } from "../gateway.ts";
import type { ChatAttachment } from "../ui-types.ts";
import { generateUUID } from "../uuid.ts";
import {
  formatMissingOperatorReadScopeMessage,
  isMissingOperatorReadScopeError,
} from "./scope-errors.ts";

const SILENT_REPLY_PATTERN = /^\s*NO_REPLY\s*$/;
export const METACLAW_APPROVAL_SOURCE_TOOL = "metaclaw-approval";

function isSilentReplyStream(text: string): boolean {
  return SILENT_REPLY_PATTERN.test(text);
}
/** Client-side defense-in-depth: detect assistant messages whose text is purely NO_REPLY. */
function isAssistantSilentReply(message: unknown): boolean {
  if (!message || typeof message !== "object") {
    return false;
  }
  const entry = message as Record<string, unknown>;
  const role = typeof entry.role === "string" ? entry.role.toLowerCase() : "";
  if (role !== "assistant") {
    return false;
  }
  // entry.text takes precedence — matches gateway extractAssistantTextForSilentCheck
  if (typeof entry.text === "string") {
    return isSilentReplyStream(entry.text);
  }
  const text = extractText(message);
  return typeof text === "string" && isSilentReplyStream(text);
}

function isHiddenInternalSystemMessage(message: unknown): boolean {
  if (!message || typeof message !== "object") {
    return false;
  }
  const entry = message as Record<string, unknown>;
  if (entry.role !== "user") {
    return false;
  }
  const provenance = normalizeInputProvenance(entry.provenance);
  return (
    provenance?.kind === "internal_system" &&
    provenance.sourceTool === METACLAW_APPROVAL_SOURCE_TOOL
  );
}

export type ChatState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  sessionKey: string;
  chatLoading: boolean;
  chatMessages: unknown[];
  chatThinkingLevel: string | null;
  chatSending: boolean;
  chatMessage: string;
  chatAttachments: ChatAttachment[];
  chatRunId: string | null;
  chatStream: string | null;
  chatStreamStartedAt: number | null;
  lastError: string | null;
  execApprovalQueue?: ExecApprovalRequest[];
  execApprovalsForm?: ExecApprovalsFile | null;
  execApprovalsSnapshot?: ExecApprovalsSnapshot | null;
  execApprovalError?: string | null;
};

function normalizeCommandForPolicy(command: string): string {
  return command.trim().toLowerCase();
}

function matchesPolicyCommand(pattern: string, commandText: string, commandHead: string): boolean {
  const normalizedPattern = normalizeCommandForPolicy(pattern);
  return (
    normalizedPattern.length > 0 &&
    (normalizedPattern === normalizeCommandForPolicy(commandText) ||
      normalizedPattern === normalizeCommandForPolicy(commandHead))
  );
}

function resolveExecApprovalsFile(state: ChatState): ExecApprovalsFile {
  return state.execApprovalsForm ?? state.execApprovalsSnapshot?.file ?? {};
}

function resolveSuggestedCommandMode(
  state: ChatState,
  commandText: string,
): "allow" | "ask" | "deny" {
  const file = resolveExecApprovalsFile(state);
  const commandHead = extractCommandHead(commandText);

  const allowlist = Array.isArray(file.commandAllowlist) ? file.commandAllowlist : [];
  if (allowlist.some((pattern) => matchesPolicyCommand(pattern, commandText, commandHead))) {
    return "allow";
  }

  const commandRules = file.commandRules ?? {};
  for (const [pattern, mode] of Object.entries(commandRules)) {
    if (matchesPolicyCommand(pattern, commandText, commandHead)) {
      return mode;
    }
  }

  return file.defaultCommandMode ?? "ask";
}

function hashText(value: string): string {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash.toString(16);
}

function buildAssistantFallbackApprovalId(
  sessionKey: string,
  commandText: string,
  timestamp: number,
): string {
  return `assistant-fallback:${hashText(`${sessionKey}:${timestamp}:${commandText}`)}`;
}

function isAssistantFallbackApproval(entry: ExecApprovalRequest): boolean {
  return entry.source === "assistant-fallback";
}

function removeAssistantFallbackApprovals(state: ChatState) {
  if (!Array.isArray(state.execApprovalQueue)) {
    return;
  }
  state.execApprovalQueue = state.execApprovalQueue.filter((entry) => !isAssistantFallbackApproval(entry));
}

function hasNativeExecApproval(state: ChatState): boolean {
  return Array.isArray(state.execApprovalQueue)
    ? state.execApprovalQueue.some((entry) => !isAssistantFallbackApproval(entry))
    : false;
}

function findLatestAssistantSuggestedApprovalCandidate(messages: unknown[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") {
      continue;
    }
    const entry = message as Record<string, unknown>;
    const role = typeof entry.role === "string" ? entry.role.toLowerCase() : "";
    if (role !== "assistant") {
      continue;
    }
    const candidate = parseAssistantSuggestedExecApprovalMessage(message);
    if (!candidate) {
      continue;
    }
    const timestamp =
      typeof entry.timestamp === "number" && Number.isFinite(entry.timestamp)
        ? entry.timestamp
        : index + 1;
    return { ...candidate, timestamp };
  }
  return null;
}

function reconcileAssistantSuggestedApproval(
  state: ChatState,
  opts: {
    autoContinue: boolean;
  },
) {
  removeAssistantFallbackApprovals(state);
  const candidate = findLatestAssistantSuggestedApprovalCandidate(state.chatMessages);
  if (!candidate) {
    return;
  }

  const mode = resolveSuggestedCommandMode(state, candidate.commandText);
  if (mode === "ask") {
    if (hasNativeExecApproval(state)) {
      return;
    }
    if (!Array.isArray(state.execApprovalQueue)) {
      state.execApprovalQueue = [];
    }
    state.execApprovalQueue = addExecApproval(state.execApprovalQueue, {
      id: buildAssistantFallbackApprovalId(state.sessionKey, candidate.commandText, candidate.timestamp),
      kind: "exec",
      source: "assistant-fallback",
      request: {
        command: candidate.commandText,
        sessionKey: state.sessionKey,
        host: "gateway",
        ask: candidate.detailText ?? "assistant suggested command awaiting approval",
      },
      createdAtMs: candidate.timestamp,
      expiresAtMs: candidate.timestamp + 30 * 60 * 1000,
    });
    state.execApprovalError = null;
    return;
  }

  if (!opts.autoContinue) {
    return;
  }

  const decision = mode === "allow" ? "allow-once" : "deny";
  void sendHiddenSystemChatMessage(
    state,
    buildAssistantApprovalFollowupMessage(candidate.commandText, decision),
    {
      sourceTool: METACLAW_APPROVAL_SOURCE_TOOL,
      appendAssistantErrorOnFailure: true,
    },
  ).catch((error) => {
    state.execApprovalError = `Approval follow-up failed: ${String(error)}`;
  });
}

export type ChatEventPayload = {
  runId: string;
  sessionKey: string;
  state: "delta" | "final" | "aborted" | "error";
  message?: unknown;
  errorMessage?: string;
};

function maybeResetToolStream(state: ChatState) {
  const toolHost = state as ChatState & Partial<Parameters<typeof resetToolStream>[0]>;
  if (
    toolHost.toolStreamById instanceof Map &&
    Array.isArray(toolHost.toolStreamOrder) &&
    Array.isArray(toolHost.chatToolMessages) &&
    Array.isArray(toolHost.chatStreamSegments)
  ) {
    resetToolStream(toolHost as Parameters<typeof resetToolStream>[0]);
  }
}

export async function loadChatHistory(state: ChatState) {
  if (!state.client || !state.connected) {
    return;
  }
  state.chatLoading = true;
  state.lastError = null;
  try {
    const res = await state.client.request<{ messages?: Array<unknown>; thinkingLevel?: string }>(
      "chat.history",
      {
        sessionKey: state.sessionKey,
        limit: 200,
      },
    );
    const messages = Array.isArray(res.messages) ? res.messages : [];
    state.chatMessages = messages.filter(
      (message) =>
        !isAssistantSilentReply(message) &&
        !isAssistantMetaclawApprovalPromptMessage(message) &&
        !isHiddenInternalSystemMessage(message),
    );
    reconcileAssistantSuggestedApproval(state, { autoContinue: false });
    state.chatThinkingLevel = res.thinkingLevel ?? null;
    // Clear all streaming state — history includes tool results and text
    // inline, so keeping streaming artifacts would cause duplicates.
    maybeResetToolStream(state);
    state.chatStream = null;
    state.chatStreamStartedAt = null;
  } catch (err) {
    if (isMissingOperatorReadScopeError(err)) {
      state.chatMessages = [];
      state.chatThinkingLevel = null;
      state.lastError = formatMissingOperatorReadScopeMessage("existing chat history");
    } else {
      state.lastError = String(err);
    }
  } finally {
    state.chatLoading = false;
  }
}

export async function sendHiddenSystemChatMessage(
  state: ChatState,
  message: string,
  opts: {
    sourceTool: string;
    sourceSessionKey?: string;
    appendAssistantErrorOnFailure?: boolean;
  },
): Promise<string> {
  if (!state.client || !state.connected) {
    throw new Error("Gateway is not connected.");
  }
  const trimmed = message.trim();
  if (!trimmed) {
    throw new Error("Hidden system message is empty.");
  }

  const now = Date.now();
  const runId = generateUUID();
  state.chatSending = true;
  state.lastError = null;
  state.chatRunId = runId;
  state.chatStream = "";
  state.chatStreamStartedAt = now;

  const systemInputProvenance: InputProvenance = {
    kind: "internal_system",
    sourceTool: opts.sourceTool,
    sourceSessionKey: opts.sourceSessionKey ?? state.sessionKey,
  };

  try {
    await state.client.request("chat.send", {
      sessionKey: state.sessionKey,
      message: trimmed,
      deliver: false,
      idempotencyKey: runId,
      systemInputProvenance,
    });
    return runId;
  } catch (err) {
    const error = formatConnectError(err);
    state.chatRunId = null;
    state.chatStream = null;
    state.chatStreamStartedAt = null;
    state.lastError = error;
    if (opts.appendAssistantErrorOnFailure !== false) {
      state.chatMessages = [
        ...state.chatMessages,
        {
          role: "assistant",
          content: [{ type: "text", text: "Error: " + error }],
          timestamp: Date.now(),
        },
      ];
    }
    throw new Error(error);
  } finally {
    state.chatSending = false;
  }
}

function dataUrlToBase64(dataUrl: string): { content: string; mimeType: string } | null {
  const match = /^data:([^;]+);base64,(.+)$/.exec(dataUrl);
  if (!match) {
    return null;
  }
  return { mimeType: match[1], content: match[2] };
}

type AssistantMessageNormalizationOptions = {
  roleRequirement: "required" | "optional";
  roleCaseSensitive?: boolean;
  requireContentArray?: boolean;
  allowTextField?: boolean;
};

function normalizeAssistantMessage(
  message: unknown,
  options: AssistantMessageNormalizationOptions,
): Record<string, unknown> | null {
  if (!message || typeof message !== "object") {
    return null;
  }
  const candidate = message as Record<string, unknown>;
  const roleValue = candidate.role;
  if (typeof roleValue === "string") {
    const role = options.roleCaseSensitive ? roleValue : roleValue.toLowerCase();
    if (role !== "assistant") {
      return null;
    }
  } else if (options.roleRequirement === "required") {
    return null;
  }

  if (options.requireContentArray) {
    return Array.isArray(candidate.content) ? candidate : null;
  }
  if (!("content" in candidate) && !(options.allowTextField && "text" in candidate)) {
    return null;
  }
  return candidate;
}

function normalizeAbortedAssistantMessage(message: unknown): Record<string, unknown> | null {
  return normalizeAssistantMessage(message, {
    roleRequirement: "required",
    roleCaseSensitive: true,
    requireContentArray: true,
  });
}

function normalizeFinalAssistantMessage(message: unknown): Record<string, unknown> | null {
  return normalizeAssistantMessage(message, {
    roleRequirement: "optional",
    allowTextField: true,
  });
}

function buildAssistantErrorMessage(errorMessage: string, stopReason: "error" | "aborted") {
  const trimmed = errorMessage.trim();
  return {
    role: "assistant",
    content: [{ type: "text", text: formatRawAssistantErrorForUi(trimmed) }],
    errorMessage: trimmed,
    stopReason,
    timestamp: Date.now(),
  };
}

function appendAssistantErrorMessage(
  state: ChatState,
  errorMessage: string | undefined,
  stopReason: "error" | "aborted",
) {
  const trimmed = typeof errorMessage === "string" ? errorMessage.trim() : "";
  if (!trimmed) {
    return false;
  }
  state.chatMessages = [...state.chatMessages, buildAssistantErrorMessage(trimmed, stopReason)];
  return true;
}

export async function sendChatMessage(
  state: ChatState,
  message: string,
  attachments?: ChatAttachment[],
): Promise<string | null> {
  if (!state.client || !state.connected) {
    return null;
  }
  const msg = message.trim();
  const hasAttachments = attachments && attachments.length > 0;
  if (!msg && !hasAttachments) {
    return null;
  }

  const now = Date.now();

  // Build user message content blocks
  const contentBlocks: Array<{ type: string; text?: string; source?: unknown }> = [];
  if (msg) {
    contentBlocks.push({ type: "text", text: msg });
  }
  // Add image previews to the message for display
  if (hasAttachments) {
    for (const att of attachments) {
      contentBlocks.push({
        type: "image",
        source: { type: "base64", media_type: att.mimeType, data: att.dataUrl },
      });
    }
  }

  state.chatMessages = [
    ...state.chatMessages,
    {
      role: "user",
      content: contentBlocks,
      timestamp: now,
    },
  ];

  state.chatSending = true;
  state.lastError = null;
  const runId = generateUUID();
  state.chatRunId = runId;
  state.chatStream = "";
  state.chatStreamStartedAt = now;

  // Convert attachments to API format
  const apiAttachments = hasAttachments
    ? attachments
        .map((att) => {
          const parsed = dataUrlToBase64(att.dataUrl);
          if (!parsed) {
            return null;
          }
          return {
            type: "image",
            mimeType: parsed.mimeType,
            content: parsed.content,
          };
        })
        .filter((a): a is NonNullable<typeof a> => a !== null)
    : undefined;

  try {
    await state.client.request("chat.send", {
      sessionKey: state.sessionKey,
      message: msg,
      deliver: false,
      idempotencyKey: runId,
      attachments: apiAttachments,
    });
    return runId;
  } catch (err) {
    const error = formatConnectError(err);
    state.chatRunId = null;
    state.chatStream = null;
    state.chatStreamStartedAt = null;
    state.lastError = error;
    state.chatMessages = [
      ...state.chatMessages,
      {
        role: "assistant",
        content: [{ type: "text", text: "Error: " + error }],
        timestamp: Date.now(),
      },
    ];
    return null;
  } finally {
    state.chatSending = false;
  }
}

export async function abortChatRun(state: ChatState): Promise<boolean> {
  if (!state.client || !state.connected) {
    return false;
  }
  const runId = state.chatRunId;
  try {
    await state.client.request(
      "chat.abort",
      runId ? { sessionKey: state.sessionKey, runId } : { sessionKey: state.sessionKey },
    );
    return true;
  } catch (err) {
    state.lastError = formatConnectError(err);
    return false;
  }
}

export function handleChatEvent(state: ChatState, payload?: ChatEventPayload) {
  if (!payload) {
    return null;
  }
  if (payload.sessionKey !== state.sessionKey) {
    return null;
  }

  // Final from another run (e.g. sub-agent announce): refresh history to show new message.
  // See https://github.com/openclaw/openclaw/issues/1909
  if (payload.runId && state.chatRunId && payload.runId !== state.chatRunId) {
    if (payload.state === "final") {
      const finalMessage = normalizeFinalAssistantMessage(payload.message);
      if (
        finalMessage &&
        !isAssistantSilentReply(finalMessage) &&
        !isAssistantMetaclawApprovalPromptMessage(finalMessage)
      ) {
        state.chatMessages = [...state.chatMessages, finalMessage];
        reconcileAssistantSuggestedApproval(state, { autoContinue: true });
        return null;
      }
      appendAssistantErrorMessage(state, payload.errorMessage, "error");
      return "final";
    }
    return null;
  }

  if (payload.state === "delta") {
    const next = extractText(payload.message);
    if (
      typeof next === "string" &&
      !isSilentReplyStream(next) &&
      !isAssistantMetaclawApprovalPromptMessage({
        role: "assistant",
        content: [{ type: "text", text: next }],
      })
    ) {
      state.chatStream = next;
    }
  } else if (payload.state === "final") {
    const finalMessage = normalizeFinalAssistantMessage(payload.message);
    if (
      finalMessage &&
      !isAssistantSilentReply(finalMessage) &&
      !isAssistantMetaclawApprovalPromptMessage(finalMessage)
    ) {
      state.chatMessages = [...state.chatMessages, finalMessage];
    } else if (
      state.chatStream?.trim() &&
      !isSilentReplyStream(state.chatStream) &&
      !isAssistantMetaclawApprovalPromptMessage({
        role: "assistant",
        content: [{ type: "text", text: state.chatStream }],
      })
    ) {
      state.chatMessages = [
        ...state.chatMessages,
        {
          role: "assistant",
          content: [{ type: "text", text: state.chatStream }],
          timestamp: Date.now(),
        },
      ];
    } else {
      appendAssistantErrorMessage(state, payload.errorMessage, "error");
    }
    state.chatStream = null;
    state.chatRunId = null;
    state.chatStreamStartedAt = null;
    reconcileAssistantSuggestedApproval(state, { autoContinue: true });
  } else if (payload.state === "aborted") {
    const normalizedMessage = normalizeAbortedAssistantMessage(payload.message);
    if (
      normalizedMessage &&
      !isAssistantSilentReply(normalizedMessage) &&
      !isAssistantMetaclawApprovalPromptMessage(normalizedMessage)
    ) {
      state.chatMessages = [...state.chatMessages, normalizedMessage];
    } else {
      const streamedText = state.chatStream ?? "";
      if (
        streamedText.trim() &&
        !isSilentReplyStream(streamedText) &&
        !isAssistantMetaclawApprovalPromptMessage({
          role: "assistant",
          content: [{ type: "text", text: streamedText }],
        })
      ) {
        state.chatMessages = [
          ...state.chatMessages,
          {
            role: "assistant",
            content: [{ type: "text", text: streamedText }],
            timestamp: Date.now(),
          },
        ];
      } else {
        appendAssistantErrorMessage(state, payload.errorMessage, "aborted");
      }
    }
    state.chatStream = null;
    state.chatRunId = null;
    state.chatStreamStartedAt = null;
  } else if (payload.state === "error") {
    appendAssistantErrorMessage(state, payload.errorMessage, "error");
    state.chatStream = null;
    state.chatRunId = null;
    state.chatStreamStartedAt = null;
    state.lastError = payload.errorMessage ?? "chat error";
  }
  return payload.state;
}
