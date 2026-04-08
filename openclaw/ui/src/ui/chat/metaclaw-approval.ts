import { extractText } from "./message-extract.ts";

const METACLAW_APPROVAL_ID_RE = /Approval ID:\s*([A-Za-z0-9._:-]+)/i;
const OPENCLAW_APPROVAL_ID_RE =
  /Approval required\s*\(id\s+([A-Za-z0-9._:-]+),\s*full\s+([A-Za-z0-9._:-]+)\)/i;
const OPENCLAW_APPROVE_COMMAND_RE =
  /\/approve(?:@[^\s]+)?\s+([A-Za-z0-9][A-Za-z0-9._:-]*)\s+(allow-once|allow-always|always|deny)\b/i;
const METACLAW_APPROVAL_PROMPT_RE =
  /(这个工具在调用前需要获得你的允许|this tool requires your approval|使用 approve\b|使用 reject\b|use approve\b|use reject\b|require_approval|approval required\b|reply with:\s*\/approve\b)/i;

const APPROVAL_IGNORE_LINE_RE =
  /^(approval id:|approval required\b|这个工具在调用前需要获得你的允许|使用 approve\b|使用 reject\b|如果有多个待处理的批准|if there are multiple pending approvals|this tool requires your approval|use approve\b|use reject\b|reply with:\s*\/approve\b|if the short code is ambiguous|mode:|background mode|host:|cwd:|command:|```|.*require_approval.*$)/i;

export type AssistantApprovalPromptCandidate = {
  approvalId: string;
  rawText: string;
  detailText: string | null;
};

function extractApprovalPromptTexts(message: unknown): string[] {
  if (!message || typeof message !== "object") {
    return [];
  }
  const entry = message as Record<string, unknown>;
  const texts: string[] = [];
  const pushText = (value: unknown) => {
    if (typeof value !== "string") {
      return;
    }
    const trimmed = value.trim();
    if (trimmed) {
      texts.push(trimmed);
    }
  };

  pushText(entry.text);

  const content = entry.content;
  if (typeof content === "string") {
    pushText(content);
  } else if (Array.isArray(content)) {
    for (const item of content) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const record = item as Record<string, unknown>;
      const type = typeof record.type === "string" ? record.type.toLowerCase() : "";
      if (type === "text" || type === "toolresult" || type === "tool_result") {
        pushText(record.text);
      }
    }
  }

  const extracted = extractText(message);
  if (extracted?.trim()) {
    texts.push(extracted.trim());
  }

  return texts;
}

export function parseAssistantApprovalPromptText(
  text: string,
): AssistantApprovalPromptCandidate | null {
  const normalized = text.trim();
  if (!normalized) {
    return null;
  }

  const fullOpenClawMatch = normalized.match(OPENCLAW_APPROVAL_ID_RE);
  const commandMatch = normalized.match(OPENCLAW_APPROVE_COMMAND_RE);
  const approvalIdMatch = normalized.match(METACLAW_APPROVAL_ID_RE);
  const approvalId =
    fullOpenClawMatch?.[2] ?? approvalIdMatch?.[1] ?? commandMatch?.[1] ?? "";
  if (!approvalId || !METACLAW_APPROVAL_PROMPT_RE.test(normalized)) {
    return null;
  }

  const lines = normalized
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const detailText =
    [...lines]
      .reverse()
      .find((line) => !APPROVAL_IGNORE_LINE_RE.test(line)) ?? null;

  return {
    approvalId,
    rawText: normalized,
    detailText,
  };
}

export function isMetaclawApprovalPromptText(text: string): boolean {
  return parseAssistantApprovalPromptText(text) != null;
}

export function parseAssistantApprovalPromptMessage(
  message: unknown,
): AssistantApprovalPromptCandidate | null {
  const texts = extractApprovalPromptTexts(message);
  for (let index = texts.length - 1; index >= 0; index -= 1) {
    const candidate = parseAssistantApprovalPromptText(texts[index]!);
    if (candidate) {
      return candidate;
    }
  }
  return null;
}

export function isAssistantMetaclawApprovalPromptMessage(message: unknown): boolean {
  if (!message || typeof message !== "object") {
    return false;
  }
  const entry = message as Record<string, unknown>;
  const role = typeof entry.role === "string" ? entry.role.toLowerCase() : "";
  if (role !== "assistant") {
    return false;
  }
  return parseAssistantApprovalPromptMessage(message) != null;
}
