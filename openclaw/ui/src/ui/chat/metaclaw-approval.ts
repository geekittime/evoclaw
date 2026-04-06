import { extractText } from "./message-extract.ts";

const METACLAW_APPROVAL_ID_RE = /Approval ID:\s*([A-Za-z0-9._:-]+)/i;
const METACLAW_APPROVAL_PROMPT_RE =
  /(这个工具在调用前需要获得你的允许|this tool requires your approval|使用 approve\b|使用 reject\b|use approve\b|use reject\b|require_approval)/i;

export function isMetaclawApprovalPromptText(text: string): boolean {
  const normalized = text.trim();
  if (!normalized) {
    return false;
  }
  return METACLAW_APPROVAL_ID_RE.test(normalized) && METACLAW_APPROVAL_PROMPT_RE.test(normalized);
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
  if (typeof entry.text === "string") {
    return isMetaclawApprovalPromptText(entry.text);
  }
  const text = extractText(message);
  return typeof text === "string" && isMetaclawApprovalPromptText(text);
}
