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

export type AssistantSuggestedExecApprovalCandidate = {
  commandText: string;
  rawText: string;
  detailText: string | null;
};

const SUGGESTED_EXEC_APPROVAL_RE =
  /(需要你批准|需要你确认|请你批准|请确认是否|等待你的批准|need your approval|requires your approval|please approve|awaiting your approval|approval is required)/i;
const SUGGESTED_EXEC_COMMAND_LINE_RE =
  /^(?:删除命令|执行命令|运行命令|命令|delete command|run command|command)\s*[:：]\s*(.+)$/i;

function normalizeSuggestedCommand(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  const unwrapped = trimmed.replace(/^`+|`+$/g, "").trim();
  return unwrapped.replace(/[。；;]+$/, "").trim();
}

function extractSuggestedCommand(text: string): string | null {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const match = lines[index]?.match(SUGGESTED_EXEC_COMMAND_LINE_RE);
    if (!match) {
      continue;
    }
    const command = normalizeSuggestedCommand(match[1] ?? "");
    if (command) {
      return command;
    }
  }
  return null;
}

export function extractCommandHead(command: string): string {
  const normalized = normalizeSuggestedCommand(command);
  if (!normalized) {
    return "";
  }
  const parts = normalized.split(/\s+/).filter(Boolean);
  let index = 0;
  while (index < parts.length && /^[A-Za-z_][A-Za-z0-9_]*=/.test(parts[index] ?? "")) {
    index += 1;
  }
  if ((parts[index] ?? "").toLowerCase() === "sudo") {
    index += 1;
  }
  return (parts[index] ?? "").trim();
}

export function buildAssistantApprovalFollowupMessage(
  commandText: string,
  decision: "allow-once" | "allow-always" | "deny",
): string {
  const command = normalizeSuggestedCommand(commandText);
  if (decision === "deny") {
    return [
      `用户拒绝执行这条 shell 命令：\`${command}\`。`,
      "不要执行它，也不要再次请求同一确认。",
      "请基于当前上下文继续任务，并给出替代方案或明确说明阻塞原因。",
    ].join("\n");
  }
  return [
    `用户已批准执行这条 shell 命令：\`${command}\`。`,
    "不要再次请求同一确认。",
    "现在立即调用 exec 执行这条命令，并基于执行结果继续完成当前任务。",
  ].join("\n");
}

type StructuredApprovalCandidate = {
  approvalId: string;
  detailText: string | null;
};

function parseStructuredApprovalRecord(value: unknown): StructuredApprovalCandidate | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const directApproval = record.approval;
  if (directApproval && typeof directApproval === "object" && !Array.isArray(directApproval)) {
    const approvalRecord = directApproval as Record<string, unknown>;
    const approvalId =
      typeof approvalRecord.approvalId === "string" ? approvalRecord.approvalId.trim() : "";
    if (approvalId) {
      const command =
        typeof approvalRecord.command === "string" ? approvalRecord.command.trim() : "";
      const warningText =
        typeof approvalRecord.warningText === "string" ? approvalRecord.warningText.trim() : "";
      return {
        approvalId,
        detailText: warningText || command || "OpenClaw reported a pending approval.",
      };
    }
  }

  const channelData = record.channelData;
  if (channelData && typeof channelData === "object" && !Array.isArray(channelData)) {
    const execApproval = (channelData as Record<string, unknown>).execApproval;
    if (execApproval && typeof execApproval === "object" && !Array.isArray(execApproval)) {
      const approvalRecord = execApproval as Record<string, unknown>;
      const approvalId =
        typeof approvalRecord.approvalId === "string" ? approvalRecord.approvalId.trim() : "";
      if (approvalId) {
        return {
          approvalId,
          detailText: "OpenClaw reported a pending approval.",
        };
      }
    }
  }

  const details = record.details;
  if (details && typeof details === "object" && !Array.isArray(details)) {
    const detailRecord = details as Record<string, unknown>;
    if (detailRecord.status === "approval-pending") {
      const approvalId =
        typeof detailRecord.approvalId === "string" ? detailRecord.approvalId.trim() : "";
      if (approvalId) {
        const command = typeof detailRecord.command === "string" ? detailRecord.command.trim() : "";
        const warningText =
          typeof detailRecord.warningText === "string" ? detailRecord.warningText.trim() : "";
        return {
          approvalId,
          detailText: warningText || command || "OpenClaw reported a pending approval.",
        };
      }
    }
  }

  return null;
}

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

export function parseAssistantSuggestedExecApprovalText(
  text: string,
): AssistantSuggestedExecApprovalCandidate | null {
  const normalized = text.trim();
  if (!normalized || !SUGGESTED_EXEC_APPROVAL_RE.test(normalized)) {
    return null;
  }
  const commandText = extractSuggestedCommand(normalized);
  if (!commandText) {
    return null;
  }
  const lines = normalized
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const detailText =
    [...lines]
      .reverse()
      .find((line) => !APPROVAL_IGNORE_LINE_RE.test(line) && !SUGGESTED_EXEC_COMMAND_LINE_RE.test(line)) ??
    commandText;
  return {
    commandText,
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
  if (message && typeof message === "object") {
    const entry = message as Record<string, unknown>;
    const structured = parseStructuredApprovalRecord(entry);
    if (structured) {
      return {
        approvalId: structured.approvalId,
        rawText: "",
        detailText: structured.detailText,
      };
    }
    const content = entry.content;
    if (Array.isArray(content)) {
      for (let index = content.length - 1; index >= 0; index -= 1) {
        const structuredItem = parseStructuredApprovalRecord(content[index]);
        if (structuredItem) {
          return {
            approvalId: structuredItem.approvalId,
            rawText: "",
            detailText: structuredItem.detailText,
          };
        }
      }
    }
  }

  const texts = extractApprovalPromptTexts(message);
  for (let index = texts.length - 1; index >= 0; index -= 1) {
    const candidate = parseAssistantApprovalPromptText(texts[index]!);
    if (candidate) {
      return candidate;
    }
  }
  return null;
}

export function parseAssistantSuggestedExecApprovalMessage(
  message: unknown,
): AssistantSuggestedExecApprovalCandidate | null {
  const texts = extractApprovalPromptTexts(message);
  for (let index = texts.length - 1; index >= 0; index -= 1) {
    const candidate = parseAssistantSuggestedExecApprovalText(texts[index]!);
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
