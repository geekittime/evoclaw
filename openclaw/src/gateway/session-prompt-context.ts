import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("gateway/session-prompt-context");

const DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions";
const DEEPSEEK_API_KEY = "sk-33021c0bec434de4b877c3142cc409c9";
const DEEPSEEK_MODEL = "deepseek-chat";

const MAX_SUMMARY_SOURCE_CHARS = 80_000;
const MAX_MESSAGE_TEXT_CHARS = 6_000;
const MAX_FALLBACK_SUMMARY_CHARS = 4_000;
const MAX_TOOL_ARGS_CHARS = 1_200;

function trimText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function extractMessageText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as Record<string, unknown>;
    const type = typeof record.type === "string" ? record.type.toLowerCase() : "";
    const text = typeof record.text === "string" ? record.text.trim() : "";
    if (text) {
      parts.push(text);
      continue;
    }
    if (type === "toolcall" || type === "tool_call") {
      const name = typeof record.name === "string" ? record.name.trim() : "tool";
      const args =
        record.arguments ?? record.args ?? (typeof record.input === "object" ? record.input : undefined);
      let argsText = "";
      if (args !== undefined) {
        try {
          argsText = JSON.stringify(args, null, 2);
        } catch {
          argsText = String(args);
        }
      }
      parts.push(
        `Tool call: ${name}${argsText ? `\n${argsText.slice(0, MAX_TOOL_ARGS_CHARS)}` : ""}`.trim(),
      );
      continue;
    }
    if (type === "toolresult" || type === "tool_result") {
      const name = typeof record.name === "string" ? record.name.trim() : "tool";
      parts.push(`Tool result: ${name}`);
    }
  }
  return parts.join("\n");
}

function stringifyMessage(message: unknown): string | null {
  if (!message || typeof message !== "object") {
    return null;
  }
  const entry = message as Record<string, unknown>;
  const role = typeof entry.role === "string" ? entry.role.trim().toLowerCase() : "other";
  const text = trimText(
    typeof entry.text === "string" ? entry.text : extractMessageText(entry.content),
  );
  if (!text) {
    return null;
  }
  const label =
    role === "assistant"
      ? "Assistant"
      : role === "user"
        ? "User"
        : role === "tool"
          ? "Tool"
          : role === "system"
            ? "System"
            : "Message";
  return `${label}: ${text.slice(0, MAX_MESSAGE_TEXT_CHARS)}`;
}

export function formatConversationForSummary(messages: unknown[]): string {
  const lines = messages
    .map((message) => stringifyMessage(message))
    .filter((line): line is string => Boolean(line));
  const transcript = lines.join("\n\n");
  if (transcript.length <= MAX_SUMMARY_SOURCE_CHARS) {
    return transcript;
  }
  return transcript.slice(transcript.length - MAX_SUMMARY_SOURCE_CHARS);
}

function fallbackConversationSummary(messages: unknown[]): string {
  const source = formatConversationForSummary(messages);
  if (!source) {
    return "No substantial conversation history yet.";
  }
  return source.slice(source.length - Math.min(MAX_FALLBACK_SUMMARY_CHARS, source.length));
}

async function callDeepSeekSummary(params: {
  system: string;
  user: string;
  maxTokens?: number;
}): Promise<string> {
  const response = await fetch(DEEPSEEK_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: DEEPSEEK_MODEL,
      temperature: 0.1,
      max_tokens: params.maxTokens ?? 600,
      messages: [
        { role: "system", content: params.system },
        { role: "user", content: params.user },
      ],
    }),
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`DeepSeek summary failed (${response.status}): ${text || "no body"}`);
  }
  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = trimText(payload.choices?.[0]?.message?.content);
  if (!content) {
    throw new Error("DeepSeek summary returned an empty message.");
  }
  return content;
}

export async function summarizeFeedbackIntoImportantNote(params: {
  instructionText: string;
  responseText: string;
  rating: "good" | "bad";
  feedback: string;
}): Promise<string> {
  const feedback = trimText(params.feedback) ?? "";
  const fallback =
    params.rating === "good"
      ? `User liked this answer style. Keep: ${(feedback || params.responseText).slice(0, 300)}`
      : `User disliked this answer. Improve: ${(feedback || params.responseText).slice(0, 300)}`;
  try {
    return await callDeepSeekSummary({
      system:
        "You summarize answer feedback into durable prompt notes. Return one concise bullet-style line without markdown bullets. Focus on stable preferences, corrections, or constraints that should affect future answers.",
      user: [
        `User question:\n${params.instructionText || "(none)"}`,
        `Assistant answer:\n${params.responseText || "(none)"}`,
        `Feedback rating: ${params.rating}`,
        `Feedback details:\n${feedback || "(empty)"}`,
        "Write one short durable note for future prompts.",
      ].join("\n\n"),
      maxTokens: 180,
    });
  } catch (error) {
    log.warn(`feedback summary fallback: ${error instanceof Error ? error.message : String(error)}`);
    return fallback;
  }
}

export async function summarizeConversationHistory(params: {
  messages: unknown[];
  instructions?: string;
}): Promise<string> {
  const source = formatConversationForSummary(params.messages);
  const fallback = fallbackConversationSummary(params.messages);
  try {
    return await callDeepSeekSummary({
      system:
        "You compress conversation history for future continuation. Preserve goals, decisions, constraints, user preferences, selected skills, approvals or denials, file paths, and unresolved next steps. Omit filler. Return a compact structured summary in plain text.",
      user: [
        params.instructions?.trim()
          ? `Compression instructions:\n${params.instructions.trim()}`
          : undefined,
        "Conversation transcript:",
        source || "(empty)",
      ]
        .filter(Boolean)
        .join("\n\n"),
      maxTokens: 900,
    });
  } catch (error) {
    log.warn(`conversation summary fallback: ${error instanceof Error ? error.message : String(error)}`);
    return fallback;
  }
}
