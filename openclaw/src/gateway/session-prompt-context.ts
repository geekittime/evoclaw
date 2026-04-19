import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("gateway/session-prompt-context");

const DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions";
const DEEPSEEK_API_KEY = "sk-33021c0bec434de4b877c3142cc409c9";
const DEEPSEEK_MODEL = "deepseek-chat";

const MAX_SUMMARY_SOURCE_CHARS = 120_000;
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
  if (role === "system") {
    return null;
  }
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
  const fallbackLines =
    params.rating === "good"
      ? [
          `Preserve this preference in future answers: ${(feedback || params.responseText).slice(0, 240)}`,
        ]
      : [
          `Avoid repeating this issue in future answers: ${(feedback || params.responseText).slice(0, 240)}`,
        ];
  const fallback = fallbackLines.join("\n");
  try {
    return await callDeepSeekSummary({
      system:
        [
          "You maintain a persistent IMPORTANT-NOTES memory for future prompts.",
          "Analyze the user question, the assistant answer, the good/bad rating, and the user's written feedback together.",
          "Extract durable lessons for future prompts: user preferences, style expectations, corrections, constraints, and things the assistant should pay attention to.",
          "Keep only stable guidance that can generalize across future turns.",
          "Do not include transient task details, file paths, timestamps, or one-off narration unless they imply a durable rule.",
          "Prefer imperative guidance such as 'Start with ...', 'Avoid ...', 'Always ...', or 'When ..., ...'.",
          "Return 1 to 3 short plain-text lines, one durable note per line.",
          "Do not use markdown bullets, numbering, headings, or extra commentary.",
        ].join(" "),
      user: [
        `User question:\n${params.instructionText || "(none)"}`,
        `Assistant answer:\n${params.responseText || "(none)"}`,
        `Feedback rating: ${params.rating}`,
        `Feedback details:\n${feedback || "(empty)"}`,
        "Summarize the durable experience from this interaction for IMPORTANT-NOTES.",
        "Focus on user preferences and things the assistant should pay attention to in future replies.",
      ].join("\n\n"),
      maxTokens: 260,
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
        [
          "You are writing a continuation summary for one chat session.",
          "The transcript can contain user requests, assistant replies, tool calls, and tool results.",
          "Summarize only the main task-relevant interaction: what the user asked for, what the assistant concluded or attempted, what tools were called, what the meaningful tool results were, what decisions were made, and what remains unresolved.",
          "Write a concise but sufficiently informative summary that another assistant can continue from immediately.",
          "Focus on the main task thread and the most important turns, not every minor exchange.",
          "Preserve concrete facts from the session when they matter for future work, especially important answers, meaningful tool outcomes, approvals or denials, file changes, and unresolved next steps.",
          "Do not omit tool outcomes if they changed the state of the task.",
          "Do not retain low-signal detail such as repeated confirmations, routine phrasing, boilerplate, exhaustive command arguments, or long raw outputs unless they materially affect the task state.",
          "Compress repetitive tool chatter into a short description instead of replaying every small step.",
          "Do not summarize user preferences, style rules, global important-notes, or any durable memory policy here unless they were explicitly discussed as part of the session conversation.",
          "Do not summarize repository boilerplate or standing context such as IDENTITY.md, USER.md, SOUL.md, MEMORY.md, memory files, HEARTBEAT files, workspace primers, or skills lists unless the session itself directly discussed or edited them as the task.",
          "Do not turn the output into a rigid sectioned template unless the content truly benefits from it.",
          "Prefer 1 to 3 readable paragraphs. Use short bullets only if unresolved next steps are clearer that way.",
          "Return plain text only.",
        ].join(" "),
      user: [
        params.instructions?.trim()
          ? `Compression instructions:\n${params.instructions.trim()}`
          : undefined,
        "Please summarize the full session, including the user/assistant conversation and the tool execution process.",
        "Focus on the actual dialogue and task progress, not standing background files or global configuration context.",
        "Conversation transcript:",
        source || "(empty)",
      ]
        .filter(Boolean)
        .join("\n\n"),
      maxTokens: 1400,
    });
  } catch (error) {
    log.warn(`conversation summary fallback: ${error instanceof Error ? error.message : String(error)}`);
    return fallback;
  }
}

export async function summarizeTaskState(params: {
  messages: unknown[];
  existingSummary?: string;
  instructions?: string;
}): Promise<string> {
  const source = formatConversationForSummary(params.messages);
  const fallback = trimText(params.existingSummary) ?? fallbackConversationSummary(params.messages);
  try {
    return await callDeepSeekSummary({
      system:
        [
          "You maintain the current TASK STATE for a personal OpenClaw agent session.",
          "Summarize only actionable state needed to continue the current task.",
          "Include current user goal, completed steps, important tool results, files or paths touched, approvals or denials, current blockers, and the likely next step.",
          "Do not include stable user preferences, global important-notes, skill lists, identity files, or general background unless they are directly part of the active task.",
          "Keep it concise, concrete, and update-oriented.",
          "Return plain text only.",
        ].join(" "),
      user: [
        params.instructions?.trim()
          ? `Task-state instructions:\n${params.instructions.trim()}`
          : undefined,
        params.existingSummary?.trim()
          ? `Existing session summary:\n${params.existingSummary.trim()}`
          : undefined,
        "Conversation and tool transcript:",
        source || "(empty)",
        "Write the current task state for the next turn.",
      ]
        .filter(Boolean)
        .join("\n\n"),
      maxTokens: 700,
    });
  } catch (error) {
    log.warn(`task state summary fallback: ${error instanceof Error ? error.message : String(error)}`);
    return fallback;
  }
}
