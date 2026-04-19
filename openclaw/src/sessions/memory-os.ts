import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { SessionPromptContext } from "../config/sessions/types.js";
import { resolveStateDir } from "../config/paths.js";
import { loadGlobalImportantNotes } from "./global-important-notes.js";

const MEMORY_OS_FILENAME = "memory-os.json";
const MEMORY_OS_VERSION = 1;
const MAX_SHORT_TERM_PAGES = 12;
const MAX_SEGMENTS_PER_SESSION = 24;
const MAX_LONG_TERM_NOTES = 256;
const MAX_RETRIEVED_SEGMENTS = 3;
const MAX_RETRIEVED_NOTES = 6;
const MAX_RETRIEVED_PAGES = 2;
const MAX_SEGMENT_SUMMARY_CHARS = 4_000;
const MAX_TASK_STATE_CHARS = 2_000;
const MAX_PAGE_CONTENT_CHARS = 1_600;
const KEYWORD_LIMIT = 18;
const CJK_TOKEN_PATTERN = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+/u;
const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "this",
  "that",
  "from",
  "into",
  "were",
  "have",
  "has",
  "been",
  "will",
  "would",
  "should",
  "could",
  "about",
  "after",
  "before",
  "when",
  "where",
  "which",
  "what",
  "then",
  "than",
  "them",
  "they",
  "there",
  "their",
  "your",
  "ours",
  "ourselves",
  "using",
  "used",
  "user",
  "assistant",
  "tool",
  "result",
  "call",
  "session",
  "notes",
  "summary",
  "current",
  "task",
  "state",
  "history",
  "important",
  "memory",
  "openclaw",
  "reply",
  "prompt",
]);

export type MemoryOsShortTermPage = {
  id: string;
  title: string;
  content: string;
  keywords: string[];
  updatedAt: number;
};

export type MemoryOsSegment = {
  id: string;
  sessionKey: string;
  title: string;
  summary: string;
  taskState?: string;
  keywords: string[];
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  heat: number;
  source: "manual" | "auto";
};

export type MemoryOsLongTermNote = {
  id: string;
  content: string;
  keywords: string[];
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  sourceSessionKeys: string[];
};

export type MemoryOsSessionState = {
  shortTermPages: MemoryOsShortTermPage[];
  segmentIds: string[];
  lastUpdatedAt?: number;
  lastCompactedAt?: number;
};

export type MemoryOsStore = {
  version: number;
  sessions: Record<string, MemoryOsSessionState>;
  segments: Record<string, MemoryOsSegment>;
  longTermNotes: Record<string, MemoryOsLongTermNote>;
};

export type MemoryOsSessionSnapshot = {
  session_id: string;
  short_term_page_count: number;
  mid_term_segment_count: number;
  long_term_note_count: number;
  latest_segment_title?: string | null;
  latest_segment_summary?: string | null;
  latest_updated_at?: number | null;
};

type NormalizedMessage = {
  role: string;
  text: string;
  timestamp: number;
};

type OrderedNormalizedMessage = NormalizedMessage & {
  order: number;
};

function trimOptionalText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function clampTail(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  return value.slice(value.length - maxChars);
}

function normalizeLine(value: string): string {
  return value.trim().replace(/^[-*•\d.)\s]+/u, "").trim();
}

function splitLines(value: string | undefined): string[] {
  return (value ?? "")
    .split(/\r?\n/u)
    .map(normalizeLine)
    .filter(Boolean);
}

function normalizeSummaryText(value: string | undefined, maxChars: number): string | undefined {
  const trimmed = trimOptionalText(value);
  if (!trimmed) {
    return undefined;
  }
  return clampTail(trimmed, maxChars);
}

function normalizeKeywords(value: string): string[] {
  const counts = new Map<string, number>();
  const rawTokens =
    value
      .toLowerCase()
      .match(/[\p{Letter}\p{Number}_-]+/gu)
      ?.map((token) => token.trim())
      .filter(Boolean) ?? [];
  const tokens: string[] = [];
  for (const token of rawTokens) {
    if (CJK_TOKEN_PATTERN.test(token)) {
      if (token.length <= 2) {
        tokens.push(token);
        continue;
      }
      for (let index = 0; index < token.length - 1; index += 1) {
        tokens.push(token.slice(index, index + 2));
      }
      continue;
    }
    if (token.length >= 2) {
      tokens.push(token);
    }
  }
  for (const token of tokens) {
    if (STOP_WORDS.has(token)) {
      continue;
    }
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, KEYWORD_LIMIT)
    .map(([token]) => token);
}

function scoreKeywords(queryKeywords: string[], candidateKeywords: string[]): number {
  if (queryKeywords.length === 0 || candidateKeywords.length === 0) {
    return 0;
  }
  const querySet = new Set(queryKeywords);
  let overlap = 0;
  for (const keyword of candidateKeywords) {
    if (querySet.has(keyword)) {
      overlap += 1;
    }
  }
  return overlap / Math.max(1, Math.min(queryKeywords.length, candidateKeywords.length));
}

function deriveTitle(value: string | undefined, fallback: string): string {
  const source = trimOptionalText(value) ?? fallback;
  const firstSentence =
    source
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .find(Boolean)
      ?.split(/[。！？.!?]/u)[0]
      ?.trim() ?? fallback;
  return firstSentence.length > 80 ? `${firstSentence.slice(0, 77)}...` : firstSentence;
}

function extractMessageText(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
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
      const args = record.arguments ?? record.args ?? record.input;
      let argsText = "";
      if (args !== undefined) {
        try {
          argsText = JSON.stringify(args, null, 2);
        } catch {
          argsText = String(args);
        }
      }
      parts.push(`Tool call: ${name}${argsText ? `\n${argsText}` : ""}`.trim());
      continue;
    }
    if (type === "toolresult" || type === "tool_result") {
      const name = typeof record.name === "string" ? record.name.trim() : "tool";
      const resultText = typeof record.output === "string" ? record.output.trim() : "";
      parts.push(`Tool result: ${name}${resultText ? `\n${resultText}` : ""}`.trim());
    }
  }
  return parts.join("\n");
}

function normalizeMessage(message: unknown): NormalizedMessage | null {
  if (!message || typeof message !== "object") {
    return null;
  }
  const record = message as Record<string, unknown>;
  const role =
    typeof record.role === "string"
      ? record.role.trim().toLowerCase()
      : typeof record.kind === "string"
        ? record.kind.trim().toLowerCase()
        : "message";
  const text = trimOptionalText(
    typeof record.text === "string" ? record.text : extractMessageText(record.content),
  );
  if (!text) {
    return null;
  }
  const timestamp =
    typeof record.timestamp === "number" && Number.isFinite(record.timestamp)
      ? Math.floor(record.timestamp)
      : Date.now();
  return { role, text: clampTail(text, MAX_PAGE_CONTENT_CHARS), timestamp };
}

function labelForRole(role: string): string {
  switch (role) {
    case "assistant":
      return "Assistant";
    case "user":
      return "User";
    case "tool":
    case "toolresult":
    case "tool_result":
      return "Tool";
    default:
      return "Message";
  }
}

function buildShortTermPages(messages: unknown[], sessionKey: string): MemoryOsShortTermPage[] {
  const normalized = messages
    .map((message, order) => {
      const normalizedMessage = normalizeMessage(message);
      return normalizedMessage ? { ...normalizedMessage, order } : null;
    })
    .filter((message): message is OrderedNormalizedMessage => Boolean(message))
    .sort((left, right) => left.timestamp - right.timestamp || left.order - right.order);
  const pages: MemoryOsShortTermPage[] = [];
  let current: NormalizedMessage[] = [];

  const flush = () => {
    if (current.length === 0) {
      return;
    }
    const updatedAt = current[current.length - 1]?.timestamp ?? Date.now();
    const content = current
      .map((item) => `${labelForRole(item.role)}: ${item.text}`)
      .join("\n\n");
    const titleSource =
      current.find((item) => item.role === "user")?.text ??
      current.find((item) => item.role === "assistant")?.text ??
      current[0]?.text ??
      "Recent interaction";
    const title = deriveTitle(titleSource, "Recent interaction");
    pages.push({
      id: `${sessionKey}:stm:${updatedAt}:${pages.length}`,
      title,
      content: clampTail(content, MAX_PAGE_CONTENT_CHARS),
      keywords: normalizeKeywords(content),
      updatedAt,
    });
    current = [];
  };

  for (const item of normalized) {
    if (item.role === "user" && current.length > 0) {
      flush();
    }
    current.push(item);
  }
  flush();
  return pages.slice(-MAX_SHORT_TERM_PAGES);
}

function createEmptyStore(): MemoryOsStore {
  return {
    version: MEMORY_OS_VERSION,
    sessions: {},
    segments: {},
    longTermNotes: {},
  };
}

function normalizeStore(value: unknown): MemoryOsStore {
  if (!value || typeof value !== "object") {
    return createEmptyStore();
  }
  const store = value as Partial<MemoryOsStore>;
  return {
    version: MEMORY_OS_VERSION,
    sessions: store.sessions && typeof store.sessions === "object" ? store.sessions : {},
    segments: store.segments && typeof store.segments === "object" ? store.segments : {},
    longTermNotes:
      store.longTermNotes && typeof store.longTermNotes === "object" ? store.longTermNotes : {},
  };
}

export function resolveMemoryOsPath(env: NodeJS.ProcessEnv = process.env): string {
  return path.join(resolveStateDir(env), "prompt-context", MEMORY_OS_FILENAME);
}

export function loadMemoryOsStore(env: NodeJS.ProcessEnv = process.env): MemoryOsStore {
  const filePath = resolveMemoryOsPath(env);
  try {
    if (!fs.existsSync(filePath)) {
      return createEmptyStore();
    }
    const raw = fs.readFileSync(filePath, "utf8");
    if (!raw.trim()) {
      return createEmptyStore();
    }
    return normalizeStore(JSON.parse(raw));
  } catch {
    return createEmptyStore();
  }
}

export function saveMemoryOsStore(
  store: MemoryOsStore,
  env: NodeJS.ProcessEnv = process.env,
): MemoryOsStore {
  const filePath = resolveMemoryOsPath(env);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(store, null, 2), "utf8");
  return store;
}

export function syncMemoryOsShortTermPages(params: {
  sessionKey: string;
  messages: unknown[];
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
}): MemoryOsStore {
  const store = loadMemoryOsStore(params.env);
  const updatedAt = params.updatedAt ?? Date.now();
  const sessionState = store.sessions[params.sessionKey] ?? {
    shortTermPages: [],
    segmentIds: [],
  };
  sessionState.shortTermPages = buildShortTermPages(params.messages, params.sessionKey);
  sessionState.lastUpdatedAt = updatedAt;
  store.sessions[params.sessionKey] = sessionState;
  return saveMemoryOsStore(store, params.env);
}

function normalizeLongTermNotesIntoStore(params: {
  store: MemoryOsStore;
  content?: string;
  sessionKey?: string;
  updatedAt?: number;
}): void {
  const lines = splitLines(params.content);
  if (lines.length === 0) {
    return;
  }
  const existingByContent = new Map<string, MemoryOsLongTermNote>();
  for (const note of Object.values(params.store.longTermNotes)) {
    existingByContent.set(note.content.trim().toLowerCase(), note);
  }
  for (const line of lines) {
    const normalized = line.toLowerCase();
    const updatedAt = params.updatedAt;
    const existing = existingByContent.get(normalized);
    if (existing) {
      if (typeof updatedAt === "number") {
        existing.updatedAt = updatedAt;
      }
      if (params.sessionKey && !existing.sourceSessionKeys.includes(params.sessionKey)) {
        existing.sourceSessionKeys.push(params.sessionKey);
      }
      continue;
    }
    const createdAt = updatedAt ?? Date.now();
    const note: MemoryOsLongTermNote = {
      id: crypto.randomUUID(),
      content: line,
      keywords: normalizeKeywords(line),
      createdAt,
      updatedAt: createdAt,
      accessCount: 0,
      sourceSessionKeys: params.sessionKey ? [params.sessionKey] : [],
    };
    params.store.longTermNotes[note.id] = note;
    existingByContent.set(normalized, note);
  }
  const sortedIds = Object.values(params.store.longTermNotes)
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .map((note) => note.id);
  for (const staleId of sortedIds.slice(MAX_LONG_TERM_NOTES)) {
    delete params.store.longTermNotes[staleId];
  }
}

function upsertSessionSegment(params: {
  store: MemoryOsStore;
  sessionKey: string;
  summary: string;
  taskState?: string;
  source: "manual" | "auto";
  updatedAt: number;
}): MemoryOsSegment | undefined {
  const normalizedSummary = normalizeSummaryText(params.summary, MAX_SEGMENT_SUMMARY_CHARS);
  const normalizedTaskState = normalizeSummaryText(params.taskState, MAX_TASK_STATE_CHARS);
  if (!normalizedSummary && !normalizedTaskState) {
    return undefined;
  }
  const effectiveSummary = normalizedSummary ?? normalizedTaskState ?? "Session segment";
  const keywords = normalizeKeywords(`${normalizedSummary}\n${normalizedTaskState ?? ""}`);
  const sessionState = params.store.sessions[params.sessionKey] ?? {
    shortTermPages: [],
    segmentIds: [],
  };
  const candidateSegments = sessionState.segmentIds
    .map((id) => params.store.segments[id])
    .filter((segment): segment is MemoryOsSegment => Boolean(segment));

  let bestMatch: MemoryOsSegment | undefined;
  let bestScore = 0;
  for (const segment of candidateSegments) {
    const score = scoreKeywords(keywords, segment.keywords);
    if (score > bestScore) {
      bestScore = score;
      bestMatch = segment;
    }
  }

  if (bestMatch && bestScore >= 0.45) {
    bestMatch.summary = effectiveSummary;
    bestMatch.taskState = normalizedTaskState;
    bestMatch.keywords = keywords;
    bestMatch.updatedAt = params.updatedAt;
    bestMatch.heat += 1;
    bestMatch.source = params.source;
    return bestMatch;
  }

  const segment: MemoryOsSegment = {
    id: crypto.randomUUID(),
    sessionKey: params.sessionKey,
    title: deriveTitle(normalizedTaskState ?? effectiveSummary, "Session segment"),
    summary: effectiveSummary,
    ...(normalizedTaskState ? { taskState: normalizedTaskState } : {}),
    keywords,
    createdAt: params.updatedAt,
    updatedAt: params.updatedAt,
    accessCount: 0,
    heat: 1,
    source: params.source,
  };
  params.store.segments[segment.id] = segment;
  sessionState.segmentIds = [...sessionState.segmentIds, segment.id]
    .map((id) => params.store.segments[id])
    .filter((item): item is MemoryOsSegment => Boolean(item))
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, MAX_SEGMENTS_PER_SESSION)
    .map((item) => item.id);
  params.store.sessions[params.sessionKey] = sessionState;
  for (const [segmentId, memory] of Object.entries(params.store.segments)) {
    if (memory.sessionKey !== params.sessionKey) {
      continue;
    }
    if (!sessionState.segmentIds.includes(segmentId)) {
      delete params.store.segments[segmentId];
    }
  }
  return segment;
}

export function updateMemoryOsFromConversation(params: {
  sessionKey: string;
  messages: unknown[];
  summary: string;
  taskState?: string;
  source: "manual" | "auto";
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
}): MemoryOsStore {
  const updatedAt = params.updatedAt ?? Date.now();
  const store = syncMemoryOsShortTermPages({
    sessionKey: params.sessionKey,
    messages: params.messages,
    updatedAt,
    env: params.env,
  });
  const sessionState = store.sessions[params.sessionKey] ?? {
    shortTermPages: [],
    segmentIds: [],
  };
  sessionState.lastCompactedAt = updatedAt;
  store.sessions[params.sessionKey] = sessionState;
  upsertSessionSegment({
    store,
    sessionKey: params.sessionKey,
    summary: params.summary,
    taskState: params.taskState,
    source: params.source,
    updatedAt,
  });
  normalizeLongTermNotesIntoStore({
    store,
    content: loadGlobalImportantNotes({ env: params.env }).content,
    updatedAt,
  });
  return saveMemoryOsStore(store, params.env);
}

export function updateMemoryOsFromFeedback(params: {
  sessionKey: string;
  summary: string;
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
}): MemoryOsStore {
  const store = loadMemoryOsStore(params.env);
  normalizeLongTermNotesIntoStore({
    store,
    content: params.summary,
    sessionKey: params.sessionKey,
    updatedAt: params.updatedAt ?? Date.now(),
  });
  return saveMemoryOsStore(store, params.env);
}

export function syncMemoryOsFromTranscript(params: {
  sessionKey: string;
  messages: unknown[];
  env?: NodeJS.ProcessEnv;
}): MemoryOsStore {
  return syncMemoryOsShortTermPages({
    sessionKey: params.sessionKey,
    messages: params.messages,
    env: params.env,
  });
}

function selectTopLongTermNotes(params: {
  store: MemoryOsStore;
  queryKeywords: string[];
}): MemoryOsLongTermNote[] {
  return Object.values(params.store.longTermNotes)
    .map((note) => ({
      note,
      score: scoreKeywords(params.queryKeywords, note.keywords) + note.accessCount * 0.02,
    }))
    .sort((left, right) => right.score - left.score || right.note.updatedAt - left.note.updatedAt)
    .slice(0, MAX_RETRIEVED_NOTES)
    .map((item) => item.note);
}

function selectTopSegments(params: {
  store: MemoryOsStore;
  sessionKey: string;
  queryKeywords: string[];
}): MemoryOsSegment[] {
  return Object.values(params.store.segments)
    .map((segment) => ({
      segment,
      score:
        scoreKeywords(params.queryKeywords, segment.keywords) +
        (segment.sessionKey === params.sessionKey ? 0.4 : 0) +
        segment.heat * 0.03,
    }))
    .sort(
      (left, right) =>
        right.score - left.score || right.segment.updatedAt - left.segment.updatedAt,
    )
    .slice(0, MAX_RETRIEVED_SEGMENTS)
    .map((item) => item.segment);
}

export function buildMemoryOsPromptAddition(params: {
  sessionKey: string;
  queryText: string;
  promptContext?: SessionPromptContext;
  env?: NodeJS.ProcessEnv;
}): string | undefined {
  const store = loadMemoryOsStore(params.env);
  const globalImportantNotes = loadGlobalImportantNotes({ env: params.env }).content;
  normalizeLongTermNotesIntoStore({
    store,
    content: globalImportantNotes,
  });
  const sessionState = store.sessions[params.sessionKey];
  const queryKeywords = normalizeKeywords(params.queryText);
  const shortTermPages = (sessionState?.shortTermPages ?? []).slice(-MAX_RETRIEVED_PAGES);
  const currentSummary = trimOptionalText(params.promptContext?.contextSummary);
  const currentTaskState = trimOptionalText(params.promptContext?.taskState);
  const segments = selectTopSegments({
    store,
    sessionKey: params.sessionKey,
    queryKeywords,
  }).filter((segment) => {
    const sameSummary = trimOptionalText(segment.summary) === currentSummary;
    const sameTaskState = trimOptionalText(segment.taskState) === currentTaskState;
    return !(sameSummary && sameTaskState);
  });
  const globalNoteSet = new Set(splitLines(globalImportantNotes).map((line) => line.toLowerCase()));
  const longTermNotes = selectTopLongTermNotes({
    store,
    queryKeywords,
  }).filter((note) => !globalNoteSet.has(note.content.trim().toLowerCase()));

  for (const segment of segments) {
    segment.accessCount += 1;
  }
  for (const note of longTermNotes) {
    note.accessCount += 1;
  }
  saveMemoryOsStore(store, params.env);

  const sections: string[] = [];
  if (shortTermPages.length > 0) {
    sections.push(
      [
        "### Short-Term Memory",
        ...shortTermPages.map((page) => `- ${page.title}: ${page.content}`),
      ].join("\n"),
    );
  }
  if (segments.length > 0) {
    sections.push(
      [
        "### Mid-Term Episodic Memory",
        ...segments.map((segment) =>
          [
            `- ${segment.title}`,
            segment.summary,
            segment.taskState ? `Task state: ${segment.taskState}` : undefined,
          ]
            .filter(Boolean)
            .join("\n"),
        ),
      ].join("\n\n"),
    );
  }
  if (longTermNotes.length > 0) {
    sections.push(
      [
        "### Long-Term Memory",
        ...longTermNotes.map((note) => `- ${note.content}`),
      ].join("\n"),
    );
  }

  if (sections.length === 0) {
    return undefined;
  }
  return [
    "## Memory OS Retrieved Context",
    "Use these retrieved memories as supporting context. Prefer newer, task-relevant items when conflicts appear. Treat them as execution memory rather than user-visible reply content.",
    ...sections,
  ].join("\n\n");
}

export function getMemoryOsSessionSnapshot(params: {
  sessionKey: string;
  env?: NodeJS.ProcessEnv;
}): MemoryOsSessionSnapshot {
  const store = loadMemoryOsStore(params.env);
  const sessionState = store.sessions[params.sessionKey];
  const segments = (sessionState?.segmentIds ?? [])
    .map((id) => store.segments[id])
    .filter((segment): segment is MemoryOsSegment => Boolean(segment))
    .sort((left, right) => right.updatedAt - left.updatedAt);
  return {
    session_id: params.sessionKey,
    short_term_page_count: sessionState?.shortTermPages.length ?? 0,
    mid_term_segment_count: segments.length,
    long_term_note_count: Object.keys(store.longTermNotes).length,
    latest_segment_title: segments[0]?.title ?? null,
    latest_segment_summary: segments[0]?.summary ?? null,
    latest_updated_at: segments[0]?.updatedAt ?? sessionState?.lastUpdatedAt ?? null,
  };
}
