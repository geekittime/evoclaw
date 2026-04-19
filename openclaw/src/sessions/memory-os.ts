import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { SessionPromptContext } from "../config/sessions/types.js";
import { resolveStateDir } from "../config/paths.js";
import { loadGlobalImportantNotes } from "./global-important-notes.js";

const MEMORY_OS_FILENAME = "memory-os.json";
const MEMORY_OS_VERSION = 2;
const MAX_SHORT_TERM_PAGES = 12;
const MAX_TOTAL_PAGES = 256;
const MAX_SEGMENTS_PER_SESSION = 24;
const MAX_LONG_TERM_NOTES = 256;
const MAX_SEGMENT_SUMMARY_CHARS = 4_000;
const MAX_TASK_STATE_CHARS = 2_000;
const MAX_PAGE_CONTENT_CHARS = 1_600;
const MAX_LONG_TERM_NOTE_CHARS = 600;
const DEFAULT_MEMORY_OS_CHAR_BUDGET = 3_600;
const DEFAULT_SHORT_TERM_CHAR_BUDGET = 900;
const DEFAULT_MID_TERM_CHAR_BUDGET = 1_700;
const DEFAULT_LONG_TERM_CHAR_BUDGET = 1_000;
const KEYWORD_LIMIT = 18;
const DERIVED_SEGMENT_WINDOW = 3;
const MIN_PAGES_FOR_SEGMENT_PROMOTION = 2;
const SEGMENT_PROMOTION_THRESHOLD = 3.5;
const NOTE_PROMOTION_COOLDOWN_MS = 5 * 60 * 1_000;
const HEAT_DECAY_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1_000;
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
  sessionKey: string;
  title: string;
  content: string;
  keywords: string[];
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  heat: number;
  lastRetrievedAt?: number;
};

export type MemoryOsSegment = {
  id: string;
  sessionKey: string;
  title: string;
  summary: string;
  taskState?: string;
  keywords: string[];
  pageIds: string[];
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  heat: number;
  source: "manual" | "auto" | "derived";
  promotionCount: number;
  promotedToLongTermAt?: number;
  lastRetrievedAt?: number;
};

export type MemoryOsLongTermNote = {
  id: string;
  content: string;
  keywords: string[];
  createdAt: number;
  updatedAt: number;
  accessCount: number;
  heat: number;
  promotionCount: number;
  sourceSessionKeys: string[];
  sourceSegmentIds: string[];
  lastRetrievedAt?: number;
};

export type MemoryOsSessionState = {
  stmPageIds: string[];
  segmentIds: string[];
  lastUpdatedAt?: number;
  lastCompactedAt?: number;
};

export type MemoryOsStore = {
  version: number;
  pages: Record<string, MemoryOsShortTermPage>;
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

type LegacyShortTermPage = {
  id?: string;
  title?: string;
  content?: string;
  keywords?: unknown;
  updatedAt?: number;
};

type LegacySessionState = {
  shortTermPages?: LegacyShortTermPage[];
  stmPageIds?: string[];
  segmentIds?: string[];
  lastUpdatedAt?: number;
  lastCompactedAt?: number;
};

type NormalizedMessage = {
  role: string;
  text: string;
  timestamp: number;
};

type OrderedNormalizedMessage = NormalizedMessage & {
  order: number;
};

type RetrievalCandidate<T> = {
  item: T;
  text: string;
  score: number;
  chars: number;
};

function estimateCharBudgetFromTokenBudget(tokenBudget?: number | null): number {
  if (typeof tokenBudget !== "number" || !Number.isFinite(tokenBudget) || tokenBudget <= 0) {
    return DEFAULT_MEMORY_OS_CHAR_BUDGET;
  }
  const approxChars = Math.floor(tokenBudget * 4);
  return Math.max(1_800, Math.min(7_200, Math.floor(approxChars * 0.12)));
}

function trimOptionalText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function clampText(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  if (maxChars <= 160) {
    return `${value.slice(0, Math.max(0, maxChars - 3))}...`;
  }
  const head = Math.floor((maxChars - 7) * 0.6);
  const tail = Math.max(32, maxChars - head - 7);
  return `${value.slice(0, head)}\n...\n${value.slice(value.length - tail)}`;
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
  return clampText(trimmed, maxChars);
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

function stableHash(parts: string[]): string {
  return crypto.createHash("sha1").update(parts.join("\u241f")).digest("hex").slice(0, 20);
}

function buildPageId(sessionKey: string, title: string, content: string): string {
  return `${sessionKey}:page:${stableHash([sessionKey, title, content])}`;
}

function buildSegmentId(sessionKey: string, pageIds: string[]): string {
  return `${sessionKey}:segment:${stableHash([sessionKey, ...pageIds])}`;
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
      const resultText =
        typeof record.output === "string"
          ? record.output.trim()
          : typeof record.text === "string"
            ? record.text.trim()
            : "";
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
  return { role, text: clampText(text, MAX_PAGE_CONTENT_CHARS), timestamp };
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
    const finalContent = clampText(content, MAX_PAGE_CONTENT_CHARS);
    pages.push({
      id: buildPageId(sessionKey, title, finalContent),
      sessionKey,
      title,
      content: finalContent,
      keywords: normalizeKeywords(`${title}\n${finalContent}`),
      createdAt: updatedAt,
      updatedAt,
      accessCount: 0,
      heat: 1,
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
    pages: {},
    sessions: {},
    segments: {},
    longTermNotes: {},
  };
}

function coerceStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function coercePage(sessionKey: string, value: unknown): MemoryOsShortTermPage | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const title = trimOptionalText(typeof record.title === "string" ? record.title : undefined);
  const content = trimOptionalText(typeof record.content === "string" ? record.content : undefined);
  if (!title || !content) {
    return null;
  }
  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? Math.floor(record.updatedAt)
      : Date.now();
  const createdAt =
    typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
      ? Math.floor(record.createdAt)
      : updatedAt;
  return {
    id:
      typeof record.id === "string" && record.id.trim().length > 0
        ? record.id
        : buildPageId(sessionKey, title, content),
    sessionKey,
    title,
    content: clampText(content, MAX_PAGE_CONTENT_CHARS),
    keywords:
      Array.isArray(record.keywords) && record.keywords.every((item) => typeof item === "string")
        ? (record.keywords as string[])
        : normalizeKeywords(`${title}\n${content}`),
    createdAt,
    updatedAt,
    accessCount:
      typeof record.accessCount === "number" && Number.isFinite(record.accessCount)
        ? record.accessCount
        : 0,
    heat: typeof record.heat === "number" && Number.isFinite(record.heat) ? record.heat : 1,
    ...(typeof record.lastRetrievedAt === "number" && Number.isFinite(record.lastRetrievedAt)
      ? { lastRetrievedAt: Math.floor(record.lastRetrievedAt) }
      : {}),
  };
}

function coerceSegment(value: unknown): MemoryOsSegment | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const id = trimOptionalText(typeof record.id === "string" ? record.id : undefined);
  const sessionKey = trimOptionalText(
    typeof record.sessionKey === "string" ? record.sessionKey : undefined,
  );
  const title = trimOptionalText(typeof record.title === "string" ? record.title : undefined);
  const summary = trimOptionalText(typeof record.summary === "string" ? record.summary : undefined);
  if (!id || !sessionKey || !title || !summary) {
    return null;
  }
  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? Math.floor(record.updatedAt)
      : Date.now();
  const createdAt =
    typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
      ? Math.floor(record.createdAt)
      : updatedAt;
  return {
    id,
    sessionKey,
    title,
    summary: clampText(summary, MAX_SEGMENT_SUMMARY_CHARS),
    ...(trimOptionalText(typeof record.taskState === "string" ? record.taskState : undefined)
      ? {
          taskState: clampText(
            trimOptionalText(typeof record.taskState === "string" ? record.taskState : undefined)!,
            MAX_TASK_STATE_CHARS,
          ),
        }
      : {}),
    keywords:
      Array.isArray(record.keywords) && record.keywords.every((item) => typeof item === "string")
        ? (record.keywords as string[])
        : normalizeKeywords(
            `${title}\n${summary}\n${typeof record.taskState === "string" ? record.taskState : ""}`,
          ),
    pageIds: coerceStringArray(record.pageIds),
    createdAt,
    updatedAt,
    accessCount:
      typeof record.accessCount === "number" && Number.isFinite(record.accessCount)
        ? record.accessCount
        : 0,
    heat: typeof record.heat === "number" && Number.isFinite(record.heat) ? record.heat : 1,
    source:
      record.source === "manual" || record.source === "auto" || record.source === "derived"
        ? record.source
        : "derived",
    promotionCount:
      typeof record.promotionCount === "number" && Number.isFinite(record.promotionCount)
        ? record.promotionCount
        : 0,
    ...(typeof record.promotedToLongTermAt === "number" &&
    Number.isFinite(record.promotedToLongTermAt)
      ? { promotedToLongTermAt: Math.floor(record.promotedToLongTermAt) }
      : {}),
    ...(typeof record.lastRetrievedAt === "number" && Number.isFinite(record.lastRetrievedAt)
      ? { lastRetrievedAt: Math.floor(record.lastRetrievedAt) }
      : {}),
  };
}

function coerceLongTermNote(value: unknown): MemoryOsLongTermNote | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const id = trimOptionalText(typeof record.id === "string" ? record.id : undefined);
  const content = trimOptionalText(typeof record.content === "string" ? record.content : undefined);
  if (!id || !content) {
    return null;
  }
  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? Math.floor(record.updatedAt)
      : Date.now();
  const createdAt =
    typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
      ? Math.floor(record.createdAt)
      : updatedAt;
  return {
    id,
    content: clampText(content, MAX_LONG_TERM_NOTE_CHARS),
    keywords:
      Array.isArray(record.keywords) && record.keywords.every((item) => typeof item === "string")
        ? (record.keywords as string[])
        : normalizeKeywords(content),
    createdAt,
    updatedAt,
    accessCount:
      typeof record.accessCount === "number" && Number.isFinite(record.accessCount)
        ? record.accessCount
        : 0,
    heat: typeof record.heat === "number" && Number.isFinite(record.heat) ? record.heat : 1,
    promotionCount:
      typeof record.promotionCount === "number" && Number.isFinite(record.promotionCount)
        ? record.promotionCount
        : 0,
    sourceSessionKeys: coerceStringArray(record.sourceSessionKeys),
    sourceSegmentIds: coerceStringArray(record.sourceSegmentIds),
    ...(typeof record.lastRetrievedAt === "number" && Number.isFinite(record.lastRetrievedAt)
      ? { lastRetrievedAt: Math.floor(record.lastRetrievedAt) }
      : {}),
  };
}

function ensureSessionState(store: MemoryOsStore, sessionKey: string): MemoryOsSessionState {
  store.sessions[sessionKey] ??= {
    stmPageIds: [],
    segmentIds: [],
  };
  return store.sessions[sessionKey];
}

function normalizeStore(value: unknown): MemoryOsStore {
  if (!value || typeof value !== "object") {
    return createEmptyStore();
  }
  const raw = value as Record<string, unknown>;
  const store = createEmptyStore();

  const existingPages =
    raw.pages && typeof raw.pages === "object"
      ? (raw.pages as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  for (const page of Object.values(existingPages)) {
    const normalized = coercePage(
      trimOptionalText(
        typeof (page as Record<string, unknown>).sessionKey === "string"
          ? ((page as Record<string, unknown>).sessionKey as string)
          : undefined,
      ) ?? "unknown",
      page,
    );
    if (normalized) {
      store.pages[normalized.id] = normalized;
    }
  }

  const rawSessions =
    raw.sessions && typeof raw.sessions === "object"
      ? (raw.sessions as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  for (const [sessionKey, sessionValue] of Object.entries(rawSessions)) {
    const legacy = (sessionValue as LegacySessionState | undefined) ?? {};
    const sessionState: MemoryOsSessionState = {
      stmPageIds: [],
      segmentIds: coerceStringArray(legacy.segmentIds),
      ...(typeof legacy.lastUpdatedAt === "number" && Number.isFinite(legacy.lastUpdatedAt)
        ? { lastUpdatedAt: Math.floor(legacy.lastUpdatedAt) }
        : {}),
      ...(typeof legacy.lastCompactedAt === "number" && Number.isFinite(legacy.lastCompactedAt)
        ? { lastCompactedAt: Math.floor(legacy.lastCompactedAt) }
        : {}),
    };
    if (Array.isArray(legacy.stmPageIds) && legacy.stmPageIds.length > 0) {
      sessionState.stmPageIds = coerceStringArray(legacy.stmPageIds);
    } else if (Array.isArray(legacy.shortTermPages)) {
      for (const legacyPage of legacy.shortTermPages) {
        const normalizedPage = coercePage(sessionKey, legacyPage);
        if (!normalizedPage) {
          continue;
        }
        store.pages[normalizedPage.id] = normalizedPage;
        sessionState.stmPageIds.push(normalizedPage.id);
      }
    }
    store.sessions[sessionKey] = sessionState;
  }

  const rawSegments =
    raw.segments && typeof raw.segments === "object"
      ? (raw.segments as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  for (const [segmentId, segmentValue] of Object.entries(rawSegments)) {
    const normalized = coerceSegment({
      ...(segmentValue as Record<string, unknown>),
      id: segmentId,
    });
    if (normalized) {
      store.segments[normalized.id] = normalized;
      ensureSessionState(store, normalized.sessionKey);
      if (!store.sessions[normalized.sessionKey].segmentIds.includes(normalized.id)) {
        store.sessions[normalized.sessionKey].segmentIds.push(normalized.id);
      }
    }
  }

  const rawNotes =
    raw.longTermNotes && typeof raw.longTermNotes === "object"
      ? (raw.longTermNotes as Record<string, unknown>)
      : ({} as Record<string, unknown>);
  for (const [noteId, noteValue] of Object.entries(rawNotes)) {
    const normalized = coerceLongTermNote({
      ...(noteValue as Record<string, unknown>),
      id: noteId,
    });
    if (normalized) {
      store.longTermNotes[normalized.id] = normalized;
    }
  }

  for (const [sessionKey, sessionState] of Object.entries(store.sessions)) {
    sessionState.stmPageIds = sessionState.stmPageIds.filter((id) => Boolean(store.pages[id]));
    sessionState.segmentIds = sessionState.segmentIds.filter((id) => {
      const segment = store.segments[id];
      return Boolean(segment && segment.sessionKey === sessionKey);
    });
  }

  return store;
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
  const normalized = normalizeStore(store);
  normalized.version = MEMORY_OS_VERSION;
  const filePath = resolveMemoryOsPath(env);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(normalized, null, 2), "utf8");
  return normalized;
}

function upsertPagesIntoStore(store: MemoryOsStore, pages: MemoryOsShortTermPage[], previousIds: string[]) {
  const previousSet = new Set(previousIds);
  for (const page of pages) {
    const existing = store.pages[page.id];
    if (existing) {
      store.pages[page.id] = {
        ...existing,
        title: page.title,
        content: page.content,
        keywords: page.keywords,
        updatedAt: Math.max(existing.updatedAt, page.updatedAt),
        ...(previousSet.has(page.id) ? {} : { heat: existing.heat + 1 }),
      };
      continue;
    }
    store.pages[page.id] = page;
  }
}

function buildDerivedSegmentSummaryFromPages(pages: MemoryOsShortTermPage[]): string | undefined {
  if (pages.length < MIN_PAGES_FOR_SEGMENT_PROMOTION) {
    return undefined;
  }
  const titles = pages.map((page) => page.title).filter(Boolean);
  const keyLines = pages
    .flatMap((page) => splitLines(page.content))
    .filter((line, index, all) => all.indexOf(line) === index)
    .slice(0, 4);
  const parts: string[] = [];
  if (titles.length > 0) {
    parts.push(`Recent interaction chain: ${titles.join(" -> ")}.`);
  }
  if (keyLines.length > 0) {
    parts.push(`Key interaction points: ${keyLines.join(" / ")}.`);
  }
  const combined = parts.join(" ");
  return trimOptionalText(clampText(combined, 900));
}

function findMatchingSegment(
  store: MemoryOsStore,
  sessionKey: string,
  pageIds: string[],
  keywords: string[],
): MemoryOsSegment | undefined {
  const pageSignature = pageIds.join("|");
  let bestMatch: MemoryOsSegment | undefined;
  let bestScore = 0;
  for (const segment of Object.values(store.segments)) {
    if (segment.sessionKey !== sessionKey) {
      continue;
    }
    if (segment.pageIds.join("|") === pageSignature) {
      return segment;
    }
    const score = scoreKeywords(keywords, segment.keywords);
    if (score > bestScore) {
      bestScore = score;
      bestMatch = segment;
    }
  }
  return bestScore >= 0.72 ? bestMatch : undefined;
}

function segmentRetentionScore(segment: MemoryOsSegment): number {
  const decayedHeat = decayHeat(segment.heat, segment.lastRetrievedAt ?? segment.updatedAt, Date.now());
  return (
    segment.promotionCount * 4 +
    decayedHeat * 1.5 +
    segment.accessCount * 1.2 +
    (segment.source === "manual" ? 1.5 : segment.source === "auto" ? 1 : 0)
  );
}

function noteRetentionScore(note: MemoryOsLongTermNote): number {
  const decayedHeat = decayHeat(note.heat, note.lastRetrievedAt ?? note.updatedAt, Date.now());
  return note.promotionCount * 4 + decayedHeat * 1.5 + note.accessCount * 1.2;
}

function pageRetentionScore(page: MemoryOsShortTermPage): number {
  const decayedHeat = decayHeat(page.heat, page.lastRetrievedAt ?? page.updatedAt, Date.now());
  return decayedHeat * 1.2 + page.accessCount;
}

function decayHeat(heat: number, lastTouchedAt: number | undefined, now: number): number {
  const safeHeat = Number.isFinite(heat) ? Math.max(0.5, heat) : 1;
  if (!lastTouchedAt || !Number.isFinite(lastTouchedAt) || lastTouchedAt <= 0 || now <= lastTouchedAt) {
    return safeHeat;
  }
  const elapsed = now - lastTouchedAt;
  const factor = Math.pow(0.5, elapsed / HEAT_DECAY_HALF_LIFE_MS);
  return Math.max(0.5, safeHeat * factor);
}

function pruneColdSegments(store: MemoryOsStore, sessionKey: string) {
  const sessionState = ensureSessionState(store, sessionKey);
  const segments = sessionState.segmentIds
    .map((id) => store.segments[id])
    .filter((segment): segment is MemoryOsSegment => Boolean(segment))
    .sort(
      (left, right) =>
        segmentRetentionScore(right) - segmentRetentionScore(left) ||
        right.updatedAt - left.updatedAt,
    );
  const keep = new Set(segments.slice(0, MAX_SEGMENTS_PER_SESSION).map((segment) => segment.id));
  sessionState.segmentIds = segments.slice(0, MAX_SEGMENTS_PER_SESSION).map((segment) => segment.id);
  for (const segment of segments) {
    if (!keep.has(segment.id)) {
      delete store.segments[segment.id];
    }
  }
}

function pruneColdPages(store: MemoryOsStore) {
  const protectedPageIds = new Set<string>();
  for (const sessionState of Object.values(store.sessions)) {
    for (const pageId of sessionState.stmPageIds.slice(-MAX_SHORT_TERM_PAGES)) {
      protectedPageIds.add(pageId);
    }
    sessionState.stmPageIds = sessionState.stmPageIds
      .filter((id) => Boolean(store.pages[id]))
      .slice(-MAX_SHORT_TERM_PAGES);
  }
  for (const segment of Object.values(store.segments)) {
    for (const pageId of segment.pageIds) {
      protectedPageIds.add(pageId);
    }
  }
  const allPageIds = Object.keys(store.pages);
  if (allPageIds.length <= MAX_TOTAL_PAGES) {
    return;
  }
  const removable = allPageIds
    .map((id) => store.pages[id]!)
    .filter((page) => !protectedPageIds.has(page.id))
    .sort(
      (left, right) =>
        pageRetentionScore(left) - pageRetentionScore(right) || left.updatedAt - right.updatedAt,
    );
  let remaining = allPageIds.length;
  for (const page of removable) {
    if (remaining <= MAX_TOTAL_PAGES) {
      break;
    }
    delete store.pages[page.id];
    remaining -= 1;
  }
}

function pruneColdLongTermNotes(store: MemoryOsStore) {
  const notes = Object.values(store.longTermNotes).sort(
    (left, right) =>
      noteRetentionScore(right) - noteRetentionScore(left) || right.updatedAt - left.updatedAt,
  );
  for (const stale of notes.slice(MAX_LONG_TERM_NOTES)) {
    delete store.longTermNotes[stale.id];
  }
}

function evictColdMemory(store: MemoryOsStore) {
  for (const sessionKey of Object.keys(store.sessions)) {
    pruneColdSegments(store, sessionKey);
  }
  pruneColdPages(store);
  pruneColdLongTermNotes(store);
}

function upsertSegment(params: {
  store: MemoryOsStore;
  sessionKey: string;
  pageIds: string[];
  summary?: string;
  taskState?: string;
  source: "manual" | "auto" | "derived";
  updatedAt: number;
}): MemoryOsSegment | undefined {
  const normalizedSummary = normalizeSummaryText(params.summary, MAX_SEGMENT_SUMMARY_CHARS);
  const normalizedTaskState = normalizeSummaryText(params.taskState, MAX_TASK_STATE_CHARS);
  if (!normalizedSummary && !normalizedTaskState && params.pageIds.length < MIN_PAGES_FOR_SEGMENT_PROMOTION) {
    return undefined;
  }
  const pages = params.pageIds.map((id) => params.store.pages[id]).filter(Boolean);
  const derivedSummary = normalizedSummary ?? buildDerivedSegmentSummaryFromPages(pages);
  if (!derivedSummary && !normalizedTaskState) {
    return undefined;
  }
  const effectiveSummary = derivedSummary ?? normalizedTaskState ?? "Session segment";
  const keywords = normalizeKeywords(
    `${effectiveSummary}\n${normalizedTaskState ?? ""}\n${pages.map((page) => page.content).join("\n")}`,
  );
  const match = findMatchingSegment(params.store, params.sessionKey, params.pageIds, keywords);
  const sessionState = ensureSessionState(params.store, params.sessionKey);
  if (match) {
    match.title = deriveTitle(normalizedTaskState ?? effectiveSummary, "Session segment");
    match.summary = effectiveSummary;
    if (normalizedTaskState) {
      match.taskState = normalizedTaskState;
    }
    match.keywords = keywords;
    match.pageIds = [...new Set([...match.pageIds, ...params.pageIds])].slice(-MAX_SHORT_TERM_PAGES);
    match.updatedAt = params.updatedAt;
    match.heat += 1;
    match.source = params.source === "derived" ? match.source : params.source;
    if (!sessionState.segmentIds.includes(match.id)) {
      sessionState.segmentIds.push(match.id);
    }
    return match;
  }

  const segment: MemoryOsSegment = {
    id: buildSegmentId(params.sessionKey, params.pageIds),
    sessionKey: params.sessionKey,
    title: deriveTitle(normalizedTaskState ?? effectiveSummary, "Session segment"),
    summary: effectiveSummary,
    ...(normalizedTaskState ? { taskState: normalizedTaskState } : {}),
    keywords,
    pageIds: params.pageIds.slice(-MAX_SHORT_TERM_PAGES),
    createdAt: params.updatedAt,
    updatedAt: params.updatedAt,
    accessCount: 0,
    heat: 1,
    source: params.source,
    promotionCount: 0,
  };
  params.store.segments[segment.id] = segment;
  sessionState.segmentIds.push(segment.id);
  pruneColdSegments(params.store, params.sessionKey);
  return params.store.segments[segment.id];
}

function promoteRecentPagesToDerivedSegment(store: MemoryOsStore, sessionKey: string, updatedAt: number) {
  const sessionState = ensureSessionState(store, sessionKey);
  const pageIds = sessionState.stmPageIds.slice(-DERIVED_SEGMENT_WINDOW);
  if (pageIds.length < MIN_PAGES_FOR_SEGMENT_PROMOTION) {
    return;
  }
  upsertSegment({
    store,
    sessionKey,
    pageIds,
    source: "derived",
    updatedAt,
  });
}

function normalizeLongTermNotesIntoStore(params: {
  store: MemoryOsStore;
  content?: string;
  sessionKey?: string;
  sourceSegmentId?: string;
  updatedAt?: number;
}): void {
  const lines = splitLines(params.content).map((line) => clampText(line, MAX_LONG_TERM_NOTE_CHARS));
  if (lines.length === 0) {
    return;
  }
  const existingByContent = new Map<string, MemoryOsLongTermNote>();
  for (const note of Object.values(params.store.longTermNotes)) {
    existingByContent.set(note.content.trim().toLowerCase(), note);
  }
  for (const line of lines) {
    const normalized = line.toLowerCase();
    const existing = existingByContent.get(normalized);
    if (existing) {
      if (typeof params.updatedAt === "number") {
        existing.updatedAt = params.updatedAt;
      }
      if (params.sessionKey && !existing.sourceSessionKeys.includes(params.sessionKey)) {
        existing.sourceSessionKeys.push(params.sessionKey);
      }
      if (params.sourceSegmentId && !existing.sourceSegmentIds.includes(params.sourceSegmentId)) {
        existing.sourceSegmentIds.push(params.sourceSegmentId);
      }
      continue;
    }
    const createdAt = params.updatedAt ?? Date.now();
    const note: MemoryOsLongTermNote = {
      id: crypto.randomUUID(),
      content: line,
      keywords: normalizeKeywords(line),
      createdAt,
      updatedAt: createdAt,
      accessCount: 0,
      heat: 1,
      promotionCount: params.sourceSegmentId ? 1 : 0,
      sourceSessionKeys: params.sessionKey ? [params.sessionKey] : [],
      sourceSegmentIds: params.sourceSegmentId ? [params.sourceSegmentId] : [],
    };
    params.store.longTermNotes[note.id] = note;
    existingByContent.set(normalized, note);
  }
  pruneColdLongTermNotes(params.store);
}

function shouldPromoteSegmentToLongTerm(segment: MemoryOsSegment, updatedAt: number): boolean {
  const score =
    segment.accessCount * 1.2 +
    segment.heat +
    (segment.source === "manual" ? 0.75 : segment.source === "auto" ? 0.5 : 0);
  if (score < SEGMENT_PROMOTION_THRESHOLD) {
    return false;
  }
  if (!segment.promotedToLongTermAt) {
    return true;
  }
  return updatedAt - segment.promotedToLongTermAt >= NOTE_PROMOTION_COOLDOWN_MS;
}

function deriveLongTermNoteFromSegment(segment: MemoryOsSegment): string {
  const parts = [segment.summary];
  if (segment.taskState && !segment.summary.includes(segment.taskState)) {
    parts.push(`Relevant task state: ${segment.taskState}`);
  }
  return clampText(parts.join("\n"), MAX_LONG_TERM_NOTE_CHARS);
}

function promoteSegmentsToLongTermNotes(store: MemoryOsStore, updatedAt: number, sessionKey?: string) {
  const candidates = Object.values(store.segments)
    .filter((segment) => (!sessionKey ? true : segment.sessionKey === sessionKey))
    .sort(
      (left, right) =>
        segmentRetentionScore(right) - segmentRetentionScore(left) || right.updatedAt - left.updatedAt,
    );
  for (const segment of candidates) {
    if (!shouldPromoteSegmentToLongTerm(segment, updatedAt)) {
      continue;
    }
    normalizeLongTermNotesIntoStore({
      store,
      content: deriveLongTermNoteFromSegment(segment),
      sessionKey: segment.sessionKey,
      sourceSegmentId: segment.id,
      updatedAt,
    });
    segment.promotedToLongTermAt = updatedAt;
    segment.promotionCount += 1;
  }
}

function sessionRecencyScore<T extends { updatedAt: number; lastRetrievedAt?: number }>(item: T): number {
  const recent = item.lastRetrievedAt ?? item.updatedAt;
  return recent / 1_000_000_000_000;
}

function selectCandidatesWithinBudget<T>(
  candidates: RetrievalCandidate<T>[],
  sectionBudget: number,
): RetrievalCandidate<T>[] {
  const selected: RetrievalCandidate<T>[] = [];
  let used = 0;
  for (const candidate of candidates) {
    if (candidate.chars > sectionBudget && selected.length > 0) {
      continue;
    }
    if (used + candidate.chars > sectionBudget && selected.length > 0) {
      continue;
    }
    selected.push(candidate);
    used += candidate.chars;
  }
  return selected;
}

function fillRemainingBudget<T>(
  existing: RetrievalCandidate<T>[],
  candidates: RetrievalCandidate<T>[],
  totalBudget: number,
): RetrievalCandidate<T>[] {
  const selectedIds = new Set(existing.map((candidate) => candidate.text));
  let used = existing.reduce((sum, candidate) => sum + candidate.chars, 0);
  const result = [...existing];
  for (const candidate of candidates) {
    if (selectedIds.has(candidate.text)) {
      continue;
    }
    if (used + candidate.chars > totalBudget) {
      continue;
    }
    result.push(candidate);
    used += candidate.chars;
    selectedIds.add(candidate.text);
  }
  return result;
}

function selectShortTermCandidates(params: {
  store: MemoryOsStore;
  sessionKey: string;
  queryKeywords: string[];
}): RetrievalCandidate<MemoryOsShortTermPage>[] {
  const sessionState = params.store.sessions[params.sessionKey];
  const pages = (sessionState?.stmPageIds ?? [])
    .map((id) => params.store.pages[id])
    .filter((page): page is MemoryOsShortTermPage => Boolean(page));
  return pages
    .map((page) => {
      const text = `- ${page.title}: ${page.content}`;
      return {
        item: page,
        text,
        chars: text.length,
        score:
          scoreKeywords(params.queryKeywords, page.keywords) +
          page.heat * 0.08 +
          page.accessCount * 0.05 +
          sessionRecencyScore(page),
      };
    })
    .sort((left, right) => right.score - left.score || right.item.updatedAt - left.item.updatedAt);
}

function selectSegmentCandidates(params: {
  store: MemoryOsStore;
  sessionKey: string;
  queryKeywords: string[];
  currentSummary?: string;
  currentTaskState?: string;
}): RetrievalCandidate<MemoryOsSegment>[] {
  return Object.values(params.store.segments)
    .filter((segment) => {
      const sameSummary = trimOptionalText(segment.summary) === trimOptionalText(params.currentSummary);
      const sameTaskState =
        trimOptionalText(segment.taskState) === trimOptionalText(params.currentTaskState);
      return !(sameSummary && sameTaskState);
    })
    .map((segment) => {
      const text = [
        `- ${segment.title}`,
        segment.summary,
        segment.taskState ? `Task state: ${segment.taskState}` : undefined,
      ]
        .filter(Boolean)
        .join("\n");
      return {
        item: segment,
        text,
        chars: text.length,
        score:
          scoreKeywords(params.queryKeywords, segment.keywords) +
          (segment.sessionKey === params.sessionKey ? 0.5 : 0) +
          segment.heat * 0.12 +
          segment.accessCount * 0.08 +
          segment.promotionCount * 0.15 +
          sessionRecencyScore(segment),
      };
    })
    .sort((left, right) => right.score - left.score || right.item.updatedAt - left.item.updatedAt);
}

function selectLongTermCandidates(params: {
  store: MemoryOsStore;
  queryKeywords: string[];
  globalImportantNotes: string;
}): RetrievalCandidate<MemoryOsLongTermNote>[] {
  const globalNoteSet = new Set(splitLines(params.globalImportantNotes).map((line) => line.toLowerCase()));
  return Object.values(params.store.longTermNotes)
    .filter((note) => !globalNoteSet.has(note.content.trim().toLowerCase()))
    .map((note) => ({
      item: note,
      text: `- ${note.content}`,
      chars: note.content.length + 2,
      score:
        scoreKeywords(params.queryKeywords, note.keywords) +
        note.heat * 0.12 +
        note.accessCount * 0.08 +
        note.promotionCount * 0.2 +
        sessionRecencyScore(note),
    }))
    .sort((left, right) => right.score - left.score || right.item.updatedAt - left.item.updatedAt);
}

function selectBudgetedMemoryContext(params: {
  shortTermCandidates: RetrievalCandidate<MemoryOsShortTermPage>[];
  segmentCandidates: RetrievalCandidate<MemoryOsSegment>[];
  longTermCandidates: RetrievalCandidate<MemoryOsLongTermNote>[];
  charBudget: number;
}) {
  const totalBudget = Math.max(1_200, params.charBudget);
  const sectionBudgets = {
    shortTerm: Math.min(DEFAULT_SHORT_TERM_CHAR_BUDGET, Math.floor(totalBudget * 0.28)),
    midTerm: Math.min(DEFAULT_MID_TERM_CHAR_BUDGET, Math.floor(totalBudget * 0.47)),
    longTerm: Math.min(DEFAULT_LONG_TERM_CHAR_BUDGET, Math.floor(totalBudget * 0.25)),
  };

  let shortTerm = selectCandidatesWithinBudget(params.shortTermCandidates, sectionBudgets.shortTerm);
  let midTerm = selectCandidatesWithinBudget(params.segmentCandidates, sectionBudgets.midTerm);
  let longTerm = selectCandidatesWithinBudget(params.longTermCandidates, sectionBudgets.longTerm);

  const combinedBudget =
    sectionBudgets.shortTerm + sectionBudgets.midTerm + sectionBudgets.longTerm;
  shortTerm = fillRemainingBudget(
    shortTerm,
    params.shortTermCandidates,
    combinedBudget - midTerm.reduce((sum, item) => sum + item.chars, 0) -
      longTerm.reduce((sum, item) => sum + item.chars, 0),
  );
  midTerm = fillRemainingBudget(
    midTerm,
    params.segmentCandidates,
    combinedBudget - shortTerm.reduce((sum, item) => sum + item.chars, 0) -
      longTerm.reduce((sum, item) => sum + item.chars, 0),
  );
  longTerm = fillRemainingBudget(
    longTerm,
    params.longTermCandidates,
    combinedBudget - shortTerm.reduce((sum, item) => sum + item.chars, 0) -
      midTerm.reduce((sum, item) => sum + item.chars, 0),
  );

  return {
    shortTerm,
    midTerm,
    longTerm,
  };
}

export function syncMemoryOsShortTermPages(params: {
  sessionKey: string;
  messages: unknown[];
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
}): MemoryOsStore {
  const store = loadMemoryOsStore(params.env);
  const updatedAt = params.updatedAt ?? Date.now();
  const sessionState = ensureSessionState(store, params.sessionKey);
  const previousIds = [...sessionState.stmPageIds];
  const pages = buildShortTermPages(params.messages, params.sessionKey);
  upsertPagesIntoStore(store, pages, previousIds);
  sessionState.stmPageIds = pages.map((page) => page.id).slice(-MAX_SHORT_TERM_PAGES);
  sessionState.lastUpdatedAt = updatedAt;
  promoteRecentPagesToDerivedSegment(store, params.sessionKey, updatedAt);
  evictColdMemory(store);
  return saveMemoryOsStore(store, params.env);
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
  const sessionState = ensureSessionState(store, params.sessionKey);
  sessionState.lastCompactedAt = updatedAt;
  upsertSegment({
    store,
    sessionKey: params.sessionKey,
    pageIds: sessionState.stmPageIds.slice(-DERIVED_SEGMENT_WINDOW),
    summary: params.summary,
    taskState: params.taskState,
    source: params.source,
    updatedAt,
  });
  normalizeLongTermNotesIntoStore({
    store,
    content: loadGlobalImportantNotes({ env: params.env }).content,
  });
  promoteSegmentsToLongTermNotes(store, updatedAt, params.sessionKey);
  evictColdMemory(store);
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
  evictColdMemory(store);
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

export function buildMemoryOsPromptAddition(params: {
  sessionKey: string;
  queryText: string;
  promptContext?: SessionPromptContext;
  env?: NodeJS.ProcessEnv;
  charBudget?: number;
  tokenBudget?: number;
}): string | undefined {
  const store = loadMemoryOsStore(params.env);
  const globalImportantNotes = loadGlobalImportantNotes({ env: params.env }).content;
  normalizeLongTermNotesIntoStore({
    store,
    content: globalImportantNotes,
  });
  const queryKeywords = normalizeKeywords(
    [params.queryText, params.promptContext?.taskState, params.promptContext?.contextSummary]
      .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      .join("\n"),
  );

  const selected = selectBudgetedMemoryContext({
    shortTermCandidates: selectShortTermCandidates({
      store,
      sessionKey: params.sessionKey,
      queryKeywords,
    }),
    segmentCandidates: selectSegmentCandidates({
      store,
      sessionKey: params.sessionKey,
      queryKeywords,
      currentSummary: params.promptContext?.contextSummary,
      currentTaskState: params.promptContext?.taskState,
    }),
    longTermCandidates: selectLongTermCandidates({
      store,
      queryKeywords,
      globalImportantNotes,
    }),
    charBudget:
      params.charBudget ??
      estimateCharBudgetFromTokenBudget(params.tokenBudget) ??
      DEFAULT_MEMORY_OS_CHAR_BUDGET,
  });

  const now = Date.now();
  for (const candidate of selected.shortTerm) {
    candidate.item.accessCount += 1;
    candidate.item.heat += 0.2;
    candidate.item.lastRetrievedAt = now;
  }
  for (const candidate of selected.midTerm) {
    candidate.item.accessCount += 1;
    candidate.item.heat += 0.5;
    candidate.item.lastRetrievedAt = now;
  }
  for (const candidate of selected.longTerm) {
    candidate.item.accessCount += 1;
    candidate.item.heat += 0.25;
    candidate.item.lastRetrievedAt = now;
  }
  promoteSegmentsToLongTermNotes(store, now);
  evictColdMemory(store);
  saveMemoryOsStore(store, params.env);

  const sections: string[] = [];
  if (selected.shortTerm.length > 0) {
    sections.push(["### Short-Term Memory", ...selected.shortTerm.map((candidate) => candidate.text)].join("\n"));
  }
  if (selected.midTerm.length > 0) {
    sections.push(["### Mid-Term Episodic Memory", ...selected.midTerm.map((candidate) => candidate.text)].join("\n\n"));
  }
  if (selected.longTerm.length > 0) {
    sections.push(["### Long-Term Memory", ...selected.longTerm.map((candidate) => candidate.text)].join("\n"));
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
    short_term_page_count: sessionState?.stmPageIds.length ?? 0,
    mid_term_segment_count: segments.length,
    long_term_note_count: Object.keys(store.longTermNotes).length,
    latest_segment_title: segments[0]?.title ?? null,
    latest_segment_summary: segments[0]?.summary ?? null,
    latest_updated_at: segments[0]?.updatedAt ?? sessionState?.lastUpdatedAt ?? null,
  };
}
