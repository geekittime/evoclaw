import fs from "node:fs";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";

const MAX_GLOBAL_IMPORTANT_NOTES_CHARS = 24_000;
const GLOBAL_IMPORTANT_NOTES_FILENAME = "important-notes.md";
const LEGACY_GLOBAL_IMPORTANT_NOTES_FILENAME = "important-notes.json";

export type GlobalImportantNotesStore = {
  content?: string;
  updatedAt?: number;
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

function normalizeSummaryLines(value: string): string[] {
  return value
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .map((line) => line.replace(/^[-*•\d.)\s]+/u, "").trim())
    .filter(Boolean);
}

export function resolveGlobalImportantNotesPath(env: NodeJS.ProcessEnv = process.env): string {
  return path.join(resolveStateDir(env), "prompt-context", GLOBAL_IMPORTANT_NOTES_FILENAME);
}

function resolveLegacyGlobalImportantNotesPath(env: NodeJS.ProcessEnv = process.env): string {
  return path.join(resolveStateDir(env), "prompt-context", LEGACY_GLOBAL_IMPORTANT_NOTES_FILENAME);
}

function normalizeStoredImportantNotes(value: unknown): GlobalImportantNotesStore {
  if (!value || typeof value !== "object") {
    return {};
  }
  const record = value as { content?: unknown; updatedAt?: unknown };
  const content = trimOptionalText(typeof record.content === "string" ? record.content : undefined);
  const updatedAt =
    typeof record.updatedAt === "number" && Number.isFinite(record.updatedAt)
      ? Math.floor(record.updatedAt)
      : undefined;
  return {
    ...(content ? { content } : {}),
    ...(updatedAt ? { updatedAt } : {}),
  };
}

function readGlobalImportantNotesFile(filePath: string): GlobalImportantNotesStore {
  try {
    if (!fs.existsSync(filePath)) {
      return {};
    }
    const raw = fs.readFileSync(filePath, "utf8");
    if (!raw.trim()) {
      return {};
    }
    return {
      content: trimOptionalText(raw),
      updatedAt: Math.floor(fs.statSync(filePath).mtimeMs),
    };
  } catch {
    return {};
  }
}

function writeGlobalImportantNotesFile(filePath: string, next: GlobalImportantNotesStore): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, next.content ?? "", "utf8");
}

function readLegacyGlobalImportantNotesFile(filePath: string): GlobalImportantNotesStore {
  try {
    if (!fs.existsSync(filePath)) {
      return {};
    }
    const raw = fs.readFileSync(filePath, "utf8");
    if (!raw.trim()) {
      return {};
    }
    return normalizeStoredImportantNotes(JSON.parse(raw));
  } catch {
    return {};
  }
}

export function loadGlobalImportantNotes(params?: {
  env?: NodeJS.ProcessEnv;
  seedFromLegacyNotes?: string | undefined;
}): GlobalImportantNotesStore {
  const filePath = resolveGlobalImportantNotesPath(params?.env);
  const current = readGlobalImportantNotesFile(filePath);
  if (current.content) {
    return current;
  }
  const legacyFile = readLegacyGlobalImportantNotesFile(resolveLegacyGlobalImportantNotesPath(params?.env));
  const seeded = trimOptionalText(legacyFile.content ?? params?.seedFromLegacyNotes);
  if (!seeded) {
    return current;
  }
  const next = {
    content: clampTail(seeded, MAX_GLOBAL_IMPORTANT_NOTES_CHARS),
    updatedAt: legacyFile.updatedAt ?? Date.now(),
  };
  writeGlobalImportantNotesFile(filePath, next);
  return next;
}

export function appendGlobalImportantNote(params: {
  summary: string;
  updatedAt?: number;
  env?: NodeJS.ProcessEnv;
  seedFromLegacyNotes?: string | undefined;
}): GlobalImportantNotesStore {
  const summary = trimOptionalText(params.summary);
  const current = loadGlobalImportantNotes({
    env: params.env,
    seedFromLegacyNotes: params.seedFromLegacyNotes,
  });
  if (!summary) {
    return current;
  }
  const nextLines = normalizeSummaryLines(summary);
  if (nextLines.length === 0) {
    return current;
  }
  const currentLines = (current.content ?? "")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);
  const mergedLines = [...currentLines];
  for (const line of nextLines) {
    const alreadyPresent = mergedLines.some((currentLine) => currentLine === `- ${line}` || currentLine === line);
    if (!alreadyPresent) {
      mergedLines.push(`- ${line}`);
    }
  }
  const nextContent = mergedLines.join("\n");
  const next = {
    content: clampTail(nextContent, MAX_GLOBAL_IMPORTANT_NOTES_CHARS),
    updatedAt: params.updatedAt ?? Date.now(),
  };
  writeGlobalImportantNotesFile(resolveGlobalImportantNotesPath(params.env), next);
  return next;
}

export function buildGlobalImportantNotesPromptAddition(params?: {
  env?: NodeJS.ProcessEnv;
  seedFromLegacyNotes?: string | undefined;
}): string | undefined {
  const content = loadGlobalImportantNotes(params).content;
  if (!content) {
    return undefined;
  }
  return [
    "## Important Notes (High Priority)",
    "These notes are durable operator guidance distilled from prior feedback across all sessions.",
    "You MUST read and follow them on every turn unless they directly conflict with higher-priority system or developer instructions.",
    "Treat them as persistent preferences, constraints, and lessons learned. Pay close attention before responding.",
    "",
    content,
  ].join("\n");
}
