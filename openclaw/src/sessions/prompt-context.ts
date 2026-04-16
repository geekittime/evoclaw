import type {
  SessionCustomSkillRecord,
  SessionEntry,
  SessionFeedbackRecord,
  SessionPromptContext,
} from "../config/sessions/types.js";
import { matchesSkillFilter, normalizeSkillFilter } from "../agents/skills/filter.js";

export const SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD = 200_000;
const MAX_CONTEXT_SUMMARY_CHARS = 16_000;
const MAX_SESSION_NOTES_CHARS = 10_000;
const MAX_TASK_STATE_CHARS = 8_000;
const MAX_SESSION_FEEDBACK_RECORDS = 48;
const MAX_SKILL_SELECTION_HISTORY = 32;
const MAX_CUSTOM_SKILLS = 32;
const MAX_CUSTOM_SKILL_NAME_CHARS = 120;
const MAX_CUSTOM_SKILL_CONTENT_CHARS = 12_000;
export const RUNTIME_GUIDANCE_START_MARKER = "[[OPENCLAW_RUNTIME_GUIDANCE_START]]";
export const RUNTIME_GUIDANCE_END_MARKER = "[[OPENCLAW_RUNTIME_GUIDANCE_END]]";
const LEGACY_RUNTIME_GUIDANCE_BLOCK_PREFIXES = [
  "## Runtime Guidance For This Turn",
  "## Important Notes (High Priority)",
  "## Enabled Session Skills",
  "## Session Custom Skills",
  "## Session Notes",
  "## Current Task State",
  "## Conversation Summary",
  "## Additional Runtime Context",
  "## Group Chat Context",
  "## Subagent Context",
] as const;

function trimOptionalText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

export function wrapRuntimeGuidanceForPrompt(addition: string, body: string): string {
  const normalizedAddition = addition.trim();
  const normalizedBody = body.trim();
  if (!normalizedAddition) {
    return body;
  }
  return [
    RUNTIME_GUIDANCE_START_MARKER,
    "## Runtime Guidance For This Turn",
    "Read and follow these notes and enabled skills before answering the user.",
    normalizedAddition,
    RUNTIME_GUIDANCE_END_MARKER,
    normalizedBody,
  ]
    .filter(Boolean)
    .join("\n\n");
}

export function stripInjectedRuntimeGuidanceFromPrompt(value: string | undefined): string | undefined {
  const text = trimOptionalText(value);
  if (!text) {
    return undefined;
  }
  const start = text.indexOf(RUNTIME_GUIDANCE_START_MARKER);
  const end = text.indexOf(RUNTIME_GUIDANCE_END_MARKER);
  if (start !== 0 || end < 0) {
    return stripLegacyInjectedRuntimeGuidanceFromPrompt(text);
  }
  const remaining = text.slice(end + RUNTIME_GUIDANCE_END_MARKER.length).trim();
  return remaining || undefined;
}

function isLegacyRuntimeGuidanceBlock(block: string): boolean {
  const trimmed = block.trim();
  return LEGACY_RUNTIME_GUIDANCE_BLOCK_PREFIXES.some((prefix) => trimmed.startsWith(prefix));
}

function stripLegacyInjectedRuntimeGuidanceFromPrompt(text: string): string {
  const blocks = text
    .split(/\n{2,}/u)
    .map((block) => block.trim())
    .filter(Boolean);
  if (blocks.length < 2 || !isLegacyRuntimeGuidanceBlock(blocks[0] ?? "")) {
    return text;
  }
  let index = 0;
  while (index < blocks.length && isLegacyRuntimeGuidanceBlock(blocks[index] ?? "")) {
    index += 1;
  }
  if (index <= 0 || index >= blocks.length) {
    return text;
  }
  return blocks.slice(index).join("\n\n").trim() || text;
}

function clampTail(value: string, maxChars: number): string {
  if (value.length <= maxChars) {
    return value;
  }
  return value.slice(value.length - maxChars);
}

function splitNonEmptyLines(value: string): string[] {
  return value
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);
}

function appendDedupedLines(params: {
  current?: string;
  addition?: string;
  maxChars: number;
}): string | undefined {
  const lines = splitNonEmptyLines(params.current ?? "");
  const seen = new Set(lines.map((line) => line.toLowerCase()));
  for (const line of splitNonEmptyLines(params.addition ?? "")) {
    const key = line.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    lines.push(line);
  }
  const joined = lines.join("\n");
  const clamped = clampTail(joined, params.maxChars).trim();
  return clamped || undefined;
}

function normalizeCustomSkillName(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }
  return trimmed.slice(0, MAX_CUSTOM_SKILL_NAME_CHARS);
}

function normalizeCustomSkillContent(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }
  return clampTail(trimmed, MAX_CUSTOM_SKILL_CONTENT_CHARS);
}

function describeCustomSkillContent(value: string): string {
  const firstLine = value
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .find(Boolean);
  if (!firstLine) {
    return "Custom session skill.";
  }
  return firstLine.length > 140 ? `${firstLine.slice(0, 137)}...` : firstLine;
}

export function resolveSessionCustomSkills(
  entry?: Pick<SessionEntry, "promptContext"> | null,
): SessionCustomSkillRecord[] {
  const raw = Array.isArray(entry?.promptContext?.customSkills) ? entry.promptContext.customSkills : [];
  const seen = new Set<string>();
  const normalized: SessionCustomSkillRecord[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const name = normalizeCustomSkillName(
      typeof (item as { name?: unknown }).name === "string"
        ? (item as { name: string }).name
        : undefined,
    );
    const content = normalizeCustomSkillContent(
      typeof (item as { content?: unknown }).content === "string"
        ? (item as { content: string }).content
        : undefined,
    );
    if (!name || !content) {
      continue;
    }
    const key = name.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    normalized.push({
      name,
      content,
      createdAt:
        typeof (item as { createdAt?: unknown }).createdAt === "number" &&
        Number.isFinite((item as { createdAt: number }).createdAt)
          ? Math.floor((item as { createdAt: number }).createdAt)
          : Date.now(),
      updatedAt:
        typeof (item as { updatedAt?: unknown }).updatedAt === "number" &&
        Number.isFinite((item as { updatedAt: number }).updatedAt)
          ? Math.floor((item as { updatedAt: number }).updatedAt)
          : Date.now(),
    });
  }
  return normalized.slice(-MAX_CUSTOM_SKILLS);
}

export function resolveSelectedSessionCustomSkills(
  entry?: Pick<SessionEntry, "promptContext"> | null,
): SessionCustomSkillRecord[] {
  const selected = new Set(resolveSessionSelectedSkillNames(entry) ?? []);
  if (selected.size === 0) {
    return [];
  }
  return resolveSessionCustomSkills(entry).filter((skill) => selected.has(skill.name));
}

export function describeSessionCustomSkill(skill: Pick<SessionCustomSkillRecord, "content">): string {
  return describeCustomSkillContent(skill.content);
}

export function getSessionPromptContext(entry?: SessionEntry | null): SessionPromptContext | undefined {
  return entry?.promptContext;
}

export function resolveSessionSelectedSkillNames(
  entry?: Pick<SessionEntry, "promptContext"> | null,
): string[] | undefined {
  const selected = normalizeSkillFilter(entry?.promptContext?.selectedSkillNames);
  return selected === undefined ? undefined : selected;
}

export function matchesSessionSelectedSkillNames(params: {
  entry?: Pick<SessionEntry, "promptContext" | "skillsSnapshot"> | null;
  fallbackSkillFilter?: string[];
}): boolean {
  const selected = resolveSessionSelectedSkillNames(params.entry);
  return matchesSkillFilter(
    params.entry?.skillsSnapshot?.skillFilter,
    selected ?? params.fallbackSkillFilter,
  );
}

export function buildSessionPromptContextAddition(
  entry?: Pick<SessionEntry, "promptContext"> | null,
): string | undefined {
  const selectedSkillNames = resolveSessionSelectedSkillNames(entry) ?? [];
  const selectedCustomSkills = resolveSelectedSessionCustomSkills(entry);
  const sessionNotes = trimOptionalText(entry?.promptContext?.sessionNotes);
  const taskState = trimOptionalText(entry?.promptContext?.taskState);
  const contextSummary = trimOptionalText(entry?.promptContext?.contextSummary);
  const sections: string[] = [];
  if (selectedSkillNames.length > 0) {
    sections.push(
      [
        "## Enabled Session Skills",
        "The operator explicitly enabled these skills for this session.",
        "You should actively apply them while planning, tool use, and answering.",
        `Enabled skills: ${selectedSkillNames.join(", ")}.`,
      ].join("\n"),
    );
  }
  if (selectedCustomSkills.length > 0) {
    sections.push(
      [
        "## Session Custom Skills",
        "These custom skills were added by the operator for this session and should be treated as active guidance.",
        ...selectedCustomSkills.map((skill) => `### ${skill.name}\n${skill.content}`),
      ].join("\n\n"),
    );
  }
  if (sessionNotes) {
    sections.push(
      [
        "## Session Notes",
        "These notes were learned inside this session and are lower priority than global Important Notes.",
        sessionNotes,
      ].join("\n"),
    );
  }
  if (taskState) {
    sections.push(
      [
        "## Current Task State",
        "Use this as the working state for the current session. Keep it consistent with newer user messages and tool results.",
        taskState,
      ].join("\n"),
    );
  }
  if (contextSummary) {
    sections.push(`## Conversation Summary\n${contextSummary}`);
  }
  return sections.length > 0 ? sections.join("\n\n") : undefined;
}

export function appendExtraSystemPrompt(
  basePrompt: string | undefined,
  addition: string | undefined,
): string | undefined {
  const parts = [trimOptionalText(basePrompt), trimOptionalText(addition)].filter(
    (value): value is string => Boolean(value),
  );
  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

export function buildUpdatedPromptContextFromFeedback(params: {
  current?: SessionPromptContext;
  record: SessionFeedbackRecord;
  noteSummary?: string;
}): SessionPromptContext {
  const nextFeedbackRecords = [
    ...(params.current?.feedbackRecords ?? []),
    params.record,
  ].slice(-MAX_SESSION_FEEDBACK_RECORDS);
  const sessionNotes = appendDedupedLines({
    current: params.current?.sessionNotes,
    addition: params.noteSummary,
    maxChars: MAX_SESSION_NOTES_CHARS,
  });
  const updatedAt = params.record.createdAt || Date.now();

  return {
    ...params.current,
    sessionNotes,
    sessionNotesUpdatedAt: sessionNotes ? updatedAt : params.current?.sessionNotesUpdatedAt,
    feedbackRecords: nextFeedbackRecords,
  };
}

export function buildUpdatedPromptContextForSkillSelection(params: {
  current?: SessionPromptContext;
  selectedSkillNames: string[];
  customized: boolean;
  updatedAt?: number;
}): SessionPromptContext {
  const updatedAt = params.updatedAt ?? Date.now();
  const selectedSkillNames = normalizeSkillFilter(params.selectedSkillNames) ?? [];
  const skillSelectionHistory = [
    ...(params.current?.skillSelectionHistory ?? []),
    { updatedAt, selectedSkillNames },
  ].slice(-MAX_SKILL_SELECTION_HISTORY);
  return {
    ...params.current,
    selectedSkillNames,
    selectionCustomized: params.customized,
    skillSelectionHistory,
  };
}

export function buildUpdatedPromptContextForCustomSkillAdd(params: {
  current?: SessionPromptContext;
  name: string;
  content: string;
  updatedAt?: number;
}): SessionPromptContext {
  const updatedAt = params.updatedAt ?? Date.now();
  const name = normalizeCustomSkillName(params.name);
  const content = normalizeCustomSkillContent(params.content);
  if (!name || !content) {
    throw new Error("Custom skill name and content are required.");
  }
  const nextSkills = resolveSessionCustomSkills({ promptContext: params.current });
  const existingIndex = nextSkills.findIndex((skill) => skill.name.toLowerCase() === name.toLowerCase());
  if (existingIndex >= 0) {
    const existing = nextSkills[existingIndex]!;
    nextSkills[existingIndex] = {
      ...existing,
      name,
      content,
      updatedAt,
    };
  } else {
    nextSkills.push({
      name,
      content,
      createdAt: updatedAt,
      updatedAt,
    });
  }
  return {
    ...params.current,
    customSkills: nextSkills.slice(-MAX_CUSTOM_SKILLS),
  };
}

export function buildUpdatedPromptContextForSummary(params: {
  current?: SessionPromptContext;
  summary: string;
  source: "manual" | "auto";
  tokenCount?: number;
  updatedAt?: number;
}): SessionPromptContext {
  const updatedAt = params.updatedAt ?? Date.now();
  const summary = clampTail(params.summary.trim(), MAX_CONTEXT_SUMMARY_CHARS);
  return {
    ...params.current,
    contextSummary: summary || undefined,
    contextSummaryUpdatedAt: summary ? updatedAt : params.current?.contextSummaryUpdatedAt,
    contextSummarySource: summary ? params.source : params.current?.contextSummarySource,
    contextSummaryTokenCount:
      typeof params.tokenCount === "number" && Number.isFinite(params.tokenCount) && params.tokenCount > 0
        ? Math.floor(params.tokenCount)
        : params.current?.contextSummaryTokenCount,
  };
}

export function buildUpdatedPromptContextForTaskState(params: {
  current?: SessionPromptContext;
  taskState: string;
  source: "manual" | "auto";
  updatedAt?: number;
}): SessionPromptContext {
  const updatedAt = params.updatedAt ?? Date.now();
  const taskState = clampTail(params.taskState.trim(), MAX_TASK_STATE_CHARS);
  return {
    ...params.current,
    taskState: taskState || undefined,
    taskStateUpdatedAt: taskState ? updatedAt : params.current?.taskStateUpdatedAt,
    taskStateSource: taskState ? params.source : params.current?.taskStateSource,
  };
}

export function shouldAutoRefreshContextSummary(params: {
  entry?: Pick<SessionEntry, "promptContext" | "totalTokens" | "totalTokensFresh"> | null;
  threshold?: number;
}): boolean {
  if (!params.entry || params.entry.totalTokensFresh === false) {
    return false;
  }
  const totalTokens =
    typeof params.entry.totalTokens === "number" && Number.isFinite(params.entry.totalTokens)
      ? Math.floor(params.entry.totalTokens)
      : 0;
  if (totalTokens < (params.threshold ?? SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD)) {
    return false;
  }
  const summarizedAt =
    typeof params.entry.promptContext?.contextSummaryTokenCount === "number"
      ? params.entry.promptContext.contextSummaryTokenCount
      : 0;
  return totalTokens > summarizedAt;
}
