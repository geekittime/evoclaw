import type {
  SessionEntry,
  SessionFeedbackRecord,
  SessionPromptContext,
} from "../config/sessions/types.js";
import { matchesSkillFilter, normalizeSkillFilter } from "../agents/skills/filter.js";

export const SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD = 200_000;
const MAX_IMPORTANT_NOTES_CHARS = 12_000;
const MAX_CONTEXT_SUMMARY_CHARS = 16_000;
const MAX_SESSION_FEEDBACK_RECORDS = 48;
const MAX_SKILL_SELECTION_HISTORY = 32;

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
  const importantNotes = trimOptionalText(entry?.promptContext?.importantNotes);
  const contextSummary = trimOptionalText(entry?.promptContext?.contextSummary);
  const sections: string[] = [];
  if (importantNotes) {
    sections.push(`## Important Notes\n${importantNotes}`);
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
  const now = params.record.createdAt;
  const nextFeedbackRecords = [
    ...(params.current?.feedbackRecords ?? []),
    params.record,
  ].slice(-MAX_SESSION_FEEDBACK_RECORDS);
  const noteSummary = trimOptionalText(params.noteSummary);
  const existingNotes = trimOptionalText(params.current?.importantNotes);
  const importantNotes =
    noteSummary && existingNotes ? `${existingNotes}\n- ${noteSummary}` : noteSummary ?? existingNotes;

  return {
    ...params.current,
    feedbackRecords: nextFeedbackRecords,
    importantNotes: importantNotes ? clampTail(importantNotes, MAX_IMPORTANT_NOTES_CHARS) : undefined,
    importantNotesUpdatedAt:
      importantNotes || params.current?.importantNotesUpdatedAt ? now : params.current?.importantNotesUpdatedAt,
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
