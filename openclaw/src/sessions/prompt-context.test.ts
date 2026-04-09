import { describe, expect, it } from "vitest";
import {
  SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD,
  buildSessionPromptContextAddition,
  buildUpdatedPromptContextForSkillSelection,
  buildUpdatedPromptContextForSummary,
  buildUpdatedPromptContextFromFeedback,
  shouldAutoRefreshContextSummary,
} from "./prompt-context.js";

describe("session prompt context helpers", () => {
  it("records feedback and appends durable important notes", () => {
    const next = buildUpdatedPromptContextFromFeedback({
      current: {
        importantNotes: "Existing preference",
        feedbackRecords: [
          {
            id: "old",
            createdAt: 1,
            rating: "good",
          },
        ],
      },
      record: {
        id: "new",
        createdAt: 2,
        rating: "bad",
        feedback: "Answer should greet first.",
        instructionText: "hi",
        responseText: "What can I do for you?",
        turn: 3,
        summary: "Start with a brief greeting when the user says hi.",
      },
      noteSummary: "Start with a brief greeting when the user says hi.",
    });

    expect(next.feedbackRecords).toHaveLength(2);
    expect(next.feedbackRecords?.at(-1)?.feedback).toBe("Answer should greet first.");
    expect(next.importantNotes).toBe("Existing preference");
    expect(next.importantNotesUpdatedAt).toBeUndefined();
  });

  it("records skill selections and preserves customization state", () => {
    const next = buildUpdatedPromptContextForSkillSelection({
      current: {
        selectedSkillNames: ["old-skill"],
        selectionCustomized: true,
        skillSelectionHistory: [{ updatedAt: 1, selectedSkillNames: ["old-skill"] }],
      },
      selectedSkillNames: ["security-triage", "code-review"],
      customized: true,
      updatedAt: 10,
    });

    expect(next.selectedSkillNames).toEqual(["security-triage", "code-review"]);
    expect(next.selectionCustomized).toBe(true);
    expect(next.skillSelectionHistory).toEqual([
      { updatedAt: 1, selectedSkillNames: ["old-skill"] },
      { updatedAt: 10, selectedSkillNames: ["security-triage", "code-review"] },
    ]);
  });

  it("builds the injected prompt addition from notes and compressed history", () => {
    const prompt = buildSessionPromptContextAddition({
      promptContext: {
        selectedSkillNames: ["security-triage", "code-review", "session-rules"],
        contextSummary: "We already inspected the repo and fixed the build.",
        customSkills: [
          {
            name: "session-rules",
            content: "Always explain risks before deleting files.",
            createdAt: 1,
            updatedAt: 1,
          },
        ],
      },
    });

    expect(prompt).toContain("## Enabled Session Skills");
    expect(prompt).toContain("security-triage, code-review");
    expect(prompt).toContain("## Session Custom Skills");
    expect(prompt).toContain("Always explain risks before deleting files.");
    expect(prompt).toContain("## Conversation Summary");
    expect(prompt).toContain("fixed the build");
  });

  it("marks a session for automatic summary refresh only when tokens exceed the threshold", () => {
    expect(
      shouldAutoRefreshContextSummary({
        entry: {
          totalTokens: SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD - 1,
          totalTokensFresh: true,
          promptContext: {},
        },
      }),
    ).toBe(false);

    expect(
      shouldAutoRefreshContextSummary({
        entry: {
          totalTokens: SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD + 10,
          totalTokensFresh: true,
          promptContext: {
            contextSummaryTokenCount: SESSION_CONTEXT_SUMMARY_TOKEN_THRESHOLD - 10,
          },
        },
      }),
    ).toBe(true);
  });

  it("stores compressed history metadata for later prompt injection", () => {
    const next = buildUpdatedPromptContextForSummary({
      current: {},
      summary: "Goal: finish the UI integration. Pending: verify delete flow.",
      source: "manual",
      tokenCount: 210_000,
      updatedAt: 20,
    });

    expect(next.contextSummary).toContain("Goal: finish the UI integration.");
    expect(next.contextSummarySource).toBe("manual");
    expect(next.contextSummaryUpdatedAt).toBe(20);
    expect(next.contextSummaryTokenCount).toBe(210_000);
  });
});
