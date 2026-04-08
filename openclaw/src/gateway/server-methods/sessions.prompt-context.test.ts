import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SessionEntry } from "../../config/sessions/types.js";
import type { GatewayRequestContext, RespondFn } from "./types.js";

const updateSessionStoreMock = vi.fn();
const resolveGatewaySessionStoreTargetMock = vi.fn();
const loadSessionEntryMock = vi.fn();
const resolveFreshestSessionEntryFromStoreKeysMock = vi.fn();
const readSessionMessagesMock = vi.fn();
const summarizeConversationHistoryMock = vi.fn();
const summarizeFeedbackIntoImportantNoteMock = vi.fn();

vi.mock("../../config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../config/config.js")>();
  return {
    ...actual,
    loadConfig: () => ({}),
  };
});

vi.mock("../../config/sessions.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../config/sessions.js")>();
  return {
    ...actual,
    updateSessionStore: (...args: unknown[]) => updateSessionStoreMock(...args),
    resolveGatewaySessionStoreTarget: (...args: unknown[]) =>
      resolveGatewaySessionStoreTargetMock(...args),
  };
});

vi.mock("../session-utils.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../session-utils.js")>();
  return {
    ...actual,
    loadSessionEntry: (...args: unknown[]) => loadSessionEntryMock(...args),
    resolveFreshestSessionEntryFromStoreKeys: (...args: unknown[]) =>
      resolveFreshestSessionEntryFromStoreKeysMock(...args),
    readSessionMessages: (...args: unknown[]) => readSessionMessagesMock(...args),
  };
});

vi.mock("../session-prompt-context.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../session-prompt-context.js")>();
  return {
    ...actual,
    summarizeConversationHistory: (...args: unknown[]) => summarizeConversationHistoryMock(...args),
    summarizeFeedbackIntoImportantNote: (...args: unknown[]) =>
      summarizeFeedbackIntoImportantNoteMock(...args),
  };
});

import { sessionsHandlers } from "./sessions.js";

describe("sessions.promptContext handlers", () => {
  let sessionEntry: SessionEntry;

  beforeEach(() => {
    sessionEntry = {
      sessionId: "sess-1",
      updatedAt: 1,
      promptContext: {
        importantNotes: "Existing durable note",
        selectedSkillNames: ["security-triage"],
        selectionCustomized: true,
      },
    };

    updateSessionStoreMock.mockReset();
    resolveGatewaySessionStoreTargetMock.mockReset();
    loadSessionEntryMock.mockReset();
    resolveFreshestSessionEntryFromStoreKeysMock.mockReset();
    readSessionMessagesMock.mockReset();
    summarizeConversationHistoryMock.mockReset();
    summarizeFeedbackIntoImportantNoteMock.mockReset();

    resolveGatewaySessionStoreTargetMock.mockReturnValue({
      canonicalKey: "agent:main:main",
      storeKeys: ["agent:main:main", "main"],
      storePath: "/tmp/sessions.json",
      agentId: "main",
    });
    resolveFreshestSessionEntryFromStoreKeysMock.mockImplementation(() => sessionEntry);
    updateSessionStoreMock.mockImplementation(
      async (_storePath: string, mutate: (store: Record<string, SessionEntry>) => Promise<SessionEntry>) => {
        const store = { "agent:main:main": sessionEntry };
        sessionEntry = await mutate(store);
        return sessionEntry;
      },
    );
    loadSessionEntryMock.mockImplementation(() => ({
      entry: sessionEntry,
      canonicalKey: "agent:main:main",
      storePath: "/tmp/sessions.json",
    }));
    readSessionMessagesMock.mockReturnValue([
      { role: "user", content: "from transcript user" },
      { role: "assistant", content: "from transcript assistant" },
    ]);
  });

  it("stores feedback summaries into important notes and feedback records", async () => {
    summarizeFeedbackIntoImportantNoteMock.mockResolvedValue(
      "Start with a brief greeting when the user says hi.",
    );

    const respond = vi.fn() as unknown as RespondFn;
    const context = {
      broadcastToConnIds: vi.fn(),
      getSessionEventSubscriberConnIds: () => new Set<string>(),
    } as unknown as GatewayRequestContext;

    await sessionsHandlers["sessions.promptContext.feedback"]({
      params: {
        key: "agent:main:main",
        turn: 7,
        rating: "bad",
        feedback: "Needs to greet first.",
        responseText: "What would you like to work on today?",
        instructionText: "hi",
      },
      respond,
      context,
    } as never);

    expect(summarizeFeedbackIntoImportantNoteMock).toHaveBeenCalledWith({
      instructionText: "hi",
      responseText: "What would you like to work on today?",
      rating: "bad",
      feedback: "Needs to greet first.",
    });
    expect(sessionEntry.promptContext?.importantNotes).toContain("Existing durable note");
    expect(sessionEntry.promptContext?.importantNotes).toContain("brief greeting");
    expect(sessionEntry.promptContext?.feedbackRecords).toHaveLength(1);
    expect(sessionEntry.promptContext?.feedbackRecords?.[0]?.instructionText).toBe("hi");
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        ok: true,
        key: "agent:main:main",
        summary: "Start with a brief greeting when the user says hi.",
        promptContext: expect.objectContaining({
          importantNotes: expect.objectContaining({
            content: expect.stringContaining("brief greeting"),
          }),
          feedbackRecords: expect.arrayContaining([
            expect.objectContaining({
              instructionText: "hi",
              responseText: "What would you like to work on today?",
            }),
          ]),
        }),
      }),
      undefined,
    );
  });

  it("compacts the provided visible messages and stores the summary for later prompt injection", async () => {
    summarizeConversationHistoryMock.mockImplementation(
      async ({ messages }: { messages: unknown[] }) => {
        expect(messages).toEqual([
          { role: "user", content: "Please inspect the repo and delete stale files." },
          { role: "assistant", content: "I inspected it and identified two stale files." },
          {
            role: "assistant",
            content: [
              { type: "toolcall", name: "exec", arguments: { command: "ls temp0" } },
              { type: "toolresult", name: "exec", text: "hi.py\nhh.py\ntemp\n" },
            ],
          },
        ]);
        return "Goal: clean temp0. Completed: listed files. Pending: confirm deletion of hi.py and hh.py.";
      },
    );

    const respond = vi.fn() as unknown as RespondFn;
    const context = {
      broadcastToConnIds: vi.fn(),
      getSessionEventSubscriberConnIds: () => new Set<string>(),
    } as unknown as GatewayRequestContext;

    await sessionsHandlers["sessions.promptContext.compact"]({
      params: {
        key: "agent:main:main",
        source: "manual",
        instructions: "Keep approvals and pending file operations.",
        messages: [
          { role: "user", content: "Please inspect the repo and delete stale files." },
          { role: "assistant", content: "I inspected it and identified two stale files." },
          {
            role: "assistant",
            content: [
              { type: "toolcall", name: "exec", arguments: { command: "ls temp0" } },
              { type: "toolresult", name: "exec", text: "hi.py\nhh.py\ntemp\n" },
            ],
          },
        ],
      },
      respond,
      context,
    } as never);

    expect(sessionEntry.promptContext?.contextSummary).toContain("Goal: clean temp0.");
    expect(sessionEntry.promptContext?.contextSummarySource).toBe("manual");
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        ok: true,
        key: "agent:main:main",
        summary: expect.stringContaining("Pending: confirm deletion"),
        promptContext: expect.objectContaining({
          contextSummary: expect.objectContaining({
            content: expect.stringContaining("Goal: clean temp0."),
          }),
        }),
      }),
      undefined,
    );
  });
});
