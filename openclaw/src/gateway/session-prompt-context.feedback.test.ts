import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("summarizeFeedbackIntoImportantNote", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("sends question, answer, rating, and user feedback to DeepSeek with a durable-note prompt", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        messages?: Array<{ role?: string; content?: string }>;
      };
      const system = body.messages?.[0]?.content ?? "";
      const user = body.messages?.[1]?.content ?? "";

      expect(system).toContain("Analyze the user question, the assistant answer, the good/bad rating, and the user's written feedback together.");
      expect(system).toContain("Return 1 to 3 short plain-text lines");
      expect(user).toContain("User question:\n请删除 temp0 目录中的 py 文件");
      expect(user).toContain("Assistant answer:\n我先检查目录内容，再删除 py 文件。");
      expect(user).toContain("Feedback rating: bad");
      expect(user).toContain("Feedback details:\n应该先直接说清楚会删除哪些文件");
      expect(user).toContain("Focus on user preferences and things the assistant should pay attention to in future replies.");

      return {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content:
                  "Before destructive actions, clearly name the affected files.\nStart with a concise confirmation when the user asks for deletion.",
              },
            },
          ],
        }),
      } as Response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { summarizeFeedbackIntoImportantNote } = await import("./session-prompt-context.js");
    const summary = await summarizeFeedbackIntoImportantNote({
      instructionText: "请删除 temp0 目录中的 py 文件",
      responseText: "我先检查目录内容，再删除 py 文件。",
      rating: "bad",
      feedback: "应该先直接说清楚会删除哪些文件",
    });

    expect(summary).toContain("Before destructive actions, clearly name the affected files.");
    expect(summary).toContain("Start with a concise confirmation");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
