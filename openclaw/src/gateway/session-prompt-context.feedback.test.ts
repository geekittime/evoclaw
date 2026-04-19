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

describe("summarizeConversationHistory", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("asks DeepSeek for a detailed session summary including user turns, assistant replies, and tool execution", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        max_tokens?: number;
        messages?: Array<{ role?: string; content?: string }>;
      };
      const system = body.messages?.[0]?.content ?? "";
      const user = body.messages?.[1]?.content ?? "";

      expect(body.max_tokens).toBe(1400);
      expect(system).toContain("Write a concise but sufficiently informative summary");
      expect(system).toContain("Focus on the main task thread and the most important turns");
      expect(system).toContain("Do not retain low-signal detail");
      expect(system).toContain("Do not summarize repository boilerplate or standing context such as IDENTITY.md, USER.md, SOUL.md");
      expect(system).toContain("Do not summarize user preferences, style rules, global important-notes");
      expect(user).toContain("Please summarize the full session, including the user/assistant conversation and the tool execution process.");
      expect(user).toContain("Focus on the actual dialogue and task progress, not standing background files or global configuration context.");
      expect(user).toContain("User: 今天南京天气怎么样？");
      expect(user).toContain("Assistant: 今天南京多云");
      expect(user).toContain("Tool: Tool call: weather");
      expect(user).toContain("Tool: Tool result: weather");

      return {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content:
                  "用户先询问南京天气，助手通过 weather 工具查询后给出了多云和温度范围。随后用户继续让助手检查 temp0 目录，工具结果表明目录中存在 hi.py 与 hh.py，后续删除动作仍待确认。",
              },
            },
          ],
        }),
      } as Response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { summarizeConversationHistory } = await import("./session-prompt-context.js");
    const summary = await summarizeConversationHistory({
      instructions: "Keep weather facts and file-operation state.",
      messages: [
        { role: "user", content: "今天南京天气怎么样？" },
        { role: "assistant", content: "今天南京多云，气温 18 到 24 度。" },
        {
          role: "tool",
          content: [{ type: "toolcall", name: "weather", arguments: { location: "Nanjing" } }],
        },
        {
          role: "tool",
          content: [{ type: "toolresult", name: "weather", text: "Cloudy, 18-24C" }],
        },
      ],
    });

    expect(summary).toContain("今天南京天气怎么样");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("excludes system/runtime guidance messages from conversation summaries", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        messages?: Array<{ role?: string; content?: string }>;
      };
      const user = body.messages?.[1]?.content ?? "";

      expect(user).toContain("User: 帮我检查 temp0 目录");
      expect(user).toContain("Assistant: 我先查看目录内容。");
      expect(user).not.toContain("Runtime Guidance For This Turn");
      expect(user).not.toContain("Important Notes (High Priority)");

      return {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: "用户要求检查 temp0 目录，助手准备先查看目录内容。",
              },
            },
          ],
        }),
      } as Response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { summarizeConversationHistory } = await import("./session-prompt-context.js");
    const summary = await summarizeConversationHistory({
      messages: [
        { role: "system", content: "## Runtime Guidance For This Turn\n## Important Notes (High Priority)" },
        { role: "user", content: "帮我检查 temp0 目录" },
        { role: "assistant", content: "我先查看目录内容。" },
      ],
    });

    expect(summary).toContain("temp0");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
