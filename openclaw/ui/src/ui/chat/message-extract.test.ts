import { describe, expect, it } from "vitest";
import {
  extractText,
  extractTextCached,
  extractThinking,
  extractThinkingCached,
} from "./message-extract.ts";

describe("extractTextCached", () => {
  it("matches extractText output", () => {
    const message = {
      role: "assistant",
      content: [{ type: "text", text: "Hello there" }],
    };
    expect(extractTextCached(message)).toBe(extractText(message));
  });

  it("returns consistent output for repeated calls", () => {
    const message = {
      role: "user",
      content: "plain text",
    };
    expect(extractTextCached(message)).toBe("plain text");
    expect(extractTextCached(message)).toBe("plain text");
  });

  it("strips assistant relevant-memories scaffolding", () => {
    const message = {
      role: "assistant",
      content: [
        {
          type: "text",
          text: [
            "<relevant-memories>",
            "Internal memory context",
            "</relevant-memories>",
            "Final user answer",
          ].join("\n"),
        },
      ],
    };
    expect(extractText(message)).toBe("Final user answer");
    expect(extractTextCached(message)).toBe("Final user answer");
  });

  it("strips runtime guidance from displayed user text and keeps only the raw prompt", () => {
    const message = {
      role: "user",
      content: [
        {
          type: "text",
          text: [
            "[[OPENCLAW_RUNTIME_GUIDANCE_START]]",
            "",
            "## Runtime Guidance For This Turn",
            "",
            "## Important Notes (High Priority)",
            "Always greet first.",
            "",
            "## Enabled Session Skills",
            "Enabled skills: safe-delete.",
            "",
            "[[OPENCLAW_RUNTIME_GUIDANCE_END]]",
            "",
            "请帮我删除 temp0 目录中的 py 文件",
          ].join("\n"),
        },
      ],
    };

    expect(extractText(message)).toBe("请帮我删除 temp0 目录中的 py 文件");
  });

  it("strips legacy unwrapped runtime guidance from displayed user text", () => {
    const message = {
      role: "user",
      content: [
        {
          type: "text",
          text: [
            "## Important Notes (High Priority)",
            "Always greet first.",
            "",
            "## Enabled Session Skills",
            "Enabled skills: safe-delete.",
            "",
            "请帮我删除 temp0 目录中的 py 文件",
          ].join("\n"),
        },
      ],
    };

    expect(extractText(message)).toBe("请帮我删除 temp0 目录中的 py 文件");
  });

  it("strips leading user timestamp prefixes from displayed text", () => {
    const message = {
      role: "user",
      content: [{ type: "text", text: "[Thu 2026-04-09 18:59 GMT+8] 你现在在哪个文件夹" }],
    };

    expect(extractText(message)).toBe("你现在在哪个文件夹");
  });
});

describe("extractThinkingCached", () => {
  it("matches extractThinking output", () => {
    const message = {
      role: "assistant",
      content: [{ type: "thinking", thinking: "Plan A" }],
    };
    expect(extractThinkingCached(message)).toBe(extractThinking(message));
  });

  it("returns consistent output for repeated calls", () => {
    const message = {
      role: "assistant",
      content: [{ type: "thinking", thinking: "Plan A" }],
    };
    expect(extractThinkingCached(message)).toBe("Plan A");
    expect(extractThinkingCached(message)).toBe("Plan A");
  });
});
