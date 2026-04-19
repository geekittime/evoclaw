import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  buildMemoryOsPromptAddition,
  getMemoryOsSessionSnapshot,
  loadMemoryOsStore,
  syncMemoryOsShortTermPages,
  updateMemoryOsFromConversation,
  updateMemoryOsFromFeedback,
} from "./memory-os.js";

describe("memory-os", () => {
  let stateDir: string;

  beforeEach(() => {
    stateDir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-memory-os-"));
    process.env.OPENCLAW_STATE_DIR = stateDir;
    fs.mkdirSync(path.join(stateDir, "prompt-context"), { recursive: true });
    fs.writeFileSync(
      path.join(stateDir, "prompt-context", "important-notes.md"),
      "- 删除前先明确列出受影响文件\n- 对危险命令保持谨慎\n",
      "utf8",
    );
  });

  afterEach(() => {
    delete process.env.OPENCLAW_STATE_DIR;
    fs.rmSync(stateDir, { recursive: true, force: true });
  });

  it("stores STM pages, MTM segments, and LTM notes", () => {
    syncMemoryOsShortTermPages({
      sessionKey: "agent:main:main",
      messages: [
        { role: "user", content: "帮我删除 temp0 目录下的 py 文件" },
        { role: "assistant", content: "我先列出目录里的 py 文件" },
        {
          role: "toolresult",
          content: [{ type: "toolcall", name: "exec", arguments: { command: "find temp0 -name '*.py'" } }],
        },
      ],
    });
    updateMemoryOsFromConversation({
      sessionKey: "agent:main:main",
      messages: [
        { role: "user", content: "帮我删除 temp0 目录下的 py 文件" },
        { role: "assistant", content: "我先列出目录里的 py 文件" },
        {
          role: "toolresult",
          content: [{ type: "toolresult", name: "exec", text: "hh.py\nhi.py" }],
        },
      ],
      summary: "用户想删除 temp0 下的 py 文件，当前已经列出 hh.py 和 hi.py。",
      taskState: "当前任务：删除 temp0 下的 py 文件。已完成：列出文件。下一步：确认删除。",
      source: "manual",
    });
    updateMemoryOsFromFeedback({
      sessionKey: "agent:main:main",
      summary: "删除文件前先列出具体文件名并说明影响范围。",
    });

    const store = loadMemoryOsStore();
    const snapshot = getMemoryOsSessionSnapshot({ sessionKey: "agent:main:main" });

    expect(store.sessions["agent:main:main"]?.shortTermPages.length).toBeGreaterThan(0);
    expect(snapshot.short_term_page_count).toBeGreaterThan(0);
    expect(snapshot.mid_term_segment_count).toBeGreaterThan(0);
    expect(snapshot.long_term_note_count).toBeGreaterThan(0);
  });

  it("builds retrieved context with recent pages, segments, and long-term notes", () => {
    updateMemoryOsFromConversation({
      sessionKey: "agent:main:main",
      messages: [
        { role: "user", content: "现在运行一下 pwd 指令" },
        { role: "assistant", content: "我准备执行 pwd 来确认当前目录。" },
        {
          role: "toolresult",
          content: [{ type: "toolresult", name: "exec", text: "/home/kangshijia/.openclaw/workspace" }],
        },
      ],
      summary: "本 session 刚刚确认了当前目录是 OpenClaw workspace。",
      taskState: "当前任务：确认工作目录。已完成：运行 pwd。",
      source: "auto",
    });
    updateMemoryOsFromFeedback({
      sessionKey: "agent:main:main",
      summary: "在工作目录相关任务中，优先先确认当前 cwd 再继续执行文件操作。",
    });

    const addition = buildMemoryOsPromptAddition({
      sessionKey: "agent:main:main",
      queryText: "现在继续在当前工作目录里查找 py 文件",
    });

    expect(addition).toContain("## Memory OS Retrieved Context");
    expect(addition).toContain("### Short-Term Memory");
    expect(addition).toContain("### Mid-Term Episodic Memory");
    expect(addition).toContain("### Long-Term Memory");
  });

  it("does not duplicate global important notes or the current session summary in retrieved context", () => {
    updateMemoryOsFromConversation({
      sessionKey: "agent:main:main",
      messages: [
        { role: "user", content: "删除前先列出受影响文件" },
        { role: "assistant", content: "我会先列出文件，再执行删除。" },
      ],
      summary: "本 session 当前在处理删除前的文件确认流程。",
      taskState: "当前任务：删除前先列出受影响文件。",
      source: "manual",
    });

    const addition = buildMemoryOsPromptAddition({
      sessionKey: "agent:main:main",
      queryText: "现在继续删除前的确认流程",
      promptContext: {
        contextSummary: "本 session 当前在处理删除前的文件确认流程。",
        taskState: "当前任务：删除前先列出受影响文件。",
      },
    });

    expect(addition).toBeTruthy();
    expect(addition).not.toContain("- 删除前先明确列出受影响文件");
    expect(addition).not.toContain("本 session 当前在处理删除前的文件确认流程。");
    expect(addition).not.toContain("当前任务：删除前先列出受影响文件。");
  });

  it("does not refresh long-term note timestamps during retrieval-only reads", () => {
    updateMemoryOsFromFeedback({
      sessionKey: "agent:main:main",
      summary: "遇到删除任务时先列出受影响文件。",
      updatedAt: 1234,
    });

    const before = loadMemoryOsStore();
    const noteBefore = Object.values(before.longTermNotes).find((note) =>
      note.content.includes("先列出受影响文件"),
    );

    expect(noteBefore?.updatedAt).toBe(1234);

    buildMemoryOsPromptAddition({
      sessionKey: "agent:main:main",
      queryText: "继续删除前确认流程",
    });

    const after = loadMemoryOsStore();
    const noteAfter = Object.values(after.longTermNotes).find((note) =>
      note.content.includes("先列出受影响文件"),
    );

    expect(noteAfter?.updatedAt).toBe(1234);
  });

  it("does not create MTM segments when both summary and task state are empty", () => {
    updateMemoryOsFromConversation({
      sessionKey: "agent:main:main",
      messages: [{ role: "user", content: "hi" }],
      summary: "   ",
      taskState: "   ",
      source: "manual",
      updatedAt: 999,
    });

    const snapshot = getMemoryOsSessionSnapshot({ sessionKey: "agent:main:main" });
    expect(snapshot.mid_term_segment_count).toBe(0);
    expect(snapshot.latest_segment_title).toBeNull();
  });
});
