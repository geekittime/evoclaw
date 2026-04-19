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

    expect(store.sessions["agent:main:main"]?.stmPageIds.length).toBeGreaterThan(0);
    expect(snapshot.short_term_page_count).toBeGreaterThan(0);
    expect(snapshot.mid_term_segment_count).toBeGreaterThan(0);
    expect(snapshot.long_term_note_count).toBeGreaterThan(0);
  });

  it("promotes repeated STM pages into derived MTM segments and then into LTM", () => {
    const sessionKey = "agent:main:main";
    const messages = [
      { role: "user", content: "先检查 temp0 目录，再准备删除其中的 py 文件" },
      { role: "assistant", content: "我先检查目录并确认待删除文件。" },
      {
        role: "toolresult",
        content: [{ type: "toolresult", name: "exec", text: "hh.py\nhi.py\ntemp\n" }],
      },
      { role: "user", content: "继续，说明删除前的风险。" },
      { role: "assistant", content: "删除会影响 hh.py 和 hi.py，需要审批。" },
    ];

    syncMemoryOsShortTermPages({ sessionKey, messages, updatedAt: 1000 });
    let store = loadMemoryOsStore();
    expect(Object.keys(store.segments).length).toBeGreaterThan(0);

    const firstSegment = Object.values(store.segments)[0];
    expect(firstSegment?.source).toBe("derived");

    buildMemoryOsPromptAddition({
      sessionKey,
      queryText: "继续处理 temp0 删除任务，并保留审批上下文",
    });
    buildMemoryOsPromptAddition({
      sessionKey,
      queryText: "继续处理 temp0 删除任务，并保留审批上下文",
    });

    store = loadMemoryOsStore();
    const promoted = Object.values(store.segments).find((segment) => segment.promotedToLongTermAt);
    expect(promoted).toBeTruthy();
    expect(Object.values(store.longTermNotes).some((note) => note.sourceSegmentIds.includes(promoted!.id))).toBe(
      true,
    );
  });

  it("uses a budget-aware scheduler instead of fixed top-k retrieval", () => {
    const sessionKey = "agent:main:main";
    for (let index = 0; index < 6; index += 1) {
      updateMemoryOsFromConversation({
        sessionKey,
        messages: [
          { role: "user", content: `任务 ${index}：检查 repo 并处理临时文件` },
          { role: "assistant", content: `我已完成第 ${index} 次检查。` },
        ],
        summary: `第 ${index} 次任务总结：检查 repo，确认临时文件位置并准备下一步。`,
        taskState: `当前任务状态 ${index}：已确认文件位置，待执行后续操作。`,
        source: "manual",
        updatedAt: 10_000 + index,
      });
    }
    updateMemoryOsFromFeedback({
      sessionKey,
      summary: "继续 repo 清理任务时，优先保留最近的删除风险与待处理文件信息。",
      updatedAt: 20_000,
    });

    const addition = buildMemoryOsPromptAddition({
      sessionKey,
      queryText: "继续 repo 清理任务，重点关注最近确认过的待删文件和风险",
      charBudget: 900,
    });

    expect(addition).toContain("## Memory OS Retrieved Context");
    expect(addition!.length).toBeLessThan(1_600);
    expect(addition).toContain("### Mid-Term Episodic Memory");
    expect(addition).toContain("### Long-Term Memory");
  });

  it("derives runtime memory budget from token budget when char budget is not provided", () => {
    const sessionKey = "agent:main:main";
    updateMemoryOsFromConversation({
      sessionKey,
      messages: [
        { role: "user", content: "请继续 repo 清理任务，并保留关键文件确认结果。" },
        { role: "assistant", content: "我会保留关键结果并继续。" },
      ],
      summary:
        "我们已经检查了 repo，确认需要保留最近的文件确认结果、删除风险、以及未完成的后续步骤。",
      taskState:
        "当前任务：继续 repo 清理。已完成：确认关键文件。待完成：下一步清理动作和审批。",
      source: "manual",
      updatedAt: 50_000,
    });
    updateMemoryOsFromFeedback({
      sessionKey,
      summary: "继续 repo 清理时，应优先保留最近的文件确认结果和删除风险。",
      updatedAt: 50_100,
    });

    const smaller = buildMemoryOsPromptAddition({
      sessionKey,
      queryText: "继续 repo 清理任务",
      tokenBudget: 512,
    });
    const larger = buildMemoryOsPromptAddition({
      sessionKey,
      queryText: "继续 repo 清理任务",
      tokenBudget: 8192,
    });

    expect(smaller).toBeTruthy();
    expect(larger).toBeTruthy();
    expect(smaller!.length).toBeLessThanOrEqual(larger!.length);
  });

  it("evicts low-retention pages when the page store grows too large", () => {
    const sessionKey = "agent:main:main";
    for (let index = 0; index < 40; index += 1) {
      syncMemoryOsShortTermPages({
        sessionKey,
        messages: [
          { role: "user", content: `用户提问 ${index}` },
          { role: "assistant", content: `回复 ${index}` },
        ],
        updatedAt: index,
      });
    }

    const store = loadMemoryOsStore();
    expect(Object.keys(store.pages).length).toBeLessThanOrEqual(256);
    expect(store.sessions[sessionKey]?.stmPageIds.length).toBeLessThanOrEqual(12);
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

  it("migrates legacy version-1 memory-os data to the new schema", () => {
    fs.writeFileSync(
      path.join(stateDir, "prompt-context", "memory-os.json"),
      JSON.stringify(
        {
          version: 1,
          sessions: {
            "agent:main:main": {
              shortTermPages: [
                {
                  id: "stm-1",
                  title: "legacy page",
                  content: "User: hi\nAssistant: hello",
                  keywords: ["legacy", "page"],
                  updatedAt: 1,
                },
              ],
              segmentIds: ["seg-1"],
              lastUpdatedAt: 1,
              lastCompactedAt: 1,
            },
          },
          segments: {
            "seg-1": {
              id: "seg-1",
              sessionKey: "agent:main:main",
              title: "legacy segment",
              summary: "legacy summary",
              keywords: ["legacy", "summary"],
              createdAt: 1,
              updatedAt: 1,
              accessCount: 0,
              heat: 1,
              source: "manual",
            },
          },
          longTermNotes: {
            "ltm-1": {
              id: "ltm-1",
              content: "legacy note",
              keywords: ["legacy", "note"],
              createdAt: 1,
              updatedAt: 1,
              accessCount: 0,
              sourceSessionKeys: ["agent:main:main"],
            },
          },
        },
        null,
        2,
      ),
      "utf8",
    );

    const store = loadMemoryOsStore();
    expect(store.version).toBe(2);
    expect(Object.keys(store.pages)).toHaveLength(1);
    expect(store.sessions["agent:main:main"]?.stmPageIds).toHaveLength(1);
    expect(store.segments["seg-1"]?.pageIds).toEqual([]);
  });
});
