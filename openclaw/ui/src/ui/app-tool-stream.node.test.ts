import { beforeAll, describe, expect, it, vi } from "vitest";
import { handleAgentEvent, type FallbackStatus, type ToolStreamEntry } from "./app-tool-stream.ts";
import type { ExecApprovalRequest } from "./controllers/exec-approval.ts";

type ToolStreamHost = Parameters<typeof handleAgentEvent>[0];
type MutableHost = ToolStreamHost & {
  compactionStatus?: unknown;
  compactionClearTimer?: number | null;
  fallbackStatus?: FallbackStatus | null;
  fallbackClearTimer?: number | null;
  execApprovalQueue?: ExecApprovalRequest[];
};

function createHost(overrides?: Partial<MutableHost>): MutableHost {
  return {
    sessionKey: "main",
    chatRunId: null,
    chatStream: null,
    chatStreamStartedAt: null,
    chatStreamSegments: [],
    toolStreamById: new Map<string, ToolStreamEntry>(),
    toolStreamOrder: [],
    chatToolMessages: [],
    toolStreamSyncTimer: null,
    execApprovalQueue: [],
    compactionStatus: null,
    compactionClearTimer: null,
    fallbackStatus: null,
    fallbackClearTimer: null,
    ...overrides,
  };
}

describe("app-tool-stream fallback lifecycle handling", () => {
  beforeAll(() => {
    const globalWithWindow = globalThis as typeof globalThis & {
      window?: Window & typeof globalThis;
    };
    if (!globalWithWindow.window) {
      globalWithWindow.window = globalThis as unknown as Window & typeof globalThis;
    }
  });

  it("accepts session-scoped fallback lifecycle events when no run is active", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: {
        phase: "fallback",
        selectedProvider: "fireworks",
        selectedModel: "fireworks/minimax-m2p5",
        activeProvider: "deepinfra",
        activeModel: "moonshotai/Kimi-K2.5",
        reasonSummary: "rate limit",
      },
    });

    expect(host.fallbackStatus?.selected).toBe("fireworks/minimax-m2p5");
    expect(host.fallbackStatus?.active).toBe("deepinfra/moonshotai/Kimi-K2.5");
    expect(host.fallbackStatus?.reason).toBe("rate limit");
    vi.useRealTimers();
  });

  it("rejects idle fallback lifecycle events for other sessions", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "agent:other:main",
      data: {
        phase: "fallback",
        selectedProvider: "fireworks",
        selectedModel: "fireworks/minimax-m2p5",
        activeProvider: "deepinfra",
        activeModel: "moonshotai/Kimi-K2.5",
      },
    });

    expect(host.fallbackStatus).toBeNull();
    vi.useRealTimers();
  });

  it("auto-clears fallback status after toast duration", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: {
        phase: "fallback",
        selectedProvider: "fireworks",
        selectedModel: "fireworks/minimax-m2p5",
        activeProvider: "deepinfra",
        activeModel: "moonshotai/Kimi-K2.5",
      },
    });

    expect(host.fallbackStatus).not.toBeNull();
    vi.advanceTimersByTime(7_999);
    expect(host.fallbackStatus).not.toBeNull();
    vi.advanceTimersByTime(1);
    expect(host.fallbackStatus).toBeNull();
    vi.useRealTimers();
  });

  it("builds previous fallback label from provider + model on fallback_cleared", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: {
        phase: "fallback_cleared",
        selectedProvider: "fireworks",
        selectedModel: "fireworks/minimax-m2p5",
        activeProvider: "fireworks",
        activeModel: "fireworks/minimax-m2p5",
        previousActiveProvider: "deepinfra",
        previousActiveModel: "moonshotai/Kimi-K2.5",
      },
    });

    expect(host.fallbackStatus?.phase).toBe("cleared");
    expect(host.fallbackStatus?.previous).toBe("deepinfra/moonshotai/Kimi-K2.5");
    vi.useRealTimers();
  });

  it("keeps compaction in retry-pending state until the matching lifecycle end", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "start" },
    });

    expect(host.compactionStatus).toEqual({
      phase: "active",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: null,
    });

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 2,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "end", willRetry: true, completed: true },
    });

    expect(host.compactionStatus).toEqual({
      phase: "retrying",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: null,
    });
    expect(host.compactionClearTimer).toBeNull();

    handleAgentEvent(host, {
      runId: "run-2",
      seq: 3,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "end" },
    });

    expect(host.compactionStatus).toEqual({
      phase: "retrying",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: null,
    });

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 4,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "end" },
    });

    expect(host.compactionStatus).toEqual({
      phase: "complete",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: expect.any(Number),
    });
    expect(host.compactionClearTimer).not.toBeNull();

    vi.advanceTimersByTime(5_000);
    expect(host.compactionStatus).toBeNull();
    expect(host.compactionClearTimer).toBeNull();

    vi.useRealTimers();
  });

  it("treats lifecycle error as terminal for retry-pending compaction", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "start" },
    });

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 2,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "end", willRetry: true, completed: true },
    });

    expect(host.compactionStatus).toEqual({
      phase: "retrying",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: null,
    });

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 3,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "error", error: "boom" },
    });

    expect(host.compactionStatus).toEqual({
      phase: "complete",
      runId: "run-1",
      startedAt: expect.any(Number),
      completedAt: expect.any(Number),
    });
    expect(host.compactionClearTimer).not.toBeNull();

    vi.advanceTimersByTime(5_000);
    expect(host.compactionStatus).toBeNull();
    expect(host.compactionClearTimer).toBeNull();

    vi.useRealTimers();
  });

  it("does not surface retrying or complete when retry compaction failed", () => {
    vi.useFakeTimers();
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 1,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "start" },
    });

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 2,
      stream: "compaction",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "end", willRetry: true, completed: false },
    });

    expect(host.compactionStatus).toBeNull();
    expect(host.compactionClearTimer).toBeNull();

    handleAgentEvent(host, {
      runId: "run-1",
      seq: 3,
      stream: "lifecycle",
      ts: Date.now(),
      sessionKey: "main",
      data: { phase: "error", error: "boom" },
    });

    expect(host.compactionStatus).toBeNull();
    expect(host.compactionClearTimer).toBeNull();

    vi.useRealTimers();
  });
});

describe("app-tool-stream exec approval metadata", () => {
  beforeAll(() => {
    const globalWithWindow = globalThis as typeof globalThis & {
      window?: Window & typeof globalThis;
    };
    if (!globalWithWindow.window) {
      globalWithWindow.window = globalThis as unknown as Window & typeof globalThis;
    }
  });

  it("preserves structured approval metadata in tool stream messages", () => {
    const host = createHost();

    handleAgentEvent(host, {
      runId: "run-approval",
      seq: 1,
      stream: "tool",
      ts: Date.now(),
      sessionKey: "main",
      data: {
        phase: "result",
        toolCallId: "tool-exec-approval",
        name: "exec",
        result: {
          details: {
            status: "approval-pending",
            approvalId: "appr_full_123",
            approvalSlug: "appr1234",
            host: "gateway",
            command: "rm -rf /tmp/demo/*",
            cwd: "/tmp/demo",
            allowedDecisions: ["allow-once", "allow-always", "deny"],
          },
        },
      },
    });

    expect(host.chatToolMessages).toHaveLength(1);
    const message = host.chatToolMessages[0] as { content?: Array<Record<string, unknown>> };
    const toolResult = Array.isArray(message.content)
      ? message.content.find((item) => item.type === "toolresult")
      : null;
    expect(toolResult).toMatchObject({
      approval: {
        approvalId: "appr_full_123",
        approvalSlug: "appr1234",
        host: "gateway",
        command: "rm -rf /tmp/demo/*",
      },
    });
    expect(host.execApprovalQueue).toHaveLength(1);
    expect(host.execApprovalQueue?.[0]).toMatchObject({
      id: "appr_full_123",
      request: {
        command: "rm -rf /tmp/demo/*",
        cwd: "/tmp/demo",
        host: "gateway",
        sessionKey: "main",
      },
    });
  });
});
