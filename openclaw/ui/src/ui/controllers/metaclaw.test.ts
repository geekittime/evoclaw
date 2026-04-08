/* @vitest-environment jsdom */

import { describe, expect, it, vi } from "vitest";
import type { GatewayBrowserClient } from "../gateway.ts";
import type { ExecApprovalRequest } from "./exec-approval.ts";
import type { ExecApprovalsSnapshot } from "./exec-approvals.ts";
import {
  compactMetaclawConversationHistory,
  createInitialMetaclawSectionsState,
  loadMetaclawState,
  resolveMetaclawApproval,
  saveMetaclawSandboxPolicy,
  saveMetaclawSkillSelection,
  submitMetaclawFeedback,
  waitForMetaclawApprovalResolution,
  type MetaclawState,
} from "./metaclaw.ts";

function createExecSnapshot(
  overrides: Partial<ExecApprovalsSnapshot> = {},
): ExecApprovalsSnapshot {
  return {
    path: "/tmp/exec-approvals.json",
    exists: true,
    hash: "hash-1",
    file: {
      version: 1,
      commandAllowlist: ["pwd"],
      pathAllowlist: ["/workspace/tmp"],
      pathBlocklist: ["/secret"],
      defaultCommandMode: "ask",
      commandRules: { rm: "deny" },
    },
    ...overrides,
  };
}

function createExecApproval(
  id: string,
  command: string,
  createdAtMs: number,
  sessionKey = "agent:main:main",
): ExecApprovalRequest {
  return {
    id,
    kind: "exec",
    createdAtMs,
    expiresAtMs: createdAtMs + 60_000,
    request: {
      command,
      sessionKey,
      security: "full",
      ask: "always",
      resolvedPath: "/usr/bin/sh",
    },
  };
}

function createClient(
  handler: (method: string, params: Record<string, unknown>) => Promise<unknown> | unknown,
): GatewayBrowserClient {
  return {
    request: vi.fn(async (method: string, params: Record<string, unknown>) => handler(method, params)),
  } as unknown as GatewayBrowserClient;
}

function createState(
  overrides: Partial<MetaclawState> = {},
  handler?: (method: string, params: Record<string, unknown>) => Promise<unknown> | unknown,
): MetaclawState {
  return {
    client: handler ? createClient(handler) : null,
    connected: handler ? true : false,
    sessionKey: "agent:main:main",
    metaclawApiBase: "http://127.0.0.1:30000",
    metaclawToken: "",
    metaclawLoading: false,
    metaclawSaving: false,
    metaclawError: null,
    metaclawConnected: false,
    metaclawSkills: [],
    metaclawSelectedSkillNames: [],
    metaclawSelectionCustomized: false,
    metaclawLatestInjectedSkills: [],
    metaclawImportantNotes: null,
    metaclawContextSummary: null,
    metaclawPendingApprovals: [],
    metaclawSandboxPolicy: null,
    metaclawSections: createInitialMetaclawSectionsState(),
    execApprovalQueue: [],
    ...overrides,
  };
}

describe("metaclaw controller in native OpenClaw mode", () => {
  it("loads skills, notes, summary, approvals, and sandbox policy through OpenClaw RPC", async () => {
    const state = createState({}, async (method) => {
      switch (method) {
        case "skills.status":
          return {
            skills: [
              { name: "security-triage", description: "desc", source: "workspace" },
              { name: "code-review", description: "desc", source: "workspace" },
            ],
          };
        case "sessions.promptContext.get":
          return {
            ok: true,
            key: "agent:main:main",
            selectedSkillNames: ["security-triage"],
            selectionCustomized: true,
            latestInjectedSkills: ["security-triage"],
            importantNotes: {
              name: "important-notes",
              description: "notes",
              content: "Remember the last user preference.",
            },
            contextSummary: {
              session_id: "agent:main:main",
              content: "Compressed summary of the chat so far.",
              has_summary: true,
            },
          };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });

    state.execApprovalQueue = [createExecApproval("appr-1", "rm -rf /tmp/demo", 2_000)];

    await loadMetaclawState(state);

    expect(state.metaclawConnected).toBe(true);
    expect(state.metaclawError).toBeNull();
    expect(state.metaclawSkills.map((skill) => skill.name)).toEqual([
      "code-review",
      "security-triage",
    ]);
    expect(state.metaclawSelectedSkillNames).toEqual(["security-triage"]);
    expect(state.metaclawImportantNotes?.content).toContain("Remember the last user preference.");
    expect(state.metaclawContextSummary?.content).toContain("Compressed summary");
    expect(state.metaclawSandboxPolicy).toEqual({
      command_allowlist: ["pwd"],
      path_allowlist: ["/workspace/tmp"],
      command_rules: { rm: "deny" },
      default_command_mode: "ask",
      path_blocklist: ["/secret"],
    });
    expect(state.metaclawPendingApprovals.map((item) => item.approval_id)).toEqual(["appr-1"]);
    expect(state.metaclawSections.skills.status).toBe("ready");
    expect(state.metaclawSections.pendingApprovals.status).toBe("ready");
    expect(state.metaclawSections.sandboxPolicy.status).toBe("ready");
  });

  it("falls back to empty local state when the OpenClaw gateway is disconnected", async () => {
    const state = createState({
      connected: false,
      execApprovalQueue: [createExecApproval("appr-offline", "pwd", 1_000)],
    });

    await loadMetaclawState(state);

    expect(state.metaclawConnected).toBe(false);
    expect(state.metaclawError).toBeNull();
    expect(state.metaclawSections).toEqual(createInitialMetaclawSectionsState());
    expect(state.metaclawPendingApprovals.map((item) => item.approval_id)).toEqual([
      "appr-offline",
    ]);
  });

  it("surfaces only the newest exec approval for the current session", async () => {
    const state = createState({}, async (method) => {
      switch (method) {
        case "skills.status":
          return { skills: [] };
        case "sessions.promptContext.get":
          return { ok: true, key: "agent:main:main" };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });
    state.execApprovalQueue = [
      createExecApproval("appr-new", "rm -rf /tmp/new", 2_000),
      createExecApproval("appr-old", "rm -rf /tmp/old", 1_000),
      createExecApproval("appr-other-session", "pwd", 3_000, "agent:other:main"),
    ];

    await loadMetaclawState(state);

    expect(state.metaclawPendingApprovals).toHaveLength(1);
    expect(state.metaclawPendingApprovals[0]?.approval_id).toBe("appr-new");
  });

  it("treats legacy main-session approval aliases as the current session", async () => {
    const state = createState({}, async (method) => {
      switch (method) {
        case "skills.status":
          return { skills: [] };
        case "sessions.promptContext.get":
          return { ok: true, key: "agent:main:main" };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });
    state.execApprovalQueue = [createExecApproval("appr-main-alias", "rm -rf /tmp/new", 2_000, "main")];

    await loadMetaclawState(state);

    expect(state.metaclawPendingApprovals).toHaveLength(1);
    expect(state.metaclawPendingApprovals[0]?.approval_id).toBe("appr-main-alias");
  });

  it("persists user-selected skills through the prompt-context RPC", async () => {
    const state = createState({}, async (method, params) => {
      switch (method) {
        case "sessions.promptContext.skills.set":
          expect(params).toEqual({
            key: "agent:main:main",
            selectedSkillNames: ["code-review", "security-triage"],
            selectionCustomized: true,
          });
          return { ok: true };
        case "skills.status":
          return {
            skills: [
              { name: "security-triage", description: "desc", source: "workspace" },
              { name: "code-review", description: "desc", source: "workspace" },
            ],
          };
        case "sessions.promptContext.get":
          return {
            ok: true,
            key: "agent:main:main",
            selectedSkillNames: ["code-review", "security-triage"],
            selectionCustomized: true,
            latestInjectedSkills: ["code-review", "security-triage"],
          };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });

    await saveMetaclawSkillSelection(state, ["code-review", "security-triage"]);

    expect(state.metaclawSelectedSkillNames).toEqual(["code-review", "security-triage"]);
    expect(state.metaclawLatestInjectedSkills).toEqual(["code-review", "security-triage"]);
    expect(state.metaclawSelectionCustomized).toBe(true);
  });

  it("includes instruction text when submitting answer feedback", async () => {
    const state = createState({}, async (method, params) => {
      expect(method).toBe("sessions.promptContext.feedback");
      expect(params).toEqual({
        key: "agent:main:main",
        turn: 7,
        rating: "bad",
        feedback: "Needs to greet first.",
        responseText: "What would you like to work on today?",
        instructionText: "hi",
      });
      return {
        ok: true,
        key: "agent:main:main",
        session_id: "agent:main:main",
        turn: 7,
        rating: "bad",
        summary: "Start with a brief greeting when the user says hi.",
      };
    });

    const result = await submitMetaclawFeedback(
      state,
      7,
      "bad",
      "Needs to greet first.",
      "What would you like to work on today?",
      "hi",
    );

    expect(result.ok).toBe(true);
    expect(result.skill_content).toContain("brief greeting");
  });

  it("compacts the current session transcript into stored context summary", async () => {
    const state = createState({}, async (method, params) => {
      switch (method) {
        case "sessions.promptContext.compact":
          expect(params).toMatchObject({
            key: "agent:main:main",
            source: "manual",
            messages: [
              { role: "user", content: "Please summarize this discussion." },
              { role: "assistant", content: "Here is the latest result." },
            ],
          });
          return {
            ok: true,
            key: "agent:main:main",
            session_id: "agent:main:main",
            summary: "Compressed summary text.",
            has_summary: true,
          };
        case "skills.status":
          return { skills: [] };
        case "sessions.promptContext.get":
          return {
            ok: true,
            key: "agent:main:main",
            contextSummary: {
              session_id: "agent:main:main",
              content: "Compressed summary text.",
              has_summary: true,
            },
          };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });

    const result = await compactMetaclawConversationHistory(state, [
      { role: "user", content: "Please summarize this discussion." },
      { role: "assistant", content: "Here is the latest result." },
    ]);

    expect(result.summary).toBe("Compressed summary text.");
    expect(state.metaclawContextSummary?.content).toBe("Compressed summary text.");
  });

  it("saves command and path policy through exec approvals storage", async () => {
    let savedSnapshot = createExecSnapshot();
    const state = createState({}, async (method, params) => {
      switch (method) {
        case "exec.approvals.get":
          return savedSnapshot;
        case "exec.approvals.set":
          expect(params).toEqual({
            file: {
              version: 1,
              commandAllowlist: ["ls", "pwd"],
              pathAllowlist: ["/workspace/tmp"],
              pathBlocklist: ["/secret", "/tmp/private"],
              defaultCommandMode: "ask",
              commandRules: { rm: "deny", ls: "allow" },
            },
            baseHash: "hash-1",
          });
          savedSnapshot = createExecSnapshot({
            hash: "hash-2",
            file: (params as { file: ExecApprovalsSnapshot["file"] }).file,
          });
          return { ok: true };
        case "skills.status":
          return { skills: [] };
        case "sessions.promptContext.get":
          return { ok: true, key: "agent:main:main" };
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });

    await saveMetaclawSandboxPolicy(state, {
      command_allowlist: ["ls", "pwd"],
      path_allowlist: ["/workspace/tmp"],
      command_rules: { rm: "deny", ls: "allow" },
      default_command_mode: "ask",
      path_blocklist: ["/secret", "/tmp/private"],
    });

    expect(state.metaclawSandboxPolicy).toEqual({
      command_allowlist: ["ls", "pwd"],
      path_allowlist: ["/workspace/tmp"],
      command_rules: { rm: "deny", ls: "allow" },
      default_command_mode: "ask",
      path_blocklist: ["/secret", "/tmp/private"],
    });
  });

  it("resolves approvals through native exec.approval.resolve and waits for queue removal", async () => {
    let resolveCalls = 0;
    const state = createState({}, async (method, params) => {
      switch (method) {
        case "exec.approval.resolve":
          resolveCalls += 1;
          expect(params).toEqual({
            id: "appr_505b16cb41c5",
            decision: "allow-once",
          });
          state.execApprovalQueue = [];
          return { ok: true };
        case "skills.status":
          return { skills: [] };
        case "sessions.promptContext.get":
          return { ok: true, key: "agent:main:main" };
        case "exec.approvals.get":
          return createExecSnapshot();
        default:
          throw new Error(`Unexpected request: ${method}`);
      }
    });
    state.execApprovalQueue = [
      createExecApproval("appr_505b16cb41c5", "rm -rf /tmp/demo", 1_000),
    ];

    await resolveMetaclawApproval(state, "appr_505b16cb41c5", "approve");
    await waitForMetaclawApprovalResolution(state, "appr_505b16cb41c5", [], {
      timeoutMs: 100,
      pollIntervalMs: 0,
    });

    expect(resolveCalls).toBe(1);
    expect(state.metaclawPendingApprovals).toEqual([]);
  });
});
