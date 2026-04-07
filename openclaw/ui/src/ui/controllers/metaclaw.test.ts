/* @vitest-environment jsdom */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  compactMetaclawConversationHistory,
  createInitialMetaclawSectionsState,
  loadMetaclawState,
  resolveMetaclawApproval,
  submitMetaclawFeedback,
  waitForMetaclawApprovalResolution,
  type MetaclawState,
} from "./metaclaw.ts";

function createState(): MetaclawState {
  return {
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
  };
}

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

describe("loadMetaclawState", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/openclaw/chat");
    Object.defineProperty(window, "__OPENCLAW_CONTROL_UI_BASE_PATH__", {
      configurable: true,
      value: "/openclaw",
      writable: true,
    });
  });

  it("keeps MetaClaw connected when only sandbox sections are unavailable", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain")) {
        const headers = new Headers(init?.headers);
        expect(headers.get("Content-Type")).toBeNull();
        expect(headers.get("Accept")).toBe("application/json");
        expect(headers.get("X-OpenClaw-MetaClaw-Upstream")).toBe("http://127.0.0.1:30000");
        return jsonResponse({
          skills: [{ name: "security-triage", description: "desc", category: "security" }],
          selection_customized: false,
          selected_skill_names: [],
          latest_injected_skills: ["security-triage"],
          important_notes: {
            name: "important-notes",
            description: "notes",
            content: "Remember the operator preference.",
          },
        });
      }
      if (
        url.endsWith(
          "/openclaw/__openclaw/metaclaw/v1/sandbox/pending?session_id=agent%3Amain%3Amain",
        )
      ) {
        return jsonResponse({ detail: "sandbox approvals are not enabled" }, { status: 503 });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/sandbox/whitelist")) {
        return jsonResponse({ detail: "sandbox whitelist is not enabled" }, { status: 503 });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    await loadMetaclawState(state);

    expect(state.metaclawConnected).toBe(true);
    expect(state.metaclawError).toBeNull();
    expect(state.metaclawSkills).toHaveLength(1);
    expect(state.metaclawImportantNotes?.name).toBe("important-notes");
    expect(state.metaclawSections.skills.status).toBe("ready");
    expect(state.metaclawSections.pendingApprovals.status).toBe("unavailable");
    expect(state.metaclawSections.sandboxPolicy.status).toBe("unavailable");

    vi.unstubAllGlobals();
  });

  it("reports gateway proxy reachability when fetch fails before a response arrives", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    await loadMetaclawState(state);

    expect(state.metaclawConnected).toBe(false);
    expect(state.metaclawError).toContain("Unable to reach MetaClaw via gateway proxy");
    expect(state.metaclawSections.skills.status).toBe("error");

    vi.unstubAllGlobals();
  });

  it("keeps only the latest pending approval in UI state", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain")) {
        return jsonResponse({
          skills: [],
          selection_customized: true,
          selected_skill_names: [],
          latest_injected_skills: [],
          important_notes: null,
        });
      }
      if (
        url.endsWith(
          "/openclaw/__openclaw/metaclaw/v1/sandbox/pending?session_id=agent%3Amain%3Amain",
        )
      ) {
        return jsonResponse({
          pending: [
            {
              approval_id: "appr_old",
              session_id: "agent:main:main",
              status: "pending",
              created_at: "2026-04-06 18:00:00",
              updated_at: "2026-04-06 18:00:00",
            },
            {
              approval_id: "appr_new",
              session_id: "agent:main:main",
              status: "pending",
              created_at: "2026-04-06 18:01:00",
              updated_at: "2026-04-06 18:01:00",
            },
          ],
        });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/sandbox/whitelist")) {
        return jsonResponse({
          command_allowlist: [],
          path_allowlist: [],
          command_rules: {},
          default_command_mode: "ask",
          path_blocklist: [],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });

    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    await loadMetaclawState(state);

    expect(state.metaclawPendingApprovals).toHaveLength(1);
    expect(state.metaclawPendingApprovals[0]?.approval_id).toBe("appr_new");

    vi.unstubAllGlobals();
  });

  it("includes instruction text when submitting answer feedback", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      expect(body).toMatchObject({
        session_id: "agent:main:main",
        turn: 7,
        rating: "bad",
        feedback: "Needs to greet first.",
        response_text: "What would you like to work on today?",
        instruction_text: "hi",
      });
      return jsonResponse({
        ok: true,
        session_id: "agent:main:main",
        turn: 7,
        rating: "bad",
        skill_updated: true,
        skill_name: "important-notes",
        skill_description: "notes",
        skill_content: "Remember to greet first.",
      });
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    const result = await submitMetaclawFeedback(
      state,
      7,
      "bad",
      "Needs to greet first.",
      "What would you like to work on today?",
      "hi",
    );

    expect(result.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    vi.unstubAllGlobals();
  });

  it("compacts visible chat history into the session context summary", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/context-summary/compact")) {
        const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
        expect(body).toMatchObject({
          session_id: "agent:main:main",
        });
        expect(Array.isArray(body.messages)).toBe(true);
        return jsonResponse({
          ok: true,
          session_id: "agent:main:main",
          summary: "Compressed summary text.",
          has_summary: true,
        });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain")) {
        return jsonResponse({
          skills: [],
          selection_customized: true,
          selected_skill_names: [],
          latest_injected_skills: [],
          important_notes: null,
          context_summary: {
            session_id: "agent:main:main",
            content: "Compressed summary text.",
            has_summary: true,
          },
        });
      }
      if (
        url.endsWith(
          "/openclaw/__openclaw/metaclaw/v1/sandbox/pending?session_id=agent%3Amain%3Amain",
        )
      ) {
        return jsonResponse({ pending: [] });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/sandbox/whitelist")) {
        return jsonResponse({
          command_allowlist: [],
          path_allowlist: [],
          command_rules: {},
          default_command_mode: "ask",
          path_blocklist: [],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    const result = await compactMetaclawConversationHistory(state, [
      { role: "user", content: "Please summarize this discussion." },
      { role: "assistant", content: "Here is the latest result." },
    ]);

    expect(result.summary).toBe("Compressed summary text.");
    expect(state.metaclawContextSummary?.content).toBe("Compressed summary text.");

    vi.unstubAllGlobals();
  });

  it("retries feedback against fallback MetaClaw sessions when the primary session has no turn record", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      if (body.session_id === "agent:main:main" && body.turn === 7) {
        return jsonResponse({ detail: "target turn record not found" }, { status: 404 });
      }
      if (body.session_id === "agent:main:main" && body.turn === null) {
        return jsonResponse({ detail: "target turn record not found" }, { status: 404 });
      }
      expect(body).toMatchObject({
        session_id: "tui-deepseek-chat",
        turn: 7,
        rating: "good",
        feedback: "Good answer.",
      });
      return jsonResponse({
        ok: true,
        session_id: "tui-deepseek-chat",
        turn: 7,
        rating: "good",
        skill_updated: true,
        skill_name: "important-notes",
        skill_description: "notes",
        skill_content: "Remember the last preference.",
      });
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    const result = await submitMetaclawFeedback(
      state,
      7,
      "good",
      "Good answer.",
      "Hi there.",
      "hi",
      ["tui-deepseek-chat"],
    );

    expect(result.session_id).toBe("tui-deepseek-chat");
    expect(fetchMock).toHaveBeenCalledTimes(3);

    vi.unstubAllGlobals();
  });

  it("retries approval resolution against fallback MetaClaw sessions when the primary session has no pending record", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/sandbox/approve")) {
        const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
        if (body.session_id === "agent:main:main") {
          return jsonResponse({ detail: "pending approval not found" }, { status: 404 });
        }
        expect(body).toMatchObject({
          session_id: "tui-deepseek-chat",
          approval_id: "appr_505b16cb41c5",
        });
        return jsonResponse({ ok: true });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain")) {
        return jsonResponse({
          skills: [{ name: "security-triage", description: "desc", category: "security" }],
          selection_customized: true,
          selected_skill_names: ["security-triage"],
          latest_injected_skills: ["security-triage"],
          important_notes: {
            name: "important-notes",
            description: "notes",
            content: "Remember the operator preference.",
          },
        });
      }
      if (
        url.endsWith(
          "/openclaw/__openclaw/metaclaw/v1/sandbox/pending?session_id=agent%3Amain%3Amain",
        )
      ) {
        return jsonResponse({ pending: [] });
      }
      if (url.endsWith("/openclaw/__openclaw/metaclaw/v1/sandbox/whitelist")) {
        return jsonResponse({
          command_allowlist: [],
          path_allowlist: [],
          command_rules: {},
          default_command_mode: "ask",
          path_blocklist: [],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    await resolveMetaclawApproval(
      state,
      "appr_505b16cb41c5",
      "approve",
      ["tui-deepseek-chat"],
    );

    expect(state.metaclawError).toBeNull();
    expect(fetchMock).toHaveBeenCalled();

    vi.unstubAllGlobals();
  });

  it("waits until an inline approval disappears from pending state before succeeding", async () => {
    let pendingCalls = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (
        url.endsWith(
          "/openclaw/__openclaw/metaclaw/v1/sandbox/pending?session_id=agent%3Amain%3Amain",
        )
      ) {
        pendingCalls += 1;
        return jsonResponse({
          pending:
            pendingCalls === 1
              ? [
                  {
                    approval_id: "appr_pending",
                    session_id: "agent:main:main",
                    status: "pending",
                    created_at: "2026-04-07 10:00:00",
                    updated_at: "2026-04-07 10:00:00",
                  },
                ]
              : [],
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const state = createState();
    await waitForMetaclawApprovalResolution(state, "appr_pending", [], {
      timeoutMs: 1000,
      pollIntervalMs: 0,
    });

    expect(state.metaclawPendingApprovals).toEqual([]);
    expect(pendingCalls).toBe(2);

    vi.unstubAllGlobals();
  });
});
