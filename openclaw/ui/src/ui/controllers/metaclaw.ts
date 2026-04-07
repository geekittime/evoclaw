import { CONTROL_UI_METACLAW_PROXY_PREFIX } from "../../../../src/gateway/control-ui-contract.js";
import { resolveAgentIdFromSessionKey } from "../../../../src/routing/session-key.js";
import type { GatewayBrowserClient } from "../gateway.ts";
import { inferBasePathFromPathname, normalizeBasePath } from "../navigation.ts";
import type { ExecApprovalRequest } from "./exec-approval.ts";
import type { ExecApprovalsFile, ExecApprovalsSnapshot } from "./exec-approvals.ts";

type MetaclawSettings = {
  apiBase: string;
  token: string;
};

export type MetaclawSkillEntry = {
  name: string;
  description: string;
  category: string;
};

export type MetaclawImportantNotes = {
  name: string;
  description: string;
  content: string;
};

export type MetaclawContextSummary = {
  session_id: string;
  content: string;
  has_summary: boolean;
};

export type MetaclawSkillsPayload = {
  skills: MetaclawSkillEntry[];
  selection_customized: boolean;
  selected_skill_names: string[];
  latest_injected_skills: string[];
  important_notes: MetaclawImportantNotes;
  context_summary?: MetaclawContextSummary | null;
};

export type MetaclawPendingApproval = {
  approval_id: string;
  session_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  decisions?: Array<{
    tool_name?: string;
    action?: string;
    reason?: string;
    command?: string;
    paths?: string[];
  }>;
};

export type MetaclawSandboxPolicy = {
  command_allowlist: string[];
  path_allowlist: string[];
  command_rules: Record<string, "allow" | "ask" | "deny">;
  default_command_mode: "allow" | "ask" | "deny";
  path_blocklist: string[];
};

export type MetaclawFeedbackResponse = {
  ok: boolean;
  session_id: string;
  turn: number;
  rating: "good" | "bad";
  skill_updated: boolean;
  skill_name: string;
  skill_description: string;
  skill_content: string;
};

export type MetaclawContextSummaryResponse = {
  ok: boolean;
  session_id: string;
  summary: string;
  has_summary: boolean;
};

export type MetaclawSectionStatus = "idle" | "ready" | "unavailable" | "error";

export type MetaclawSectionState = {
  status: MetaclawSectionStatus;
  message: string | null;
};

export type MetaclawSectionsState = {
  skills: MetaclawSectionState;
  pendingApprovals: MetaclawSectionState;
  sandboxPolicy: MetaclawSectionState;
};

export type MetaclawState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  sessionKey: string;
  metaclawApiBase: string;
  metaclawToken: string;
  metaclawLoading: boolean;
  metaclawSaving: boolean;
  metaclawError: string | null;
  metaclawConnected: boolean;
  metaclawSkills: MetaclawSkillEntry[];
  metaclawSelectedSkillNames: string[];
  metaclawSelectionCustomized: boolean;
  metaclawLatestInjectedSkills: string[];
  metaclawImportantNotes: MetaclawImportantNotes | null;
  metaclawContextSummary: MetaclawContextSummary | null;
  metaclawPendingApprovals: MetaclawPendingApproval[];
  metaclawSandboxPolicy: MetaclawSandboxPolicy | null;
  metaclawSections: MetaclawSectionsState;
  execApprovalQueue?: ExecApprovalRequest[];
};

type SessionsPromptContextGetResult = {
  ok: boolean;
  key: string;
  selectedSkillNames?: string[];
  selectionCustomized?: boolean;
  latestInjectedSkills?: string[];
  importantNotes?: MetaclawImportantNotes | null;
  contextSummary?: MetaclawContextSummary | null;
  feedbackRecords?: unknown[];
  skillSelectionHistory?: unknown[];
};

type SessionsPromptContextFeedbackResult = {
  ok: boolean;
  key: string;
  session_id: string;
  turn: number | null;
  rating: "good" | "bad";
  summary: string;
};

type SkillsStatusResult = {
  skills?: Array<{
    name: string;
    description: string;
    source?: string;
  }>;
};

const SETTINGS_KEY = "openclaw.control.metaclaw.v1";
const METACLAW_UPSTREAM_HEADER = "X-OpenClaw-MetaClaw-Upstream";
const FALLBACK_METACLAW_API_BASE = "http://127.0.0.1:30000";

function resolveBuiltMetaclawApiBase() {
  const fromVite = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env
    ?.VITE_METACLAW_UPSTREAM;
  return (fromVite ?? "").trim();
}

class MetaclawRequestError extends Error {
  readonly statusCode: number | null;
  readonly reachable: boolean;

  constructor(
    message: string,
    opts?: {
      statusCode?: number | null;
      reachable?: boolean;
    },
  ) {
    super(message);
    this.name = "MetaclawRequestError";
    this.statusCode = opts?.statusCode ?? null;
    this.reachable = opts?.reachable ?? false;
  }
}

function createSectionState(
  status: MetaclawSectionStatus = "idle",
  message: string | null = null,
): MetaclawSectionState {
  return { status, message };
}

export function createInitialMetaclawSectionsState(): MetaclawSectionsState {
  return {
    skills: createSectionState(),
    pendingApprovals: createSectionState(),
    sandboxPolicy: createSectionState(),
  };
}

function resolveCurrentAgentId(sessionKey: string) {
  return resolveAgentIdFromSessionKey(sessionKey) || "main";
}

function mapExecApprovalQueue(
  queue: ExecApprovalRequest[] | undefined,
  sessionKey: string,
): MetaclawPendingApproval[] {
  if (!Array.isArray(queue) || queue.length === 0) {
    return [];
  }
  const relevant = queue.filter(
    (entry) =>
      entry.kind === "exec" &&
      (!entry.request.sessionKey || entry.request.sessionKey === sessionKey),
  );
  if (relevant.length === 0) {
    return [];
  }
  const latest = relevant[0]!;
  return [
    {
      approval_id: latest.id,
      session_id: latest.request.sessionKey ?? sessionKey,
      status: "pending",
      created_at: new Date(latest.createdAtMs).toISOString(),
      updated_at: new Date(latest.createdAtMs).toISOString(),
      decisions: [
        {
          tool_name: latest.kind === "plugin" ? latest.pluginTitle ?? "plugin" : "exec",
          action: "ask",
          reason: latest.request.security ?? latest.request.ask ?? "",
          command: latest.request.command,
          paths: latest.request.resolvedPath ? [latest.request.resolvedPath] : undefined,
        },
      ],
    },
  ];
}

function buildSandboxPolicyFromSnapshot(
  snapshot: ExecApprovalsSnapshot | null,
): MetaclawSandboxPolicy {
  const file: ExecApprovalsFile = snapshot?.file ?? {};
  return {
    command_allowlist: [...(file.commandAllowlist ?? [])],
    path_allowlist: [...(file.pathAllowlist ?? [])],
    command_rules: { ...(file.commandRules ?? {}) },
    default_command_mode: file.defaultCommandMode ?? "ask",
    path_blocklist: [...(file.pathBlocklist ?? [])],
  };
}

function applySandboxPolicyToApprovalsFile(
  file: ExecApprovalsFile | undefined,
  policy: MetaclawSandboxPolicy,
): ExecApprovalsFile {
  return {
    ...(file ?? { version: 1 }),
    version: 1,
    commandAllowlist: [...policy.command_allowlist],
    pathAllowlist: [...policy.path_allowlist],
    pathBlocklist: [...policy.path_blocklist],
    defaultCommandMode: policy.default_command_mode,
    commandRules: { ...policy.command_rules },
  };
}

function defaultApiBase() {
  return resolveBuiltMetaclawApiBase() || FALLBACK_METACLAW_API_BASE;
}

function resolveControlUiBasePath() {
  if (typeof window === "undefined") {
    return "";
  }
  const configured = normalizeBasePath(
    (window as Window & { __OPENCLAW_CONTROL_UI_BASE_PATH__?: string })
      .__OPENCLAW_CONTROL_UI_BASE_PATH__ ?? "",
  );
  if (configured) {
    return configured;
  }
  return normalizeBasePath(inferBasePathFromPathname(window.location.pathname));
}

function resolveProxyBase() {
  if (typeof window === "undefined") {
    return CONTROL_UI_METACLAW_PROXY_PREFIX;
  }
  const basePath = resolveControlUiBasePath();
  const proxyPath = basePath
    ? `${basePath}${CONTROL_UI_METACLAW_PROXY_PREFIX}`
    : CONTROL_UI_METACLAW_PROXY_PREFIX;
  return new URL(proxyPath, window.location.origin).toString().replace(/\/+$/, "");
}

export function loadMetaclawSettings(): MetaclawSettings {
  if (typeof window === "undefined") {
    return { apiBase: defaultApiBase(), token: "" };
  }
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    if (!raw) {
      return { apiBase: defaultApiBase(), token: "" };
    }
    const parsed = JSON.parse(raw) as Partial<MetaclawSettings>;
    const storedApiBase =
      typeof parsed.apiBase === "string" && parsed.apiBase.trim() ? parsed.apiBase.trim() : "";
    const resolvedDefault = defaultApiBase();
    return {
      apiBase:
        storedApiBase &&
        !(
          storedApiBase === FALLBACK_METACLAW_API_BASE &&
          resolvedDefault !== FALLBACK_METACLAW_API_BASE
        )
          ? storedApiBase
          : resolvedDefault,
      token: typeof parsed.token === "string" ? parsed.token : "",
    };
  } catch {
    return { apiBase: defaultApiBase(), token: "" };
  }
}

export function persistMetaclawSettings(settings: MetaclawSettings) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function normalizeBase(apiBase: string) {
  return (apiBase || "").trim().replace(/\/+$/, "");
}

function toMetaclawRequestError(error: unknown): MetaclawRequestError {
  if (error instanceof MetaclawRequestError) {
    return error;
  }
  if (error instanceof Error) {
    return new MetaclawRequestError(error.message);
  }
  return new MetaclawRequestError(String(error));
}

function classifySectionError(error: MetaclawRequestError): MetaclawSectionState {
  const lowered = error.message.toLowerCase();
  if (
    error.statusCode === 404 ||
    error.statusCode === 501 ||
    error.statusCode === 503 ||
    lowered.includes("not enabled") ||
    lowered.includes("disabled") ||
    lowered.includes("not available")
  ) {
    return createSectionState("unavailable", error.message);
  }
  return createSectionState("error", error.message);
}

function firstSectionMessage(sections: MetaclawSectionsState): string | null {
  return (
    sections.skills.message ??
    sections.pendingApprovals.message ??
    sections.sandboxPolicy.message ??
    null
  );
}

function updateSkillsState(state: MetaclawState, payload: MetaclawSkillsPayload) {
  state.metaclawSkills = Array.isArray(payload.skills) ? payload.skills : [];
  state.metaclawSelectedSkillNames = Array.isArray(payload.selected_skill_names)
    ? payload.selected_skill_names
    : [];
  state.metaclawSelectionCustomized = Boolean(payload.selection_customized);
  state.metaclawLatestInjectedSkills = Array.isArray(payload.latest_injected_skills)
    ? payload.latest_injected_skills
    : [];
  state.metaclawImportantNotes = payload.important_notes ?? null;
  state.metaclawContextSummary = payload.context_summary ?? null;
}

function retainLatestPendingApproval(
  pending: MetaclawPendingApproval[],
): MetaclawPendingApproval[] {
  if (pending.length === 0) {
    return [];
  }
  // The operator UI only surfaces the newest approval request so stale prompts
  // do not keep resurfacing after a decision is made.
  return [pending[pending.length - 1]!];
}

function sortPendingApprovals(
  pending: MetaclawPendingApproval[],
): MetaclawPendingApproval[] {
  return [...pending].sort((left, right) =>
    String(left.created_at ?? "").localeCompare(String(right.created_at ?? "")),
  );
}

async function fetchPendingApprovals(
  state: Pick<MetaclawState, "sessionKey" | "metaclawApiBase" | "metaclawToken">,
  fallbackSessionIds: string[] = [],
): Promise<MetaclawPendingApproval[]> {
  const attempts: string[] = [];
  const seen = new Set<string>();
  const addAttempt = (sessionId: string) => {
    const normalized = sessionId.trim();
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    attempts.push(normalized);
  };
  addAttempt(state.sessionKey);
  for (const fallbackSessionId of fallbackSessionIds) {
    addAttempt(fallbackSessionId);
  }

  const pendingById = new Map<string, MetaclawPendingApproval>();
  let reachable = false;
  let lastError: unknown = null;

  for (const sessionId of attempts) {
    try {
      const result = await metaclawRequest<{ pending: MetaclawPendingApproval[] }>(
        state,
        `/v1/sandbox/pending?session_id=${encodeURIComponent(sessionId)}`,
      );
      reachable = true;
      for (const item of Array.isArray(result.pending) ? result.pending : []) {
        const approvalId = String(item.approval_id ?? "").trim();
        if (!approvalId) {
          continue;
        }
        pendingById.set(approvalId, item);
      }
    } catch (error) {
      const requestError = toMetaclawRequestError(error);
      lastError = requestError;
      reachable ||= requestError.reachable;
    }
  }

  if (!reachable && lastError) {
    throw toMetaclawRequestError(lastError);
  }

  return sortPendingApprovals([...pendingById.values()]);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function readMetaclawErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as { detail?: unknown; message?: unknown };
      if (typeof payload.detail === "string" && payload.detail.trim()) {
        return payload.detail.trim();
      }
      if (typeof payload.message === "string" && payload.message.trim()) {
        return payload.message.trim();
      }
    } catch {
      // Fall through to raw text below.
    }
  }
  const text = (await response.text()).trim();
  return text || `${response.status} ${response.statusText}`;
}

async function metaclawRequest<T>(
  state: Pick<MetaclawState, "metaclawApiBase" | "metaclawToken">,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const upstreamBase = normalizeBase(state.metaclawApiBase);
  if (!upstreamBase) {
    throw new MetaclawRequestError("MetaClaw API URL is empty");
  }

  const proxyUrl = `${resolveProxyBase()}${path}`;
  const headers = new Headers(init?.headers);
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  if (init?.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set(METACLAW_UPSTREAM_HEADER, upstreamBase);
  if (state.metaclawToken.trim()) {
    headers.set("Authorization", `Bearer ${state.metaclawToken.trim()}`);
  }

  let response: Response;
  try {
    response = await fetch(proxyUrl, {
      ...init,
      headers,
      credentials: "same-origin",
      cache: "no-store",
    });
  } catch (error) {
    const suffix = error instanceof Error && error.message ? ` ${error.message}` : "";
    throw new MetaclawRequestError(
      `Unable to reach MetaClaw via gateway proxy (${proxyUrl}). Check that OpenClaw is serving the proxy route and MetaClaw is reachable at ${upstreamBase}.${suffix}`,
      { reachable: false },
    );
  }

  if (!response.ok) {
    const message = await readMetaclawErrorMessage(response);
    throw new MetaclawRequestError(message, {
      statusCode: response.status,
      reachable: true,
    });
  }

  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new MetaclawRequestError(`MetaClaw returned invalid JSON for ${path}`, {
      statusCode: response.status,
      reachable: true,
    });
  }
}

function applyMutationError(state: MetaclawState, error: unknown) {
  const requestError = toMetaclawRequestError(error);
  state.metaclawConnected = requestError.reachable;
  state.metaclawError = requestError.message;
}

export async function loadMetaclawState(state: MetaclawState) {
  if (!state.client || !state.connected) {
    state.metaclawLoading = false;
    state.metaclawConnected = false;
    state.metaclawError = null;
    state.metaclawSections = createInitialMetaclawSectionsState();
    state.metaclawPendingApprovals = mapExecApprovalQueue(state.execApprovalQueue, state.sessionKey);
    return;
  }
  state.metaclawLoading = true;
  state.metaclawError = null;

  const sections = createInitialMetaclawSectionsState();
  let connected = state.connected;

  try {
    const [skillsStatusResult, promptContextResult, policyResult] = await Promise.allSettled([
      state.client.request<SkillsStatusResult>("skills.status", {
        agentId: resolveCurrentAgentId(state.sessionKey),
      }),
      state.client.request<SessionsPromptContextGetResult>("sessions.promptContext.get", {
        key: state.sessionKey,
      }),
      state.client.request<ExecApprovalsSnapshot>("exec.approvals.get", {}),
    ]);

    if (skillsStatusResult.status === "fulfilled" && promptContextResult.status === "fulfilled") {
      const skillEntries = Array.isArray(skillsStatusResult.value.skills)
        ? skillsStatusResult.value.skills
            .map((skill) => ({
              name: skill.name,
              description: skill.description,
              category: skill.source ?? "workspace",
            }))
            .sort((left, right) => left.name.localeCompare(right.name))
        : [];
      state.metaclawSkills = skillEntries;
      state.metaclawSelectedSkillNames = Array.isArray(promptContextResult.value.selectedSkillNames)
        ? promptContextResult.value.selectedSkillNames
        : [];
      state.metaclawSelectionCustomized = promptContextResult.value.selectionCustomized === true;
      state.metaclawLatestInjectedSkills = Array.isArray(promptContextResult.value.latestInjectedSkills)
        ? promptContextResult.value.latestInjectedSkills
        : [];
      state.metaclawImportantNotes = promptContextResult.value.importantNotes ?? null;
      state.metaclawContextSummary = promptContextResult.value.contextSummary ?? null;
      sections.skills = createSectionState("ready");
      connected = true;
    } else {
      const requestError = toMetaclawRequestError(
        skillsStatusResult.status === "rejected" ? skillsStatusResult.reason : promptContextResult.reason,
      );
      sections.skills = classifySectionError(requestError);
      connected ||= requestError.reachable;
    }

    state.metaclawPendingApprovals = mapExecApprovalQueue(state.execApprovalQueue, state.sessionKey);
    if (state.connected) {
      sections.pendingApprovals = createSectionState("ready");
    } else {
      sections.pendingApprovals = createSectionState("error", "OpenClaw gateway is disconnected.");
    }

    if (policyResult.status === "fulfilled") {
      state.metaclawSandboxPolicy = buildSandboxPolicyFromSnapshot(policyResult.value);
      sections.sandboxPolicy = createSectionState("ready");
      connected = true;
    } else {
      const requestError = toMetaclawRequestError(policyResult.reason);
      sections.sandboxPolicy = classifySectionError(requestError);
      connected ||= requestError.reachable;
    }

    state.metaclawSections = sections;
    state.metaclawConnected = connected;
    state.metaclawError = connected ? null : firstSectionMessage(sections);
  } finally {
    state.metaclawLoading = false;
  }
}

export async function saveMetaclawSkillSelection(
  state: MetaclawState,
  skillNames: string[] | null,
) {
  if (!state.client || !state.connected) {
    throw new Error("OpenClaw gateway is not connected.");
  }
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await state.client.request("sessions.promptContext.skills.set", {
      key: state.sessionKey,
      selectedSkillNames: Array.isArray(skillNames) ? skillNames : [],
      selectionCustomized: true,
    });
    await loadMetaclawState(state);
  } catch (error) {
    state.metaclawError = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.metaclawSaving = false;
  }
}

export async function compactMetaclawConversationHistory(
  state: MetaclawState,
  messages: unknown[],
) {
  if (!state.client || !state.connected) {
    throw new Error("OpenClaw gateway is not connected.");
  }
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    const result = await state.client.request<{
      ok: boolean;
      session_id: string;
      summary: string;
      has_summary: boolean;
    }>("sessions.promptContext.compact", {
      key: state.sessionKey,
      source: "manual",
      instructions:
        Array.isArray(messages) && messages.length > 0
          ? "Summarize the current chat for future continuation while keeping goals, files, approvals, and user preferences."
          : undefined,
    });
    state.metaclawContextSummary = {
      session_id: result.session_id,
      content: result.summary,
      has_summary: result.has_summary,
    };
    await loadMetaclawState(state);
    return result;
  } catch (error) {
    applyMutationError(state, error);
    throw error;
  } finally {
    state.metaclawSaving = false;
  }
}

export async function submitMetaclawFeedback(
  state: Pick<MetaclawState, "sessionKey" | "metaclawApiBase" | "metaclawToken">,
  turn: number | null,
  rating: "good" | "bad",
  feedback: string,
  responseText: string,
  instructionText: string,
  fallbackSessionIds: string[] = [],
) {
  void fallbackSessionIds;
  const client = (state as MetaclawState & { client?: GatewayBrowserClient | null }).client;
  const connected = (state as MetaclawState & { connected?: boolean }).connected;
  if (!client || !connected) {
    throw new Error("OpenClaw gateway is not connected.");
  }
  const result = await client.request<SessionsPromptContextFeedbackResult>(
    "sessions.promptContext.feedback",
    {
      key: state.sessionKey,
      turn,
      rating,
      feedback,
      responseText,
      instructionText,
    },
  );
  return {
    ok: result.ok,
    session_id: result.session_id,
    turn: result.turn ?? 0,
    rating: result.rating,
    skill_updated: true,
    skill_name: "important-notes",
    skill_description: "Durable notes compiled from answer feedback.",
    skill_content: result.summary,
  };
}

export async function resolveMetaclawApproval(
  state: MetaclawState,
  approvalId: string,
  decision: "approve" | "reject",
  fallbackSessionIds: string[] = [],
) {
  void fallbackSessionIds;
  if (!state.client || !state.connected) {
    throw new Error("OpenClaw gateway is not connected.");
  }
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await state.client.request("exec.approval.resolve", {
      id: approvalId,
      decision: decision === "approve" ? "allow-once" : "deny",
    });
    await loadMetaclawState(state);
  } catch (error) {
    state.metaclawError = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.metaclawSaving = false;
  }
}

export async function waitForMetaclawApprovalResolution(
  state: MetaclawState,
  approvalId: string,
  fallbackSessionIds: string[] = [],
  opts: {
    timeoutMs?: number;
    pollIntervalMs?: number;
  } = {},
) {
  void fallbackSessionIds;
  const timeoutMs = opts.timeoutMs ?? 2500;
  const pollIntervalMs = opts.pollIntervalMs ?? 250;
  const startedAt = Date.now();

  while (true) {
    state.metaclawPendingApprovals = mapExecApprovalQueue(state.execApprovalQueue, state.sessionKey);
    if (!state.metaclawPendingApprovals.some((item) => item.approval_id === approvalId)) {
      state.metaclawError = null;
      return;
    }
    if (Date.now() - startedAt >= timeoutMs) {
      throw new Error(`MetaClaw did not consume approval ${approvalId}.`);
    }
    await delay(pollIntervalMs);
  }
}

export async function saveMetaclawSandboxPolicy(
  state: MetaclawState,
  policy: MetaclawSandboxPolicy,
) {
  if (!state.client || !state.connected) {
    throw new Error("OpenClaw gateway is not connected.");
  }
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    const snapshot = await state.client.request<ExecApprovalsSnapshot>("exec.approvals.get", {});
    const file = applySandboxPolicyToApprovalsFile(snapshot.file, policy);
    await state.client.request("exec.approvals.set", {
      file,
      baseHash: snapshot.hash,
    });
    await loadMetaclawState(state);
  } catch (error) {
    state.metaclawError = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    state.metaclawSaving = false;
  }
}

export async function addMetaclawWhitelistEntry(
  state: MetaclawState,
  type: "command" | "path",
  value: string,
) {
  const nextValue = value.trim();
  if (!nextValue) {
    return;
  }
  const current = state.metaclawSandboxPolicy ?? buildSandboxPolicyFromSnapshot(null);
  const next: MetaclawSandboxPolicy =
    type === "command"
      ? {
          ...current,
          command_allowlist: [...new Set([...current.command_allowlist, nextValue])].sort(),
        }
      : {
          ...current,
          path_allowlist: [...new Set([...current.path_allowlist, nextValue])].sort(),
        };
  await saveMetaclawSandboxPolicy(state, next);
}

export async function removeMetaclawWhitelistEntry(
  state: MetaclawState,
  type: "command" | "path",
  value: string,
) {
  const current = state.metaclawSandboxPolicy ?? buildSandboxPolicyFromSnapshot(null);
  const next: MetaclawSandboxPolicy =
    type === "command"
      ? {
          ...current,
          command_allowlist: current.command_allowlist.filter((entry) => entry !== value),
        }
      : {
          ...current,
          path_allowlist: current.path_allowlist.filter((entry) => entry !== value),
        };
  await saveMetaclawSandboxPolicy(state, next);
}
