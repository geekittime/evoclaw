import { CONTROL_UI_METACLAW_PROXY_PREFIX } from "../../../../src/gateway/control-ui-contract.js";
import { inferBasePathFromPathname, normalizeBasePath } from "../navigation.ts";

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

export type MetaclawSkillsPayload = {
  skills: MetaclawSkillEntry[];
  selection_customized: boolean;
  selected_skill_names: string[];
  latest_injected_skills: string[];
  important_notes: MetaclawImportantNotes;
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
  metaclawPendingApprovals: MetaclawPendingApproval[];
  metaclawSandboxPolicy: MetaclawSandboxPolicy | null;
  metaclawSections: MetaclawSectionsState;
};

const SETTINGS_KEY = "openclaw.control.metaclaw.v1";
const METACLAW_UPSTREAM_HEADER = "X-OpenClaw-MetaClaw-Upstream";

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

function defaultApiBase() {
  return "http://127.0.0.1:30000";
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
    return {
      apiBase:
        typeof parsed.apiBase === "string" && parsed.apiBase.trim()
          ? parsed.apiBase.trim()
          : defaultApiBase(),
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
  state.metaclawLoading = true;
  state.metaclawError = null;

  const sections = createInitialMetaclawSectionsState();
  let connected = false;

  try {
    const [skillsResult, pendingResult, policyResult] = await Promise.allSettled([
      metaclawRequest<MetaclawSkillsPayload>(
        state,
        `/v1/skills?session_id=${encodeURIComponent(state.sessionKey)}`,
      ),
      metaclawRequest<{ pending: MetaclawPendingApproval[] }>(
        state,
        `/v1/sandbox/pending?session_id=${encodeURIComponent(state.sessionKey)}`,
      ),
      metaclawRequest<MetaclawSandboxPolicy>(state, "/v1/sandbox/whitelist"),
    ]);

    if (skillsResult.status === "fulfilled") {
      updateSkillsState(state, skillsResult.value);
      sections.skills = createSectionState("ready");
      connected = true;
    } else {
      const requestError = toMetaclawRequestError(skillsResult.reason);
      sections.skills = classifySectionError(requestError);
      connected ||= requestError.reachable;
    }

    if (pendingResult.status === "fulfilled") {
      state.metaclawPendingApprovals = Array.isArray(pendingResult.value.pending)
        ? pendingResult.value.pending
        : [];
      sections.pendingApprovals = createSectionState("ready");
      connected = true;
    } else {
      const requestError = toMetaclawRequestError(pendingResult.reason);
      sections.pendingApprovals = classifySectionError(requestError);
      connected ||= requestError.reachable;
    }

    if (policyResult.status === "fulfilled") {
      state.metaclawSandboxPolicy = policyResult.value ?? null;
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
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await metaclawRequest<MetaclawSkillsPayload>(state, "/v1/skills/selection", {
      method: "PUT",
      body: JSON.stringify({
        session_id: state.sessionKey,
        skill_names: skillNames,
      }),
    });
    await loadMetaclawState(state);
  } catch (error) {
    applyMutationError(state, error);
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
  const attempts: Array<{ sessionId: string; turn: number | null }> = [];
  const seen = new Set<string>();
  const addAttempt = (sessionId: string, attemptTurn: number | null) => {
    const normalizedSessionId = sessionId.trim();
    if (!normalizedSessionId) {
      return;
    }
    const key = `${normalizedSessionId}::${attemptTurn == null ? "none" : attemptTurn}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    attempts.push({ sessionId: normalizedSessionId, turn: attemptTurn });
  };

  addAttempt(state.sessionKey, turn);
  if (turn != null) {
    addAttempt(state.sessionKey, null);
  }
  for (const fallbackSessionId of fallbackSessionIds) {
    addAttempt(fallbackSessionId, turn);
    if (turn != null) {
      addAttempt(fallbackSessionId, null);
    }
  }

  let lastError: unknown = null;
  for (const attempt of attempts) {
    try {
      return await metaclawRequest<MetaclawFeedbackResponse>(state, "/v1/feedback", {
        method: "POST",
        body: JSON.stringify({
          session_id: attempt.sessionId,
          turn: attempt.turn,
          rating,
          feedback,
          response_text: responseText,
          instruction_text: instructionText,
        }),
      });
    } catch (error) {
      const requestError = toMetaclawRequestError(error);
      lastError = requestError;
      const shouldRetry =
        requestError.statusCode === 404 &&
        requestError.message.toLowerCase().includes("target turn record not found");
      if (!shouldRetry) {
        throw requestError;
      }
    }
  }

  throw toMetaclawRequestError(lastError);
}

export async function resolveMetaclawApproval(
  state: MetaclawState,
  approvalId: string,
  decision: "approve" | "reject",
) {
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await metaclawRequest(state, `/v1/sandbox/${decision}`, {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionKey,
        approval_id: approvalId,
      }),
    });
    await loadMetaclawState(state);
  } catch (error) {
    applyMutationError(state, error);
  } finally {
    state.metaclawSaving = false;
  }
}

export async function saveMetaclawSandboxPolicy(
  state: MetaclawState,
  policy: MetaclawSandboxPolicy,
) {
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await metaclawRequest(state, "/v1/sandbox/policy", {
      method: "PUT",
      body: JSON.stringify(policy),
    });
    await loadMetaclawState(state);
  } catch (error) {
    applyMutationError(state, error);
  } finally {
    state.metaclawSaving = false;
  }
}

export async function addMetaclawWhitelistEntry(
  state: MetaclawState,
  type: "command" | "path",
  value: string,
) {
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await metaclawRequest(state, "/v1/sandbox/whitelist", {
      method: "POST",
      body: JSON.stringify({ type, value }),
    });
    await loadMetaclawState(state);
  } catch (error) {
    applyMutationError(state, error);
  } finally {
    state.metaclawSaving = false;
  }
}

export async function removeMetaclawWhitelistEntry(
  state: MetaclawState,
  type: "command" | "path",
  value: string,
) {
  state.metaclawSaving = true;
  state.metaclawError = null;
  try {
    await metaclawRequest(state, "/v1/sandbox/whitelist", {
      method: "DELETE",
      body: JSON.stringify({ type, value }),
    });
    await loadMetaclawState(state);
  } catch (error) {
    applyMutationError(state, error);
  } finally {
    state.metaclawSaving = false;
  }
}
