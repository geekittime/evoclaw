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
};

const SETTINGS_KEY = "openclaw.control.metaclaw.v1";

function defaultApiBase() {
  const protocol = typeof window !== "undefined" && window.location.protocol === "https:" ? "https" : "http";
  const hostname = typeof window !== "undefined" ? window.location.hostname : "localhost";
  return `${protocol}://${hostname}:30000`;
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
      apiBase: typeof parsed.apiBase === "string" && parsed.apiBase.trim() ? parsed.apiBase.trim() : defaultApiBase(),
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

async function metaclawRequest<T>(
  state: Pick<MetaclawState, "metaclawApiBase" | "metaclawToken">,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const apiBase = normalizeBase(state.metaclawApiBase);
  if (!apiBase) {
    throw new Error("MetaClaw API URL is empty");
  }
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  if (state.metaclawToken.trim()) {
    headers.set("Authorization", `Bearer ${state.metaclawToken.trim()}`);
  }
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function loadMetaclawState(state: MetaclawState) {
  state.metaclawLoading = true;
  state.metaclawError = null;
  try {
    const [skillsPayload, pendingPayload, policyPayload] = await Promise.all([
      metaclawRequest<MetaclawSkillsPayload>(state, `/v1/skills?session_id=${encodeURIComponent(state.sessionKey)}`),
      metaclawRequest<{ pending: MetaclawPendingApproval[] }>(
        state,
        `/v1/sandbox/pending?session_id=${encodeURIComponent(state.sessionKey)}`,
      ),
      metaclawRequest<MetaclawSandboxPolicy>(state, "/v1/sandbox/whitelist"),
    ]);
    state.metaclawSkills = Array.isArray(skillsPayload.skills) ? skillsPayload.skills : [];
    state.metaclawSelectedSkillNames = Array.isArray(skillsPayload.selected_skill_names)
      ? skillsPayload.selected_skill_names
      : [];
    state.metaclawSelectionCustomized = Boolean(skillsPayload.selection_customized);
    state.metaclawLatestInjectedSkills = Array.isArray(skillsPayload.latest_injected_skills)
      ? skillsPayload.latest_injected_skills
      : [];
    state.metaclawImportantNotes = skillsPayload.important_notes ?? null;
    state.metaclawPendingApprovals = Array.isArray(pendingPayload.pending) ? pendingPayload.pending : [];
    state.metaclawSandboxPolicy = policyPayload ?? null;
    state.metaclawConnected = true;
  } catch (err) {
    state.metaclawConnected = false;
    state.metaclawError = err instanceof Error ? err.message : String(err);
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
  } catch (err) {
    state.metaclawError = err instanceof Error ? err.message : String(err);
  } finally {
    state.metaclawSaving = false;
  }
}

export async function submitMetaclawFeedback(
  state: Pick<MetaclawState, "sessionKey" | "metaclawApiBase" | "metaclawToken">,
  turn: number | null,
  rating: "good" | "bad",
  feedback: string,
) {
  return metaclawRequest<Record<string, unknown>>(state, "/v1/feedback", {
    method: "POST",
    body: JSON.stringify({
      session_id: state.sessionKey,
      turn,
      rating,
      feedback,
    }),
  });
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
  } catch (err) {
    state.metaclawError = err instanceof Error ? err.message : String(err);
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
  } catch (err) {
    state.metaclawError = err instanceof Error ? err.message : String(err);
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
  } catch (err) {
    state.metaclawError = err instanceof Error ? err.message : String(err);
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
  } catch (err) {
    state.metaclawError = err instanceof Error ? err.message : String(err);
  } finally {
    state.metaclawSaving = false;
  }
}
