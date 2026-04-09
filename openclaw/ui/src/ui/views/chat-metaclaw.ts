import { html, nothing, type TemplateResult } from "lit";
import {
  parseAssistantApprovalPromptText,
  type AssistantApprovalPromptCandidate,
} from "../chat/metaclaw-approval.ts";
import { extractText } from "../chat/message-extract.ts";
import { normalizeRoleForGrouping } from "../chat/message-normalizer.ts";
import type {
  MetaclawContextSummary,
  MetaclawFeedbackResponse,
  MetaclawSectionState,
} from "../controllers/metaclaw.ts";
import { icons } from "../icons.ts";
import type { MessageGroup } from "../types/chat-types.ts";

export type ChatMetaclawProps = {
  apiBase: string;
  token: string;
  loading: boolean;
  saving: boolean;
  compactingHistory: boolean;
  connected: boolean;
  error: string | null;
  sections: {
    skills: MetaclawSectionState;
    pendingApprovals: MetaclawSectionState;
    sandboxPolicy: MetaclawSectionState;
  };
  pendingApprovals: Array<{
    approval_id: string;
    created_at: string;
    decisions?: Array<{
      tool_name?: string;
      command?: string;
      reason?: string;
      action?: string;
      paths?: string[];
    }>;
  }>;
  sandboxPolicy: {
    command_allowlist: string[];
    path_allowlist: string[];
    command_rules: Record<string, "allow" | "ask" | "deny">;
    default_command_mode: "allow" | "ask" | "deny";
    path_blocklist: string[];
  } | null;
  skills: Array<{ name: string; description: string; category: string }>;
  selectedSkillNames: string[];
  selectionCustomized: boolean;
  latestInjectedSkills: string[];
  importantNotes: { name: string; description: string; content: string } | null;
  contextSummary: MetaclawContextSummary | null;
  onApiBaseChange: (value: string) => void;
  onTokenChange: (value: string) => void;
  onRefresh: () => void;
  onCompactHistory: () => Promise<void> | void;
  onApprove: (approvalId: string) => Promise<void> | void;
  onReject: (approvalId: string) => Promise<void> | void;
  onSavePolicy: (policy: {
    command_allowlist: string[];
    path_allowlist: string[];
    command_rules: Record<string, "allow" | "ask" | "deny">;
    default_command_mode: "allow" | "ask" | "deny";
    path_blocklist: string[];
  }) => void;
  onAddWhitelistEntry: (type: "command" | "path", value: string) => void;
  onRemoveWhitelistEntry: (type: "command" | "path", value: string) => void;
  onSaveSkillSelection: (skillNames: string[] | null) => void;
  onAddSkill: (name: string, content: string) => Promise<void> | void;
  onSubmitFeedback: (
    turn: number | null,
    rating: "good" | "bad",
    feedback: string,
    responseText: string,
    instructionText: string,
  ) => Promise<MetaclawFeedbackResponse>;
};

export type ChatMetaclawViewState = {
  studioExpanded: boolean;
  dismissedApprovalIds: string[];
  approvalPromptSubmittingId: string | null;
  approvalPromptMessageId: string | null;
  approvalPromptMessage: string;
  feedbackTargetKey: string | null;
  feedbackTargetTurn: number | null;
  feedbackRating: "good" | "bad";
  feedbackText: string;
  feedbackSaving: boolean;
  feedbackMessage: string;
  feedbackMessageTone: "success" | "danger" | null;
  metaclawRuleCommand: string;
  metaclawRuleMode: "allow" | "ask" | "deny";
  metaclawWhitelistCommand: string;
  metaclawWhitelistPath: string;
  metaclawBlockedPath: string;
  metaclawNewSkillName: string;
  metaclawNewSkillContent: string;
  metaclawSkillMessage: string;
  metaclawSkillMessageTone: "success" | "danger" | null;
};

export function createChatMetaclawViewState(): ChatMetaclawViewState {
  return {
    studioExpanded: false,
    dismissedApprovalIds: [],
    approvalPromptSubmittingId: null,
    approvalPromptMessageId: null,
    approvalPromptMessage: "",
    feedbackTargetKey: null,
    feedbackTargetTurn: null,
    feedbackRating: "good",
    feedbackText: "",
    feedbackSaving: false,
    feedbackMessage: "",
    feedbackMessageTone: null,
    metaclawRuleCommand: "",
    metaclawRuleMode: "ask",
    metaclawWhitelistCommand: "",
    metaclawWhitelistPath: "",
    metaclawBlockedPath: "",
    metaclawNewSkillName: "",
    metaclawNewSkillContent: "",
    metaclawSkillMessage: "",
    metaclawSkillMessageTone: null,
  };
}

export function resetChatMetaclawViewState(state: ChatMetaclawViewState) {
  Object.assign(state, createChatMetaclawViewState());
}

function extractAssistantTurn(group: MessageGroup): number | null {
  for (let index = group.messages.length - 1; index >= 0; index -= 1) {
    const message = group.messages[index]?.message as Record<string, unknown> | undefined;
    const rawTurn = message?.turn ?? message?.metaclaw_turn;
    if (typeof rawTurn === "number" && Number.isFinite(rawTurn)) {
      return rawTurn;
    }
    if (typeof rawTurn === "string" && rawTurn.trim()) {
      const parsed = Number(rawTurn);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return null;
}

function sectionBadge(section: MetaclawSectionState, readyLabel = "Live") {
  const label =
    section.status === "ready"
      ? readyLabel
      : section.status === "unavailable"
        ? "Unavailable"
        : section.status === "error"
          ? "Needs attention"
          : "Syncing";
  return html`
    <span class="metaclaw-status-pill metaclaw-status-pill--${section.status}"> ${label} </span>
  `;
}

function renderSectionCallout(section: MetaclawSectionState, fallback: string) {
  if (section.status === "ready" || section.status === "idle") {
    return nothing;
  }
  return html`
    <div class="callout ${section.status === "error" ? "danger" : ""}">
      ${section.message ?? fallback}
    </div>
  `;
}

function sortedCommandRules(
  policy: NonNullable<ChatMetaclawProps["sandboxPolicy"]>,
): Array<[string, "allow" | "ask" | "deny"]> {
  return Object.entries(policy.command_rules).sort(([left], [right]) => left.localeCompare(right));
}

function appendUnique(values: string[], value: string): string[] {
  const next = value.trim();
  if (!next) {
    return values;
  }
  const lowered = next.toLowerCase();
  if (values.some((item) => item.toLowerCase() === lowered)) {
    return values;
  }
  return [...values, next];
}

function toggleSkill(skillName: string, checked: boolean, props: ChatMetaclawProps) {
  const selected = new Set(props.selectedSkillNames);
  if (checked) {
    selected.add(skillName);
  } else {
    selected.delete(skillName);
  }
  props.onSaveSkillSelection([...selected].sort((left, right) => left.localeCompare(right)));
}

function assistantFeedbackLabel(turn: number | null, responseText: string) {
  if (turn != null) {
    return `How was answer #${turn}?`;
  }
  const snippet = responseText.trim().slice(0, 48);
  return snippet ? `How was this answer?` : "How was this answer?";
}

function extractAssistantResponseText(group: MessageGroup): string {
  const parts: string[] = [];
  for (const entry of group.messages) {
    const text = extractText(entry.message);
    if (text?.trim()) {
      parts.push(text.trim());
    }
  }
  return parts.join("\n\n").trim();
}

export const parseMetaclawApprovalPromptCandidate = parseAssistantApprovalPromptText;

function renderConnectionPanel(props: ChatMetaclawProps): TemplateResult {
  const degradedCount = Object.values(props.sections).filter(
    (section) => section.status === "unavailable" || section.status === "error",
  ).length;
  const statusLabel = !props.connected ? "Offline" : degradedCount > 0 ? "Partial" : "Connected";

  return html`
    <section class="metaclaw-panel metaclaw-panel--connection">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Connection</div>
          <div class="metaclaw-panel__sub">This panel is now powered directly by OpenClaw.</div>
        </div>
        <span
          class="metaclaw-status-pill metaclaw-status-pill--${props.connected
            ? degradedCount
              ? "warn"
              : "ready"
            : "error"}"
        >
          ${statusLabel}
        </span>
      </div>
      <div class="metaclaw-field-grid">
        <label class="field">
          <span>Legacy URL</span>
          <input
            class="input"
            .value=${props.apiBase}
            @input=${(event: Event) =>
              props.onApiBaseChange((event.target as HTMLInputElement).value)}
            placeholder="Unused in native OpenClaw mode"
          />
        </label>
        <label class="field">
          <span>Bearer token</span>
          <input
            class="input"
            .value=${props.token}
            @input=${(event: Event) =>
              props.onTokenChange((event.target as HTMLInputElement).value)}
            placeholder="Unused in native OpenClaw mode"
          />
        </label>
      </div>
      <div class="metaclaw-inline-stats">
        <span class="chip">${props.token.trim() ? "Token configured" : "No token"}</span>
        <span class="chip">${props.connected ? "Gateway reachable" : "Gateway unreachable"}</span>
        <span class="chip">${degradedCount} limited section${degradedCount === 1 ? "" : "s"}</span>
      </div>
      ${props.error ? html`<div class="callout danger">${props.error}</div>` : nothing}
    </section>
  `;
}

function renderPendingApprovalsPanel(props: ChatMetaclawProps): TemplateResult {
  const section = props.sections.pendingApprovals;
  return html`
    <section class="metaclaw-panel">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Pending Approvals</div>
          <div class="metaclaw-panel__sub">
            Review commands that are waiting for an explicit operator decision.
          </div>
        </div>
        ${sectionBadge(section)}
      </div>
      ${renderSectionCallout(
        section,
        "Pending approvals are unavailable for this OpenClaw session.",
      )}
      ${section.status === "ready"
        ? props.pendingApprovals.length === 0
          ? html`<div class="metaclaw-empty">No pending approvals.</div>`
          : html`
              <div class="metaclaw-approval-list">
                ${props.pendingApprovals.map(
                  (item) => html`
                    <article class="metaclaw-approval">
                      <div class="metaclaw-approval__head">
                        <div class="mono">${item.approval_id}</div>
                        <div class="muted">${item.created_at}</div>
                      </div>
                      <div class="metaclaw-approval__body">
                        ${(item.decisions ?? []).map(
                          (decision) => html`
                            <div class="metaclaw-approval__decision">
                              <strong>${decision.tool_name ?? "tool"}</strong>
                              <span class="mono"
                                >${decision.command ?? "No command text provided"}</span
                              >
                              ${decision.paths?.length
                                ? html`
                                    <span class="muted"> Paths: ${decision.paths.join(", ")} </span>
                                  `
                                : nothing}
                              <span class="muted">${decision.reason ?? decision.action ?? ""}</span>
                            </div>
                          `,
                        )}
                      </div>
                      <div class="metaclaw-approval__actions">
                        <button
                          class="btn primary"
                          type="button"
                          ?disabled=${props.saving}
                          @click=${async () => {
                            try {
                              await props.onApprove(item.approval_id);
                            } catch {
                              requestUpdate();
                            }
                          }}
                        >
                          Approve
                        </button>
                        <button
                          class="btn danger"
                          type="button"
                          ?disabled=${props.saving}
                          @click=${async () => {
                            try {
                              await props.onReject(item.approval_id);
                            } catch {
                              requestUpdate();
                            }
                          }}
                        >
                          Reject
                        </button>
                      </div>
                    </article>
                  `,
                )}
              </div>
            `
        : nothing}
    </section>
  `;
}

export function renderMetaclawPendingApprovalsInline(
  props: ChatMetaclawProps | undefined,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
  fallbackPrompt: AssistantApprovalPromptCandidate | null = null,
): TemplateResult | typeof nothing {
  const visiblePendingApprovals = props?.pendingApprovals.filter(
    (item) => !viewState.dismissedApprovalIds.includes(item.approval_id),
  );
  const fallbackApprovalId = fallbackPrompt?.approvalId ?? null;
  const fallbackVisible =
    !!fallbackApprovalId && !viewState.dismissedApprovalIds.includes(fallbackApprovalId);
  if (
    !props ||
    viewState.studioExpanded ||
    ((!visiblePendingApprovals || visiblePendingApprovals.length === 0) && !fallbackVisible) ||
    (props.sections.pendingApprovals.status !== "ready" &&
      (!visiblePendingApprovals || visiblePendingApprovals.length === 0) &&
      !fallbackVisible)
  ) {
    return nothing;
  }

  const renderApprovalActions = (approvalId: string) => html`
    <div class="metaclaw-approval__actions">
      <button
        class="btn primary"
        type="button"
        ?disabled=${props.saving}
        @click=${async () => {
          try {
            await props.onApprove(approvalId);
            viewState.dismissedApprovalIds = appendUnique(viewState.dismissedApprovalIds, approvalId);
          } catch {
            requestUpdate();
          }
        }}
      >
        Approve
      </button>
      <button
        class="btn danger"
        type="button"
        ?disabled=${props.saving}
        @click=${async () => {
          try {
            await props.onReject(approvalId);
            viewState.dismissedApprovalIds = appendUnique(viewState.dismissedApprovalIds, approvalId);
          } catch {
            requestUpdate();
          }
        }}
      >
        Reject
      </button>
    </div>
  `;

  if ((!visiblePendingApprovals || visiblePendingApprovals.length === 0) && fallbackPrompt && fallbackVisible) {
    return html`
      <section class="metaclaw-inline-approvals">
        <div class="metaclaw-inline-approvals__head">
          <div>
            <div class="metaclaw-panel__title">Pending Command Approvals</div>
            <div class="metaclaw-panel__sub">
              The agent requested a restricted command. Approve or reject it here without opening
              the session studio.
            </div>
          </div>
          <span class="metaclaw-status-pill metaclaw-status-pill--warn">1 waiting</span>
        </div>
        <div class="metaclaw-approval-list">
          <article class="metaclaw-approval">
            <div class="metaclaw-approval__head">
              <div class="mono">${fallbackPrompt.approvalId}</div>
            </div>
            <div class="metaclaw-approval__body">
              <div class="metaclaw-approval__decision">
                <strong>exec</strong>
                <span class="mono"
                  >${fallbackPrompt.detailText ?? "OpenClaw reported a pending approval."}</span
                >
              </div>
            </div>
            ${renderApprovalActions(fallbackPrompt.approvalId)}
          </article>
        </div>
      </section>
    `;
  }

  return html`
    <section class="metaclaw-inline-approvals">
      <div class="metaclaw-inline-approvals__head">
        <div>
          <div class="metaclaw-panel__title">Pending Command Approvals</div>
          <div class="metaclaw-panel__sub">
            The agent requested restricted commands. Approve or reject them here without opening
            the session studio.
          </div>
        </div>
        <span class="metaclaw-status-pill metaclaw-status-pill--warn">
          ${visiblePendingApprovals.length} waiting
        </span>
      </div>
      <div class="metaclaw-approval-list">
        ${visiblePendingApprovals.map(
          (item) => html`
            <article class="metaclaw-approval">
              <div class="metaclaw-approval__head">
                <div class="mono">${item.approval_id}</div>
                <div class="muted">${item.created_at}</div>
              </div>
              <div class="metaclaw-approval__body">
                ${(item.decisions ?? []).map(
                  (decision) => html`
                    <div class="metaclaw-approval__decision">
                      <strong>${decision.tool_name ?? "tool"}</strong>
                      <span class="mono">${decision.command ?? "No command text provided"}</span>
                      ${decision.paths?.length
                        ? html`<span class="muted">Paths: ${decision.paths.join(", ")}</span>`
                        : nothing}
                      <span class="muted">${decision.reason ?? decision.action ?? ""}</span>
                    </div>
                  `,
                )}
              </div>
              ${renderApprovalActions(item.approval_id)}
            </article>
          `,
        )}
      </div>
    </section>
  `;
}

export function renderMetaclawPendingApprovalPrompt(
  props: ChatMetaclawProps | undefined,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
  fallbackPrompt: MetaclawApprovalPromptCandidate | null = null,
): TemplateResult | typeof nothing {
  if (!props) {
    return nothing;
  }

  const activeApprovalId = fallbackPrompt?.approvalId ?? null;
  if (!activeApprovalId) {
    return nothing;
  }
  const activePendingApproval =
    props.sections.pendingApprovals.status === "ready"
      ? props.pendingApprovals.find((item) => item.approval_id === activeApprovalId) ?? null
      : null;

  const queueCount = activePendingApproval
    ? props.pendingApprovals.filter((item) => item.approval_id !== activeApprovalId).length + 1
    : 1;
  const isSubmitting = viewState.approvalPromptSubmittingId === activeApprovalId;
  const promptMessage =
    viewState.approvalPromptMessageId === activeApprovalId ? viewState.approvalPromptMessage : "";
  const handleDecision = async (decision: "approve" | "reject") => {
    viewState.approvalPromptSubmittingId = activeApprovalId;
    viewState.approvalPromptMessageId = null;
    viewState.approvalPromptMessage = "";
    requestUpdate();
    try {
      await (decision === "approve"
        ? props.onApprove(activeApprovalId)
        : props.onReject(activeApprovalId));
      viewState.dismissedApprovalIds = appendUnique(
        viewState.dismissedApprovalIds,
        activeApprovalId,
      );
    } catch (error) {
      viewState.approvalPromptMessageId = activeApprovalId;
      viewState.approvalPromptMessage = error instanceof Error ? error.message : String(error);
    } finally {
      viewState.approvalPromptSubmittingId = null;
      requestUpdate();
    }
  };

  return html`
    <div class="exec-approval-overlay" role="dialog" aria-modal="true" aria-live="polite">
      <div class="exec-approval-card">
        <div class="exec-approval-header">
          <div>
            <div class="exec-approval-title">OpenClaw approval needed</div>
            <div class="exec-approval-sub">
              ${activePendingApproval
                ? "A sandboxed command is waiting for your decision."
                : "The assistant reported an approval request that still needs a decision."}
            </div>
          </div>
          ${queueCount > 1 ? html`<div class="exec-approval-queue">${queueCount} pending</div>` : nothing}
        </div>
        <div class="exec-approval-command mono">${activeApprovalId}</div>
        ${activePendingApproval
          ? html`
              <div class="metaclaw-approval__body">
                ${(activePendingApproval.decisions ?? []).map(
                  (decision) => html`
                    <div class="metaclaw-approval__decision">
                      <strong>${decision.tool_name ?? "tool"}</strong>
                      <span class="mono">${decision.command ?? "No command text provided"}</span>
                      ${decision.paths?.length
                        ? html`<span class="muted">Paths: ${decision.paths.join(", ")}</span>`
                        : nothing}
                      <span class="muted">${decision.reason ?? decision.action ?? ""}</span>
                    </div>
                  `,
                )}
              </div>
            `
          : html`
              <div class="metaclaw-approval__body">
                <div class="metaclaw-approval__decision">
                  <strong>assistant</strong>
                  <span class="mono"
                    >${fallbackPrompt?.detailText ?? "OpenClaw reported a pending approval."}</span
                  >
                  <span class="muted">${fallbackPrompt?.rawText ?? ""}</span>
                </div>
              </div>
            `}
        <div class="exec-approval-actions">
          <button
            class="btn primary"
            type="button"
            ?disabled=${props.saving || isSubmitting}
            @click=${() => void handleDecision("approve")}
          >
            Approve
          </button>
          <button
            class="btn danger"
            type="button"
            ?disabled=${props.saving || isSubmitting}
            @click=${() => void handleDecision("reject")}
          >
            Reject
          </button>
        </div>
        ${promptMessage ? html`<div class="callout danger">${promptMessage}</div>` : nothing}
      </div>
    </div>
  `;
}

function renderCommandPolicyPanel(
  props: ChatMetaclawProps,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult {
  const section = props.sections.sandboxPolicy;
  const policy = props.sandboxPolicy;
  const rules = policy ? sortedCommandRules(policy) : [];
  const allowCount = rules.filter(([, mode]) => mode === "allow").length;
  const askCount = rules.filter(([, mode]) => mode === "ask").length;
  const denyCount = rules.filter(([, mode]) => mode === "deny").length;

  return html`
    <section class="metaclaw-panel">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Command Policy</div>
          <div class="metaclaw-panel__sub">
            Define the default command behavior, then override specific commands with allow, ask, or
            deny.
          </div>
        </div>
        ${sectionBadge(section)}
      </div>
      ${renderSectionCallout(
        section,
        "Command policy controls are unavailable for this OpenClaw session.",
      )}
      ${policy
        ? html`
            <label class="field">
              <span>Default command mode</span>
              <select
                .value=${policy.default_command_mode}
                @change=${(event: Event) =>
                  props.onSavePolicy({
                    ...policy,
                    default_command_mode: (event.target as HTMLSelectElement).value as
                      | "allow"
                      | "ask"
                      | "deny",
                  })}
              >
                <option value="allow">Allow</option>
                <option value="ask">Ask</option>
                <option value="deny">Deny</option>
              </select>
            </label>
            <div class="metaclaw-inline-stats">
              <span class="chip">Allow ${allowCount}</span>
              <span class="chip">Ask ${askCount}</span>
              <span class="chip chip--danger">Deny ${denyCount}</span>
            </div>
            <div class="metaclaw-rule-list">
              ${rules.length === 0
                ? html`<div class="metaclaw-empty">No command-specific rules yet.</div>`
                : rules.map(
                    ([command, mode]) => html`
                      <div class="metaclaw-rule-row">
                        <div class="metaclaw-rule-row__main">
                          <span class="mono">${command}</span>
                        </div>
                        <div class="metaclaw-rule-row__actions">
                          <select
                            .value=${mode}
                            @change=${(event: Event) => {
                              const nextMode = (event.target as HTMLSelectElement).value as
                                | "allow"
                                | "ask"
                                | "deny";
                              props.onSavePolicy({
                                ...policy,
                                command_rules: {
                                  ...policy.command_rules,
                                  [command]: nextMode,
                                },
                              });
                            }}
                          >
                            <option value="allow">Allow</option>
                            <option value="ask">Ask</option>
                            <option value="deny">Deny</option>
                          </select>
                          <button
                            class="btn btn--ghost"
                            type="button"
                            ?disabled=${props.saving}
                            @click=${() => {
                              const nextRules = { ...policy.command_rules };
                              delete nextRules[command];
                              props.onSavePolicy({
                                ...policy,
                                command_rules: nextRules,
                              });
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    `,
                  )}
            </div>
            <div class="metaclaw-inline-form">
              <input
                class="input"
                .value=${viewState.metaclawRuleCommand}
                @input=${(event: Event) => {
                  viewState.metaclawRuleCommand = (event.target as HTMLInputElement).value;
                  requestUpdate();
                }}
                placeholder="Add a command rule, e.g. pwd"
              />
              <select
                .value=${viewState.metaclawRuleMode}
                @change=${(event: Event) => {
                  viewState.metaclawRuleMode = (event.target as HTMLSelectElement).value as
                    | "allow"
                    | "ask"
                    | "deny";
                  requestUpdate();
                }}
              >
                <option value="allow">Allow</option>
                <option value="ask">Ask</option>
                <option value="deny">Deny</option>
              </select>
              <button
                class="btn"
                type="button"
                ?disabled=${!viewState.metaclawRuleCommand.trim() || props.saving}
                @click=${() => {
                  const command = viewState.metaclawRuleCommand.trim();
                  viewState.metaclawRuleCommand = "";
                  requestUpdate();
                  props.onSavePolicy({
                    ...policy,
                    command_rules: {
                      ...policy.command_rules,
                      [command]: viewState.metaclawRuleMode,
                    },
                  });
                }}
              >
                Save Rule
              </button>
            </div>
          `
        : nothing}
    </section>
  `;
}

function renderAccessListsPanel(
  props: ChatMetaclawProps,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult {
  const section = props.sections.sandboxPolicy;
  const policy = props.sandboxPolicy;

  return html`
    <section class="metaclaw-panel">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Access Lists</div>
          <div class="metaclaw-panel__sub">
            Manage always-allow commands, allowlisted paths, and blocked paths without slash
            commands.
          </div>
        </div>
        ${sectionBadge(section)}
      </div>
      ${renderSectionCallout(
        section,
        "Access list controls are unavailable for this OpenClaw session.",
      )}
      ${policy
        ? html`
            <div class="metaclaw-list-block">
              <div class="metaclaw-list-block__title">Command allowlist</div>
              <div class="metaclaw-chip-group">
                ${policy.command_allowlist.length === 0
                  ? html`<span class="metaclaw-empty">No always-allow commands.</span>`
                  : policy.command_allowlist.map(
                      (command) => html`
                        <button
                          class="chip"
                          type="button"
                          ?disabled=${props.saving}
                          @click=${() => props.onRemoveWhitelistEntry("command", command)}
                        >
                          ${command} ${icons.x}
                        </button>
                      `,
                    )}
              </div>
              <div class="metaclaw-inline-form">
                <input
                  class="input"
                  .value=${viewState.metaclawWhitelistCommand}
                  @input=${(event: Event) => {
                    viewState.metaclawWhitelistCommand = (event.target as HTMLInputElement).value;
                    requestUpdate();
                  }}
                  placeholder="Always allow command"
                />
                <button
                  class="btn"
                  type="button"
                  ?disabled=${!viewState.metaclawWhitelistCommand.trim() || props.saving}
                  @click=${() => {
                    const value = viewState.metaclawWhitelistCommand.trim();
                    viewState.metaclawWhitelistCommand = "";
                    requestUpdate();
                    props.onAddWhitelistEntry("command", value);
                  }}
                >
                  Add
                </button>
              </div>
            </div>

            <div class="metaclaw-list-block">
              <div class="metaclaw-list-block__title">Path allowlist</div>
              <div class="metaclaw-chip-group">
                ${policy.path_allowlist.length === 0
                  ? html`<span class="metaclaw-empty">No allowlisted paths.</span>`
                  : policy.path_allowlist.map(
                      (path) => html`
                        <button
                          class="chip"
                          type="button"
                          ?disabled=${props.saving}
                          @click=${() => props.onRemoveWhitelistEntry("path", path)}
                        >
                          ${path} ${icons.x}
                        </button>
                      `,
                    )}
              </div>
              <div class="metaclaw-inline-form">
                <input
                  class="input"
                  .value=${viewState.metaclawWhitelistPath}
                  @input=${(event: Event) => {
                    viewState.metaclawWhitelistPath = (event.target as HTMLInputElement).value;
                    requestUpdate();
                  }}
                  placeholder="Allow path"
                />
                <button
                  class="btn"
                  type="button"
                  ?disabled=${!viewState.metaclawWhitelistPath.trim() || props.saving}
                  @click=${() => {
                    const value = viewState.metaclawWhitelistPath.trim();
                    viewState.metaclawWhitelistPath = "";
                    requestUpdate();
                    props.onAddWhitelistEntry("path", value);
                  }}
                >
                  Add
                </button>
              </div>
            </div>

            <div class="metaclaw-list-block">
              <div class="metaclaw-list-block__title">Blocked paths</div>
              <div class="metaclaw-chip-group">
                ${policy.path_blocklist.length === 0
                  ? html`<span class="metaclaw-empty">No blocked paths.</span>`
                  : policy.path_blocklist.map(
                      (path) => html`
                        <button
                          class="chip chip--danger"
                          type="button"
                          ?disabled=${props.saving}
                          @click=${() =>
                            props.onSavePolicy({
                              ...policy,
                              path_blocklist: policy.path_blocklist.filter((item) => item !== path),
                            })}
                        >
                          ${path} ${icons.x}
                        </button>
                      `,
                    )}
              </div>
              <div class="metaclaw-inline-form">
                <input
                  class="input"
                  .value=${viewState.metaclawBlockedPath}
                  @input=${(event: Event) => {
                    viewState.metaclawBlockedPath = (event.target as HTMLInputElement).value;
                    requestUpdate();
                  }}
                  placeholder="Block path"
                />
                <button
                  class="btn danger"
                  type="button"
                  ?disabled=${!viewState.metaclawBlockedPath.trim() || props.saving}
                  @click=${() => {
                    const value = viewState.metaclawBlockedPath.trim();
                    viewState.metaclawBlockedPath = "";
                    requestUpdate();
                    props.onSavePolicy({
                      ...policy,
                      path_blocklist: appendUnique(policy.path_blocklist, value),
                    });
                  }}
                >
                  Block
                </button>
              </div>
            </div>
          `
        : nothing}
    </section>
  `;
}

function renderSkillsPanel(
  props: ChatMetaclawProps,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult {
  const section = props.sections.skills;
  const selected = new Set(props.selectedSkillNames);
  const activeCount = props.selectedSkillNames.length;
  const customSkillCount = props.skills.filter((skill) => skill.category === "session").length;

  return html`
    <section class="metaclaw-panel metaclaw-panel--skills">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Skills</div>
          <div class="metaclaw-panel__sub">
            Enable the skills this chat session should inject into the prompt.
          </div>
        </div>
        ${sectionBadge(section)}
      </div>
      ${renderSectionCallout(section, "Skill management is unavailable for this OpenClaw session.")}
      ${section.status === "ready"
        ? html`
            <div class="metaclaw-inline-stats">
              <span class="chip">${activeCount === 0 ? "No skills selected" : "Custom selection"}</span>
              <span class="chip">${activeCount} / ${props.skills.length} active</span>
              <span class="chip">${customSkillCount} custom</span>
            </div>
            ${props.latestInjectedSkills.length
              ? html`
                  <div class="metaclaw-list-block">
                    <div class="metaclaw-list-block__title">Last injected</div>
                    <div class="metaclaw-chip-group">
                      ${props.latestInjectedSkills.map(
                        (skill) => html`<span class="chip">${skill}</span>`,
                      )}
                    </div>
                  </div>
                `
              : nothing}
            <div class="metaclaw-skill-list metaclaw-skill-list--scrollable">
              ${props.skills.length === 0
                ? html`<div class="metaclaw-empty">No skills available.</div>`
                : props.skills.map(
                    (skill) => html`
                      <label class="metaclaw-skill">
                        <input
                          type="checkbox"
                          .checked=${selected.has(skill.name)}
                          @change=${(event: Event) =>
                            toggleSkill(
                              skill.name,
                              (event.target as HTMLInputElement).checked,
                              props,
                            )}
                        />
                        <span class="metaclaw-skill__body">
                          <strong>${skill.name}</strong>
                          <small>${skill.category}</small>
                          <span>${skill.description}</span>
                        </span>
                      </label>
                    `,
                  )}
            </div>
            <div class="metaclaw-approval__actions">
              <button
                class="btn"
                type="button"
                ?disabled=${props.saving}
                @click=${() =>
                  props.onSaveSkillSelection(
                    props.skills
                      .map((skill) => skill.name)
                      .sort((left, right) => left.localeCompare(right)),
                  )}
              >
                Use All Skills
              </button>
              <button
                class="btn"
                type="button"
                ?disabled=${props.saving}
                @click=${() => props.onSaveSkillSelection([])}
              >
                Disable All
              </button>
            </div>
            ${renderSkillComposer(props, viewState, requestUpdate)}
          `
        : nothing}
    </section>
  `;
}

function renderSkillComposer(
  props: ChatMetaclawProps,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult {
  return html`
    <div class="metaclaw-list-block">
      <div class="metaclaw-list-block__title">Add Custom Skill</div>
      <div class="metaclaw-list-block__sub">
        Create a session-only skill and immediately make it available in this prompt.
      </div>
      <div class="metaclaw-inline-form metaclaw-inline-form--stack">
        <input
          class="input"
          .value=${viewState.metaclawNewSkillName}
          @input=${(event: Event) => {
            viewState.metaclawNewSkillName = (event.target as HTMLInputElement).value;
            requestUpdate();
          }}
          placeholder="Skill name"
        />
        <textarea
          class="metaclaw-feedback__input metaclaw-skill-composer__content"
          .value=${viewState.metaclawNewSkillContent}
          @input=${(event: Event) => {
            viewState.metaclawNewSkillContent = (event.target as HTMLTextAreaElement).value;
            requestUpdate();
          }}
          placeholder="Skill content"
        ></textarea>
        <div class="metaclaw-approval__actions">
          <button
            class="btn"
            type="button"
            ?disabled=${!viewState.metaclawNewSkillName.trim() ||
            !viewState.metaclawNewSkillContent.trim() ||
            props.saving}
            @click=${async () => {
              viewState.metaclawSkillMessage = "";
              viewState.metaclawSkillMessageTone = null;
              requestUpdate();
              try {
                await props.onAddSkill(
                  viewState.metaclawNewSkillName.trim(),
                  viewState.metaclawNewSkillContent.trim(),
                );
                viewState.metaclawNewSkillName = "";
                viewState.metaclawNewSkillContent = "";
                viewState.metaclawSkillMessage = "Skill added successfully.";
                viewState.metaclawSkillMessageTone = "success";
              } catch (error) {
                viewState.metaclawSkillMessage = error instanceof Error ? error.message : String(error);
                viewState.metaclawSkillMessageTone = "danger";
              }
              requestUpdate();
            }}
          >
            Add Skill
          </button>
        </div>
      </div>
      ${viewState.metaclawSkillMessage
        ? html`
            <div
              class="callout ${viewState.metaclawSkillMessageTone === "danger" ? "danger" : ""}"
            >
              ${viewState.metaclawSkillMessage}
            </div>
          `
        : nothing}
    </div>
  `;
}

function renderNotesPanel(props: ChatMetaclawProps): TemplateResult {
  return html`
    <section class="metaclaw-panel metaclaw-panel--notes">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">
            ${props.importantNotes?.name ?? "important-notes"}
          </div>
          <div class="metaclaw-panel__sub">
            ${props.importantNotes?.description ??
            "Persistent notes distilled from user feedback on previous answers."}
          </div>
        </div>
        ${sectionBadge(props.sections.skills, "Synced")}
      </div>
      <pre class="metaclaw-notes">
${props.importantNotes?.content ?? "No important notes yet."}</pre
      >
    </section>
  `;
}

function renderContextSummaryPanel(props: ChatMetaclawProps): TemplateResult {
  const summary = props.contextSummary?.content?.trim() ?? "";
  return html`
    <section class="metaclaw-panel metaclaw-panel--notes">
      <div class="metaclaw-panel__head">
        <div>
          <div class="metaclaw-panel__title">Compressed History</div>
          <div class="metaclaw-panel__sub">
            Manual conversation compression is injected into future prompts for this session.
          </div>
        </div>
        <span class="metaclaw-status-pill metaclaw-status-pill--${summary ? "ready" : "idle"}">
          ${summary ? "Active" : "Empty"}
        </span>
      </div>
      <pre class="metaclaw-notes">
${summary || "No compressed history has been stored for this session yet."}</pre
      >
    </section>
  `;
}

export function renderMetaclawStudio(
  props: ChatMetaclawProps | undefined,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult | typeof nothing {
  if (!props) {
    return nothing;
  }

  const pendingCount = props.pendingApprovals.length;
  const activeSkillCount = props.selectedSkillNames.length;
  const blockedPathCount = props.sandboxPolicy?.path_blocklist.length ?? 0;
  const studioExpanded = viewState.studioExpanded;
  const toggleLabel = studioExpanded ? "Hide Session Studio" : "Show Session Studio";

  return html`
    <div class="metaclaw-toggle-row">
      <button
        class="metaclaw-toggle"
        type="button"
        aria-expanded=${studioExpanded ? "true" : "false"}
        aria-controls="metaclaw-studio-panel"
        @click=${() => {
          viewState.studioExpanded = !viewState.studioExpanded;
          requestUpdate();
        }}
      >
        <span class="metaclaw-toggle__label">
          ${studioExpanded ? icons.panelLeftClose : icons.panelLeftOpen}
          ${toggleLabel}
        </span>
        <span class="metaclaw-toggle__meta">
          ${pendingCount} approvals · ${activeSkillCount}/${props.skills.length} skills ·
          ${blockedPathCount} blocked paths
        </span>
      </button>
    </div>

    ${studioExpanded
      ? html`
          <section id="metaclaw-studio-panel" class="metaclaw-studio">
            <div class="metaclaw-studio__banner">
              <div>
                <div class="metaclaw-studio__eyebrow">OpenClaw Session Studio</div>
                <div class="metaclaw-studio__title">
                  Feedback, approvals, prompt skills, and safety policy
                </div>
                <div class="metaclaw-studio__subtitle">
                  Give answer-level feedback next to assistant messages, approve commands with
                  buttons, and tune session rules without slash commands.
                </div>
              </div>
              <div class="metaclaw-studio__actions">
                <span
                  class="metaclaw-status-pill metaclaw-status-pill--${props.connected
                    ? "ready"
                    : "error"}"
                >
                  ${props.connected ? "OpenClaw ready" : "OpenClaw offline"}
                </span>
                <button
                  class="btn btn--ghost"
                  type="button"
                  ?disabled=${props.loading || props.saving || props.compactingHistory}
                  @click=${props.onCompactHistory}
                >
                  ${props.compactingHistory ? icons.loader : icons.scrollText}
                  ${props.compactingHistory ? "Compressing" : "Compress history"}
                </button>
                <button
                  class="btn btn--ghost"
                  type="button"
                  ?disabled=${props.loading || props.saving}
                  @click=${props.onRefresh}
                >
                  ${icons.refresh} Refresh
                </button>
              </div>
            </div>

            <div class="metaclaw-kpis">
              <article class="metaclaw-kpi">
                <strong>${pendingCount}</strong>
                <span>Pending command approvals</span>
              </article>
              <article class="metaclaw-kpi">
                <strong>${activeSkillCount}/${props.skills.length}</strong>
                <span>Active skills in this session</span>
              </article>
              <article class="metaclaw-kpi">
                <strong>${blockedPathCount}</strong>
                <span>Blocked paths</span>
              </article>
            </div>

            <div class="metaclaw-studio__grid">
              ${renderPendingApprovalsPanel(props)} ${renderCommandPolicyPanel(
                props,
                viewState,
                requestUpdate,
              )}
              ${renderAccessListsPanel(props, viewState, requestUpdate)} ${renderSkillsPanel(
                props,
                viewState,
                requestUpdate,
              )}
              ${renderNotesPanel(props)} ${renderContextSummaryPanel(props)}
            </div>
          </section>
        `
      : nothing}
  `;
}

function openFeedbackComposer(
  targetKey: string,
  turn: number | null,
  rating: "good" | "bad",
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
) {
  const isSameComposer =
    viewState.feedbackTargetKey === targetKey && viewState.feedbackRating === rating;
  if (isSameComposer) {
    viewState.feedbackTargetKey = null;
    viewState.feedbackTargetTurn = null;
    viewState.feedbackText = "";
    viewState.feedbackMessage = "";
    viewState.feedbackMessageTone = null;
    requestUpdate();
    return;
  }
  viewState.feedbackTargetKey = targetKey;
  viewState.feedbackTargetTurn = turn;
  viewState.feedbackRating = rating;
  viewState.feedbackText = "";
  viewState.feedbackMessage = "";
  viewState.feedbackMessageTone = null;
  requestUpdate();
}

export function renderAssistantFeedback(
  group: MessageGroup,
  instructionText: string,
  props: ChatMetaclawProps | undefined,
  viewState: ChatMetaclawViewState,
  requestUpdate: () => void,
): TemplateResult | typeof nothing {
  if (!props || normalizeRoleForGrouping(group.role) !== "assistant") {
    return nothing;
  }

  const turn = extractAssistantTurn(group);
  const responseText = extractAssistantResponseText(group);
  if (!responseText) {
    return nothing;
  }
  const feedbackKey =
    turn != null ? `turn:${turn}` : `assistant:${group.key}:${group.messages.length}:${responseText.length}`;

  const isOpen = viewState.feedbackTargetKey === feedbackKey;
  const calloutClass =
    viewState.feedbackMessageTone === "danger" ? "callout danger" : "callout success";

  return html`
    <div class="metaclaw-feedback">
      <div class="metaclaw-feedback__bar">
        <div>
          <div class="metaclaw-feedback__title">${assistantFeedbackLabel(turn, responseText)}</div>
          <div class="metaclaw-feedback__sub">
            Feedback is summarized into important-notes for future prompts.
          </div>
        </div>
        <div class="metaclaw-feedback__actions">
          <button
            class="btn ${isOpen && viewState.feedbackRating === "good" ? "primary" : ""}"
            type="button"
            ?disabled=${viewState.feedbackSaving || !props.connected}
            @click=${() =>
              openFeedbackComposer(feedbackKey, turn, "good", viewState, requestUpdate)}
          >
            ${icons.check} Good
          </button>
          <button
            class="btn ${isOpen && viewState.feedbackRating === "bad" ? "danger" : ""}"
            type="button"
            ?disabled=${viewState.feedbackSaving || !props.connected}
            @click=${() =>
              openFeedbackComposer(feedbackKey, turn, "bad", viewState, requestUpdate)}
          >
            ${icons.x} Bad
          </button>
        </div>
      </div>

      ${isOpen
        ? html`
            <div class="metaclaw-feedback__composer">
              <textarea
                class="metaclaw-feedback__input"
                .value=${viewState.feedbackText}
                @input=${(event: Event) => {
                  viewState.feedbackText = (event.target as HTMLTextAreaElement).value;
                  requestUpdate();
                }}
                placeholder=${viewState.feedbackRating === "good"
                  ? "What should the agent keep doing next time?"
                  : "What should the agent do differently next time?"}
              ></textarea>
              <div class="metaclaw-feedback__actions">
                <button
                  class="btn primary"
                  type="button"
                  ?disabled=${viewState.feedbackSaving}
                  @click=${async () => {
                    viewState.feedbackSaving = true;
                    viewState.feedbackMessage = "";
                    viewState.feedbackMessageTone = null;
                    requestUpdate();
                    try {
                      const result = await props.onSubmitFeedback(
                        turn,
                        viewState.feedbackRating,
                        viewState.feedbackText.trim(),
                        responseText,
                        instructionText,
                      );
                      viewState.feedbackText = "";
                      viewState.feedbackMessage = result.skill_updated
                        ? `Summarized into ${result.skill_name || "important-notes"}.`
                        : "Feedback recorded.";
                      viewState.feedbackMessageTone = "success";
                    } catch (error) {
                      viewState.feedbackMessage =
                        error instanceof Error ? error.message : String(error);
                      viewState.feedbackMessageTone = "danger";
                    } finally {
                      viewState.feedbackSaving = false;
                      requestUpdate();
                    }
                  }}
                >
                  Save Feedback
                </button>
                <button
                  class="btn"
                  type="button"
                  ?disabled=${viewState.feedbackSaving}
                  @click=${() => {
                    viewState.feedbackTargetKey = null;
                    viewState.feedbackTargetTurn = null;
                    viewState.feedbackText = "";
                    viewState.feedbackMessage = "";
                    viewState.feedbackMessageTone = null;
                    requestUpdate();
                  }}
                >
                  Close
                </button>
              </div>
              ${viewState.feedbackMessage
                ? html`<div class=${calloutClass}>${viewState.feedbackMessage}</div>`
                : nothing}
            </div>
          `
        : nothing}
    </div>
  `;
}
