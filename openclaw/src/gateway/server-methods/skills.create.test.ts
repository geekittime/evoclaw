import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RespondFn } from "./types.js";

const loadConfigMock = vi.fn();
const resolveDefaultAgentIdMock = vi.fn();
const resolveAgentWorkspaceDirMock = vi.fn();
const listAgentIdsMock = vi.fn();
const loadWorkspaceSkillEntriesMock = vi.fn();

vi.mock("../../config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../config/config.js")>();
  return {
    ...actual,
    loadConfig: (...args: unknown[]) => loadConfigMock(...args),
  };
});

vi.mock("../../agents/agent-scope.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../agents/agent-scope.js")>();
  return {
    ...actual,
    resolveDefaultAgentId: (...args: unknown[]) => resolveDefaultAgentIdMock(...args),
    resolveAgentWorkspaceDir: (...args: unknown[]) => resolveAgentWorkspaceDirMock(...args),
    listAgentIds: (...args: unknown[]) => listAgentIdsMock(...args),
  };
});

vi.mock("../../agents/skills.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../agents/skills.js")>();
  return {
    ...actual,
    loadWorkspaceSkillEntries: (...args: unknown[]) => loadWorkspaceSkillEntriesMock(...args),
  };
});

import { skillsHandlers } from "./skills.js";

describe("skills.create", () => {
  const tempDirs: string[] = [];

  beforeEach(() => {
    loadConfigMock.mockReset();
    resolveDefaultAgentIdMock.mockReset();
    resolveAgentWorkspaceDirMock.mockReset();
    listAgentIdsMock.mockReset();
    loadWorkspaceSkillEntriesMock.mockReset();

    loadConfigMock.mockReturnValue({});
    resolveDefaultAgentIdMock.mockReturnValue("main");
    listAgentIdsMock.mockReturnValue(["main"]);
    loadWorkspaceSkillEntriesMock.mockReturnValue([]);
  });

  afterEach(() => {
    for (const dir of tempDirs.splice(0, tempDirs.length)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  function createWorkspaceDir() {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-skills-create-"));
    tempDirs.push(dir);
    return dir;
  }

  it("creates a workspace SKILL.md from title and content", async () => {
    const workspaceDir = createWorkspaceDir();
    resolveAgentWorkspaceDirMock.mockReturnValue(workspaceDir);

    const respond = vi.fn() as unknown as RespondFn;
    await skillsHandlers["skills.create"]({
      params: {
        title: "Repo Safety",
        content: "Always inspect destructive side effects before deleting files.",
      },
      respond,
    } as never);

    const skillsRoot = path.join(workspaceDir, "skills");
    const createdDir = fs.readdirSync(skillsRoot)[0];
    expect(createdDir).toBe("repo-safety");
    const skillFile = path.join(skillsRoot, createdDir!, "SKILL.md");
    const content = fs.readFileSync(skillFile, "utf8");
    expect(content).toContain('name: "Repo Safety"');
    expect(content).toContain(
      'description: "Always inspect destructive side effects before deleting files."',
    );
    expect(content).toContain("# Repo Safety");
    expect(content).toContain("Always inspect destructive side effects before deleting files.");
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        ok: true,
        name: "Repo Safety",
        filePath: skillFile,
      }),
      undefined,
    );
  });

  it("rejects duplicate skill names in the workspace library", async () => {
    const workspaceDir = createWorkspaceDir();
    resolveAgentWorkspaceDirMock.mockReturnValue(workspaceDir);
    loadWorkspaceSkillEntriesMock.mockReturnValue([
      {
        skill: {
          name: "Repo Safety",
        },
      },
    ]);

    const respond = vi.fn() as unknown as RespondFn;
    await skillsHandlers["skills.create"]({
      params: {
        title: "Repo Safety",
        content: "Another version.",
      },
      respond,
    } as never);

    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({
        message: 'skill "Repo Safety" already exists',
      }),
    );
  });
});
