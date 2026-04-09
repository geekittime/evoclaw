import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  appendGlobalImportantNote,
  buildGlobalImportantNotesPromptAddition,
  loadGlobalImportantNotes,
  resolveGlobalImportantNotesPath,
} from "./global-important-notes.js";

const tempRoots: string[] = [];

function createEnv() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-important-notes-"));
  tempRoots.push(dir);
  return {
    ...process.env,
    OPENCLAW_STATE_DIR: dir,
  };
}

afterEach(() => {
  for (const dir of tempRoots.splice(0, tempRoots.length)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe("global important notes", () => {
  it("seeds the global file from legacy session notes when empty", () => {
    const env = createEnv();
    const loaded = loadGlobalImportantNotes({
      env,
      seedFromLegacyNotes: "Existing legacy note",
    });

    expect(loaded.content).toBe("Existing legacy note");
    expect(fs.existsSync(resolveGlobalImportantNotesPath(env))).toBe(true);
    expect(resolveGlobalImportantNotesPath(env).endsWith("important-notes.md")).toBe(true);
    expect(fs.readFileSync(resolveGlobalImportantNotesPath(env), "utf8")).toContain(
      "Existing legacy note",
    );
  });

  it("appends new feedback summaries and exposes them as a prompt section", () => {
    const env = createEnv();
    appendGlobalImportantNote({
      env,
      seedFromLegacyNotes: "Keep replies concise.",
      summary:
        "Start with a short greeting when the user says hi.\nAvoid skipping the exact command before deletion.",
      updatedAt: 123,
    });

    const loaded = loadGlobalImportantNotes({ env });
    expect(loaded.content).toContain("Keep replies concise.");
    expect(loaded.content).toContain("Start with a short greeting");
    expect(loaded.content).toContain("Avoid skipping the exact command before deletion.");

    const prompt = buildGlobalImportantNotesPromptAddition({ env });
    expect(prompt).toContain("## Important Notes (High Priority)");
    expect(prompt).toContain("You MUST read and follow them on every turn");
    expect(prompt).toContain("Start with a short greeting");
  });

  it("migrates content from the legacy json store into important-notes.md", () => {
    const env = createEnv();
    const promptDir = path.join(String(env.OPENCLAW_STATE_DIR), "prompt-context");
    fs.mkdirSync(promptDir, { recursive: true });
    fs.writeFileSync(
      path.join(promptDir, "important-notes.json"),
      JSON.stringify(
        {
          content: "Legacy JSON note",
          updatedAt: 456,
        },
        null,
        2,
      ),
      "utf8",
    );

    const loaded = loadGlobalImportantNotes({ env });

    expect(loaded.content).toBe("Legacy JSON note");
    expect(fs.readFileSync(resolveGlobalImportantNotesPath(env), "utf8")).toContain(
      "Legacy JSON note",
    );
  });
});
