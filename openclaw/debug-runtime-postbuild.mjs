import fs from "node:fs";
import path from "node:path";
import { copyBundledPluginMetadata } from "./scripts/copy-bundled-plugin-metadata.mjs";
import { copyPluginSdkRootAlias } from "./scripts/copy-plugin-sdk-root-alias.mjs";
import { writeTextFileIfChanged } from "./scripts/runtime-postbuild-shared.mjs";
import { stageBundledPluginRuntimeDeps } from "./scripts/stage-bundled-plugin-runtime-deps.mjs";
import { stageBundledPluginRuntime } from "./scripts/stage-bundled-plugin-runtime.mjs";
import { writeOfficialChannelCatalog } from "./scripts/write-official-channel-catalog.mjs";

const ROOT = process.cwd();
const ROOT_RUNTIME_ALIAS_PATTERN = /^(?<base>.+\.(?:runtime|contract))-[A-Za-z0-9_-]+\.js$/u;

function writeStableRootRuntimeAliases(params = {}) {
  const rootDir = params.rootDir ?? ROOT;
  const distDir = path.join(rootDir, "dist");
  const fsImpl = params.fs ?? fs;
  let entries = [];
  try {
    entries = fsImpl.readdirSync(distDir, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const match = entry.name.match(ROOT_RUNTIME_ALIAS_PATTERN);
    if (!match?.groups?.base) continue;
    const aliasPath = path.join(distDir, `${match.groups.base}.js`);
    writeTextFileIfChanged(aliasPath, `export * from "./${entry.name}";\n`);
  }
}

const STATIC_EXTENSION_ASSETS = [
  {
    src: "extensions/acpx/src/runtime-internals/mcp-proxy.mjs",
    dest: "dist/extensions/acpx/mcp-proxy.mjs",
  },
  {
    src: "extensions/diffs/assets/viewer-runtime.js",
    dest: "dist/extensions/diffs/assets/viewer-runtime.js",
  },
];

function copyStaticExtensionAssets(params = {}) {
  const rootDir = params.rootDir ?? ROOT;
  const assets = params.assets ?? STATIC_EXTENSION_ASSETS;
  const fsImpl = params.fs ?? fs;
  const warn = params.warn ?? console.warn;
  for (const { src, dest } of assets) {
    const srcPath = path.join(rootDir, src);
    const destPath = path.join(rootDir, dest);
    if (fsImpl.existsSync(srcPath)) {
      fsImpl.mkdirSync(path.dirname(destPath), { recursive: true });
      fsImpl.copyFileSync(srcPath, destPath);
    } else {
      warn(`[runtime-postbuild] static asset not found, skipping: ${src}`);
    }
  }
}

console.log("1 copyPluginSdkRootAlias");
copyPluginSdkRootAlias({ rootDir: ROOT });

console.log("2 copyBundledPluginMetadata");
copyBundledPluginMetadata({ rootDir: ROOT });

console.log("3 writeOfficialChannelCatalog");
writeOfficialChannelCatalog({ rootDir: ROOT });

console.log("4 stageBundledPluginRuntimeDeps");
stageBundledPluginRuntimeDeps({ rootDir: ROOT });

console.log("5 stageBundledPluginRuntime");
stageBundledPluginRuntime({ rootDir: ROOT });

console.log("6 writeStableRootRuntimeAliases");
writeStableRootRuntimeAliases({ rootDir: ROOT });

console.log("7 copyStaticExtensionAssets");
copyStaticExtensionAssets({ rootDir: ROOT });

console.log("done");
