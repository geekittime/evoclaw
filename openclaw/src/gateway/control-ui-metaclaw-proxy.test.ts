import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { createGatewayHttpServer } from "./server-http.js";

describe("handleControlUiMetaclawProxyRequest", () => {
  const servers: Array<{ close: () => void }> = [];

  afterEach(async () => {
    await Promise.all(
      servers.map(
        (server) =>
          new Promise<void>((resolve) => {
            server.close(() => resolve());
          }),
      ),
    );
    servers.length = 0;
  });

  async function startServer(
    handler: Parameters<typeof createServer>[0],
  ): Promise<{ server: ReturnType<typeof createServer>; port: number }> {
    const server = createServer(handler);
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
    return { server, port: (server.address() as AddressInfo).port };
  }

  it("proxies MetaClaw requests through the gateway on the same origin", async () => {
    const upstream = await startServer((req, res) => {
      if (req.url !== "/v1/skills?session_id=agent%3Amain%3Amain") {
        res.statusCode = 404;
        res.end("unexpected path");
        return;
      }
      res.statusCode = 200;
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(
        JSON.stringify({
          skills: [],
          selection_customized: false,
          selected_skill_names: [],
          latest_injected_skills: [],
          important_notes: {
            name: "important-notes",
            description: "notes",
            content: "",
          },
        }),
      );
    });

    const gateway = createGatewayHttpServer({
      canvasHost: null,
      clients: new Set(),
      controlUiEnabled: true,
      controlUiBasePath: "",
      openAiChatCompletionsEnabled: false,
      openResponsesEnabled: false,
      handleHooksRequest: async () => false,
      resolvedAuth: { mode: "none", allowTailscale: false },
    });
    servers.push(gateway);
    await new Promise<void>((resolve) => gateway.listen(0, "127.0.0.1", () => resolve()));
    const gatewayPort = (gateway.address() as AddressInfo).port;

    const response = await fetch(
      `http://127.0.0.1:${gatewayPort}/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain`,
      {
        headers: {
          "X-OpenClaw-MetaClaw-Upstream": `http://127.0.0.1:${upstream.port}`,
        },
      },
    );

    expect(response.status).toBe(200);
    const payload = (await response.json()) as { important_notes?: { name?: string } };
    expect(payload.important_notes?.name).toBe("important-notes");
  });
});
