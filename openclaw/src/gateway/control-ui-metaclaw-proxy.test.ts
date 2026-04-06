import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { createGatewayRequest } from "./hooks-test-helpers.js";
import { createGatewayHttpServer } from "./server-http.js";

describe("handleControlUiMetaclawProxyRequest", () => {
  const servers: Array<ReturnType<typeof createServer>> = [];

  afterEach(async () => {
    await Promise.all(
      servers.map(
        (server) =>
          new Promise<void>((resolve, reject) => {
            server.closeAllConnections?.();
            server.close((error) => (error ? reject(error) : resolve()));
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
    await new Promise<void>((resolve, reject) =>
      server.listen(0, "127.0.0.1", (error?: Error) => (error ? reject(error) : resolve())),
    );
    return { server, port: (server.address() as AddressInfo).port };
  }

  async function dispatchGatewayRequest(
    server: ReturnType<typeof createGatewayHttpServer>,
    req: IncomingMessage,
  ): Promise<{ statusCode: number; headers: Map<string, string>; body: string }> {
    return await new Promise((resolve, reject) => {
      const headers = new Map<string, string>();
      let body = "";
      const res = {
        headersSent: false,
        statusCode: 200,
        setHeader(name: string, value: string) {
          headers.set(name.toLowerCase(), value);
        },
        end(chunk?: unknown) {
          if (typeof chunk === "string") {
            body = chunk;
          } else if (chunk instanceof Uint8Array) {
            body = Buffer.from(chunk).toString("utf8");
          } else if (chunk != null) {
            body = String(chunk);
          }
          resolve({
            statusCode: this.statusCode,
            headers,
            body,
          });
        },
      } as unknown as ServerResponse;

      try {
        server.emit("request", req, res);
      } catch (error) {
        reject(error);
      }
    });
  }

  it("proxies MetaClaw requests through the gateway request pipeline", async () => {
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

    const req = createGatewayRequest({
      method: "GET",
      path: "/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain",
      headers: {
        "x-openclaw-metaclaw-upstream": `http://127.0.0.1:${upstream.port}`,
      },
    });

    const response = await dispatchGatewayRequest(gateway, req);

    expect(response.statusCode).toBe(200);
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(JSON.parse(response.body)).toMatchObject({
      important_notes: {
        name: "important-notes",
      },
    });
  });
});
