import type { IncomingMessage, ServerResponse } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";

describe("handleControlUiMetaclawProxyRequest", () => {
  afterEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
    vi.doUnmock("../infra/fetch.js");
  });

  async function dispatchProxyRequest(
    req: IncomingMessage,
  ): Promise<{ handled: boolean; statusCode: number; headers: Map<string, string>; body: string }> {
    return await new Promise(async (resolve, reject) => {
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
            handled: true,
            statusCode: this.statusCode,
            headers,
            body,
          });
        },
      } as unknown as ServerResponse;

      try {
        const { handleControlUiMetaclawProxyRequest } = await import(
          "./control-ui-metaclaw-proxy.js"
        );
        const handled = await handleControlUiMetaclawProxyRequest(req, res);
        if (!handled) {
          resolve({
            handled: false,
            statusCode: res.statusCode,
            headers,
            body,
          });
        }
      } catch (error) {
        reject(error);
      }
    });
  }

  it("proxies MetaClaw requests through the control-ui proxy handler", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
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
        {
          status: 200,
          headers: {
            "Content-Type": "application/json; charset=utf-8",
          },
        },
      ),
    );

    vi.doMock("../infra/fetch.js", () => ({
      resolveFetch: () => fetchMock,
    }));

    const req = {
      method: "GET",
      url: "/__openclaw/metaclaw/v1/skills?session_id=agent%3Amain%3Amain",
      headers: {
        host: "localhost:18789",
        "x-openclaw-metaclaw-upstream": "http://127.0.0.1:30000",
      },
    } as IncomingMessage;

    const response = await dispatchProxyRequest(req);

    expect(response.handled).toBe(true);
    expect(response.statusCode).toBe(200);
    expect(response.headers.get("content-type")).toContain("application/json");
    expect(JSON.parse(response.body)).toMatchObject({
      important_notes: {
        name: "important-notes",
      },
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:30000/v1/skills?session_id=agent%3Amain%3Amain",
      expect.objectContaining({
        method: "GET",
      }),
    );
  });
});
