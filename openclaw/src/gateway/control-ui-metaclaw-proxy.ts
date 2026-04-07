import type { IncomingMessage, ServerResponse } from "node:http";
import {
  DEFAULT_WEBHOOK_BODY_TIMEOUT_MS,
  readRequestBodyWithLimit,
  requestBodyErrorToText,
  isRequestBodyLimitError,
} from "../infra/http-body.js";
import { resolveFetch } from "../infra/fetch.js";
import { CONTROL_UI_METACLAW_PROXY_PREFIX } from "./control-ui-contract.js";
import { normalizeControlUiBasePath } from "./control-ui-shared.js";

const FALLBACK_METACLAW_UPSTREAM = "http://127.0.0.1:30000";
const DEFAULT_METACLAW_UPSTREAM =
  process.env.OPENCLAW_METACLAW_UPSTREAM?.trim() || FALLBACK_METACLAW_UPSTREAM;
const MAX_PROXY_BODY_BYTES = 1024 * 1024;
const ALLOWED_METHODS = new Set(["GET", "HEAD", "POST", "PUT", "DELETE"]);
const FORWARDED_REQUEST_HEADERS = new Set(["accept", "authorization", "content-type"]);
const FORWARDED_RESPONSE_HEADERS = new Set(["content-type", "cache-control"]);
const METACLAW_UPSTREAM_HEADER = "x-openclaw-metaclaw-upstream";
const proxyFetch = resolveFetch();

function sendPlainText(res: ServerResponse, statusCode: number, body: string) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end(body);
}

function getHeader(req: IncomingMessage, name: string): string | undefined {
  const raw = req.headers[name];
  if (Array.isArray(raw)) {
    return raw[0];
  }
  return typeof raw === "string" ? raw : undefined;
}

function resolveProxyPath(basePath: string): string {
  return basePath ? `${basePath}${CONTROL_UI_METACLAW_PROXY_PREFIX}` : CONTROL_UI_METACLAW_PROXY_PREFIX;
}

function normalizeUpstreamBase(rawValue: string | undefined): URL {
  const resolved = (rawValue ?? DEFAULT_METACLAW_UPSTREAM).trim() || DEFAULT_METACLAW_UPSTREAM;
  let parsed: URL;
  try {
    parsed = new URL(resolved);
  } catch {
    throw new Error("Invalid MetaClaw API URL");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("MetaClaw API URL must use http or https");
  }
  if (parsed.username || parsed.password) {
    throw new Error("MetaClaw API URL must not include credentials");
  }
  parsed.hash = "";
  parsed.pathname = parsed.pathname.replace(/\/+$/, "");
  return parsed;
}

function buildUpstreamUrl(reqUrl: URL, proxyPrefix: string, upstreamBase: URL): string {
  const suffix = reqUrl.pathname.slice(proxyPrefix.length);
  const pathname = `${upstreamBase.pathname}${suffix}`.replace(/\/{2,}/g, "/") || "/";
  const upstreamUrl = new URL(upstreamBase.toString());
  upstreamUrl.pathname = pathname.startsWith("/") ? pathname : `/${pathname}`;
  upstreamUrl.search = reqUrl.search;
  return upstreamUrl.toString();
}

function copyResponseHeaders(res: ServerResponse, upstream: Response) {
  for (const [name, value] of upstream.headers.entries()) {
    if (FORWARDED_RESPONSE_HEADERS.has(name.toLowerCase())) {
      res.setHeader(name, value);
    }
  }
}

async function readProxyBody(req: IncomingMessage): Promise<string | undefined> {
  const method = (req.method ?? "GET").toUpperCase();
  if (method === "GET" || method === "HEAD") {
    return undefined;
  }
  const body = await readRequestBodyWithLimit(req, {
    maxBytes: MAX_PROXY_BODY_BYTES,
    timeoutMs: DEFAULT_WEBHOOK_BODY_TIMEOUT_MS,
  });
  return body.length > 0 ? body : undefined;
}

export async function handleControlUiMetaclawProxyRequest(
  req: IncomingMessage,
  res: ServerResponse,
  opts?: { basePath?: string },
): Promise<boolean> {
  const urlRaw = req.url;
  if (!urlRaw) {
    return false;
  }

  const basePath = normalizeControlUiBasePath(opts?.basePath);
  const proxyPrefix = resolveProxyPath(basePath);
  const reqUrl = new URL(urlRaw, "http://localhost");
  if (reqUrl.pathname !== proxyPrefix && !reqUrl.pathname.startsWith(`${proxyPrefix}/`)) {
    return false;
  }

  const method = (req.method ?? "GET").toUpperCase();
  if (!ALLOWED_METHODS.has(method)) {
    res.statusCode = 405;
    res.setHeader("Allow", Array.from(ALLOWED_METHODS).join(", "));
    sendPlainText(res, 405, "Method Not Allowed");
    return true;
  }

  let upstreamBase: URL;
  try {
    upstreamBase = normalizeUpstreamBase(getHeader(req, METACLAW_UPSTREAM_HEADER));
  } catch (error) {
    sendPlainText(res, 400, error instanceof Error ? error.message : String(error));
    return true;
  }

  let body: string | undefined;
  try {
    body = await readProxyBody(req);
  } catch (error) {
    if (isRequestBodyLimitError(error)) {
      sendPlainText(res, error.statusCode, requestBodyErrorToText(error.code));
      return true;
    }
    sendPlainText(res, 400, error instanceof Error ? error.message : String(error));
    return true;
  }

  const headers = new Headers();
  for (const [name, value] of Object.entries(req.headers)) {
    if (!FORWARDED_REQUEST_HEADERS.has(name.toLowerCase())) {
      continue;
    }
    if (Array.isArray(value)) {
      for (const entry of value) {
        headers.append(name, entry);
      }
      continue;
    }
    if (typeof value === "string") {
      headers.set(name, value);
    }
  }

  if (body != null && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }

  let upstreamResponse: Response;
  try {
    if (!proxyFetch) {
      sendPlainText(res, 500, "Fetch API unavailable in this runtime");
      return true;
    }
    upstreamResponse = await proxyFetch(buildUpstreamUrl(reqUrl, proxyPrefix, upstreamBase), {
      method,
      headers,
      body,
    });
  } catch (error) {
    sendPlainText(res, 502, error instanceof Error ? error.message : "MetaClaw upstream unavailable");
    return true;
  }

  res.statusCode = upstreamResponse.status;
  copyResponseHeaders(res, upstreamResponse);
  const responseBody = await upstreamResponse.arrayBuffer();
  res.end(Buffer.from(responseBody));
  return true;
}
