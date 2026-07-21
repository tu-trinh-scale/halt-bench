import http from "node:http";
import { URL } from "node:url";

const realBaseUrl = process.env.REAL_LITELLM_URL;
const stats = {
  llm_call_count: 0,
  error_count: 0,
  status_counts: {},
  stripped_params: {},
  recent_errors: [],
};

if (!realBaseUrl) {
  console.error("REAL_LITELLM_URL is required");
  process.exit(1);
}

function sanitizePayload(payload) {
  if (payload && typeof payload === "object") {
    if ("tool_choice" in payload) {
      delete payload.tool_choice;
      stats.stripped_params.tool_choice = (stats.stripped_params.tool_choice || 0) + 1;
    }
    // @ai-sdk/openai-compatible sends maxTokens (camelCase); translate to max_tokens so
    // LiteLLM can forward it correctly to all backends (Anthropic, vLLM, etc. reject maxTokens).
    if ("maxTokens" in payload) {
      if (!("max_tokens" in payload)) {
        payload.max_tokens = payload.maxTokens;
      }
      delete payload.maxTokens;
      stats.stripped_params.maxTokens_translated = (stats.stripped_params.maxTokens_translated || 0) + 1;
    }
    payload.drop_params = true;
    if (typeof payload.model === "string" && payload.model.toLowerCase().includes("gemini")) {
      if ("reasoning" in payload) {
        delete payload.reasoning;
        stats.stripped_params.reasoning = (stats.stripped_params.reasoning || 0) + 1;
      }
      if ("reasoning_effort" in payload) {
        delete payload.reasoning_effort;
        stats.stripped_params.reasoning_effort = (stats.stripped_params.reasoning_effort || 0) + 1;
      }
    }
    // Claude/Anthropic models reject requests whose message list ends with an
    // assistant message ("This model does not support assistant message prefill.
    // The conversation must end with a user message.").
    //
    // OpenCode <=1.14.x uses assistant-prefill continuations between tool-call
    // steps, which works for OpenAI/GPT/Gemini but is a hard 400 for Anthropic.
    // Strip any trailing assistant message here so the conversation always ends
    // with a user or tool message before reaching the Anthropic backend.
    if (
      typeof payload.model === "string" &&
      payload.model.toLowerCase().includes("claude") &&
      Array.isArray(payload.messages) &&
      payload.messages.length > 0
    ) {
      const last = payload.messages[payload.messages.length - 1];
      if (last && last.role === "assistant") {
        payload.messages = payload.messages.slice(0, -1);
        stats.stripped_params.claude_prefill_stripped =
          (stats.stripped_params.claude_prefill_stripped || 0) + 1;
      }
    }
  }
  return payload;
}

function addRecentError(message) {
  const text = String(message || "").slice(0, 2000);
  if (!text) return;
  stats.recent_errors.push(text);
  if (stats.recent_errors.length > 20) stats.recent_errors.shift();
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/__health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    if (req.method === "GET" && req.url === "/__stats") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(stats));
      return;
    }
    const target = new URL(req.url || "/", realBaseUrl);
    const bodyChunks = [];
    for await (const chunk of req) bodyChunks.push(chunk);
    let body = Buffer.concat(bodyChunks).toString("utf8");
    if (body) {
      try {
        body = JSON.stringify(sanitizePayload(JSON.parse(body)));
      } catch {
        // keep raw body for non-json endpoints
      }
    }

    const upstreamHeaders = new Headers(req.headers);
    upstreamHeaders.delete("host");
    upstreamHeaders.delete("content-length");
    upstreamHeaders.delete("connection");
    upstreamHeaders.delete("transfer-encoding");

    const upstream = await fetch(target, {
      method: req.method,
      headers: upstreamHeaders,
      body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
    });
    if (req.method === "POST") {
      stats.llm_call_count += 1;
    }
    stats.status_counts[String(upstream.status)] = (stats.status_counts[String(upstream.status)] || 0) + 1;
    const text = await upstream.text();
    if (upstream.status >= 400) {
      stats.error_count += 1;
      addRecentError(`HTTP ${upstream.status}: ${text}`);
    }
    res.writeHead(upstream.status, Object.fromEntries(upstream.headers.entries()));
    res.end(text);
  } catch (err) {
    stats.error_count += 1;
    addRecentError(`proxy error: ${String(err)}`);
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: String(err) }));
  }
});

server.listen(0, "127.0.0.1", () => {
  const address = server.address();
  if (!address || typeof address === "string") {
    console.error("Unable to bind proxy port");
    process.exit(1);
  }
  process.stdout.write(`PROXY_PORT=${address.port}\n`);
});

