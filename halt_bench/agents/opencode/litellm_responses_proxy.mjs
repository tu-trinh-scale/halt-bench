/**
 * Drop-params proxy that also translates /chat/completions → /responses
 * for models whose names contain "gpt" or start with "llama_experimental_reasoning" / "openai".
 *
 * All drop-params behaviour from litellm_drop_params_proxy.mjs is preserved:
 *   - tool_choice stripped
 *   - maxTokens (camelCase) → max_tokens
 *   - drop_params: true injected
 *   - reasoning / reasoning_effort stripped for Gemini models
 *
 * For qualifying models, additionally:
 *   - POST /v1/chat/completions → POST /v1/responses
 *   - messages array translated to responses input format
 *   - max_tokens renamed to max_output_tokens
 *   - Response translated back to chat.completion shape
 *   - If original request had stream:true, a synthetic SSE stream is returned
 */

import http from "node:http";
import { URL } from "node:url";

import { appendFileSync } from "node:fs";

const realBaseUrl = process.env.REAL_LITELLM_URL;
if (!realBaseUrl) {
  console.error("REAL_LITELLM_URL is required");
  process.exit(1);
}

// Optional debug log file — set RESPONSES_PROXY_LOG_FILE env var to enable
const logFile = process.env.RESPONSES_PROXY_LOG_FILE || null;

const stats = {
  llm_call_count: 0,
  responses_call_count: 0,
  chat_call_count: 0,
  error_count: 0,
  status_counts: {},
  stripped_params: {},
  recent_errors: [],
};

function addRecentError(message) {
  const text = String(message || "").slice(0, 2000);
  if (!text) return;
  stats.recent_errors.push(text);
  if (stats.recent_errors.length > 20) stats.recent_errors.shift();
}

function dbgLog(msg) {
  const line = `[responses-proxy] ${msg}\n`;
  process.stderr.write(line);
  if (logFile) {
    try { appendFileSync(logFile, line); } catch {}
  }
}

// ---------------------------------------------------------------------------
// Model filter
// ---------------------------------------------------------------------------

function shouldUseResponses(model) {
  if (!model || typeof model !== "string") return false;
  const m = model.toLowerCase();
  return m.includes("gpt") || m.startsWith("llama_experimental_reasoning") || m.startsWith("openai");
}

// ---------------------------------------------------------------------------
// Drop-params sanitization (identical to litellm_drop_params_proxy.mjs)
// ---------------------------------------------------------------------------

function sanitizePayload(payload) {
  if (!payload || typeof payload !== "object") return payload;
  if ("tool_choice" in payload) {
    delete payload.tool_choice;
    stats.stripped_params.tool_choice = (stats.stripped_params.tool_choice || 0) + 1;
  }
  if ("maxTokens" in payload) {
    if (!("max_tokens" in payload)) payload.max_tokens = payload.maxTokens;
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
  return payload;
}

// ---------------------------------------------------------------------------
// Request translation: chat/completions → responses
// ---------------------------------------------------------------------------

function translateMessages(messages) {
  if (!Array.isArray(messages)) return [];
  const result = [];
  for (const msg of messages) {
    const role = String(msg?.role || "");
    if (role === "tool") {
      // Tool result → top-level function_call_output item
      const content = msg.content;
      const output = typeof content === "string"
        ? content
        : Array.isArray(content)
          ? content.map(c => (typeof c === "string" ? c : (c?.text || JSON.stringify(c)))).join("")
          : JSON.stringify(content ?? "");
      result.push({
        type: "function_call_output",
        call_id: msg.tool_call_id || "",
        output,
      });
    } else if (role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      // Assistant turn with tool calls:
      //   optional text part → assistant message
      //   each tool_call → top-level function_call item
      if (msg.content) {
        result.push({
          role: "assistant",
          content: [{ type: "text", text: String(msg.content) }],
        });
      }
      for (const tc of msg.tool_calls) {
        result.push({
          type: "function_call",
          id: tc.id || "",
          call_id: tc.id || "",
          name: tc.function?.name || "",
          arguments: tc.function?.arguments || "{}",
        });
      }
    } else if (role === "assistant") {
      // Regular assistant text
      const text = typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map(c => (typeof c === "string" ? c : (c?.text || ""))).join("")
          : "";
      result.push({
        role: "assistant",
        content: [{ type: "text", text }],
      });
    } else {
      // user / system — pass through; content can be string or array
      result.push(msg);
    }
  }
  return result;
}

function translateTools(tools) {
  // /chat/completions tools: {type:"function", function:{name,description,parameters}}
  // /responses tools:        {type:"function", name, description, parameters}  (flat)
  if (!Array.isArray(tools)) return tools;
  return tools.map(t => {
    if (t?.type === "function" && t.function && !t.name) {
      const { name, description, parameters, strict } = t.function;
      const flat = { type: "function", name };
      if (description !== undefined) flat.description = description;
      if (parameters !== undefined) flat.parameters = parameters;
      if (strict !== undefined) flat.strict = strict;
      return flat;
    }
    return t; // already flat or unknown — pass through
  });
}

function buildResponsesRequest(chatPayload) {
  const { messages, stream, max_tokens, drop_params, tools, ...rest } = chatPayload;
  const req = {
    ...rest,
    input: translateMessages(messages || []),
    drop_params: true,
  };
  if (max_tokens !== undefined) req.max_output_tokens = max_tokens;
  if (tools !== undefined) req.tools = translateTools(tools);
  return { payload: req, wasStreaming: !!stream };
}

// ---------------------------------------------------------------------------
// Response translation: responses → chat.completion
// ---------------------------------------------------------------------------

function translateResponsesOutput(data, originalModel) {
  const output = Array.isArray(data.output) ? data.output : [];
  let textContent = null;
  const toolCalls = [];

  for (const item of output) {
    if (item?.type === "message") {
      const parts = Array.isArray(item.content) ? item.content : [];
      const texts = parts
        .filter(c => c?.type === "output_text" || c?.type === "text")
        .map(c => c.text || "");
      if (texts.length > 0) textContent = texts.join("");
    } else if (item?.type === "function_call") {
      toolCalls.push({
        id: item.call_id || item.id || "",
        type: "function",
        function: {
          name: item.name || "",
          arguments: item.arguments || "{}",
        },
      });
    }
  }

  const message = { role: "assistant" };
  if (toolCalls.length > 0) {
    message.content = null;
    message.tool_calls = toolCalls;
  } else {
    message.content = textContent ?? "";
  }

  const finishReason = toolCalls.length > 0 ? "tool_calls" : "stop";

  const usageRaw = data.usage || {};
  const usage = {
    prompt_tokens: usageRaw.input_tokens || 0,
    completion_tokens: usageRaw.output_tokens || 0,
    total_tokens: usageRaw.total_tokens || (usageRaw.input_tokens || 0) + (usageRaw.output_tokens || 0),
  };

  return {
    id: String(data.id || "").replace(/^resp_/, "chatcmpl-") || `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: data.model || originalModel,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage,
  };
}

// ---------------------------------------------------------------------------
// SSE stream synthesis (for when the original request had stream:true)
// ---------------------------------------------------------------------------

function buildSseStream(chatCompletion) {
  const msg = chatCompletion.choices[0]?.message || {};
  const base = {
    id: chatCompletion.id,
    object: "chat.completion.chunk",
    created: chatCompletion.created,
    model: chatCompletion.model,
  };

  const parts = [];

  // 1. Role announcement
  parts.push({ ...base, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });

  // 2. Text content chunk (if any)
  if (typeof msg.content === "string" && msg.content) {
    parts.push({ ...base, choices: [{ index: 0, delta: { content: msg.content }, finish_reason: null }] });
  }

  // 3. Tool call chunks
  if (Array.isArray(msg.tool_calls)) {
    for (let i = 0; i < msg.tool_calls.length; i++) {
      const tc = msg.tool_calls[i];
      // Name + id announced first
      parts.push({
        ...base,
        choices: [{
          index: 0,
          delta: {
            tool_calls: [{
              index: i,
              id: tc.id,
              type: "function",
              function: { name: tc.function.name, arguments: "" },
            }],
          },
          finish_reason: null,
        }],
      });
      // Arguments chunk
      if (tc.function.arguments) {
        parts.push({
          ...base,
          choices: [{
            index: 0,
            delta: { tool_calls: [{ index: i, function: { arguments: tc.function.arguments } }] },
            finish_reason: null,
          }],
        });
      }
    }
  }

  // 4. Final stop chunk
  const finishReason = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0
    ? "tool_calls"
    : "stop";
  parts.push({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
    usage: chatCompletion.usage,
  });

  return parts.map(p => `data: ${JSON.stringify(p)}\n\n`).join("") + "data: [DONE]\n\n";
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

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

    const bodyChunks = [];
    for await (const chunk of req) bodyChunks.push(chunk);
    const rawBody = Buffer.concat(bodyChunks).toString("utf8");

    let parsed = null;
    try { if (rawBody) parsed = JSON.parse(rawBody); } catch {}

    const isChatCompletions =
      req.method === "POST" &&
      (req.url === "/v1/chat/completions" || req.url === "/chat/completions");

    if (isChatCompletions && parsed && shouldUseResponses(parsed.model)) {
      // Apply drop-params first (tool_choice, maxTokens, drop_params, Gemini reasoning)
      sanitizePayload(parsed);

      const { payload: responsesPayload, wasStreaming } = buildResponsesRequest(parsed);

      // Debug logging
      const dbgTools = responsesPayload.tools;
      dbgLog(`model=${responsesPayload.model} tools=${dbgTools ? JSON.stringify(dbgTools) : "none"} streaming=${wasStreaming}`);

      // Build target URL: replace chat/completions with responses
      const targetPath = (req.url || "").replace(/chat\/completions/, "responses");
      const target = new URL(targetPath, realBaseUrl);

      const upstreamHeaders = new Headers(req.headers);
      upstreamHeaders.delete("host");
      upstreamHeaders.delete("content-length");
      upstreamHeaders.delete("connection");
      upstreamHeaders.delete("transfer-encoding");
      upstreamHeaders.set("content-type", "application/json");

      const upstream = await fetch(target.toString(), {
        method: "POST",
        headers: upstreamHeaders,
        body: JSON.stringify(responsesPayload),
      });

      const responseText = await upstream.text();

      stats.llm_call_count += 1;
      stats.responses_call_count += 1;
      stats.status_counts[String(upstream.status)] = (stats.status_counts[String(upstream.status)] || 0) + 1;
      dbgLog(`upstream status=${upstream.status} response_preview=${responseText.slice(0, 500)}`);

      if (!upstream.ok) {
        stats.error_count += 1;
        addRecentError(`HTTP ${upstream.status} (/responses): ${responseText.slice(0, 2000)}`);
        res.writeHead(upstream.status, { "Content-Type": "application/json" });
        res.end(responseText);
        return;
      }

      const responseData = JSON.parse(responseText);
      const chatCompletion = translateResponsesOutput(responseData, parsed.model);
      dbgLog(`translated finish_reason=${chatCompletion.choices?.[0]?.finish_reason} tool_calls=${JSON.stringify(chatCompletion.choices?.[0]?.message?.tool_calls ?? null)}`);

      if (wasStreaming) {
        const sseBody = buildSseStream(chatCompletion);
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        });
        res.end(sseBody);
      } else {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(chatCompletion));
      }
    } else {
      // Pass through with drop-params sanitization only
      if (parsed) sanitizePayload(parsed);
      const body = parsed !== null ? JSON.stringify(parsed) : rawBody;

      const target = new URL(req.url || "/", realBaseUrl);
      const upstreamHeaders = new Headers(req.headers);
      upstreamHeaders.delete("host");
      upstreamHeaders.delete("content-length");
      upstreamHeaders.delete("connection");
      upstreamHeaders.delete("transfer-encoding");

      const upstream = await fetch(target.toString(), {
        method: req.method,
        headers: upstreamHeaders,
        body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
      });

      const text = await upstream.text();
      if (req.method === "POST") {
        stats.llm_call_count += 1;
        stats.chat_call_count += 1;
        stats.status_counts[String(upstream.status)] = (stats.status_counts[String(upstream.status)] || 0) + 1;
        if (upstream.status >= 400) {
          stats.error_count += 1;
          addRecentError(`HTTP ${upstream.status} (/chat/completions): ${text.slice(0, 2000)}`);
        }
      }
      res.writeHead(upstream.status, Object.fromEntries(upstream.headers.entries()));
      res.end(text);
    }
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
