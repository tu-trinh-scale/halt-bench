import { stdin, stdout, stderr } from "node:process";

const SIDECAR_URL = process.env.SIDECAR_URL;
const TASK_INSTANCE_ID = process.env.TASK_INSTANCE_ID || "";
const CANT_ANSWER = "can't answer (perhaps transient hiccup)";

if (!SIDECAR_URL) {
  stderr.write("Missing SIDECAR_URL for ask_human bridge\n");
}

function write(msg) {
  stdout.write(JSON.stringify(msg) + "\n");
}

function readQuestion(args) {
  if (typeof args?.question === "string" && args.question.trim()) {
    return args.question.trim();
  }
  return "";
}

// Maximum ms to wait for the sidecar to respond.  The sidecar itself allows
// EVAL_TIMEOUT_S (300 s) × EVAL_MAX_RETRIES (3) = 900 s for vllm/LiteLLM calls
// before returning CANT_ANSWER.  1000 s here gives it comfortable headroom while
// still guaranteeing the session never hangs indefinitely if the sidecar itself
// becomes unreachable or its timeout mechanism fails.
const SIDECAR_FETCH_TIMEOUT_MS = 1_000_000;

async function sidecarAsk(question) {
  if (!SIDECAR_URL || !TASK_INSTANCE_ID) {
    return CANT_ANSWER;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SIDECAR_FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(`${SIDECAR_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        instance_id: TASK_INSTANCE_ID,
        native_event_type: "haltbench.mcp.ask_human",
      }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!response.ok) return CANT_ANSWER;
    const payload = await response.json();
    return payload?.resolution || payload?.response || CANT_ANSWER;
  } catch {
    clearTimeout(timer);
    return CANT_ANSWER;
  }
}

async function handleMessage(msg) {
  if (!msg || typeof msg !== "object") return;
  if (msg.method === "initialize") {
    write({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "haltbench-ask-human-bridge", version: "1.0.0" },
      },
    });
    return;
  }
  if (msg.method === "tools/list") {
    write({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        tools: [
          {
            name: "ask_human",
            description: "Ask user clarification question through blocker registry",
            inputSchema: {
              type: "object",
              properties: { question: { type: "string" } },
              required: ["question"],
            },
          },
        ],
      },
    });
    return;
  }
  if (msg.method === "tools/call") {
    const question = readQuestion(msg.params?.arguments ?? {});
    const resolution = question ? await sidecarAsk(question) : CANT_ANSWER;
    write({
      jsonrpc: "2.0",
      id: msg.id,
      result: {
        content: [{ type: "text", text: resolution }],
        structuredContent: { resolution, question },
      },
    });
    return;
  }
}

let buffer = "";
stdin.setEncoding("utf8");
stdin.on("data", async (chunk) => {
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    try {
      const msg = JSON.parse(line);
      await handleMessage(msg);
    } catch (err) {
      stderr.write(`Bridge parse error: ${String(err)}\n`);
    }
  }
});

