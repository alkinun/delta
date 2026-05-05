"""
proxy.py — Bridges Zed's chat completions requests to vLLM's raw completions API.

Zed  →  POST /v1/chat/completions  →  [this proxy]
                                           ↓
                                   builds your prompt format
                                           ↓
                               POST /v1/completions → vLLM
                                           ↓
                                   wraps response for Zed

Usage:
    python proxy.py

Environment variables:
    VLLM_URL    vLLM base URL (default: http://localhost:8000)
    MODEL_NAME  model name as registered in vLLM (default: delta)
    PORT        port this proxy listens on (default: 8001)
"""

import os
import re
import json
import time
import uuid
import asyncio
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

VLLM_URL   = os.getenv("VLLM_URL",   "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "delta")
PORT       = int(os.getenv("PORT",   "8001"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("proxy")

app = FastAPI()
client = httpx.AsyncClient(timeout=120.0)


# ── Prompt builder ────────────────────────────────────────────────────────────

def parse_messages(messages: list[dict]) -> tuple[str, str]:
    """
    Extract instruction + region from Zed's chat message list.

    Zed inline assistant sends:
        system: "<document>...full file...<rewrite_this>region</rewrite_this>...</document>"
        user:   "<instruction>"

    We extract the <rewrite_this> block as the REGION and the user
    message as the INSTRUCTION.
    """
    log.debug("parse_messages called with %d messages", len(messages))
    instruction = ""
    region = ""

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )

        role = msg.get("role", "")

        if role == "system":
            # Extract the full document
            doc_match = re.search(r"<document>(.*?)</document>", content, re.DOTALL)
            if doc_match:
                doc_body = doc_match.group(1)
                # Find the LAST <rewrite_this> block (Zed's tags are appended to the end)
                rewrite_matches = list(re.finditer(
                    r"<rewrite_this>(.*?)</rewrite_this>", doc_body, re.DOTALL
                ))
                rewrite_match = rewrite_matches[-1] if rewrite_matches else None
                if rewrite_match:
                    region = rewrite_match.group(1).strip("\n")
                    log.debug(
                        "Extracted from system — region: %d chars",
                        len(region),
                    )
                else:
                    log.warning("System message has <document> but no <rewrite_this>")
                    # Fall back to empty region if no rewrite_this found
                    region = ""
            else:
                log.debug("System message has no <document> tag, skipping")

        elif role == "user":
            instruction = content.strip()
            log.debug("User instruction: %r", instruction[:200])

    return region, instruction


def build_prompt(region: str, instruction: str) -> str:
    return (
        f"[REGION]\n{region}\n[/REGION]\n"
        f"[INSTRUCTION]\n{instruction}\n[/INSTRUCTION]\n"
        f"[OUTPUT]"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    log.debug("GET /v1/models — returning model=%s", MODEL_NAME)
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    log.debug("Incoming request body: %s", json.dumps(body, default=str)[:10000])

    messages = body.get("messages", [])
    stream   = body.get("stream", False)
    log.info("Chat completion request: %d messages, stream=%s", len(messages), stream)

    region, instruction = parse_messages(messages)
    log.debug("Parsed prompt — instruction: %r, region: %r",
              instruction[:200], region[:200])

    prompt = build_prompt(region, instruction)
    log.debug("Built prompt:\n%s", prompt[:3000])

    vllm_payload = {
        "model":       MODEL_NAME,
        "prompt":      prompt,
        "max_tokens":  1024,
        "temperature": 0.0,
        "stop":        ["[/OUTPUT]"],
        "stream":      stream,
    }
    log.info("Sending to vLLM: model=%s max_tokens=%s temperature=%s",
             vllm_payload["model"], vllm_payload["max_tokens"], vllm_payload["temperature"])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created       = int(time.time())

    if stream:
        async def stream_response():
            log.debug("Opening streaming connection to vLLM...")
            buffer = ""
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"

            async with client.stream(
                "POST",
                f"{VLLM_URL}/v1/completions",
                json=vllm_payload,
            ) as resp:
                log.debug("vLLM stream status: %s", resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        log.debug("vLLM stream done")
                        # Now send the complete tool call response
                        buffer = buffer.strip()
                        log.info("Sending tool call with %d chars of text", len(buffer))

                        # Start chunk with tool call info
                        start_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": "rewrite_section",
                                            "arguments": json.dumps({"replacement_text": buffer})
                                        }
                                    }]
                                },
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(start_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        vllm_chunk = json.loads(data)
                        text = vllm_chunk["choices"][0].get("text", "")
                        finish = vllm_chunk["choices"][0].get("finish_reason")
                        log.debug("Stream chunk: text=%r finish=%s", text, finish)
                        buffer += text
                        log.debug("Added to buffer: %r | Buffer now: %d chars", text, len(buffer))
                    except Exception as exc:
                        log.warning("Failed to parse streaming chunk: %s | raw: %r", exc, data[:500])
                        continue

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # Non-streaming
    log.debug("Sending non-streaming request to vLLM...")
    resp = await client.post(f"{VLLM_URL}/v1/completions", json=vllm_payload)
    log.debug("vLLM response status: %s", resp.status_code)
    resp.raise_for_status()
    vllm_data = resp.json()
    log.debug("vLLM response body: %s", json.dumps(vllm_data, default=str)[:2000])

    text = vllm_data["choices"][0]["text"].strip().rstrip('\n')
    log.info("Completion text (%d chars): %r", len(text), text[:500])

    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "rewrite_section",
                        "arguments": json.dumps({"replacement_text": text})
                    }
                }]
            },
            "finish_reason": "stop",
        }],
        "usage": vllm_data.get("usage", {}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info("Starting proxy on port %d → vLLM at %s (model=%s)", PORT, VLLM_URL, MODEL_NAME)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
