#!/usr/bin/env python3
"""OpenAI-compat shim in front of ChatGPT Codex OAuth (Responses API).

The llm-router only speaks /v1/chat/completions. Codex OAuth lives at
https://chatgpt.com/backend-api/codex and wants /responses. This process
translates simple chat completions (text in, text out) and attaches the
Hermes-managed oauth token. Streaming is forwarded as SSE when requested.

Bind: 127.0.0.1:8649
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import ClientSession, ClientTimeout, web

from agent.credential_pool import load_pool
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, _decode_jwt_claims

log = logging.getLogger("codex-bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HOST = os.environ.get("CODEX_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CODEX_BRIDGE_PORT", "8649"))
_POOL = "openai-codex"


def _pool_entry():
    pool = load_pool(_POOL)
    if pool is None or not pool.has_credentials():
        raise RuntimeError("no openai-codex credentials in hermes auth store")
    entry = pool.select()
    if entry is None:
        # still try an exhausted entry — local cooldown can be stale; let upstream 429
        entries = list(getattr(pool, "_entries", None) or [])
        entry = next(
            (
                e
                for e in entries
                if str(getattr(e, "runtime_api_key", None) or getattr(e, "access_token", "") or "").strip()
            ),
            None,
        )
    if entry is None:
        raise RuntimeError("openai-codex credentials exist but none have a token")
    return pool, entry


def _codex_auth() -> tuple[str, str, Dict[str, str]]:
    _pool, entry = _pool_entry()
    token = str(
        getattr(entry, "runtime_api_key", None)
        or getattr(entry, "access_token", "")
        or ""
    ).strip()
    if not token:
        raise RuntimeError("openai-codex entry has no access token")
    base = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "base_url", None)
        or DEFAULT_CODEX_BASE_URL
    )
    base = str(base or DEFAULT_CODEX_BASE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
    }
    account_id = getattr(entry, "account_id", None)
    if not account_id:
        claims = _decode_jwt_claims(token) or {}
        auth_claim = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claim, dict):
            account_id = auth_claim.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id.strip():
        headers["ChatGPT-Account-Id"] = account_id.strip()
    return token, base, headers


def _messages_to_input(messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    instructions_parts: List[str] = []
    items: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = "" if content is None else str(content)
        if role == "system":
            if text.strip():
                instructions_parts.append(text)
            continue
        items.append({"role": role, "content": text})
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, items


def _extract_text(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    output = payload.get("output")
    chunks: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"message", "output_text"} or item.get("role") == "assistant":
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text") or part.get("output_text") or ""
                            if text:
                                chunks.append(str(text))
                        elif isinstance(part, str):
                            chunks.append(part)
            text = item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    if chunks:
        return "".join(chunks)
    # some backends nest output_text at top level
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    return ""


def _as_chat_completion(model: str, text: str, raw_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": raw_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def handle_health(_request: web.Request) -> web.Response:
    try:
        _pool_entry()
        ok = True
        err = None
    except Exception as exc:
        ok = False
        err = str(exc)
    return web.json_response({"status": "ok" if ok else "degraded", "upstream": "openai-codex", "authenticated": ok, "error": err})


async def handle_models(_request: web.Request) -> web.Response:
    try:
        _token, base, headers = _codex_auth()
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc), "type": "auth"}}, status=401)
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as session:
        async with session.get(f"{base}/models", headers=headers) as resp:
            body = await resp.read()
            return web.Response(status=resp.status, body=body, content_type=resp.content_type or "application/json")


async def handle_chat(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": {"message": "invalid json"}}, status=400)
    try:
        _token, base, headers = _codex_auth()
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc), "type": "auth"}}, status=401)

    model = str(payload.get("model") or "gpt-5.5")
    client_wants_stream = bool(payload.get("stream"))
    instructions, items = _messages_to_input(payload.get("messages") or [])
    if not items:
        return web.json_response({"error": {"message": "messages required"}}, status=400)

    # ChatGPT Codex OAuth requires stream=true and rejects max_output_tokens.
    upstream = {
        "model": model,
        "input": items,
        "stream": True,
        "store": False,
    }
    if instructions:
        upstream["instructions"] = instructions
    if payload.get("temperature") is not None:
        upstream["temperature"] = payload["temperature"]

    timeout = ClientTimeout(total=None, sock_connect=15, sock_read=300)
    async with ClientSession(timeout=timeout) as session:
        async with session.post(f"{base}/responses", headers=headers, json=upstream) as resp:
            if client_wants_stream:
                out = web.StreamResponse(
                    status=resp.status,
                    headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"},
                )
                await out.prepare(request)
                async for chunk in resp.content.iter_any():
                    if chunk:
                        await out.write(chunk)
                await out.write_eof()
                return out

            raw = await resp.read()
            if resp.status >= 400:
                return web.Response(status=resp.status, body=raw, content_type=resp.content_type or "application/json")
            text = ""
            try:
                # Non-stream client: Codex still streams SSE. Reassemble text.
                decoded = raw.decode("utf-8", errors="replace")
                if decoded.lstrip().startswith("{"):
                    data = json.loads(decoded)
                    text = _extract_text(data)
                    return web.json_response(_as_chat_completion(model, text, data.get("id") if isinstance(data, dict) else None))
                for line in decoded.splitlines():
                    if not line.startswith("data:"):
                        continue
                    piece = line[5:].strip()
                    if not piece or piece == "[DONE]":
                        continue
                    try:
                        ev = json.loads(piece)
                    except Exception:
                        continue
                    text += _extract_text(ev) or (
                        ev.get("delta") if isinstance(ev.get("delta"), str) else ""
                    ) or ""
                    if isinstance(ev.get("delta"), dict):
                        text += str(ev["delta"].get("text") or ev["delta"].get("content") or "")
            except Exception:
                return web.Response(status=502, text='{"error":{"message":"bad upstream stream"}}', content_type="application/json")
            return web.json_response(_as_chat_completion(model, text))


def main() -> None:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/chat/completions", handle_chat)
    log.info("codex-bridge listening on http://%s:%s/v1", HOST, PORT)
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
