"""Streaming OpenAI-compatible chat client (raw httpx + SSE).

Works with llama.cpp's llama-server, vLLM, SGLang, LM Studio, Ollama's /v1 and
hosted OpenAI-compatible APIs. Server-specific knobs are only sent when the
server type has been detected (see probe.py).
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from rigcode.llm.probe import ServerInfo
from rigcode.llm.stream import ChunkAssembler, SSEDecoder
from rigcode.llm.types import AssistantTurn, Delta, LLMError

DeltaCallback = Callable[[Delta], None]


def _extract_error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:500]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err.get("detail") or err)[:800]
        if isinstance(err, str):
            return err[:800]
        if "detail" in data:
            return str(data["detail"])[:800]
        if "message" in data:
            return str(data["message"])[:800]
    return body.strip()[:500]


def classify_error(status: int | None, body: str) -> LLMError:
    msg = _extract_error_message(body)
    low = (body or "").lower()
    kind, retryable = "bad_request", False
    if status in (401, 403):
        kind = "auth"
    elif ("exceed" in low and "context" in low) or any(k in low for k in (
            "context length", "context_length", "maximum context", "context size", "too many tokens",
            "prompt is too long", "exceeds the available context", "exceed_context_size")):
        kind = "context_overflow"
    elif "max_tokens" in low or "max_completion_tokens" in low:
        kind = "max_tokens"
    elif "tool" in low and any(k in low for k in ("jinja", "not support", "unsupported", "does not support", "tool_choice")):
        kind = "tools_unsupported"
    elif status == 404:
        kind = "not_found"
    elif status in (408, 409, 425, 429, 500, 502, 503, 504) or status is None:
        kind, retryable = "server", True
    return LLMError(f"HTTP {status}: {msg}" if status else msg, status=status, kind=kind, body=body, retryable=retryable)


class LLMClient:
    def __init__(self, provider: dict, server: ServerInfo | None = None, log_path: Path | None = None):
        self.provider = provider
        self.server = server or ServerInfo()
        self.log_path = log_path
        self._http: httpx.AsyncClient | None = None
        mode = provider.get("tool_mode", "auto")
        self.tool_mode = "text" if mode == "text" else "native"

    # ------------------------------------------------------------------ setup
    @property
    def base_url(self) -> str:
        return str(self.provider.get("base_url", "")).rstrip("/")

    @property
    def model(self) -> str:
        return str(self.provider.get("model", ""))

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            timeout = httpx.Timeout(
                connect=float(self.provider.get("connect_timeout", 15)),
                read=float(self.provider.get("request_timeout", 3600)),
                write=120.0,
                pool=60.0,
            )
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
            key = self.provider.get("api_key")
            if key:
                headers["Authorization"] = f"Bearer {key}"
            headers.update(self.provider.get("extra_headers") or {})
            self._http = httpx.AsyncClient(timeout=timeout, headers=headers)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------ request
    def build_body(self, messages: list[dict], tools: list[dict] | None, *, effort: str | None,
                   max_tokens: int | None, slot: int | None = None, temperature: float | None = None,
                   extra: dict | None = None) -> dict:
        p = self.provider
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools and self.tool_mode == "native":
            body["tools"] = tools
            body["parallel_tool_calls"] = True
        if effort:
            levels = self.server.effort_levels
            if effort == "medium" and levels and "medium" not in levels and "high" in levels:
                effort = "high"     # e.g. Kimi K3's stock template treats an unknown level as max
            body["reasoning_effort"] = effort
            if effort == "none":    # K3-style templates read either flag to turn thinking off
                body["chat_template_kwargs"] = {**(p.get("extra_body") or {}).get("chat_template_kwargs", {}),
                                                "enable_thinking": False}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        temp = temperature if temperature is not None else p.get("temperature")
        if temp is not None:
            body["temperature"] = temp
        if p.get("top_p") is not None:
            body["top_p"] = p["top_p"]
        if p.get("reasoning_budget_tokens"):
            body["reasoning_budget_tokens"] = int(p["reasoning_budget_tokens"])
        if self.server.kind == "llama.cpp":
            lc = p.get("llama_cpp") or {}
            if lc.get("cache_prompt", True):
                body["cache_prompt"] = True
            if slot is not None and lc.get("slot_pinning", False):
                body["id_slot"] = slot
            if lc.get("return_progress", True):
                body["return_progress"] = True
        body.update(p.get("extra_body") or {})
        if extra:
            body.update(extra)
        return body

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None, *, on_delta: DeltaCallback,
                          assembler: ChunkAssembler, effort: str | None = None, max_tokens: int | None = None,
                          slot: int | None = None, temperature: float | None = None,
                          extra: dict | None = None) -> AssistantTurn:
        """Stream one completion. The caller owns `assembler` so it can recover the
        partial turn if this coroutine is cancelled (Ctrl+C)."""
        max_retries = int(self.provider.get("max_retries", 3))
        body = self.build_body(messages, tools, effort=effort, max_tokens=max_tokens, slot=slot,
                               temperature=temperature, extra=extra)
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                turn = await self._stream_once(body, assembler, on_delta, started)
                turn.elapsed = time.monotonic() - started
                self._log({"event": "response", "finish": turn.finish_reason, "tool_calls": len(turn.tool_calls),
                           "usage": vars(turn.usage), "elapsed": round(turn.elapsed, 2)})
                return turn
            except LLMError as e:
                self._log({"event": "error", "kind": e.kind, "status": e.status, "message": str(e)})
                if e.kind == "max_tokens" and "max_tokens" in body:
                    body.pop("max_tokens")  # let the server pick a valid limit
                    assembler.reset()
                    continue
                if not e.retryable or attempt >= max_retries:
                    raise
                attempt += 1
                delay = min(30.0, (2 ** attempt) + random.random())
                on_delta(Delta("reset", text=f"{e} - retrying in {delay:.0f}s (attempt {attempt}/{max_retries})"))
                assembler.reset()
                await asyncio.sleep(delay)

    async def _stream_once(self, body: dict, asm: ChunkAssembler, on_delta: DeltaCallback, started: float) -> AssistantTurn:
        url = f"{self.base_url}/chat/completions"
        self._log({"event": "request", "url": url, "n_messages": len(body["messages"]),
                   "tools": len(body.get("tools") or []), "params": {k: v for k, v in body.items() if k not in ("messages", "tools")},
                   "tail": body["messages"][-2:]})
        try:
            async with self._client().stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    raw = (await resp.aread()).decode("utf-8", errors="replace")
                    raise classify_error(resp.status_code, raw)
                ctype = resp.headers.get("content-type", "")
                if "text/event-stream" not in ctype and "application/json" in ctype:
                    data = json.loads((await resp.aread()).decode("utf-8", errors="replace"))
                    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
                        raise classify_error(resp.status_code, json.dumps(data))
                    for d in asm.ingest(data):
                        on_delta(d)
                else:
                    decoder = SSEDecoder()
                    async for line in resp.aiter_lines():
                        frame = decoder.feed_line(line)
                        if frame is not None and self._handle_frame(frame, asm, on_delta, started):
                            break
                    frame = decoder.flush()
                    if frame is not None:
                        self._handle_frame(frame, asm, on_delta, started)
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot connect to {self.base_url} ({e}). Is the server running and the port open?",
                           kind="connection", retryable=True) from e
        except httpx.ConnectTimeout as e:
            raise LLMError(f"Timed out connecting to {self.base_url}. A firewall may be blocking the port.",
                           kind="connection", retryable=True) from e
        except httpx.ReadTimeout as e:
            # Never auto-retry: disconnecting cancels the server's work and a retry starts from scratch.
            raise LLMError("Timed out waiting for the model (no data within provider.request_timeout).",
                           kind="timeout", retryable=False) from e
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as e:
            raise LLMError(f"Connection to the server was lost: {e}", kind="connection", retryable=True) from e
        turn, deltas = asm.finish()
        for d in deltas:
            on_delta(d)
        return turn

    def _handle_frame(self, frame, asm: ChunkAssembler, on_delta: DeltaCallback, started: float) -> bool:
        """Process one SSE frame; returns True at end of stream."""
        _event, data = frame
        if data.strip() == "[DONE]":
            return True
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return False
        if not isinstance(chunk, dict):
            return False
        if chunk.get("error") and not chunk.get("choices"):
            raise classify_error(None if not isinstance(chunk["error"], dict) else chunk["error"].get("code"), data)
        for d in asm.ingest(chunk):
            if asm.turn.first_token_latency is None and d.kind in ("reasoning", "content", "tool_start"):
                asm.turn.first_token_latency = time.monotonic() - started
                on_delta(Delta("first_token"))
            on_delta(d)
        return False

    def _log(self, record: dict) -> None:
        if not self.log_path:
            return
        try:
            record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ simple one-shot
    async def complete(self, messages: list[dict], *, effort: str | None = "low", max_tokens: int | None = None,
                       on_delta: DeltaCallback | None = None, tools: list[dict] | None = None,
                       slot: int | None = None) -> AssistantTurn:
        """Convenience: one completion without the agent loop (warm-up, probes)."""
        asm = ChunkAssembler()
        return await self.stream_chat(messages, tools, on_delta=on_delta or (lambda d: None), assembler=asm,
                                      effort=effort, max_tokens=max_tokens, slot=slot)
