"""A scripted OpenAI-compatible streaming server (llama.cpp flavoured) for tests.

    server = MockServer([Reply(tool_calls=[("Read", {"file_path": "a.py"})]), Reply(content="Done.")])
    base_url = server.start()      # http://127.0.0.1:PORT/v1
    ...
    server.requests                # the JSON bodies received by /v1/chat/completions
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)     # [(name, args_dict_or_raw_str), ...]
    finish: str | None = None
    status: int = 200
    error: str | None = None
    delay: float = 0.0                                  # seconds between chunks
    prompt_tokens: int = 1000
    cached_tokens: int = 900


class MockServer:
    def __init__(self, replies=None, n_ctx: int = 32768, slots: int = 4, template: str = "{{ messages }}"):
        self.replies: list = list(replies or [])
        self.requests: list[dict] = []
        self.warmups: list[dict] = []          # max_tokens=1 requests (warm-ups and slot flushes)
        self.events: list[tuple] = []          # (kind, "start"|"end", monotonic time); kind = "warmup" | "chat"
        self.warmup_delay = 0.0
        self.n_ctx = n_ctx
        self.slots = slots
        self.template = template
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    def flushes(self) -> list[dict]:
        return [w for w in self.warmups if w["messages"] == [{"role": "user", "content": "."}]]

    def add(self, *replies) -> None:
        with self._lock:
            self.replies.extend(replies)

    def _next(self, body: dict) -> Reply:
        with self._lock:
            r = self.replies.pop(0) if self.replies else Reply(content="(no more scripted replies)")
        return r(body) if callable(r) else r

    def start(self) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep test output clean
                pass

            def _json(self, status: int, obj) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [
                        {"id": "mock-model", "object": "model", "owned_by": "llamacpp", "meta": {"n_ctx_train": 65536}}]})
                elif self.path == "/props":
                    self._json(200, {"default_generation_settings": {"n_ctx": server.n_ctx},
                                     "total_slots": server.slots, "build_info": "mock-1",
                                     "modalities": {"vision": False}, "chat_template": server.template})
                elif self.path == "/slots":
                    self._json(200, [{"id": i, "n_ctx": server.n_ctx} for i in range(server.slots)])
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                kind = "warmup" if body.get("max_tokens") == 1 else "chat"
                server.events.append((kind, "start", time.monotonic()))
                if kind == "warmup":
                    server.warmups.append(body)
                    reply = Reply(content="o", delay=server.warmup_delay)
                else:
                    server.requests.append(body)
                    reply = server._next(body)
                if reply.status != 200:
                    self._json(reply.status, {"error": {"code": reply.status, "message": reply.error or "error",
                                                        "type": "invalid_request_error"}})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for chunk in _chunks(reply):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if reply.delay:
                            time.sleep(reply.delay)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                server.events.append((kind, "end", time.monotonic()))

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


def _pieces(text: str, size: int = 7) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _chunks(r: Reply):
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": "mock-model"}

    def delta(d: dict, finish=None) -> dict:
        return {**base, "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}

    yield {**base, "choices": [], "prompt_progress": {"total": r.prompt_tokens, "cache": r.cached_tokens,
                                                      "processed": r.prompt_tokens, "time_ms": 50}}
    yield delta({"role": "assistant", "content": None})
    n_out = 0
    for p in _pieces(r.reasoning):
        n_out += 1
        yield delta({"reasoning_content": p})
    for p in _pieces(r.content):
        n_out += 1
        yield delta({"content": p})
    for i, (name, args) in enumerate(r.tool_calls):
        raw = args if isinstance(args, str) else json.dumps(args)
        call_id = uuid.uuid4().hex
        yield delta({"tool_calls": [{"index": i, "id": call_id, "type": "function",
                                     "function": {"name": name, "arguments": ""}}]})
        for p in _pieces(raw, 11):
            n_out += 1
            yield delta({"tool_calls": [{"index": i, "function": {"arguments": p}}]})
    finish = r.finish or ("tool_calls" if r.tool_calls else "stop")
    yield delta({}, finish)
    yield {**base, "choices": [],
           "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": max(1, n_out),
                     "total_tokens": r.prompt_tokens + n_out,
                     "prompt_tokens_details": {"cached_tokens": r.cached_tokens}},
           "timings": {"cache_n": r.cached_tokens, "prompt_n": r.prompt_tokens - r.cached_tokens,
                       "prompt_ms": 100.0, "predicted_n": max(1, n_out), "predicted_ms": 200.0}}


Responder = Callable[[dict], Reply]
