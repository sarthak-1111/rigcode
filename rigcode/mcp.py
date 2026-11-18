"""Minimal Model Context Protocol client (stdio and Streamable HTTP transports).

Config (settings "mcp_servers" or a Claude-Code-style .mcp.json "mcpServers"):
  "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."], "env": {}}
  "remote": {"type": "http", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
Tools are exposed to the model as mcp__<server>__<tool>.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rigcode import __version__
from rigcode.llm.stream import SSEDecoder
from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.utils.text import truncate_middle

PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    pass


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


class _StdioTransport:
    def __init__(self, command: str, args: list[str], env: dict, cwd: Path):
        self.command, self.args, self.env, self.cwd = command, args, env, cwd
        self.proc: asyncio.subprocess.Process | None = None
        self.stderr_tail: list[str] = []
        self._handler = None

    async def start(self, on_message) -> None:
        exe = shutil.which(self.command) or self.command
        env = {**os.environ, **{k: str(v) for k, v in (self.env or {}).items()}}
        self.proc = await asyncio.create_subprocess_exec(
            exe, *self.args, cwd=str(self.cwd), env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024)
        self._handler = on_message
        asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                self.stderr_tail.append(f"(non-JSON stdout) {text[:200]}")
                continue
            await self._handler(msg)

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
            del self.stderr_tail[:-30]

    async def send(self, msg: dict) -> dict | None:
        if not self.proc or not self.proc.stdin or self.proc.returncode is not None:
            raise McpError("server process is not running" + self._stderr_hint())
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        return None

    def _stderr_hint(self) -> str:
        return (": " + " | ".join(self.stderr_tail[-3:])) if self.stderr_tail else ""

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
                await asyncio.wait_for(self.proc.wait(), 3)
            except (asyncio.TimeoutError, OSError, AttributeError):
                try:
                    self.proc.kill()
                except OSError:
                    pass


class _HttpTransport:
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = dict(headers or {})
        self.session_id: str | None = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=15, read=300, write=60, pool=30))
        self._handler = None
        self.stderr_tail: list[str] = []

    async def start(self, on_message) -> None:
        self._handler = on_message

    async def send(self, msg: dict) -> dict | None:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if msg.get("method") != "initialize":
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        async with self.client.stream("POST", self.url, json=msg, headers=headers) as r:
            if r.headers.get("mcp-session-id"):
                self.session_id = r.headers["mcp-session-id"]
            if r.status_code == 202 or "id" not in msg:
                return None
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", errors="replace")[:300]
                raise McpError(f"HTTP {r.status_code}: {body}")
            ctype = r.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                decoder = SSEDecoder()
                async for line in r.aiter_lines():
                    frame = decoder.feed_line(line)
                    if frame is None:
                        continue
                    try:
                        data = json.loads(frame[1])
                    except json.JSONDecodeError:
                        continue
                    if data.get("id") == msg["id"] and ("result" in data or "error" in data):
                        return data
                    await self._handler(data)
                return None
            return json.loads((await r.aread()).decode("utf-8"))

    async def close(self) -> None:
        await self.client.aclose()


@dataclass
class McpServer:
    name: str
    spec: dict
    cwd: Path
    status: str = "pending"      # pending | connected | failed
    error: str | None = None
    tools: list[dict] = field(default_factory=list)
    server_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        kind = str(self.spec.get("type") or ("http" if self.spec.get("url") else "stdio")).lower()
        if kind in ("http", "streamable-http", "streamable_http"):
            self.transport = _HttpTransport(self.spec["url"], self.spec.get("headers") or {})
        elif kind == "stdio":
            self.transport = _StdioTransport(self.spec.get("command", ""), [str(a) for a in self.spec.get("args") or []],
                                             self.spec.get("env") or {}, self.cwd)
        else:
            raise McpError(f"unsupported MCP transport type '{kind}' (use stdio or http)")

    async def _on_message(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg) and msg.get("method") is None:
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                fut.set_result(msg)
            return
        method = msg.get("method")
        if method and "id" in msg:  # server -> client request
            if method == "ping":
                result = {}
            elif method == "roots/list":
                result = {"roots": [{"uri": self.cwd.as_uri(), "name": self.cwd.name}]}
            else:
                await self.transport.send({"jsonrpc": "2.0", "id": msg["id"],
                                           "error": {"code": -32601, "message": f"{method} not supported"}})
                return
            await self.transport.send({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    async def request(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        self._next_id += 1
        rid = self._next_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        direct = await self.transport.send(msg)
        if direct is not None:
            self._pending.pop(rid, None)
            reply = direct
        else:
            try:
                reply = await asyncio.wait_for(fut, timeout)
            except asyncio.TimeoutError as e:
                self._pending.pop(rid, None)
                raise McpError(f"{method} timed out after {timeout:.0f}s") from e
        if "error" in reply:
            err = reply["error"]
            raise McpError(f"{method}: {err.get('message', err)}")
        return reply.get("result") or {}

    async def connect(self, timeout: float = 30) -> None:
        try:
            await self.transport.start(self._on_message)
            init = await self.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": "rigcode", "version": __version__}}, timeout=timeout)
            self.server_info = init.get("serverInfo") or {}
            await self.transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools, cursor = [], None
            for _ in range(20):
                res = await self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout=timeout)
                tools.extend(res.get("tools") or [])
                cursor = res.get("nextCursor")
                if not cursor:
                    break
            self.tools = tools
            self.status = "connected"
        except (McpError, OSError, httpx.HTTPError, KeyError) as e:
            self.status = "failed"
            hint = ""
            if isinstance(self.transport, _StdioTransport) and self.transport.stderr_tail:
                hint = " | " + " | ".join(self.transport.stderr_tail[-3:])
            self.error = f"{type(e).__name__}: {e}{hint}"

    async def call_tool(self, name: str, arguments: dict, timeout: float = 600) -> dict:
        return await self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    async def close(self) -> None:
        await self.transport.close()


class McpTool(Tool):
    kind = "mcp"

    def __init__(self, server: McpServer, tool: dict):
        self.server = server
        self.tool_name = tool["name"]
        self.name = f"mcp__{_sanitize(server.name)}__{_sanitize(tool['name'])}"[:64]
        self.description = (tool.get("description") or f"MCP tool {tool['name']} from {server.name}").strip()[:1200]
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        self.parameters = schema
        self.read_only = bool((tool.get("annotations") or {}).get("readOnlyHint"))

    def permission_target(self, args: dict) -> str:
        return json.dumps(args, ensure_ascii=False)[:200]

    def label(self, args: dict, ctx=None) -> str:
        return f"{self.server.name}:{self.tool_name}({json.dumps(args, ensure_ascii=False)[:60]})"

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self.server.status != "connected":
            return ToolResult.error(f"MCP server '{self.server.name}' is not connected: {self.server.error}")
        try:
            result = await self.server.call_tool(self.tool_name, args)
        except McpError as e:
            return ToolResult.error(f"MCP call failed: {e}")
        parts = []
        for item in result.get("content") or []:
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "resource":
                res = item.get("resource") or {}
                parts.append(res.get("text") or f"[resource {res.get('uri', '')}]")
            elif t == "resource_link":
                parts.append(f"[resource link: {item.get('uri', '')}]")
            elif t in ("image", "audio"):
                parts.append(f"[{t} content omitted ({item.get('mimeType', '')})]")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False, indent=2))
        text = "\n".join(parts) or "(no content)"
        limit = int(ctx.rt.settings.get("context.max_tool_output_chars", 12000))
        text, _ = truncate_middle(text, limit)
        return ToolResult(content=text, is_error=bool(result.get("isError")),
                          summary=f"{self.server.name}:{self.tool_name}")


class McpManager:
    def __init__(self, config: dict, cwd: Path):
        self.servers: dict[str, McpServer] = {}
        for name, spec in (config or {}).items():
            if not isinstance(spec, dict) or spec.get("disabled"):
                continue
            try:
                self.servers[name] = McpServer(name, spec, cwd)
            except (McpError, KeyError) as e:
                srv = McpServer.__new__(McpServer)
                srv.name, srv.spec, srv.cwd, srv.status, srv.error, srv.tools, srv.server_info = \
                    name, spec, cwd, "failed", str(e), [], {}
                self.servers[name] = srv

    async def start_all(self, timeout: float = 30) -> None:
        pending = [s for s in self.servers.values() if s.status == "pending"]
        if pending:
            await asyncio.gather(*(s.connect(timeout) for s in pending))

    def tools(self) -> list[McpTool]:
        out = []
        for s in self.servers.values():
            if s.status == "connected":
                out.extend(McpTool(s, t) for t in s.tools if t.get("name"))
        return out

    def status_lines(self) -> list[str]:
        lines = []
        for s in self.servers.values():
            if s.status == "connected":
                lines.append(f"{s.name}: connected, {len(s.tools)} tools")
            else:
                lines.append(f"{s.name}: {s.status}" + (f" - {s.error}" if s.error else ""))
        return lines

    async def close_all(self) -> None:
        for s in self.servers.values():
            if hasattr(s, "transport"):
                try:
                    await s.close()
                except Exception:  # noqa: BLE001
                    pass
