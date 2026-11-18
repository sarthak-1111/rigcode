"""`rigcode doctor` / `/doctor`: diagnose the model server connection and the local setup."""
from __future__ import annotations

import asyncio
import ipaddress
import platform
import shutil
import socket
import sys
import time
from collections.abc import Callable
from urllib.parse import urlparse

from rigcode import __version__
from rigcode.agent.runtime import MIN_LOCAL_CONTEXT
from rigcode.llm.client import LLMClient
from rigcode.llm.probe import AUTO, discover
from rigcode.llm.types import Delta, LLMError
from rigcode.tools.shell import resolve_shell
from rigcode.utils.text import fmt_duration

Out = Callable[..., None]   # out(text, style="")


def _local_subnet_hint(host: str) -> str:
    """Best guess at this machine's /24 on the server's network, for the firewall command."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 9))
            mine = s.getsockname()[0]
        net = ipaddress.ip_network(f"{mine}/24", strict=False)
        return str(net)
    except OSError:
        return "<your-subnet>/24"


async def _tcp_check(host: str, port: int, timeout: float = 5.0) -> str | None:
    try:
        _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        w.close()
        try:
            await w.wait_closed()
        except OSError:
            pass
        return None
    except asyncio.TimeoutError:
        return "timed out"
    except OSError as e:
        return str(e)


async def run_doctor(settings, project_root, out: Out, chat_test: bool = True) -> bool:
    ok = True
    out(f"rigcode doctor · rigcode {__version__} · Python {sys.version.split()[0]} · {platform.platform()}", "bold")

    out("\nConfiguration", "bold")
    for src in settings.sources:
        if src.error:
            ok = False
            out(f"  ✗ {src.scope}: {src.path} - {src.error}", "class:error")
        else:
            out(f"  {'✓' if src.loaded else '·'} {src.scope}: {src.path}{'' if src.loaded else ' (not present)'}",
                "" if src.loaded else "class:dim")

    out("\nLocal tools", "bold")
    try:
        shell = resolve_shell(settings.get("shell", "auto"))
        out(f"  ✓ shell: {shell.name} ({shell.exe})")
    except RuntimeError as e:
        ok = False
        out(f"  ✗ shell: {e}", "class:error")
    git = shutil.which("git")
    out(f"  {'✓' if git else '✗'} git: {git or 'not found (git features disabled)'}", "" if git else "class:warn")
    rg = shutil.which("rg")
    out(f"  {'✓' if rg else '·'} ripgrep: {rg or 'not found - Grep uses a slower built-in search'}",
        "" if rg else "class:dim")

    provider = dict(settings.provider)
    base = str(provider.get("base_url") or "")
    auto = base in AUTO
    out(f"\nModel server: {'auto-detect (provider.base_url is not set)' if auto else base}", "bold")
    if not auto:
        u = urlparse(base)
        host = u.hostname or ""
        port = u.port or (443 if u.scheme == "https" else 80)
        err = await _tcp_check(host, port)
        if err:
            ok = False
            out(f"  ✗ TCP {host}:{port}: {err}", "class:error")
            if "timed out" in err or "10060" in err:
                out("    The host did not answer on that port. If the server is running, a firewall is probably "
                    "blocking it. On the server machine (Ubuntu ufw):", "class:dim")
                out(f"      sudo ufw allow from {_local_subnet_hint(host)} to any port {port} proto tcp", "")
            else:
                out("    Connection refused: check that the server is running and listening on 0.0.0.0 "
                    f"(llama-server --host 0.0.0.0 --port {port}).", "class:dim")
            return ok
        out(f"  ✓ TCP {host}:{port} reachable")
    info, found = await discover(provider)
    if not info.reachable:
        ok = False
        out(f"  ✗ {info.error}", "class:error")
        return ok
    if found:
        out(f"  ✓ {found}")
    out(f"  ✓ server: {info.describe()}")
    if info.models:
        model = provider.get("model")
        listed = model in info.models
        out(f"  {'✓' if listed else '·'} models: {', '.join(info.models[:6])}"
            + ("" if listed or info.kind == "llama.cpp" else f"  (configured '{model}' is not listed)"),
            "" if listed else "class:dim")
    if info.context_window:
        small = info.context_window < MIN_LOCAL_CONTEXT
        out(f"  {'✗' if small else '✓'} context window per request: {info.context_window:,} tokens"
            + ("  - too small: long prompts get cut off" if small else ""), "class:warn" if small else "")
    elif info.kind == "ollama":
        out("  · context window: decided when Ollama loads the model (OLLAMA_CONTEXT_LENGTH). "
            "rigcode needs at least 16k; 32k or more is better.", "class:dim")
    if info.kind == "llama.cpp":
        pinned = (settings.get("provider.llama_cpp.slot_pinning", False))
        out(f"  ✓ slots: {info.total_slots or '?'} ("
            + ("pinned: main agent on the highest slot" if pinned else
               "unpinned: the server restores each agent's cached prompt from its RAM cache") + ")", "class:dim")
        if info.effort_levels:
            out(f"  ✓ template effort levels: {', '.join(info.effort_levels)}"
                + ("  (medium is sent as high)" if "medium" not in info.effort_levels else ""), "class:dim")

    if chat_test:
        out("\nChat test (tiny request, thinking off)", "bold")
        client = LLMClient(provider, info)
        first: list[float] = []
        started = time.monotonic()

        def on_delta(d: Delta) -> None:
            if d.kind == "first_token" and not first:
                first.append(time.monotonic() - started)

        try:
            turn = await client.complete([{"role": "user", "content": "Reply with exactly: OK"}], effort="none",
                                         max_tokens=24, on_delta=on_delta)
            text = (turn.content or turn.reasoning).strip().replace("\n", " ")[:60]
            ttft = f"first token after {fmt_duration(first[0])}" if first else "no streamed tokens"
            speed = f", {turn.usage.gen_tps:.1f} tok/s" if turn.usage.gen_tps else ""
            out(f"  ✓ replied {text!r} - {ttft}{speed}, total {fmt_duration(turn.elapsed)}")
            if turn.tool_calls:
                out("  · (the reply contained a tool call)", "class:dim")
        except LLMError as e:
            ok = False
            out(f"  ✗ request failed: {e}", "class:error")
        finally:
            await client.aclose()

    servers = settings.get("mcp_servers", {}) or {}
    if servers:
        out(f"\nMCP: {len(servers)} server(s) configured: {', '.join(servers)} (status: /mcp inside rigcode)", "class:dim")
    out("\n" + ("All checks passed." if ok else "Some checks failed - see above."), "class:ok" if ok else "class:error")
    return ok
