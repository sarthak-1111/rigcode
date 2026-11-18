"""Line-oriented front end for `rigcode -p` (headless) and for terminals where the
interactive prompt can't run (pipes, dumb terminals).

Headless output formats:
  text         only the final answer on stdout (progress on stderr with --verbose)
  json         one JSON object with the answer, usage and session id at the end
  stream-json  one JSON event per line as things happen
"""
from __future__ import annotations

import asyncio
import json
import re
import signal
import sys
import time

from rigcode.agent.mentions import expand_mentions
from rigcode.permissions import Decision
from rigcode.tools.base import ToolContext
from rigcode.tools.shell import build_invocation
from rigcode.ui.base import BaseUI
from rigcode.ui.commands import build_registry
from rigcode.ui.render import Renderer
from rigcode.utils.text import fmt_duration, fmt_tokens, one_line

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_STYLE_ANSI = {"class:error": "\x1b[31m", "class:warn": "\x1b[33m", "class:dim": "\x1b[2m", "class:faint": "\x1b[2m",
               "class:ok": "\x1b[32m", "bold": "\x1b[1m"}


class PlainUI(BaseUI):
    def __init__(self, settings, *, interactive: bool, output_format: str = "text", verbose: bool = False,
                 color: bool | None = None):
        self.interactive = interactive
        self.fmt = output_format
        self.verbose = verbose or interactive
        self.thinking_mode = str(settings.get("ui.thinking", "collapsed"))
        self.r = Renderer()
        self.rt = None
        self.commands = build_registry()
        self.running: dict = {}
        self.busy = False
        self.exit_requested = False
        self._mid_line = False
        self._thinking_since: float | None = None
        self._pending: list[str] = []
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        self.stream = sys.stdout if interactive else sys.stderr
        self.color = color if color is not None else bool(getattr(self.stream, "isatty", lambda: False)())

    # ---------------------------------------------------------------- output
    def _write(self, text: str, style: str = "") -> None:
        if self.fmt == "stream-json" or not self.verbose:
            return
        if self._mid_line and not text.startswith("\n"):
            text = "\n" + text
        self._mid_line = False
        if self.color and style in _STYLE_ANSI:
            text = _STYLE_ANSI[style] + text + "\x1b[0m"
        self.stream.write(text)
        self.stream.flush()

    def _event(self, kind: str, **data) -> None:
        if self.fmt == "stream-json":
            sys.stdout.write(json.dumps({"type": kind, **data}, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()

    def out_line(self, text: str, style: str = "") -> None:
        if self.fmt == "stream-json":
            self._event("notice", text=text)
        else:
            self._write(text + "\n", style)

    def out(self, fragments) -> None:
        self._write("".join(t for _, t in fragments))

    def out_ansi(self, text: str) -> None:
        self._write((text if self.color else _ANSI_RE.sub("", text)).rstrip("\n") + "\n")

    def out_markdown(self, text: str, bullet: str | None = None, style: str | None = None) -> None:
        if text.strip():
            self._write("\n" + (bullet or "") + text.strip() + "\n")

    def print_tool(self, label: str, res, output_lines: int | None = None) -> None:
        self._write(f"\n{'✗' if res.is_error else '●'} {label}\n", "class:error" if res.is_error else "bold")
        self._write(f"  ⎿  {one_line(res.summary or res.content, 160)}\n",
                    "class:error" if res.is_error else "class:dim")
        disp = res.display or {}
        if disp.get("type") == "output" and output_lines:
            lines = [l for l in disp.get("text", "").split("\n") if l.strip()][-output_lines:]
            self._write("".join(f"     {l}\n" for l in lines), "class:dim")

    def begin_activity(self, key: str, label: str) -> None:
        self._write(f"\n● {label} …\n", "class:dim")

    def end_activity(self, key: str) -> None:
        pass

    def request_exit(self) -> None:
        self.exit_requested = True

    def submit_message(self, text: str) -> None:
        self._pending.append(text)

    # ---------------------------------------------------------------- engine events
    def on_notice(self, level: str, text: str) -> None:
        if self.fmt == "stream-json":
            if level != "info":     # the init event carries the discovered server and model
                self._event("notice", level=level, text=text)
            return
        if level == "error" and not self.verbose:
            sys.stderr.write(f"error: {text}\n")
            return
        style = {"warn": "class:warn", "error": "class:error"}.get(level, "class:dim")
        prefix = {"warn": "⚠ ", "error": "✗ "}.get(level, "  ⎿ ")
        self._write(prefix + text + "\n", style)

    def on_model_start(self, agent, info: dict) -> None:
        self._thinking_since = None

    def on_delta(self, agent, d) -> None:
        if not agent.is_main:
            return
        if self.fmt == "stream-json":
            if d.kind == "content":
                self._event("text_delta", text=d.text)
            return
        if d.kind == "reasoning":
            if self._thinking_since is None:
                self._thinking_since = time.monotonic()
                if self.thinking_mode != "hidden":
                    self._write("✻ thinking…\n", "class:dim")
            if self.thinking_mode == "full" and self.verbose:
                self.stream.write(d.text)
                self.stream.flush()
                self._mid_line = not d.text.endswith("\n")
        elif d.kind == "content":
            if self.interactive:
                self.stream.write(d.text)
                self.stream.flush()
                self._mid_line = not d.text.endswith("\n")
        elif d.kind == "reset":
            self._write(f"⚠ {d.text}\n", "class:warn")

    def on_model_end(self, agent, turn) -> None:
        if self._mid_line:
            self.stream.write("\n")
            self.stream.flush()
            self._mid_line = False
        if turn is not None and self.fmt == "stream-json":
            self._event("assistant", agent=agent.name, message=turn.to_message(),
                        usage={"prompt_tokens": turn.usage.prompt_tokens, "cached_tokens": turn.usage.cached_tokens,
                               "completion_tokens": turn.usage.completion_tokens})

    def on_tool_start(self, agent, prepared, args: dict) -> None:
        self._event("tool_use", agent=agent.name, id=prepared.call.id, name=prepared.call.name, input=args)

    def on_tool_end(self, agent, prepared, result) -> None:
        if self.fmt == "stream-json":
            self._event("tool_result", agent=agent.name, id=prepared.call.id, is_error=result.is_error,
                        summary=result.summary, content=result.content[:20000])
        elif agent.is_main:
            self.print_tool(prepared.label, result)
        else:
            self._write(f"    ⎿ [{agent.name}] {one_line(prepared.label, 90)}\n", "class:dim")

    def on_compact_start(self, agent, before: int, reason: str) -> None:
        self._write(f"✻ compacting {fmt_tokens(before)} tokens of conversation…\n", "class:dim")

    def on_compact_end(self, agent, before: int, after: int, summary: str | None) -> None:
        if summary:
            self._write(f"  ⎿ compacted: {fmt_tokens(before)} → {fmt_tokens(after)} tokens\n", "class:dim")

    def on_subagent_start(self, agent, description: str) -> None:
        self._write(f"  ⎿ {agent.name} subagent: {description}\n", "class:dim")

    def on_warmup(self, state: str, data: dict | None = None) -> None:
        if state == "failed":
            self._write(f"  ⎿ cache warm-up failed: {(data or {}).get('error')}\n", "class:dim")

    # ---------------------------------------------------------------- questions
    async def _input(self, prompt: str) -> str | None:
        if not self.interactive:
            return None
        try:
            return await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    async def choose(self, title: str, options: list[str]) -> int | None:
        if not self.interactive:
            return None
        self._write(title + "\n" + "".join(f"  {i}. {o}\n" for i, o in enumerate(options, 1)))
        ans = (await self._input("choice> ") or "").strip()
        if not ans.isdigit():
            return None
        idx = int(ans) - 1
        return idx if 0 <= idx < len(options) else None

    async def ask_text(self, title: str, label: str = "") -> str | None:
        self._write(title + "\n")
        return await self._input("> ")

    async def request_permission(self, req, agent):
        if not self.interactive:
            return None
        self._write(f"\n● Permission needed: {req.label}\n", "bold")
        if req.reason:
            self._write(f"  {req.reason}\n", "class:dim")
        if req.danger:
            self._write(f"  ⚠ {req.danger}\n", "class:warn")
        pv = req.preview or {}
        if pv.get("type") == "diff":
            self._write("".join(f"  {l}\n" for l in pv.get("diff", "").split("\n")[:40]), "class:dim")
        always = req.rule_suggestion or ("all edits this session" if req.kind == "edit" else None)
        prompt = "Allow? [y]es" + (f" / [a]lways ({always})" if always else "") + " / [n]o, or type instructions: "
        a = ((await self._input(prompt)) or "").strip()
        if a.lower() in ("y", "yes", ""):
            return Decision(allow=True)
        if a.lower() in ("a", "always") and always:
            if req.kind == "edit":
                return Decision(allow=True, session_mode="acceptEdits")
            return Decision(allow=True, remember_rule=req.rule_suggestion)
        if a.lower() in ("n", "no"):
            return Decision(allow=False, stop=True)
        return Decision(allow=False, feedback=a)

    async def ask_user(self, question: str, options: list[str]) -> str | None:
        if not self.interactive:
            return None
        self._write(f"\n? {question}\n" + "".join(f"  {i}. {o}\n" for i, o in enumerate(options, 1)), "bold")
        ans = ((await self._input("answer> ")) or "").strip()
        if ans.isdigit() and options and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        return ans or "(no answer)"

    async def review_plan(self, plan: str):
        if not self.interactive:
            return None
        self._write("\n● Plan\n" + plan.strip() + "\n", "bold")
        ans = ((await self._input("Proceed? [y]es with auto-accepted edits / [m]anual approval / or type feedback: "))
               or "").strip()
        if ans.lower() in ("y", "yes"):
            return (True, "acceptEdits", None)
        if ans.lower() in ("m", "manual"):
            return (True, "default", None)
        return (False, None, ans or None)

    # ---------------------------------------------------------------- running
    async def run_turn(self, text: str):
        rt = self.rt
        expanded, notes = expand_mentions(text, rt)
        for n in notes:
            self._write(f"  ⎿ {n}\n", "class:dim")
        loop = asyncio.get_running_loop()
        task = loop.create_task(rt.submit(expanded))
        old = None
        try:
            old = signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(task.cancel))
        except (ValueError, OSError):
            pass
        self.busy = True
        try:
            stats = await task
        finally:
            self.busy = False
            if old is not None:
                signal.signal(signal.SIGINT, old)
        if self.interactive:
            self._write(f"  {fmt_duration(stats.elapsed)} · {stats.tool_calls} tool calls · {stats.stopped_reason}\n",
                        "class:faint")
        return stats

    async def run(self, rt, initial: str | None = None) -> None:
        """A simple line-based REPL (used when the interactive prompt is unavailable)."""
        self.rt = rt
        self._write(f"rigcode - {rt.client.model} @ {rt.client.base_url}. Type /help, or /exit to quit.\n", "bold")
        if initial:
            await self.handle(initial)
        while not self.exit_requested:
            line = await self._input("\n> ")
            if line is None:
                break
            await self.handle(line)
            while self._pending and not self.exit_requested:
                await self.handle(self._pending.pop(0))

    async def handle(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        rt = self.rt
        if text.startswith("/") and not text.startswith("//"):
            name, _, args = text[1:].partition(" ")
            cmd = self.commands.find(name)
            if cmd is not None:
                await cmd.run(self, rt, args.strip())
                return
            custom = rt.commands.get(name)
            if custom is not None:
                prompt = custom.render(args.strip(), rt.cwd,
                                       lambda c: build_invocation(rt.shell, c, rt.cwd, track_cwd=False))
                await self.run_turn(prompt)
                return
            if "/" not in name and "." not in name:
                self._write(f"Unknown command /{name}\n", "class:warn")
                return
        if text.startswith("!"):
            tool = rt.tools.get("Bash")
            if tool is not None:
                command = text[1:].strip()
                res = await tool.run({"command": command, "timeout": 600}, ToolContext(rt, rt.main_agent, "user-shell"))
                self._write(res.content + "\n")
                rt.main_agent.add_reminder(f"The user ran this shell command themselves:\n$ {command}\n{res.content[:6000]}")
            return
        await self.run_turn(line)
