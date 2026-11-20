"""Interactive terminal front end.

The prompt stays live while the agent works: type ahead to queue a message (it
is delivered at the agent's next step), Esc or Ctrl+C interrupts. A live area
above the prompt shows progress, streaming text, running tools and subagents;
finished output is committed to the scrollback as rendered markdown.
"""
from __future__ import annotations

import asyncio
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from rigcode import __version__
from rigcode.agent.mentions import expand_mentions
from rigcode.permissions import Decision
from rigcode.tools.base import ToolContext, ToolResult
from rigcode.ui.base import BaseUI
from rigcode.ui.render import BlockSplitter, Renderer, render_todos, term_width
from rigcode.utils.files import list_project_files
from rigcode.utils.text import fmt_duration, fmt_tokens, one_line, strip_ansi

SPINNER = "·✢✳✶✻✽✻✶✳✢"

STYLE = Style.from_dict({
    "prompt": "bold",
    "dim": "#8a8a8a",
    "faint": "#626262",
    "user": "#ffffff bg:#303030",
    "user.prefix": "#8a8a8a bg:#303030",
    "tool.bullet": "#5fd75f",
    "tool.bullet.err": "#ff5f5f",
    "tool.bullet.run": "#d7af5f",
    "tool.label": "bold",
    "error": "#ff5f5f",
    "warn": "#ffaf00",
    "ok": "#5fd75f",
    "thinking": "italic #8a8a8a",
    "spinner": "#d7875f",
    "status": "#d7875f",
    "hint": "#6c6c6c",
    # not "dialog.*": prompt_toolkit's built-in style gives every "dialog" class a blue background
    "choice.title": "bold #87afd7",
    "choice.sel": "bold #87d7ff",
    "choice.opt": "",
    "mode.plan": "#5fafaf",
    "mode.accept": "#af87ff",
    "mode.bypass": "#ff5f5f",
    "bottom-toolbar": "noreverse #6c6c6c",
    "placeholder": "#6c6c6c",
    "agent": "bold #87afd7",
    "queued": "italic #8a8a8a",
    "banner": "bold #d7875f",
})


@dataclass
class AgentView:
    agent: object
    name: str
    description: str = ""
    phase: str = "idle"        # idle | waiting | prompt | thinking | writing | calling | tools | compacting
    started: float = field(default_factory=time.monotonic)
    req_started: float = 0.0
    gen_started: float | None = None
    think_started: float | None = None
    thought_printed: bool = False
    think_tokens: int = 0
    out_tokens: int = 0
    progress: dict | None = None
    reasoning_tail: str = ""
    tool_name: str | None = None
    tool_args_chars: int = 0
    tool_calls: int = 0
    recent_tools: list[str] = field(default_factory=list)
    first_block: bool = True


@dataclass
class ToolView:
    agent: object
    label: str
    started: float
    output: str = ""


@dataclass
class Dialog:
    title: str
    options: list[str]
    future: asyncio.Future
    index: int = 0
    text_option: int | None = None
    typing: bool = False
    text_label: str = ""


def _cut(line: str, width: int) -> int:
    w = 0
    for i, ch in enumerate(line):
        w += get_cwidth(ch)
        if w > width:
            return max(1, i)
    return len(line)


def wrap_lines(text: str, width: int) -> list[str]:
    width = max(10, width)
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.expandtabs(4)
        while get_cwidth(line) > width:
            n = _cut(line, width)
            out.append(line[:n])
            line = line[n:]
        out.append(line)
    return out


def tail_lines(text: str, n: int, width: int) -> list[str]:
    clean = strip_ansi(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in clean.split("\n") if l.strip()]
    return [one_line(l, width) for l in lines[-n:]]


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


class _Completer(Completer):
    def __init__(self, ui: TerminalUI):
        self.ui = ui
        self._files: list[str] = []
        self._dirs: list[str] = []
        self._at = 0.0

    def _project_files(self) -> tuple[list[str], list[str]]:
        if time.monotonic() - self._at > 30 and self.ui.rt is not None:
            try:
                self._files = list_project_files(self.ui.rt.project_root, max_files=50_000)
            except Exception:  # noqa: BLE001
                self._files = []
            dirs = set()
            for f in self._files:
                parts = f.split("/")[:-1]
                for i in range(1, len(parts) + 1):
                    dirs.add("/".join(parts[:i]) + "/")
            self._dirs = sorted(dirs)
            self._at = time.monotonic()
        return self._files, self._dirs

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text and "\n" not in text:
            word = text[1:].lower()
            for name, desc in self.ui.command_entries():
                if name.lower().startswith(word):
                    yield Completion("/" + name, start_position=-len(text), display="/" + name,
                                     display_meta=one_line(desc, 60))
            return
        m = re.search(r"(?:^|\s)@([^\s@\"]*)$", text)
        if not m:
            return
        frag = m.group(1)
        low = frag.lower().replace("\\", "/")
        files, dirs = self._project_files()
        scored = []
        for f in dirs + files:
            fl = f.lower()
            if low and low not in fl and not _subsequence(low, fl):
                continue
            if not low:
                rank = 0 if f.count("/") <= (1 if f.endswith("/") else 0) else 5
            elif fl.startswith(low):
                rank = 0
            elif "/" + low in fl:
                rank = 1
            elif low in fl:
                rank = 2
            else:
                rank = 3
            scored.append((rank, len(f), f))
        for _, _, f in sorted(scored)[:40]:
            yield Completion(f, start_position=-len(frag), display=f)


class TerminalUI(BaseUI):
    interactive = True

    def __init__(self, settings, history_path: Path | None = None):
        from rigcode.ui.commands import build_registry
        self.settings = settings
        self.rt = None
        self.commands = build_registry()
        self.r = Renderer()
        self.thinking_mode = str(settings.get("ui.thinking", "collapsed"))
        self.notify = str(settings.get("ui.notify", "bell"))
        self.tool_output_lines = int(settings.get("ui.tool_output_lines", 6) or 6)
        self.views: dict[int, AgentView] = {}
        self.running: dict[str, ToolView] = {}
        self.dialog: Dialog | None = None
        self.queued: list[str] = []
        self.task: asyncio.Task | None = None
        self.task_label = ""
        self.task_started = 0.0
        self.splitter = BlockSplitter()
        self.think_splitter = BlockSplitter()
        self.warmup: dict = {"state": "off"}
        self.show_todos = False
        self.exit_requested = False
        self._ctx_cache = (0.0, 0)
        self._last_ctrl_c = 0.0
        self._stash = None
        self._ticker: asyncio.Task | None = None
        history = InMemoryHistory()
        if history_path is not None:
            try:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_path))
            except OSError:
                pass
        self.session = PromptSession(
            message=self._message, bottom_toolbar=self._toolbar, style=STYLE, history=history,
            completer=_Completer(self), complete_while_typing=True, auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self._bindings(), erase_when_done=True, reserve_space_for_menu=0,
            placeholder=self._placeholder, prompt_continuation=lambda w, n, s: " " * min(w, 2),
            multiline=False, wrap_lines=True, mouse_support=False)
        self.session.default_buffer.on_text_changed += self._on_text_changed
        self.session.app.timeoutlen = 0.3
        self.session.app.ttimeoutlen = 0.05

    # ================================================================== output helpers
    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def out(self, fragments: list[tuple[str, str]]) -> None:
        print_formatted_text(FormattedText(fragments), style=STYLE, end="")

    def out_line(self, text: str, style: str = "") -> None:
        self.out([(style, text + "\n")])

    def out_ansi(self, text: str) -> None:
        if text.strip():
            print_formatted_text(ANSI(text.rstrip("\n") + "\n"), end="")

    def out_markdown(self, text: str, bullet: str | None = None, style: str | None = None) -> None:
        width = term_width() - 2
        rendered = self.r.markdown(text, width=width, style=style).strip("\n")
        if not rendered.strip():
            return
        lines = rendered.split("\n")
        first = bullet if bullet is not None else "  "
        self.out_ansi("\n" + "\n".join((first if i == 0 else "  ") + l for i, l in enumerate(lines)))

    def notice(self, level: str, text: str) -> None:
        self.on_notice(level, text)

    def _invalidate(self) -> None:
        app = self.session.app
        if app.is_running:
            app.invalidate()

    def _bell(self) -> None:
        if self.notify == "bell":
            try:
                self.session.app.output.bell()
                self.session.app.output.flush()
            except Exception:  # noqa: BLE001
                pass

    def command_entries(self) -> list[tuple[str, str]]:
        return self.commands.entries(self.rt)

    # ================================================================== live area
    def _main_view(self) -> AgentView | None:
        main = self.rt.main_agent if self.rt else None
        return self.views.get(id(main)) if main is not None else None

    def _phase_text(self, v: AgentView, now: float) -> str:
        p = v.phase
        if p == "waiting":
            if self.warmup.get("state") == "running" and getattr(v.agent, "is_main", False):
                return f"Waiting for the cache warm-up to finish{self._warmup_pct()}"
            return "Waiting for the model"
        if p == "prompt":
            pr = v.progress or {}
            total, cache, done = int(pr.get("total") or 0), int(pr.get("cache") or 0), int(pr.get("processed") or 0)
            todo = max(1, total - cache)
            got = max(0, done - cache)
            text = f"Reading prompt {min(100, 100 * got // todo)}% ({fmt_tokens(got)}/{fmt_tokens(todo)} tokens"
            if cache:
                text += f", {fmt_tokens(cache)} cached"
            text += ")"
            ms = float(pr.get("time_ms") or 0)
            if got > 0 and ms > 1000 and total > done:
                text += f" · ~{fmt_duration((total - done) / (got / (ms / 1000)))} left"
            return text
        if p == "thinking":
            secs = now - (v.think_started or now)
            return f"Thinking · {v.think_tokens} tokens · {fmt_duration(secs)}"
        if p == "writing":
            rate = ""
            if v.gen_started and now - v.gen_started > 2:
                rate = f" · {(v.think_tokens + v.out_tokens) / (now - v.gen_started):.1f} tok/s"
            return f"Writing · {v.out_tokens} tokens{rate}"
        if p == "calling":
            return f"Preparing {v.tool_name or 'tool'} call · ~{fmt_tokens(v.tool_args_chars / 3.5)} tokens"
        if p == "tools":
            n = sum(1 for t in self.running.values() if t.agent is v.agent)
            return f"Running {n} tool{'s' if n != 1 else ''}" if n else "Working"
        if p == "compacting":
            return f"Compacting the conversation · {v.out_tokens} tokens"
        return "Working"

    def _warmup_pct(self) -> str:
        pr = self.warmup
        total, cache, done = int(pr.get("total") or 0), int(pr.get("cache") or 0), int(pr.get("processed") or 0)
        if total <= 0:
            return ""
        return f" ({min(100, 100 * max(0, done - cache) // max(1, total - cache))}%)"

    def _live(self, w: int) -> list[tuple[str, str]]:
        now = time.monotonic()
        rt = self.rt
        main = rt.main_agent if rt else None
        mv = self._main_view()
        out: list[tuple[str, str]] = [("", "\n")]
        if mv and mv.phase == "writing" and self.splitter.pending.strip():
            for line in wrap_lines(self.splitter.pending.strip("\n"), w - 2)[-8:]:
                out.append(("", "  " + line + "\n"))
        elif mv and mv.phase == "thinking" and self.thinking_mode != "hidden":
            pending = self.think_splitter.pending if self.thinking_mode == "full" else mv.reasoning_tail
            tail = " ".join(pending.split())
            if tail:
                for line in wrap_lines(tail, w - 4)[-2:]:
                    out.append(("class:thinking", "  " + line + "\n"))
        for cid, tv in list(self.running.items()):
            if tv.agent is not main:
                continue
            out += [("class:tool.bullet.run", "● "), ("class:tool.label", one_line(tv.label, w - 14)),
                    ("class:dim", f"  {fmt_duration(now - tv.started)}\n")]
            if tv.output:
                for i, line in enumerate(tail_lines(tv.output, 4, w - 6)):
                    out.append(("class:dim", ("  ⎿ " if i == 0 else "    ") + line + "\n"))
            for sv in list(self.views.values()):
                if getattr(sv.agent, "depth", 0) == 0 or getattr(sv.agent, "parent_call_id", "") != cid:
                    continue
                out += [("class:dim", "  ⎿ "), ("class:agent", sv.name),
                        ("class:dim", f" · {sv.tool_calls} tool calls · {fmt_duration(now - sv.started)} · "
                                      f"{one_line(self._phase_text(sv, now), max(20, w - 50))}\n")]
                for t in sv.recent_tools[-3:]:
                    out.append(("class:faint", f"      {one_line(t, w - 8)}\n"))
        if main is not None and main.todos.items:
            open_items = main.todos.open_items()
            if self.show_todos:
                out.append(("class:dim", "  Todos (ctrl+t to hide)\n"))
                for t in main.todos.items:
                    mark = {"completed": "☒", "in_progress": "▸"}.get(t.status, "☐")
                    style = "class:faint" if t.status == "completed" else ("bold" if t.status == "in_progress" else "")
                    out.append((style, f"    {mark} {one_line(t.content, w - 8)}\n"))
            elif open_items and self.busy:
                done = len(main.todos.items) - len(open_items)
                cur = next((t for t in main.todos.items if t.status == "in_progress"), None)
                text = f"  ☐ {done}/{len(main.todos.items)} done"
                if cur:
                    text += f" · ▸ {one_line(cur.active_form or cur.content, w - 30)}"
                out.append(("class:dim", text + "  (ctrl+t)\n"))
        for q in self.queued:
            out.append(("class:queued", f"  ↳ queued: {one_line(q, w - 14)}\n"))
        if self.busy:
            spin = SPINNER[int(now * 8) % len(SPINNER)]
            text = self._phase_text(mv, now) if mv and mv.phase != "idle" else (
                "Working" if self.task_label == "turn" else f"Running {self.task_label}")
            out += [("class:spinner", f"{spin} "), ("class:status", one_line(text, w - 30)),
                    ("class:dim", f"  ({fmt_duration(now - self.task_started)} · esc to interrupt)\n")]
        return out

    def _message(self):
        w = term_width() - 1
        frags: list[tuple[str, str]] = []
        if self.rt is not None and (self.busy or self.running or self.dialog or self.queued or self.show_todos):
            frags.extend(self._live(w))
        d = self.dialog
        if d is not None:
            frags.append(("class:choice.title", "\n" + d.title + "\n"))
            if d.typing:
                frags.append(("class:faint", f"  {d.text_label or 'Type your answer'} · Enter to send · Esc to go back\n"))
            else:
                for i, opt in enumerate(d.options):
                    sel = i == d.index
                    frags.append(("class:choice.sel" if sel else "class:choice.opt",
                                  f"{'❯' if sel else ' '} {i + 1}. {one_line(opt, w - 6)}\n"))
                frags.append(("class:faint", "  ↑/↓ and Enter, or press a number · Esc to cancel\n"))
        frags.append(("class:prompt", "  ↳ " if d is not None and d.typing else "> "))
        return frags

    def _placeholder(self):
        if self.dialog is not None:
            return [("class:placeholder", "Type here…")] if self.dialog.typing else ""
        if time.monotonic() - self._last_ctrl_c < 2:
            return [("class:warn", "Press Ctrl+C again to exit")]
        if self.busy:
            return [("class:placeholder", "Type to queue a message (delivered at the agent's next step)")]
        return [("class:placeholder", 'Ask rigcode…  ("/" commands · "@" files · "!" shell · "#" memory)')]

    def _ctx_tokens(self) -> int:
        at, val = self._ctx_cache
        if time.monotonic() - at > 0.5 and self.rt and self.rt.main_agent:
            val = self.rt.main_agent.context_tokens()
            self._ctx_cache = (time.monotonic(), val)
        return val

    def _toolbar(self):
        rt = self.rt
        if rt is None:
            return ""
        w = term_width() - 1
        mode = rt.permissions.mode
        if mode == "plan":
            left = [("class:mode.plan", "⏸ plan mode on"), ("class:hint", " (shift+tab to cycle)")]
        elif mode == "acceptEdits":
            left = [("class:mode.accept", "⏵⏵ accept edits on"), ("class:hint", " (shift+tab to cycle)")]
        elif mode == "bypass":
            left = [("class:mode.bypass", "⚠ bypass permissions on")]
        else:
            left = [("class:hint", "/help · shift+tab: mode")]
        parts = [rt.client.model, f"effort {rt.effort or 'default'}"]
        ctx = self._ctx_tokens()
        if ctx:
            parts.append(f"ctx {fmt_tokens(ctx)}/{fmt_tokens(rt.context_window)}")
        lu = rt.last_usage
        if lu and lu.prompt_tokens:
            parts.append(f"cache {100 * lu.cached_tokens // max(1, lu.prompt_tokens)}%")
        ws = self.warmup.get("state")
        if ws == "running":
            parts.append(f"warming cache{self._warmup_pct()}")
        right = " · ".join(parts)
        used = sum(get_cwidth(t) for _, t in left) + get_cwidth(right)
        return left + [("", " " * max(1, w - used)), ("class:hint", right)]

    def _on_text_changed(self, buf) -> None:
        text = buf.document.text_before_cursor
        completing = (text.startswith("/") and " " not in text) or bool(re.search(r"(?:^|\s)@[^\s@]*$", text))
        self.session.reserve_space_for_menu = 8 if completing else 0

    def _ensure_ticker(self) -> None:
        if self._ticker is None or self._ticker.done():
            self._ticker = asyncio.create_task(self._tick())

    async def _tick(self) -> None:
        while self.busy or self.running or self.warmup.get("state") == "running":
            self._invalidate()
            await asyncio.sleep(0.125)
        self._invalidate()

    # ================================================================== key bindings
    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        selecting = Condition(lambda: self.dialog is not None and not self.dialog.typing)
        typing = Condition(lambda: self.dialog is not None and self.dialog.typing)
        no_dialog = Condition(lambda: self.dialog is None)
        busy = Condition(lambda: self.busy)

        @kb.add("up", filter=selecting, eager=True)
        def _(e):
            self.dialog.index = (self.dialog.index - 1) % max(1, len(self.dialog.options))

        @kb.add("down", filter=selecting, eager=True)
        def _(e):
            self.dialog.index = (self.dialog.index + 1) % max(1, len(self.dialog.options))

        @kb.add("enter", filter=selecting, eager=True)
        def _(e):
            self._choose(self.dialog.index)

        for i in range(1, 10):
            @kb.add(str(i), filter=selecting, eager=True)
            def _(e, i=i):
                if i <= len(self.dialog.options):
                    self._choose(i - 1)

        @kb.add("escape", filter=selecting, eager=True)
        def _(e):
            self._resolve(None, "")

        @kb.add(Keys.Any, filter=selecting)
        def _(e):
            pass   # swallow typing while a choice is shown

        @kb.add("enter", filter=typing, eager=True)
        def _(e):
            text = e.current_buffer.text
            e.current_buffer.reset()
            self._resolve(self.dialog.text_option, text)

        @kb.add("escape", filter=typing, eager=True)
        def _(e):
            e.current_buffer.reset()
            if self.dialog.options:
                self.dialog.typing = False
            else:
                self._resolve(None, "")

        @kb.add("c-c", eager=True)
        def _(e):
            if self.dialog is not None:
                self._resolve(None, "")
                return
            if self.busy:
                self.interrupt()
                return
            buf = e.current_buffer
            if buf.text:
                buf.reset()
                return
            now = time.monotonic()
            if now - self._last_ctrl_c < 2.0:
                e.app.exit(exception=EOFError())
                return
            self._last_ctrl_c = now

            async def clear_hint():
                await asyncio.sleep(2.1)
                self._invalidate()
            asyncio.get_running_loop().create_task(clear_hint())

        @kb.add("escape", filter=busy & no_dialog)
        def _(e):
            self.interrupt()

        @kb.add("s-tab", filter=~has_completions & no_dialog)
        def _(e):
            if self.rt is not None:
                self.rt.cycle_mode()

        @kb.add("escape", "enter", filter=no_dialog | typing)
        def _(e):
            e.current_buffer.insert_text("\n")

        @kb.add("c-j", filter=no_dialog | typing)
        def _(e):
            e.current_buffer.insert_text("\n")

        @kb.add("enter", filter=no_dialog & Condition(
            lambda: self.session.default_buffer.document.text_before_cursor.endswith("\\")))
        def _(e):
            e.current_buffer.delete_before_cursor(1)
            e.current_buffer.insert_text("\n")

        @kb.add("c-o")
        def _(e):
            self.thinking_mode = "full" if self.thinking_mode != "full" else "collapsed"
            self.out_line(f"  ⎿ Thinking display: {self.thinking_mode}", "class:dim")

        @kb.add("c-t")
        def _(e):
            self.show_todos = not self.show_todos

        return kb

    # ================================================================== dialogs
    def _choose(self, index: int) -> None:
        d = self.dialog
        if d is None:
            return
        if d.text_option is not None and index == d.text_option:
            d.index = index
            d.typing = True
            return
        self._resolve(index, "")

    def _resolve(self, index: int | None, text: str) -> None:
        d = self.dialog
        if d is not None and not d.future.done():
            d.future.set_result((index, text))

    async def dialog_choice(self, title: str, options: list[str], *, text_option: int | None = None,
                            text_label: str = "", start_typing: bool = False) -> tuple[int | None, str]:
        while self.dialog is not None:          # one question at a time (parallel subagents)
            await asyncio.sleep(0.1)
        fut = asyncio.get_running_loop().create_future()
        buf = self.session.default_buffer
        self._stash = buf.document if buf.text else None
        buf.reset()
        self.dialog = Dialog(title, options, fut, text_option=text_option, typing=start_typing,
                             text_label=text_label)
        self._bell()
        self._invalidate()
        try:
            return await fut
        finally:
            self.dialog = None
            if self._stash is not None:
                buf.document = self._stash
                self._stash = None
            self._invalidate()

    async def choose(self, title: str, options: list[str]) -> int | None:
        idx, _ = await self.dialog_choice(title, options)
        return idx

    async def ask_text(self, title: str, label: str = "") -> str | None:
        idx, text = await self.dialog_choice(title, [], text_option=0, text_label=label, start_typing=True)
        return None if idx is None else text

    async def request_permission(self, req, agent):
        titles = {"exec": "Bash command", "edit": "Edit file", "web": "Web access", "mcp": "MCP tool",
                  "agent": "Subagent"}
        title = titles.get(req.kind, req.tool_name)
        if req.kind == "edit" and req.preview and req.preview.get("type") == "new_file":
            title = "Create file"
        who = "" if getattr(agent, "is_main", True) else f"  (requested by the {agent.name} subagent)"
        frags = [("", "\n"), ("class:choice.title", f"● {title}"), ("class:dim", who + "\n")]
        target = req.target or req.label
        for line in target.split("\n")[:20]:
            frags.append(("", f"  {line}\n"))
        if req.reason:
            frags.append(("class:dim", f"  {req.reason}\n"))
        if req.danger:
            frags.append(("class:error", f"  ⚠ {req.danger}\n"))
        self.out(frags)
        pv = req.preview
        if pv and pv.get("type") == "diff" and pv.get("diff"):
            self.out_ansi(self.r.ansi(self.r.diff(pv["diff"], max_lines=30, indent=2)))
        elif pv and pv.get("type") == "new_file":
            self.out_ansi(self.r.ansi(self.r.new_file(pv.get("content", ""), max_lines=15, indent=2)))
        options = ["Yes"]
        if req.kind == "edit":
            options.append("Yes, allow all edits during this session (shift+tab)")
        elif req.rule_suggestion:
            options.append(f"Yes, and don't ask again for {req.rule_suggestion} in this project")
        options.append("No, and tell rigcode what to do differently")
        questions = {"Bash command": "Allow this bash command?", "Edit file": "Allow this edit?",
                     "Create file": "Allow creating this file?", "Web access": "Allow this web request?",
                     "MCP tool": "Allow this MCP tool call?", "Subagent": "Allow this subagent?"}
        idx, text = await self.dialog_choice(questions.get(title, f"Allow {title}?"), options,
                                             text_option=len(options) - 1,
                                             text_label="What should rigcode do instead? (empty = just stop)")
        if idx is None:
            return Decision(allow=False, stop=True)
        if idx == 0:
            return Decision(allow=True)
        if idx == len(options) - 1:
            fb = text.strip()
            return Decision(allow=False, feedback=fb or None, stop=not fb)
        if req.kind == "edit":
            return Decision(allow=True, session_mode="acceptEdits")
        return Decision(allow=True, remember_rule=req.rule_suggestion)

    async def ask_user(self, question: str, options: list[str]) -> str | None:
        self.out([("", "\n"), ("class:choice.title", "? "), ("class:tool.label", question + "\n")])
        if options:
            opts = list(options) + ["Type something else"]
            idx, text = await self.dialog_choice("Your answer:", opts, text_option=len(opts) - 1)
            if idx is None:
                answer = "(the user dismissed the question without answering)"
            else:
                answer = text.strip() if idx == len(opts) - 1 else opts[idx]
        else:
            idx, text = await self.dialog_choice("Your answer:", [], text_option=0, start_typing=True)
            answer = "(the user dismissed the question without answering)" if idx is None else text.strip()
        self.out_line(f"  ⎿ {answer}", "class:dim")
        return answer

    async def review_plan(self, plan: str):
        self.out([("", "\n"), ("class:choice.title", "● Plan ready for review\n")])
        self.out_markdown(plan)
        opts = ["Yes, and auto-accept edits", "Yes, and manually approve edits", "No, keep planning"]
        idx, text = await self.dialog_choice("Would you like to proceed?", opts, text_option=2,
                                             text_label="What should change in the plan?")
        if idx is None:
            return (False, None, None)
        if idx == 0:
            return (True, "acceptEdits", None)
        if idx == 1:
            return (True, "default", None)
        return (False, None, text.strip() or None)

    def take_queued_input(self) -> str | None:
        if not self.queued:
            return None
        text = "\n\n".join(self.queued)
        self.queued.clear()
        self._echo_user(text)
        expanded, notes = expand_mentions(text, self.rt)
        for n in notes:
            self.out_line(f"  ⎿ {n}", "class:dim")
        return expanded

    # ================================================================== engine events
    def _view(self, agent) -> AgentView:
        v = self.views.get(id(agent))
        if v is None:
            v = AgentView(agent, getattr(agent, "name", "agent"))
            self.views[id(agent)] = v
        return v

    def on_notice(self, level: str, text: str) -> None:
        if level == "info":
            self.out_line(f"  ⎿ {text}", "class:dim")
        elif level == "warn":
            self.out([("", "\n"), ("class:warn", f"⚠ {text}\n")])
        else:
            self.out([("", "\n"), ("class:error", f"✗ {text}\n")])

    def on_model_start(self, agent, info: dict) -> None:
        v = self._view(agent)
        v.phase, v.req_started, v.gen_started, v.progress = "waiting", time.monotonic(), None, None
        v.think_started, v.thought_printed, v.think_tokens, v.out_tokens = None, False, 0, 0
        v.reasoning_tail, v.tool_name, v.tool_args_chars, v.first_block = "", None, 0, True
        if getattr(agent, "is_main", False):
            self.splitter = BlockSplitter()
            self.think_splitter = BlockSplitter()
        self._ensure_ticker()

    def on_delta(self, agent, d) -> None:
        v = self._view(agent)
        is_main = getattr(agent, "is_main", False)
        k = d.kind
        if k == "progress":
            v.phase, v.progress = "prompt", d.data
        elif k == "reasoning":
            now = time.monotonic()
            if v.think_started is None:
                v.think_started = now
            if v.gen_started is None:
                v.gen_started = now
            v.phase = "thinking"
            v.think_tokens += 1
            v.reasoning_tail = (v.reasoning_tail + d.text)[-800:]
            if is_main and self.thinking_mode == "full":
                for block in self.think_splitter.feed(d.text):
                    self.out_markdown(block, style="italic color(244)")
        elif k == "content":
            if v.gen_started is None:
                v.gen_started = time.monotonic()
            if is_main:
                self._end_thinking(v)
                for block in self.splitter.feed(d.text):
                    self._print_block(v, block)
            v.phase = "writing"
            v.out_tokens += 1
        elif k == "tool_start":
            if is_main:
                self._end_thinking(v)
                self._flush_text(v)
            v.phase, v.tool_name, v.tool_args_chars = "calling", d.name, 0
        elif k == "tool_args":
            v.tool_args_chars += len(d.text)
        elif k == "reset":
            self.on_notice("warn", d.text)
            v.phase, v.think_started, v.thought_printed, v.gen_started = "waiting", None, False, None
            v.think_tokens = v.out_tokens = 0
            if is_main:
                if self.splitter.pending.strip():
                    self.out_line("  ⎿ (partial reply discarded; retrying)", "class:dim")
                self.splitter = BlockSplitter()
                self.think_splitter = BlockSplitter()

    def _print_block(self, v: AgentView, block: str) -> None:
        self.out_markdown(block, bullet="● " if v.first_block else None)
        v.first_block = False

    def _flush_text(self, v: AgentView) -> None:
        rest = self.splitter.flush()
        if rest.strip():
            self._print_block(v, rest)

    def _end_thinking(self, v: AgentView) -> None:
        if v.think_started is None or v.thought_printed:
            return
        v.thought_printed = True
        if self.thinking_mode == "hidden":
            return
        if self.thinking_mode == "full":
            rest = self.think_splitter.flush()
            if rest.strip():
                self.out_markdown(rest, style="italic color(244)")
        secs = time.monotonic() - v.think_started
        self.out([("", "\n"), ("class:thinking", f"✻ Thought for {fmt_duration(secs)} ({v.think_tokens} tokens)"),
                  ("class:faint", "  ctrl+o to show thinking\n" if self.thinking_mode == "collapsed" else "\n")])

    def on_model_end(self, agent, turn) -> None:
        v = self._view(agent)
        if getattr(agent, "is_main", False):
            self._end_thinking(v)
            self._flush_text(v)
        v.phase = "tools" if turn is not None and turn.tool_calls else "idle"

    def on_usage(self, agent, usage) -> None:
        self._ctx_cache = (0.0, self._ctx_cache[1])

    def on_tool_start(self, agent, prepared, args: dict) -> None:
        self.running[prepared.call.id] = ToolView(agent, prepared.label, time.monotonic())
        v = self._view(agent)
        v.phase = "tools"
        v.tool_calls += 1
        if not getattr(agent, "is_main", False):
            v.recent_tools.append(prepared.label)
            del v.recent_tools[:-3]
        self._ensure_ticker()

    def on_tool_output(self, agent, call_id: str, text: str) -> None:
        tv = self.running.get(call_id)
        if tv is not None:
            tv.output = (tv.output + text)[-4000:]

    def on_tool_end(self, agent, prepared, result) -> None:
        self.running.pop(prepared.call.id, None)
        if getattr(agent, "is_main", False):
            self.print_tool(prepared.label, result)

    def print_tool(self, label: str, res: ToolResult, output_lines: int | None = None) -> None:
        w = term_width() - 1
        bullet = "class:tool.bullet.err" if res.is_error else "class:tool.bullet"
        summary = res.summary or one_line(res.content, 160)
        frags = [("", "\n"), (bullet, "● "), ("class:tool.label", one_line(label, w - 4) + "\n"),
                 ("class:dim", "  ⎿  "), ("class:error" if res.is_error else "class:dim", one_line(summary, w - 8) + "\n")]
        if res.is_error:
            detail = [l for l in res.content.strip().split("\n") if l.strip()]
            if len(detail) > 1 or (detail and one_line(detail[0], 300) != one_line(summary, 300)):
                for line in detail[:6]:
                    frags.append(("class:faint", "     " + one_line(line, w - 6) + "\n"))
                if len(detail) > 6:
                    frags.append(("class:faint", f"     … {len(detail) - 6} more lines\n"))
        self.out(frags)
        disp = res.display
        if not disp or res.is_error:
            return
        t = disp.get("type")
        if t == "diff" and disp.get("diff"):
            self.out_ansi(self.r.ansi(self.r.diff(disp["diff"])))
        elif t == "new_file":
            self.out_ansi(self.r.ansi(self.r.new_file(disp.get("content", ""))))
        elif t == "todos":
            self.out_ansi(self.r.ansi(render_todos(disp.get("items") or [])))
        elif t == "output" and disp.get("text", "").strip():
            n = output_lines or self.tool_output_lines
            lines = [l for l in strip_ansi(disp["text"]).replace("\r", "").split("\n")]
            while lines and not lines[-1].strip():
                lines.pop()
            shown = lines[-n:]
            body = [("class:faint", f"     … {len(lines) - n} earlier lines\n")] if len(lines) > n else []
            body += [("class:dim", "     " + one_line(l, w - 6) + "\n") for l in shown]
            self.out(body)
        elif t == "markdown" and disp.get("text", "").strip():
            rendered = self.r.markdown(disp["text"], width=w - 6, style="color(246)").strip("\n").split("\n")
            more = len(rendered) - 10
            text = "\n".join("     " + l for l in rendered[:10])
            if more > 0:
                text += f"\n     \x1b[38;5;242m… {more} more lines\x1b[0m"
            self.out_ansi(text)

    def on_compact_start(self, agent, before: int, reason: str) -> None:
        v = self._view(agent)
        v.phase, v.out_tokens, v.req_started = "compacting", 0, time.monotonic()
        if getattr(agent, "is_main", False):
            why = {"auto": "The context is getting large", "overflow": "The prompt overflowed the context window"}
            self.out([("", "\n"), ("class:dim", f"✻ {why.get(reason, 'Compacting')}: summarizing "
                                                f"{fmt_tokens(before)} tokens of conversation…\n")])
        self._ensure_ticker()

    def on_compact_delta(self, agent, d) -> None:
        v = self._view(agent)
        if d.kind == "progress":
            v.progress = d.data
        elif d.kind in ("content", "reasoning"):
            v.out_tokens += 1

    def on_compact_end(self, agent, before: int, after: int, summary: str | None) -> None:
        v = self._view(agent)
        v.phase = "idle"
        if summary and getattr(agent, "is_main", False):
            self.out_line(f"  ⎿ Conversation compacted: {fmt_tokens(before)} → {fmt_tokens(after)} tokens", "class:dim")

    def on_subagent_start(self, agent, description: str) -> None:
        v = self._view(agent)
        v.description = description
        v.started = time.monotonic()

    def on_subagent_end(self, agent, stats) -> None:
        self.views.pop(id(agent), None)

    def on_warmup(self, state: str, data: dict | None = None) -> None:
        if state == "progress":
            self.warmup.update(data or {})
            self.warmup["state"] = "running"
            return
        self.warmup = {"state": state, **(data or {})}
        if state == "start":
            self._ensure_ticker()
        elif state == "done":
            secs = (data or {}).get("seconds") or 0
            self.out_line(f"  ⎿ Prompt cache ready ({fmt_tokens((data or {}).get('prompt_tokens'))} tokens "
                          f"in {fmt_duration(secs)})", "class:faint")
        elif state == "failed":
            self.out_line(f"  ⎿ Cache warm-up failed: {(data or {}).get('error', '?')}", "class:faint")

    # ================================================================== input handling
    def _echo_user(self, text: str) -> None:
        w = term_width() - 1
        lines = wrap_lines(text.rstrip("\n"), w - 3)
        frags: list[tuple[str, str]] = [("", "\n")]
        for i, line in enumerate(lines[:40]):
            pad = " " * max(0, w - 2 - get_cwidth(line))
            frags.append(("class:user.prefix", "> " if i == 0 else "  "))
            frags.append(("class:user", line + pad + "\n"))
        if len(lines) > 40:
            frags.append(("class:faint", f"  … {len(lines) - 40} more lines\n"))
        self.out(frags)

    def dispatch(self, text: str) -> None:
        if not text.strip():
            return
        stripped = text.strip()
        if stripped.startswith("/") and not stripped.startswith("//"):
            name, _, args = stripped[1:].partition(" ")
            cmd = self.commands.find(name)
            custom = None if cmd else (self.rt.commands.get(name) if self.rt else None)
            if cmd is None and custom is None:
                if "/" in name or "." in name:     # looks like a path: treat as a normal message
                    self._send(text)
                else:
                    self.out_line(f"Unknown command /{name}. Type /help for the list.", "class:warn")
                return
            if self.busy and (custom is not None or not cmd.busy_ok):
                self.out_line(f"/{name} can't run while the agent is working. Wait, or press Esc to interrupt.",
                              "class:warn")
                return
            self._echo_user(stripped)
            if custom is not None:
                self._start_task(self._custom_command(custom, args.strip()), f"/{name}")
            elif cmd.busy_ok and self.busy:
                asyncio.get_running_loop().create_task(self._run_command(cmd, args.strip()))
            else:
                self._start_task(self._run_command(cmd, args.strip()), f"/{name}")
            return
        if stripped.startswith("!"):
            if self.busy:
                self.out_line("Shell commands can't run while the agent is working.", "class:warn")
                return
            self._echo_user(stripped)
            self._start_task(self._bash(stripped[1:].strip()), "shell command")
            return
        if stripped.startswith("#") and not stripped.startswith("##") and len(stripped) > 1 and "\n" not in stripped:
            self._echo_user(stripped)
            self._memory_add(stripped[1:].strip())
            return
        self._send(text)

    def _send(self, text: str) -> None:
        if self.busy:
            self.queued.append(text.strip("\n"))
            self._invalidate()
            return
        self.start_turn(text)

    def submit_message(self, text: str) -> None:
        """Send a message as if typed (queued if a task is running, e.g. from inside a command)."""
        self._send(text)

    def start_turn(self, text: str, echo: bool = True) -> None:
        if echo:
            self._echo_user(text)
        self._start_task(self._turn(text), "turn")

    async def _turn(self, text: str) -> None:
        await self.run_turn(text)

    async def run_turn(self, text: str):
        """Run one main-agent turn (inside the current task) and print its footer."""
        expanded, notes = expand_mentions(text, self.rt)
        for n in notes:
            self.out_line(f"  ⎿ {n}", "class:dim")
        stats = await self.rt.submit(expanded)
        self._footer(stats)
        return stats

    def _footer(self, stats) -> None:
        rt = self.rt
        if stats.stopped_reason == "interrupted":
            self.out_line("  ⎿ Interrupted · tell rigcode what to do instead", "class:warn")
            return
        parts = [fmt_duration(stats.elapsed)]
        if stats.tool_calls:
            parts.append(f"{stats.tool_calls} tool call{'s' if stats.tool_calls != 1 else ''}")
        u = stats.usage
        if u.completion_tokens:
            parts.append(f"{fmt_tokens(u.completion_tokens)} tokens out")
        if u.gen_tps:
            parts.append(f"{u.gen_tps:.1f} tok/s")
        ctx = rt.main_agent.context_tokens()
        parts.append(f"context {fmt_tokens(ctx)} ({100 * ctx // max(1, rt.context_window)}%)")
        if u.prompt_tokens:
            parts.append(f"cache hit {100 * u.cached_tokens // max(1, u.prompt_tokens)}%")
        if stats.stopped_reason not in ("done", "denied"):
            parts.append(f"stopped: {stats.stopped_reason}")
        self.out_line("  " + " · ".join(parts), "class:faint")

    def _start_task(self, coro, label: str) -> None:
        self.task_label = label
        self.task_started = time.monotonic()
        self.task = asyncio.get_running_loop().create_task(self._task_wrapper(coro))
        self._ensure_ticker()

    async def _task_wrapper(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            self.out_line("  ⎿ Interrupted", "class:warn")
        except Exception as e:  # noqa: BLE001
            self.on_notice("error", f"{type(e).__name__}: {e}")
            if self.rt and self.rt.debug:
                self.out_line(traceback.format_exc(), "class:faint")
        finally:
            elapsed = time.monotonic() - self.task_started
            self.task = None
            self.running.clear()
            for key in [k for k, v in self.views.items() if getattr(v.agent, "depth", 0) > 0]:
                self.views.pop(key, None)
            if elapsed > 20:
                self._bell()
            if self.queued and not self.exit_requested:
                text = "\n\n".join(self.queued)
                self.queued.clear()
                self.start_turn(text)
            self._invalidate()

    def interrupt(self) -> None:
        if self.busy:
            self.task.cancel()

    def begin_activity(self, key: str, label: str) -> None:
        """Show a long-running non-tool activity (e.g. /review's subagent) in the live area."""
        self.running[key] = ToolView(self.rt.main_agent, label, time.monotonic())
        self._ensure_ticker()

    def end_activity(self, key: str) -> None:
        self.running.pop(key, None)

    async def _run_command(self, cmd, args: str) -> None:
        await cmd.run(self, self.rt, args)

    async def _custom_command(self, cmd, args: str) -> None:
        from rigcode.tools.shell import build_invocation
        rt = self.rt
        prompt = cmd.render(args, rt.cwd, lambda c: build_invocation(rt.shell, c, rt.cwd, track_cwd=False))
        await self.run_turn(prompt)

    async def _bash(self, command: str) -> None:
        if not command:
            return
        rt = self.rt
        tool = rt.tools.get("Bash")
        if tool is None:
            self.out_line("The Bash tool is disabled.", "class:warn")
            return
        ctx = ToolContext(rt, rt.main_agent, "user-shell")
        self.running["user-shell"] = ToolView(rt.main_agent, f"! {command}", time.monotonic())
        try:
            res = await tool.run({"command": command, "timeout": 600}, ctx)
        finally:
            self.running.pop("user-shell", None)
        self.print_tool(f"! {command}", res, output_lines=40)
        body = res.content if len(res.content) < 6000 else res.content[:3000] + "\n[...]\n" + res.content[-2000:]
        rt.main_agent.add_reminder(f"The user ran this shell command themselves (cwd {rt.cwd}):\n$ {command}\n{body}")

    def _memory_add(self, text: str) -> None:
        root = self.rt.project_root
        target = next((root / n for n in ("AGENTS.md", "CLAUDE.md", "RIGCODE.md") if (root / n).exists()),
                      root / "AGENTS.md")
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            with open(target, "a", encoding="utf-8", newline="\n") as fh:
                if existing and not existing.endswith("\n"):
                    fh.write("\n")
                fh.write(f"- {text}\n")
        except OSError as e:
            self.out_line(f"Could not write {target}: {e}", "class:error")
            return
        self.rt.main_agent.add_reminder(f"The user added this to the project memory file {target.name}: {text}")
        self.out_line(f"  ⎿ Added to {target.name} (the agent sees it from the next message)", "class:dim")

    # ================================================================== main loop
    def welcome(self) -> None:
        rt = self.rt
        s = rt.server
        where = rt.client.base_url.replace("http://", "").replace("https://", "")
        server = s.describe() if s.reachable else "server unreachable"
        lines = [("class:banner", f"✻ rigcode {__version__}"), ("", f"  {rt.client.model} @ {where}"),
                 ("class:dim", f"  ({server})\n")]
        info = [f"cwd {rt.cwd}", f"mode {rt.permissions.mode}", f"effort {rt.effort or 'default'}"]
        if rt.memory:
            info.append(f"{len(rt.memory)} memory file{'s' if len(rt.memory) != 1 else ''}")
        n_mcp = sum(1 for x in rt.mcp.servers.values() if x.status == "connected")
        if n_mcp:
            info.append(f"{n_mcp} MCP server{'s' if n_mcp != 1 else ''}")
        lines.append(("class:dim", "  " + " · ".join(info) + "\n"))
        lines.append(("class:faint", "  /help for commands · esc to interrupt · shift+tab to change mode\n"))
        self.out(lines)
        history = len(rt.main_agent.messages) - rt.main_agent.n_prefix()
        if history:
            how = "re-reading them in the background" if rt.warmup_state == "running" else "the first reply re-reads them"
            self.out_line(f"  ⎿ Resumed session {rt.session.id} ({history} messages, "
                          f"~{fmt_tokens(rt.main_agent.context_tokens())} tokens; {how})", "class:dim")
            reply = rt.main_agent.last_assistant_text()
            if reply:
                self.out_markdown(reply[-2000:], bullet="● ")

    async def run(self, rt, initial: str | None = None) -> None:
        self.rt = rt
        self.welcome()
        if initial:
            self.dispatch(initial)
        while not self.exit_requested:
            try:
                text = await self.session.prompt_async()
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            self.dispatch(text)
        await self.shutdown()

    def request_exit(self) -> None:
        self.exit_requested = True
        app = self.session.app
        if app.is_running:
            app.exit(exception=EOFError())

    async def shutdown(self) -> None:
        self.exit_requested = True
        if self.busy:
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 5)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        if self._ticker and not self._ticker.done():
            self._ticker.cancel()
        rt = self.rt
        if rt is not None and rt.session.path.exists():
            self.out_line(f"\nResume this session with: rigcode -r {rt.session.id}", "class:faint")
