"""Built-in slash commands. Each handler is `async def handler(ui, rt, args: str)`.

`ui` is the front end (TerminalUI or PlainUI): out_line, out, out_ansi,
out_markdown, r (Renderer), choose, ask_text, run_turn, print_tool,
begin_activity/end_activity, request_exit, thinking_mode.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rich.table import Table
from rich.text import Text

from rigcode import __version__
from rigcode.agent.loop import _msg_text as message_text
from rigcode.agent.prompts import load_prompt
from rigcode.agent.session import SessionStore
from rigcode.config import add_permission_rule, settings_paths
from rigcode.permissions import MODE_ALIASES, MODE_DESCRIPTIONS, MODES, Rule
from rigcode.ui.render import render_diff
from rigcode.utils.paths import display_path, resolve_path
from rigcode.utils.text import est_tokens, fmt_duration, fmt_tokens, one_line


@dataclass
class Command:
    name: str
    description: str
    run: Callable
    aliases: tuple[str, ...] = ()
    hint: str = ""
    busy_ok: bool = False      # safe to run while the agent is working


class CommandRegistry:
    def __init__(self) -> None:
        self.commands: dict[str, Command] = {}
        self._alias: dict[str, str] = {}

    def add(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd
        for a in cmd.aliases:
            self._alias[a] = cmd.name

    def find(self, name: str) -> Command | None:
        name = name.lower()
        return self.commands.get(name) or self.commands.get(self._alias.get(name, ""))

    def entries(self, rt) -> list[tuple[str, str]]:
        out = [(c.name, c.description) for c in self.commands.values()]
        if rt is not None:
            out += [(n, f"{c.description} (custom)") for n, c in rt.commands.items() if n not in self.commands]
        return sorted(out)


_REGISTRY = CommandRegistry()


def command(name: str, description: str, *, aliases: tuple[str, ...] = (), hint: str = "", busy_ok: bool = False):
    def deco(fn):
        _REGISTRY.add(Command(name, description, fn, aliases, hint, busy_ok))
        return fn
    return deco


def build_registry() -> CommandRegistry:
    return _REGISTRY


def _table(rows: list[tuple[str, str]], key_style: str = "bold") -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(no_wrap=True)
    t.add_column()
    for k, v in rows:
        t.add_row(Text(k, style=key_style), Text(v))
    return t


def _print_rows(ui, rows: list[tuple[str, str]], indent: int = 2) -> None:
    rendered = ui.r.ansi(_table(rows)).rstrip("\n")
    ui.out_ansi("\n".join(" " * indent + l for l in rendered.split("\n")))


def _git(rt, *args: str, timeout: float = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(["git", *args], cwd=str(rt.cwd), capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)
    out = proc.stdout.decode("utf-8", errors="replace") + proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out


# ---------------------------------------------------------------- general

@command("help", "Show commands and keyboard shortcuts", aliases=("?",), busy_ok=True)
async def cmd_help(ui, rt, args: str) -> None:
    rows = [(f"/{c.name}" + (f" {c.hint}" if c.hint else ""), c.description)
            for c in sorted(_REGISTRY.commands.values(), key=lambda c: c.name)]
    ui.out_line("\nCommands", "bold")
    _print_rows(ui, rows)
    if rt.commands:
        ui.out_line("\nCustom commands", "bold")
        _print_rows(ui, [(f"/{n}" + (f" {c.argument_hint}" if c.argument_hint else ""), c.description)
                         for n, c in sorted(rt.commands.items())])
    ui.out_line("\nInput", "bold")
    _print_rows(ui, [
        ("Enter", "send (while the agent works: queue for its next step)"),
        ("Alt+Enter, Ctrl+J, \\+Enter", "new line"),
        ("Esc / Ctrl+C", "interrupt the agent; Ctrl+C twice on an empty prompt exits (or Ctrl+D)"),
        ("Shift+Tab", "cycle permission mode: default → accept edits → plan"),
        ("Ctrl+O / Ctrl+T", "toggle full thinking display / the todo list"),
        ("@path", "attach a file or directory listing to your message (Tab completes)"),
        ("!command", "run a shell command yourself; the agent sees the output"),
        ("#note", "append a note to the project's AGENTS.md memory"),
    ])


@command("exit", "Exit rigcode", aliases=("quit", "q"), busy_ok=True)
async def cmd_exit(ui, rt, args: str) -> None:
    ui.request_exit()


@command("clear", "Start a new conversation (the prompt cache is kept)", aliases=("new", "reset"))
async def cmd_clear(ui, rt, args: str) -> None:
    rt.new_conversation()
    ui.out_line("  ⎿ Started a new conversation.", "class:dim")


@command("compact", "Summarize the conversation to free up context", hint="[what to focus on]")
async def cmd_compact(ui, rt, args: str) -> None:
    await rt.main_agent.compact(args.strip() or None, reason="manual")


@command("status", "Version, model, server, session and context info", busy_ok=True)
async def cmd_status(ui, rt, args: str) -> None:
    s = rt.server
    ctx = rt.main_agent.context_tokens()
    rows = [
        ("rigcode", __version__),
        ("Model", f"{rt.client.model} (effort {rt.effort or 'server default'}, tool calls: {rt.client.tool_mode})"),
        ("Server", f"{rt.client.base_url} · " + (s.describe() if s.reachable else f"unreachable: {s.error}")),
        ("Slots", f"pinned: main {rt.main_slot}, subagents {rt.sub_slots or '-'}" if rt.main_slot is not None
         else "not pinned (the server restores each agent's cached state)"),
        ("Subagents", f"context {rt.subagent_mode()}, up to {rt.subagent_concurrency} in parallel"
         + (" (fresh only)" if rt.subagent_concurrency > 1 else "")),
        ("Session", f"{rt.session.id}  ({rt.session.path})"),
        ("Directory", f"{rt.cwd} (project root {rt.project_root})"),
        ("Permissions", f"{rt.permissions.mode} - {MODE_DESCRIPTIONS.get(rt.permissions.mode, '')}"),
        ("Context", f"{fmt_tokens(ctx)} of {fmt_tokens(rt.context_window)} tokens; auto-compact at "
                    f"{fmt_tokens(rt.compact_threshold())}"),
        ("Cache warm-up", rt.warmup_state),
    ]
    _print_rows(ui, rows)


# ---------------------------------------------------------------- context and usage

@command("context", "Show what is using the context window", busy_ok=True)
async def cmd_context(ui, rt, args: str) -> None:
    agent = rt.main_agent
    n_pre = agent.n_prefix()
    tools = est_tokens(json.dumps(rt.tool_schemas)) if rt.client.tool_mode == "native" else 0
    session = est_tokens(rt.project_context)
    core = max(0, sum(est_tokens(m.get("content", "")) for m in agent.messages[:n_pre]) - session)
    by_role = {"user": 0, "assistant": 0, "tool": 0}
    reasoning = 0
    for m in agent.messages[n_pre:]:
        role = m.get("role", "user")
        if role == "assistant" and m.get("reasoning_content"):
            reasoning += est_tokens(m["reasoning_content"])
        by_role[role] = by_role.get(role, 0) + est_tokens(message_text(m))
    total = agent.context_tokens()
    window = rt.context_window
    rows = [
        ("Tool definitions", tools), ("System prompt", core),
        ("Project context (environment, repo map, memory)", session),
        ("Your messages (incl. attachments)", by_role.get("user", 0)),
        ("Assistant replies", by_role.get("assistant", 0) - reasoning),
        ("Assistant reasoning", reasoning), ("Tool results", by_role.get("tool", 0)),
    ]
    _print_rows(ui, [(k, f"{fmt_tokens(v):>7}  {100 * v / max(1, window):5.1f}%") for k, v in rows])
    ui.out_line(f"  Total ≈ {fmt_tokens(total)} of {fmt_tokens(window)} tokens "
                f"({100 * total / max(1, window):.1f}%); auto-compact at {fmt_tokens(rt.compact_threshold())}. "
                "Estimates; the server's count is exact after each reply.", "class:dim")


@command("cost", "Token usage, cache hits and speed for this session", aliases=("usage", "stats"), busy_ok=True)
async def cmd_cost(ui, rt, args: str) -> None:
    u = rt.usage
    rows = [
        ("Requests", str(rt.requests)),
        ("Prompt tokens", f"{u.prompt_tokens:,} ({u.cached_tokens:,} from cache, "
                          f"{100 * u.cached_tokens // max(1, u.prompt_tokens)}%)"),
        ("Output tokens", f"{u.completion_tokens:,} (incl. reasoning)"),
        ("Prompt processing", f"{u.prompt_tps:.1f} tok/s" if u.prompt_tps else "-"),
        ("Generation", f"{u.gen_tps:.1f} tok/s" if u.gen_tps else "-"),
        ("Session time", fmt_duration(time.time() - rt.started)),
    ]
    _print_rows(ui, rows)


# ---------------------------------------------------------------- model and modes

@command("model", "Show the model, or switch to a profile or model name", hint="[profile|model]")
async def cmd_model(ui, rt, args: str) -> None:
    name = args.strip()
    if not name:
        s = rt.server
        ui.out_line(f"  Model: {rt.client.model} @ {rt.client.base_url}", "")
        ui.out_line("  Server: " + (s.describe() if s.reachable else f"unreachable ({s.error})"), "class:dim")
        if s.models:
            ui.out_line("  Served models: " + ", ".join(s.models), "class:dim")
        profiles = rt.settings.get("profiles", {}) or {}
        if profiles:
            ui.out_line("  Profiles: " + ", ".join(sorted(profiles)) + "  (switch with /model <profile>)", "class:dim")
        return
    desc = await rt.switch_model(name)
    ui.out_line(f"  ⎿ Now using {desc}", "class:dim")
    ui.out_line("  ⎿ The next request re-reads the whole conversation on the new model.", "class:dim")


@command("effort", "Show or set the reasoning effort", hint="[low|medium|high|max|none]", aliases=("reasoning",))
async def cmd_effort(ui, rt, args: str) -> None:
    if not args.strip():
        ui.out_line(f"  Reasoning effort: {rt.effort or 'server default'}  "
                    "(low = quick, high/max = slower and more thorough, none = no thinking)", "")
        return
    old = rt.effort
    try:
        new = rt.set_effort(args.strip())
    except ValueError as e:
        ui.out_line(str(e), "class:error")
        return
    ui.out_line(f"  ⎿ Reasoning effort: {old or 'default'} → {new or 'default'}", "class:dim")
    tokens = rt.main_agent.context_tokens()
    if new != old and tokens > 2000:
        speed = rt.usage.prompt_tps or 25.0
        ui.out_line(f"  ⎿ The effort setting is part of the prompt, so the next request re-reads the whole "
                    f"conversation (~{fmt_tokens(tokens)} tokens, about {fmt_duration(tokens / speed)} here).",
                    "class:warn")


@command("mode", "Show or set the permission mode", hint="[default|acceptEdits|plan|bypass]",
         aliases=("permission-mode",), busy_ok=True)
async def cmd_mode(ui, rt, args: str) -> None:
    want = args.strip()
    if not want:
        for m in MODES:
            mark = "❯" if m == rt.permissions.mode else " "
            ui.out_line(f"  {mark} {m:<12} {MODE_DESCRIPTIONS[m]}", "bold" if m == rt.permissions.mode else "")
        return
    if want not in MODES and want.lower() not in MODE_ALIASES and want.lower() not in [m.lower() for m in MODES]:
        ui.out_line(f"Unknown mode '{want}'. Modes: {', '.join(MODES)}", "class:error")
        return
    want = next((m for m in MODES if m.lower() == want.lower()), want)
    mode = rt.set_mode(want)
    ui.out_line(f"  ⎿ Permission mode: {mode} - {MODE_DESCRIPTIONS[mode]}", "class:dim")


@command("plan", "Switch to plan mode (read-only investigation, then a plan for approval)", busy_ok=True)
async def cmd_plan(ui, rt, args: str) -> None:
    rt.set_mode("plan")
    ui.out_line("  ⎿ Plan mode on. The agent will investigate and propose a plan before changing anything.",
                "class:dim")
    if args.strip():
        ui.submit_message(args.strip())


@command("thinking", "How to show the model's thinking", hint="[collapsed|full|hidden]", busy_ok=True)
async def cmd_thinking(ui, rt, args: str) -> None:
    want = args.strip().lower()
    if want not in ("collapsed", "full", "hidden"):
        ui.out_line(f"  Thinking display: {ui.thinking_mode} (choose collapsed, full or hidden; ctrl+o toggles)", "")
        return
    ui.thinking_mode = want
    ui.out_line(f"  ⎿ Thinking display: {want}", "class:dim")


# ---------------------------------------------------------------- permissions and tools

@command("permissions", "Show permission rules, or add one", hint="[allow|deny|ask <rule>]",
         aliases=("perms", "allowed-tools"), busy_ok=True)
async def cmd_permissions(ui, rt, args: str) -> None:
    parts = args.split(None, 1)
    if len(parts) == 2 and parts[0] in ("allow", "deny", "ask"):
        kind, rule = parts[0], parts[1].strip()
        if Rule.parse(rule) is None:
            ui.out_line(f"Invalid rule '{rule}'. Examples: Bash(npm test:*), Edit(src/**), WebFetch(domain:x.org)",
                        "class:error")
            return
        path = add_permission_rule(rt.project_root, rule, kind, "local")
        lst = rt.settings.data.setdefault("permissions", {}).setdefault(kind, [])
        if rule not in lst:
            lst.append(rule)
        ui.out_line(f"  ⎿ Added {kind} rule {rule} (saved to {path})", "class:dim")
        return
    p = rt.permissions
    ui.out_line(f"  Mode: {p.mode} - {MODE_DESCRIPTIONS[p.mode]}", "bold")
    for kind in ("allow", "ask", "deny"):
        rules = list(rt.settings.get(f"permissions.{kind}", []) or [])
        if kind == "allow":
            rules += [f"{r} (this session)" for r in p.session_allow if r not in rules]
        ui.out_line(f"  {kind}: " + (", ".join(rules) if rules else "(none)"), "")
    ui.out_line("  Workspace: " + ", ".join(str(w) for w in p.workspace), "class:dim")
    paths = settings_paths(rt.project_root)
    ui.out_line(f"  Rules live in {paths['local']} (personal) or {paths['project']} (shared). "
                "Add one with /permissions allow <rule>.", "class:dim")


@command("add-dir", "Let the agent edit files in another directory", hint="<path>")
async def cmd_add_dir(ui, rt, args: str) -> None:
    if not args.strip():
        ui.out_line("Usage: /add-dir <path>", "class:warn")
        return
    path = resolve_path(args.strip(), rt.cwd)
    if not path.is_dir():
        ui.out_line(f"Not a directory: {path}", "class:error")
        return
    if path not in rt.permissions.workspace:
        rt.permissions.workspace.append(path)
    rt.main_agent.add_reminder(f"The user added {path} to the workspace; you may work on files there too.")
    ui.out_line(f"  ⎿ Added {path} to the workspace for this session.", "class:dim")


@command("tools", "List the tools available to the agent", busy_ok=True)
async def cmd_tools(ui, rt, args: str) -> None:
    rows = [(t.name, one_line(t.description.split(". ")[0], 90)) for t in rt.tools.all()]
    _print_rows(ui, rows)


@command("agents", "List the subagents the main agent can delegate to", busy_ok=True)
async def cmd_agents(ui, rt, args: str) -> None:
    rows = []
    for name, a in sorted(rt.agents.items()):
        tools = ", ".join(a.tools) if a.tools else "all tools"
        flags = [a.source] + (["read-only"] if a.read_only else []) + ([f"effort {a.effort}"] if a.effort else [])
        rows.append((name, f"{one_line(a.description, 80)}\n[{', '.join(flags)}; {tools}]"))
    _print_rows(ui, rows or [("(none)", "")])
    mode = rt.subagent_mode()
    how = {"fork": "they start from a copy of this conversation",
           "fresh": "they start from just the system prompt and their brief",
           "auto": "they start from a copy of this conversation unless it is very large"}[mode]
    ui.out_line(f"  Subagent context: {mode} ({how}; agents.context). Add your own agents as Markdown files "
                "in .rigcode/agents/ or .claude/agents/.", "class:dim")


@command("skills", "List available skills", busy_ok=True)
async def cmd_skills(ui, rt, args: str) -> None:
    rows = [(s.name, f"{one_line(s.description, 90)} [{s.source}]") for s in sorted(rt.skills.values(), key=lambda s: s.name)]
    _print_rows(ui, rows or [("(none)", "Add skills as <name>/SKILL.md under .rigcode/skills/ or .claude/skills/")])


@command("mcp", "Show MCP server status", busy_ok=True)
async def cmd_mcp(ui, rt, args: str) -> None:
    lines = rt.mcp.status_lines()
    if not lines:
        ui.out_line("  No MCP servers configured (add them to .mcp.json or settings \"mcp_servers\").", "class:dim")
    for line in lines:
        ui.out_line("  " + line, "")


@command("memory", "Show the memory files loaded into the system prompt", busy_ok=True)
async def cmd_memory(ui, rt, args: str) -> None:
    if not rt.memory:
        ui.out_line("  No memory files loaded. Create AGENTS.md (or run /init); use #note to add lines.", "class:dim")
        return
    for m in rt.memory:
        ui.out_line(f"  {m.path}  ({m.scope}, {len(m.content):,} chars)", "")
    ui.out_line("  Edits take effect in the next session (the system prompt is fixed per session to keep the "
                "cache warm). Use #note to tell the current session too.", "class:dim")


@command("bashes", "List background shells", aliases=("shells",), busy_ok=True)
async def cmd_bashes(ui, rt, args: str) -> None:
    lines = rt.shells.describe()
    for line in lines or ["No background shells."]:
        ui.out_line("  " + line, "")


@command("todos", "Show the agent's todo list", busy_ok=True)
async def cmd_todos(ui, rt, args: str) -> None:
    ui.out_line("  " + rt.main_agent.todos.render_text().replace("\n", "\n  "), "")


# ---------------------------------------------------------------- workflows

@command("init", "Create or improve AGENTS.md for this project", hint="[instructions]")
async def cmd_init(ui, rt, args: str) -> None:
    prompt = load_prompt("init")
    if args.strip():
        prompt += f"\n\nAdditional instructions from the user: {args.strip()}"
    await ui.run_turn(prompt)


@command("commit", "Commit the current changes with a well-written message", hint="[instructions]")
async def cmd_commit(ui, rt, args: str) -> None:
    prompt = load_prompt("commit")
    if args.strip():
        prompt += f"\n\nAdditional instructions from the user: {args.strip()}"
    await ui.run_turn(prompt)


@command("review", "Review uncommitted changes with the reviewer subagent", hint="[focus]")
async def cmd_review(ui, rt, args: str) -> None:
    agent_def = rt.agents.get("reviewer")
    if agent_def is None:
        ui.out_line("The reviewer agent is not available (disabled?).", "class:error")
        return
    brief = ("Review the uncommitted changes in this repository. Start with `git status --short` and `git diff HEAD` "
             "(read untracked files that matter), then read the surrounding code as needed. Report concrete, verified "
             "problems ranked by severity with file:line references, and end with a one-line overall assessment. "
             "Skip style nitpicks unless they hide a bug.")
    if args.strip():
        brief += f"\nFocus on: {args.strip()}"
    key = "review"
    ui.begin_activity(key, "Task[reviewer](Review uncommitted changes)")
    try:
        report, stats = await rt.run_subagent(agent_def, "Review uncommitted changes", brief,
                                              SimpleNamespace(agent=rt.main_agent, call_id=key))
    finally:
        ui.end_activity(key)
    ui.out_markdown(report or "(the reviewer returned no report)", bullet="● ")
    ui.out_line(f"  ⎿ reviewer: {stats}", "class:faint")
    if report:
        rt.main_agent.add_reminder("The user ran /review. The reviewer subagent reported:\n\n" + report
                                   + "\n\nThe user may ask you to address these findings.")


@command("diff", "Show uncommitted changes", busy_ok=True)
async def cmd_diff(ui, rt, args: str) -> None:
    code, stat = await asyncio.to_thread(_git, rt, "diff", "HEAD", "--stat")
    if code != 0:
        code, stat = await asyncio.to_thread(_git, rt, "diff", "--stat")
    if code != 0:
        ui.out_line(f"git diff failed: {one_line(stat, 200)}", "class:error")
        return
    if not stat.strip():
        ui.out_line("  No uncommitted changes to tracked files.", "class:dim")
        return
    ui.out_line(stat.rstrip(), "class:dim")
    _, patch = await asyncio.to_thread(_git, rt, "diff", "HEAD")
    budget = 300
    for chunk in patch.split("diff --git ")[1:]:
        header, _, body = chunk.partition("\n")
        name = header.split(" b/")[-1] if " b/" in header else header
        ui.out_line(f"\n  {name}", "bold")
        text = render_diff(body, max_lines=max(10, min(80, budget)), indent=2)
        ui.out_ansi(ui.r.ansi(text))
        budget -= 80
        if budget <= 0:
            ui.out_line("  … more files not shown", "class:faint")
            break


@command("undo", "Revert the file changes from the agent's most recent turn that changed files",
         aliases=("rewind",))
async def cmd_undo(ui, rt, args: str) -> None:
    cp = rt.checkpoints.last_with_changes()
    if cp is None:
        ui.out_line("  Nothing to undo (no file changes recorded this session).", "class:dim")
        return
    files = [display_path(Path(p), rt.cwd) for p in cp.files]
    shown = "\n".join(f"    {f}" for f in files[:12]) + (f"\n    … {len(files) - 12} more" if len(files) > 12 else "")
    idx = await ui.choose(f"Revert {len(files)} file(s) changed during \"{one_line(cp.label, 50)}\"?\n{shown}",
                          ["Yes, revert them", "No"])
    if idx != 0:
        return
    results = rt.checkpoints.undo_last()
    for path, status in results:
        ui.out_line(f"  ⎿ {display_path(Path(path), rt.cwd)}: {status}", "class:dim")
        rt.file_state.forget(Path(path))
    rt.main_agent.add_reminder(
        f"The user reverted the file changes from the turn \"{cp.label}\": {', '.join(files)}. Those files are back "
        "to their earlier contents; Read them again before editing.")


# ---------------------------------------------------------------- sessions

@command("resume", "Resume a previous session in this project", hint="[session id]")
async def cmd_resume(ui, rt, args: str) -> None:
    if args.strip():
        path = SessionStore.resolve(rt.project_root, args.strip())
        if path is None:
            ui.out_line(f"No session matching '{args.strip()}'.", "class:error")
            return
    else:
        infos = [i for i in SessionStore.list(rt.project_root, limit=15) if i.id != rt.session.id and i.n_messages]
        if not infos:
            ui.out_line("  No previous sessions in this project.", "class:dim")
            return
        labels = [f"{time.strftime('%m-%d %H:%M', time.localtime(i.mtime))} · {i.n_messages:>3} msgs · "
                  f"{one_line(i.title or '(untitled)', 60)}" for i in infos]
        idx = await ui.choose("Resume which session?", labels)
        if idx is None:
            return
        path = infos[idx].path
    n = rt.resume(path)
    last_user = next((message_text(m) for m in reversed(rt.main_agent.messages)
                      if m.get("role") == "user" and not m.get("_meta")), "")
    how = "re-reading them in the background" if rt.warmup_state == "running" else "the first reply re-reads them"
    ui.out_line(f"  ⎿ Resumed {path.stem} ({n} messages, ~{fmt_tokens(rt.main_agent.context_tokens())} tokens; "
                f"{how}).", "class:dim")
    if last_user:
        ui.out_line(f"  ⎿ Last message: {one_line(last_user, 100)}", "class:faint")
    reply = rt.main_agent.last_assistant_text()
    if reply:
        ui.out_markdown(reply[-3000:], bullet="● ")


@command("export", "Save the conversation as a Markdown file", hint="[path]")
async def cmd_export(ui, rt, args: str) -> None:
    path = resolve_path(args.strip(), rt.cwd) if args.strip() else rt.cwd / f"rigcode-{rt.session.id}.md"
    lines = [f"# rigcode session {rt.session.id}", "", f"Model: {rt.client.model} · exported {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for m in rt.main_agent.messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            lines += ["## User" if not m.get("_meta") else "## (rigcode note)", "", message_text(m).strip(), ""]
        elif role == "assistant":
            if m.get("content"):
                lines += ["## Assistant", "", m["content"].strip(), ""]
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                lines += [f"**Tool call** `{fn.get('name')}`", "", "```json", fn.get("arguments", ""), "```", ""]
        elif role == "tool":
            body = (m.get("content") or "")[:4000]
            lines += [f"**Result** ({m.get('name', 'tool')})", "", "```", body, "```", ""]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as e:
        ui.out_line(f"Could not write {path}: {e}", "class:error")
        return
    ui.out_line(f"  ⎿ Exported to {path}", "class:dim")


# ---------------------------------------------------------------- settings and diagnostics

def _mask(data):
    if isinstance(data, dict):
        return {k: ("***" if any(s in k.lower() for s in ("key", "token", "secret", "password")) and v else _mask(v))
                for k, v in data.items()}
    if isinstance(data, list):
        return [_mask(v) for v in data]
    return data


@command("config", "Show settings, or set one for this session", hint="[key [value]]", busy_ok=True)
async def cmd_config(ui, rt, args: str) -> None:
    key, _, value = args.strip().partition(" ")
    if not key:
        for src in rt.settings.sources:
            state = "loaded" if src.loaded else ("error: " + src.error if src.error else "not present")
            ui.out_line(f"  {src.scope:<8} {src.path}  ({state})", "class:dim")
        text = json.dumps(_mask(rt.settings.data), indent=2, ensure_ascii=False)
        ui.out_ansi(ui.r.markdown(f"```json\n{text}\n```"))
        return
    if not value.strip():
        ui.out_line(f"  {key} = {json.dumps(_mask(rt.settings.get(key)), ensure_ascii=False)}", "")
        return
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.strip()
    rt.settings.set(key, parsed)
    ui.out_line(f"  ⎿ {key} = {json.dumps(parsed, ensure_ascii=False)} for this session "
                "(edit .rigcode/settings.json to make it permanent).", "class:dim")


@command("doctor", "Check the server connection, tools and configuration")
async def cmd_doctor(ui, rt, args: str) -> None:
    from rigcode.doctor import run_doctor
    await run_doctor(rt.settings, rt.project_root, lambda text, style="": ui.out_line(text, style),
                     chat_test="quick" not in args)
