"""Builds the prompt pieces.

Layout chosen for prompt-cache reuse on the server:
  [tools][system: CORE + PROJECT CONTEXT]   <- fixed for the whole session, shared by every agent
  [user: <env snapshot reminder> + first message] ...   <- append-only from here on

llama.cpp can only rewind a hybrid model's cache to a checkpoint, and it creates
them at the start of user messages. The first one is never evicted, so everything
before the first user message (tools, system prompt, project context) is reusable
for /clear, compaction and subagents. That is why the stable project context lives
in the system message, while volatile facts (date, git status) go in the first user
message: they would break the shared prefix if they lived in the system message.
"""
from __future__ import annotations

import platform
import subprocess
import time
from collections import Counter
from importlib import resources
from pathlib import Path

from rigcode.utils.files import list_project_files
from rigcode.utils.paths import find_git_root


def load_prompt(name: str) -> str:
    return (resources.files("rigcode") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def _git(root: Path, *args: str, timeout: float = 5) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace").strip()


def git_snapshot(root: Path) -> dict | None:
    git_root = find_git_root(root)
    if not git_root:
        return None
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    status = _git(root, "status", "--porcelain=v1") or ""
    lines = [l for l in status.splitlines() if l.strip()]
    log = _git(root, "log", "--oneline", "-5", "--no-decorate") or ""
    return {"root": str(git_root), "branch": branch, "changes": lines, "log": log}


def _about(n: int) -> str:
    """Rounded file count, so adding a file rarely changes the prompt (and its cache)."""
    if n < 10:
        return str(n)
    digits = len(str(n)) - 1
    return f"~{round(n, -digits):,}"


def repo_overview(root: Path, max_lines: int = 80) -> str:
    try:
        files = list_project_files(root, max_files=30_000)
    except Exception:  # noqa: BLE001
        return ""
    if not files:
        return "(empty directory)"
    top_dirs: Counter = Counter()
    sub_dirs: dict[str, Counter] = {}
    top_files = []
    for f in files:
        parts = f.split("/")
        if len(parts) == 1:
            top_files.append(f)
            continue
        top_dirs[parts[0]] += 1
        if len(parts) > 2:
            sub_dirs.setdefault(parts[0], Counter())[parts[1]] += 1
    lines: list[str] = []
    for d in sorted(top_dirs):
        lines.append(f"{d}/ ({_about(top_dirs[d])} files)")
        subs = sub_dirs.get(d)
        if subs and len(subs) <= 12:
            lines.append("  " + ", ".join(f"{s}/" for s in sorted(subs)))
        elif subs:
            shown = ", ".join(f"{s}/" for s in sorted(s for s, _ in subs.most_common(8)))
            lines.append(f"  {len(subs)} subdirectories, largest: {shown}")
    files_line = sorted(top_files)
    if len(files_line) > 30:
        files_line = files_line[:30] + ["... more files"]
    lines.extend(files_line)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    return f"(about {_about(len(files))} files, .gitignore respected)\n" + "\n".join(lines)


def build_project_context(rt) -> str:
    """Stable facts about the project. Sent in the system message, so keep anything
    that changes during a session (date, git status) out of here."""
    s = rt.settings
    parts: list[str] = []
    os_name = f"{platform.system()} {platform.release()}"
    env = [
        f"- Working directory: {rt.cwd}",
        f"- Project root: {rt.project_root}",
        f"- Platform: {os_name}; Bash tool shell: {rt.shell.name}",
        f"- Model: {rt.client.model} ({rt.server.kind} server)",
        "- The current date and git status are in a reminder in the first user message.",
    ]
    parts.append("# Environment\n" + "\n".join(env))
    if s.get("context.repo_map", True):
        overview = repo_overview(rt.project_root, int(s.get("context.repo_map_max_lines", 80)))
        if overview:
            parts.append("# Project layout\n" + overview)
    if rt.memory:
        mem = ["# Project instructions",
               "Instructions from the user's memory files. Follow them; they override default behavior."]
        for m in rt.memory:
            mem.append(f"\n## {m.path}\n{m.content}")
        parts.append("\n".join(mem))
    if rt.skills:
        lines = ["# Skills", "Specialized instructions you can load with the Skill tool when a task matches:"]
        for sk in sorted(rt.skills.values(), key=lambda x: x.name):
            lines.append(f"- {sk.name}: {sk.description}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_env_snapshot(rt, resumed: bool = False) -> str:
    """Volatile facts for the first user message of a conversation."""
    head = "This session was resumed. Current environment:" if resumed else "Environment at the start of this conversation:"
    lines = [head, f"- Date: {time.strftime('%Y-%m-%d %H:%M')} ({time.strftime('%A')})"]
    git = git_snapshot(rt.project_root)
    if not git:
        lines.append("- Git: not a git repository")
        return "\n".join(lines)
    n = len(git["changes"])
    lines.append(f"- Git branch: {git['branch']}; {n} uncommitted change{'s' if n != 1 else ''}")
    if git["changes"]:
        shown = git["changes"][:15]
        if n > 15:
            shown.append(f"... {n - 15} more")
        lines.append("\nUncommitted changes (git status --porcelain):\n" + "\n".join(shown))
    if git["log"]:
        lines.append("\nRecent commits:\n" + git["log"])
    return "\n".join(lines)


def subagent_brief(agent_def, description: str, prompt: str, *, forked: bool = False, env: str | None = None) -> str:
    """The first message a subagent sees. A forked subagent gets it as the result of the
    Task call, right after the main agent's conversation; a fresh one as its first user
    message, after the shared system prompt."""
    tools_note = ", ".join(agent_def.tools) if agent_def.tools else "all tools except Task"
    mode = (" You are read-only: do not modify files, and use Bash only for read-only commands."
            if agent_def.read_only else "")
    if forked:
        intro = (f"You are now running as the \"{agent_def.name}\" subagent, forked from the main agent's "
                 "conversation above. That conversation is background only: do not continue the main agent's "
                 "work, do not reply to the user, and do not repeat work that is already done. Work only on the "
                 "task below.")
    else:
        intro = f"You are running as the \"{agent_def.name}\" subagent, working for the main agent."
    parts = [
        f"<system-reminder>\n{intro}{mode} Tools you may use: {tools_note}. You cannot ask the user questions: if "
        "something is ambiguous, make a reasonable assumption and mention it in your report. Your final message is "
        "returned to the main agent as your report; make it self-contained and concise.\n</system-reminder>",
    ]
    if env:
        parts.append(reminder(env))
    parts.append(f"<role>\n{agent_def.prompt}\n</role>")
    parts.append(f"# Task: {description}\n\n{prompt}")
    return "\n\n".join(parts)


def compact_instruction(extra: str | None = None) -> str:
    extra_text = f"\nAdditional focus requested by the user: {extra}" if extra else ""
    return load_prompt("compact").replace("{extra}", extra_text).strip()


def post_compact_message(summary: str, recent_user: list[str], todos_text: str | None, files: list[str]) -> str:
    parts = ["<system-reminder>\nThis session was compacted to save context. The summary below replaces the "
             "earlier conversation. Continue the work from where it left off without recapping it to the user.\n"
             "</system-reminder>",
             f"<conversation-summary>\n{summary.strip()}\n</conversation-summary>"]
    if recent_user:
        parts.append("Most recent user messages (verbatim, oldest first):\n" +
                     "\n".join(f"<user-message>\n{u}\n</user-message>" for u in recent_user))
    if todos_text:
        parts.append(f"Current todo list:\n{todos_text}")
    if files:
        parts.append("Files recently read or edited: " + ", ".join(files))
    return "\n\n".join(parts)


def reminder(text: str) -> str:
    return f"<system-reminder>\n{text.strip()}\n</system-reminder>"
