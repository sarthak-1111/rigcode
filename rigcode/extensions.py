"""Discovery of user extensions: subagents, skills, memory files, slash commands.

Search order (later wins on name clashes):
  built-in  ->  ~/.rigcode/...  ->  ~/.claude/... (if compat.claude_user_dirs)
            ->  <project>/.claude/... (if compat.claude_project_dirs)  ->  <project>/.rigcode/...
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from rigcode.utils.frontmatter import as_list, parse_frontmatter
from rigcode.utils.paths import rigcode_home

EFFORTS = {"low", "medium", "high", "max", "none"}


def _roots(settings, project_root: Path, sub: str) -> list[tuple[str, Path]]:
    roots = [("user", rigcode_home() / sub)]
    if settings.get("compat.claude_user_dirs", False):
        roots.append(("user-claude", Path.home() / ".claude" / sub))
    if settings.get("compat.claude_project_dirs", True):
        roots.append(("project-claude", project_root / ".claude" / sub))
    roots.append(("project", project_root / ".rigcode" / sub))
    return roots


# ---------------------------------------------------------------- subagents

@dataclass
class AgentDef:
    name: str
    description: str
    prompt: str
    tools: list[str] | None = None      # allowlist; None = every tool except Task
    read_only: bool = False
    effort: str | None = None           # None = same as the main agent (shares the prompt cache)
    max_turns: int | None = None
    model: str | None = None            # profile name from settings.profiles
    source: str = "builtin"


def _agent_from_text(text: str, source: str, fallback_name: str) -> AgentDef | None:
    meta, body = parse_frontmatter(text)
    name = str(meta.get("name") or fallback_name).strip()
    desc = str(meta.get("description") or "").strip()
    if not name or not desc:
        return None
    tools = as_list(meta.get("tools")) or None
    if tools and any(t in ("*", "all") for t in tools):
        tools = None
    effort = str(meta.get("effort") or "").strip().lower() or None
    if effort not in EFFORTS:
        effort = None
    model = str(meta.get("model") or "").strip() or None
    if model in ("inherit", "sonnet", "opus", "haiku", "fable"):  # Claude Code model names -> inherit
        model = None
    try:
        max_turns = int(meta["max_turns"]) if meta.get("max_turns") else None
    except (TypeError, ValueError):
        max_turns = None
    read_only = meta.get("read_only")
    if read_only is None:
        read_only = bool(tools) and not any(t in ("Edit", "Write", "MultiEdit", "Bash", "NotebookEdit") for t in tools)
    return AgentDef(name=name, description=" ".join(desc.split()), prompt=body.strip(), tools=tools,
                    read_only=bool(read_only), effort=effort, max_turns=max_turns, model=model, source=source)


def load_agents(settings, project_root: Path) -> dict[str, AgentDef]:
    agents: dict[str, AgentDef] = {}
    try:
        pkg = resources.files("rigcode") / "prompts" / "agents"
        for entry in sorted(pkg.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".md"):
                a = _agent_from_text(entry.read_text(encoding="utf-8"), "builtin", entry.name[:-3])
                if a:
                    agents[a.name] = a
    except (FileNotFoundError, OSError):
        pass
    for source, root in _roots(settings, project_root, "agents"):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            try:
                a = _agent_from_text(path.read_text(encoding="utf-8"), source, path.stem)
            except OSError:
                continue
            if a:
                agents[a.name] = a
    disabled = set(settings.get("agents.disabled", []) or [])
    return {k: v for k, v in agents.items() if k not in disabled}


# ---------------------------------------------------------------- skills

@dataclass
class Skill:
    name: str
    description: str
    directory: Path
    path: Path
    source: str

    def load_body(self) -> str:
        try:
            _, body = parse_frontmatter(self.path.read_text(encoding="utf-8"))
        except OSError as e:
            return f"(could not read skill: {e})"
        return body.strip()


def load_skills(settings, project_root: Path) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    for source, root in _roots(settings, project_root, "skills"):
        if not root.is_dir():
            continue
        for skill_md in sorted(root.glob("*/SKILL.md")):
            try:
                meta, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = str(meta.get("name") or skill_md.parent.name).strip()
            desc = " ".join(str(meta.get("description") or "").split())
            if name and desc:
                skills[name] = Skill(name, desc, skill_md.parent, skill_md, source)
    return skills


# ---------------------------------------------------------------- memory files

@dataclass
class MemoryFile:
    path: Path
    content: str
    scope: str


_IMPORT_RE = re.compile(r"(?<![\w`])@((?:~|\.{1,2})?[\w./\\-]+\.\w+)")


def _expand_imports(text: str, base: Path, depth: int = 0, seen: set | None = None) -> str:
    if depth >= 3:
        return text
    seen = seen or set()
    in_fence = False
    out = []
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if in_fence:
            out.append(line)
            continue

        def repl(m: re.Match) -> str:
            raw = m.group(1)
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (base / p)
            try:
                p = p.resolve()
                key = str(p).lower()
                if key in seen or not p.is_file() or p.stat().st_size > 200_000:
                    return m.group(0)
                seen.add(key)
                inner = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return m.group(0)
            return f"\n<!-- imported from {raw} -->\n" + _expand_imports(inner, p.parent, depth + 1, seen)

        out.append(_IMPORT_RE.sub(repl, line))
    return "\n".join(out)


def load_memory(settings, project_root: Path, cwd: Path) -> list[MemoryFile]:
    names = list(settings.get("memory_files", ["AGENTS.md", "CLAUDE.md", "RIGCODE.md"]) or [])
    local_names = [n.replace(".md", ".local.md") for n in names]
    candidates: list[tuple[Path, str]] = []
    for n in ("AGENTS.md", "RIGCODE.md"):
        candidates.append((rigcode_home() / n, "user"))
    if settings.get("compat.claude_user_dirs", False):
        candidates.append((Path.home() / ".claude" / "CLAUDE.md", "user"))
    try:
        cwd_r = cwd.resolve()
    except OSError:
        cwd_r = cwd
    chain = list(reversed([cwd_r, *cwd_r.parents]))
    for d in chain:
        scope = "project" if str(d).lower().startswith(str(project_root).lower()) else "parent"
        for n in names + local_names:
            candidates.append((d / n, scope))
        for n in names:
            candidates.append((d / ".rigcode" / n, scope))
    files, seen_hash, seen_path = [], set(), set()
    per_file = 12000
    total_budget = int(settings.get("context.max_memory_chars", 20000))
    used = 0
    for path, scope in candidates:
        key = str(path).lower()
        if key in seen_path or not path.is_file():
            continue
        seen_path.add(key)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _expand_imports(text, path.parent).strip()
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if not text or h in seen_hash:
            continue
        seen_hash.add(h)
        if len(text) > per_file:
            text = text[:per_file] + "\n[... truncated ...]"
        if used + len(text) > total_budget:
            text = text[: max(0, total_budget - used)] + "\n[... truncated: memory budget reached ...]"
        used += len(text)
        files.append(MemoryFile(path, text, scope))
        if used >= total_budget:
            break
    return files


# ---------------------------------------------------------------- custom slash commands

@dataclass
class CustomCommand:
    name: str
    description: str
    body: str
    path: Path
    source: str
    argument_hint: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None

    def render(self, arguments: str, cwd: Path, shell_argv=None) -> str:
        text = self.body
        args = arguments.split()
        text = text.replace("$ARGUMENTS", arguments)
        for i in range(9, 0, -1):
            text = text.replace(f"${i}", args[i - 1] if len(args) >= i else "")
        if shell_argv and any(t == "Bash" or t.startswith("Bash(") for t in self.allowed_tools):
            def run(m: re.Match) -> str:
                try:
                    proc = subprocess.run(shell_argv(m.group(1)), cwd=str(cwd), capture_output=True, timeout=30)
                    out = proc.stdout.decode("utf-8", errors="replace") + proc.stderr.decode("utf-8", errors="replace")
                except (OSError, subprocess.SubprocessError) as e:
                    out = f"(command failed: {e})"
                return out.strip()
            text = re.sub(r"!`([^`]+)`", run, text)
        return text.strip()


def load_commands(settings, project_root: Path) -> dict[str, CustomCommand]:
    commands: dict[str, CustomCommand] = {}
    for source, root in _roots(settings, project_root, "commands"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root).with_suffix("")
            name = ":".join(rel.parts)
            try:
                meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            desc = str(meta.get("description") or "").strip() or body.strip().split("\n", 1)[0][:80]
            commands[name] = CustomCommand(
                name=name, description=desc, body=body, path=path, source=source,
                argument_hint=meta.get("argument-hint") or meta.get("argument_hint"),
                allowed_tools=as_list(meta.get("allowed-tools") or meta.get("allowed_tools")),
                model=meta.get("model"))
    return commands
