"""Permission modes and allow/ask/deny rules.

Rule syntax (same shape as Claude Code's):
  Bash                     every shell command
  Bash(npm test:*)         commands starting with "npm test"
  Bash(git status)         exactly this command
  Edit(src/**)             file edits under src/ (also covers Write/MultiEdit)
  Read(~/.ssh/**)          reads of matching paths (also covers Glob/Grep targets)
  WebFetch(domain:docs.python.org)
  mcp__github              every tool of an MCP server; mcp__github__create_issue for one tool
Order: deny > plan-mode restriction > ask > bypass > allow > mode defaults.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from rigcode.tools.bashsafety import command_matches_rule, danger_reason, suggest_rule
from rigcode.utils.globmatch import glob_match
from rigcode.utils.paths import is_within, is_within_any, resolve_path, rigcode_home

MODES = ["default", "acceptEdits", "plan", "bypass"]
MODE_ALIASES = {"accept-edits": "acceptEdits", "acceptedits": "acceptEdits", "auto": "acceptEdits",
                "yolo": "bypass", "bypasspermissions": "bypass", "dontask": "bypass", "ask": "default"}
MODE_DESCRIPTIONS = {
    "default": "asks before edits, commands and web access",
    "acceptEdits": "auto-approves file edits in the workspace",
    "plan": "read-only planning; propose a plan before changing anything",
    "bypass": "approves everything except deny rules",
}

_RULE_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*(?:\((.*)\))?\s*$", re.S)


def normalize_mode(mode: str) -> str:
    m = (mode or "default").strip()
    return m if m in MODES else MODE_ALIASES.get(m.lower(), "default")


@dataclass
class Rule:
    tool: str
    pattern: str | None
    text: str

    @classmethod
    def parse(cls, text: str) -> Rule | None:
        m = _RULE_RE.match(text or "")
        if not m:
            return None
        return cls(m.group(1), m.group(2).strip() if m.group(2) is not None else None, text.strip())


@dataclass
class PermissionRequest:
    tool_name: str
    label: str
    kind: str
    target: str
    reason: str | None = None
    danger: str | None = None
    rule_suggestion: str | None = None     # rule offered for "don't ask again"
    preview: dict | None = None            # e.g. a diff for edits


@dataclass
class Decision:
    allow: bool
    feedback: str | None = None
    stop: bool = False                      # user wants the whole turn to stop
    remember_rule: str | None = None        # persist this allow rule (project local)
    session_mode: str | None = None         # switch mode for the rest of the session


@dataclass
class PermissionManager:
    settings: object
    project_root: Path
    mode: str = "default"
    previous_mode: str = "default"
    session_allow: list[str] = field(default_factory=list)
    workspace: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = normalize_mode(self.settings.get("permissions.mode", "default"))
        extra = [resolve_path(p, self.project_root) for p in self.settings.get("permissions.additional_directories", [])]
        self.workspace = [self.project_root, *extra]

    # ---------------------------------------------------------------- mode
    def set_mode(self, mode: str) -> str:
        mode = normalize_mode(mode)
        if mode != self.mode:
            self.previous_mode = self.mode
            self.mode = mode
        return self.mode

    def cycle_mode(self) -> str:
        order = ["default", "acceptEdits", "plan"]
        nxt = order[(order.index(self.mode) + 1) % len(order)] if self.mode in order else "default"
        return self.set_mode(nxt)

    # ---------------------------------------------------------------- rules
    def _rules(self, kind: str) -> list[Rule]:
        texts = list(self.settings.get(f"permissions.{kind}", []) or [])
        if kind == "allow":
            texts += self.session_allow
        return [r for r in (Rule.parse(t) for t in texts) if r]

    def _tool_matches(self, rule: Rule, tool) -> bool:
        if rule.tool == tool.name:
            return True
        if tool.name.startswith("mcp__") and tool.name.startswith(rule.tool + "__"):
            return True
        if rule.tool in ("Edit", "Write", "MultiEdit") and tool.kind == "edit":
            return True
        if rule.tool == "Read" and tool.kind == "read" and tool.name in ("Read", "Glob", "Grep"):
            return True
        return False

    def _path_matches(self, pattern: str, raw_path: str, cwd: Path) -> bool:
        path = resolve_path(raw_path, cwd)
        pat = os.path.expanduser(pattern.strip())
        if pat.startswith("//"):
            pat = pat[1:]
        if os.path.isabs(pat) or re.match(r"^[A-Za-z]:[\\/]", pat):
            return glob_match(pat.replace("\\", "/"), str(path).replace("\\", "/"))
        if is_within(path, self.project_root):
            rel = os.path.relpath(str(path), str(self.project_root)).replace("\\", "/")
            return glob_match(pat, rel) or glob_match(pat.rstrip("/") + "/**", rel)
        return False

    def rule_matches(self, rule: Rule, tool, args: dict, cwd: Path) -> bool:
        if not self._tool_matches(rule, tool):
            return False
        if rule.pattern is None or rule.pattern in ("", "*"):
            return True
        target = tool.permission_target(args)
        if tool.kind == "exec" and tool.name == "Bash":
            return command_matches_rule(rule.pattern, target)
        if tool.kind in ("edit",) or tool.name == "Read":
            return bool(target) and self._path_matches(rule.pattern, target, cwd)
        if tool.name == "WebFetch":
            if rule.pattern.startswith("domain:"):
                host = (urlparse(target if "://" in target else "https://" + target).hostname or "").lower()
                dom = rule.pattern[7:].strip().lower()
                return host == dom or host.endswith("." + dom)
            return glob_match(rule.pattern, target)
        return glob_match(rule.pattern, target)

    # ---------------------------------------------------------------- evaluation
    def _protected(self, tool, args: dict, cwd: Path) -> str | None:
        if tool.kind != "edit":
            return None
        target = tool.permission_target(args)
        if not target:
            return None
        path = resolve_path(target, cwd)
        name = path.name.lower()
        parent = path.parent.name.lower()
        if parent in (".rigcode", ".claude") and name.startswith("settings"):
            return "modifies rigcode/agent permission settings"
        if is_within(path, rigcode_home()):
            return "modifies rigcode's global configuration"
        if ".git" in [p.lower() for p in path.parts]:
            return "modifies git internals"
        return None

    def evaluate(self, tool, args: dict, cwd: Path) -> tuple[str, str | None]:
        """Returns ("allow"|"ask"|"deny", reason)."""
        for rule in self._rules("deny"):
            if self.rule_matches(rule, tool, args, cwd):
                return "deny", f"blocked by deny rule {rule.text}"
        read_only = tool.is_read_only(args)
        if self.mode == "plan" and not read_only and tool.kind != "interact":
            return "deny", ("plan mode is active: only read-only tools may be used. Finish investigating, then call "
                            "ExitPlanMode with your plan")
        for rule in self._rules("ask"):
            if self.rule_matches(rule, tool, args, cwd):
                return "ask", f"matches ask rule {rule.text}"
        if self.mode == "bypass":
            return "allow", None
        protected = self._protected(tool, args, cwd)
        if protected:
            return "ask", protected
        for rule in self._rules("allow"):
            if self.rule_matches(rule, tool, args, cwd):
                return "allow", None
        if read_only and tool.kind != "web":
            return "allow", None
        if tool.kind == "edit":
            target = tool.permission_target(args)
            inside = bool(target) and is_within_any(resolve_path(target, cwd), self.workspace)
            if self.mode == "acceptEdits" and inside:
                return "allow", None
            return "ask", None if inside else "edits a file outside the workspace"
        return "ask", None

    def build_request(self, tool, args: dict, label: str, reason: str | None) -> PermissionRequest:
        target = tool.permission_target(args)
        danger = danger_reason(target) if tool.name == "Bash" else None
        suggestion = None
        if tool.name == "Bash":
            suggestion = suggest_rule(target)
        elif tool.name == "WebFetch":
            host = urlparse(target if "://" in target else "https://" + target).hostname
            suggestion = f"WebFetch(domain:{host})" if host else None
        elif tool.name.startswith("mcp__") or tool.kind in ("web",):
            suggestion = tool.name
        return PermissionRequest(tool_name=tool.name, label=label, kind=tool.kind, target=target,
                                 reason=reason, danger=danger, rule_suggestion=suggestion)

    def remember(self, rule: str, persist: bool = True) -> Path | None:
        if rule not in self.session_allow:
            self.session_allow.append(rule)
        if persist:
            from rigcode.config import add_permission_rule
            return add_permission_rule(self.project_root, rule, "allow", "local")
        return None
