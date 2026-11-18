"""Tool interface, result type, argument validation and the registry."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rigcode.utils.text import one_line

if TYPE_CHECKING:  # pragma: no cover
    from rigcode.agent.loop import Agent
    from rigcode.agent.runtime import Runtime


@dataclass
class ToolResult:
    content: str                      # what the model sees
    is_error: bool = False
    summary: str | None = None        # one-line UI summary ("Read 120 lines")
    display: dict | None = None       # rich UI payload: {"type": "diff"|"todos"|"output"|"markdown", ...}
    images: list[dict] = field(default_factory=list)   # [{"mime": "image/png", "data": b64}]
    stop_turn: bool = False           # end the agent turn after this result (e.g. user said stop)

    @classmethod
    def error(cls, message: str, summary: str | None = None) -> ToolResult:
        return cls(content=message, is_error=True, summary=summary or one_line(message, 100))


@dataclass
class ToolContext:
    rt: Runtime
    agent: Agent
    call_id: str = ""

    @property
    def cwd(self):
        return self.rt.cwd


class Tool:
    name: str = ""
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}
    read_only: bool = False
    # Permission category: read | edit | exec | web | agent | interact | mcp
    kind: str = "read"
    # Parameter names other harnesses use -> ours (applied before validation).
    aliases: dict[str, str] = {}

    def apply_aliases(self, args: dict) -> dict:
        for alias, canonical in self.aliases.items():
            if alias in args and canonical not in args:
                args[canonical] = args.pop(alias)
        return args

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description.strip(), "parameters": self.parameters}}

    def is_read_only(self, args: dict) -> bool:
        return self.read_only

    def permission_target(self, args: dict) -> str:
        """The string permission rules match against (path, command, URL...)."""
        return ""

    def label(self, args: dict, ctx: ToolContext | None = None) -> str:
        """Short human-readable call description for the UI: Name(arg)."""
        target = self.permission_target(args)
        return f"{self.name}({one_line(target, 70)})" if target else self.name

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------- validation

_TYPE_NAMES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _coerce(value: Any, spec: dict) -> tuple[Any, str | None]:
    typ = spec.get("type")
    if isinstance(typ, list):
        for t in typ:
            v, err = _coerce(value, {**spec, "type": t})
            if err is None:
                return v, None
        return value, f"expected one of {typ}"
    if typ is None or value is None:
        return value, None
    if typ == "integer":
        if isinstance(value, bool):
            return value, "expected integer"
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip()), None
        return value, "expected integer"
    if typ == "number":
        if isinstance(value, bool):
            return value, "expected number"
        if isinstance(value, (int, float)):
            return value, None
        try:
            return float(value), None
        except (TypeError, ValueError):
            return value, "expected number"
    if typ == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.strip().lower() in ("true", "1", "yes"), None
        if isinstance(value, int):
            return bool(value), None
        return value, "expected boolean"
    if typ == "string":
        if isinstance(value, str):
            return value, None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2), None
        return value, "expected string"
    if typ == "array":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    value = parsed
            except json.JSONDecodeError:
                return value, "expected array"
        if not isinstance(value, list):
            return value, "expected array"
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            out = []
            for i, item in enumerate(value):
                v, err = _coerce(item, item_spec)
                if err:
                    return value, f"item {i}: {err}"
                out.append(v)
            value = out
        return value, None
    if typ == "object":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    value = parsed
            except json.JSONDecodeError:
                return value, "expected object"
        if not isinstance(value, dict):
            return value, "expected object"
        if "properties" in spec:
            return validate_object(value, spec)
        return value, None
    return value, None


def validate_object(args: dict, schema: dict) -> tuple[dict, str | None]:
    props = schema.get("properties") or {}
    out = dict(args)
    missing = [k for k in schema.get("required") or [] if out.get(k) is None]
    if missing:
        return out, f"missing required parameter(s): {', '.join(missing)}"
    for key, spec in props.items():
        if key in out and out[key] is not None:
            v, err = _coerce(out[key], spec)
            if err:
                return out, f"parameter '{key}': {err}"
            if "enum" in spec and v not in spec["enum"]:
                return out, f"parameter '{key}' must be one of {spec['enum']}"
            out[key] = v
    return out, None


def validate_args(tool: Tool, args: dict) -> tuple[dict, str | None]:
    return validate_object(args, tool.parameters)


# ---------------------------------------------------------------- registry

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        if tool is None:
            # Tolerate case/underscore drift from the model (read_file -> Read handled by aliases).
            low = name.lower().replace("_", "").replace("-", "")
            for t in self._tools.values():
                if t.name.lower().replace("_", "") == low:
                    return t
            alias = TOOL_ALIASES.get(name.lower())
            if alias:
                return self._tools.get(alias)
        return tool

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def subset(self, names: list[str] | None, exclude: set[str] | None = None) -> ToolRegistry:
        reg = ToolRegistry()
        exclude = exclude or set()
        for t in self._tools.values():
            if t.name in exclude:
                continue
            if names is None or t.name in names or ("mcp" in names and t.name.startswith("mcp__")):
                reg.register(t)
        return reg

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


# Common names other harnesses use, mapped to ours.
TOOL_ALIASES = {
    "read_file": "Read", "readfile": "Read", "view": "Read", "cat": "Read", "open_file": "Read",
    "write_file": "Write", "writefile": "Write", "create_file": "Write",
    "edit_file": "Edit", "str_replace": "Edit", "strreplacefile": "Edit", "replace_in_file": "Edit",
    "str_replace_editor": "Edit", "apply_edit": "Edit",
    "multi_edit": "MultiEdit",
    "shell": "Bash", "run_shell_command": "Bash", "execute_command": "Bash", "run_command": "Bash",
    "terminal": "Bash", "exec": "Bash", "powershell": "Bash",
    "glob_search": "Glob", "find_files": "Glob", "list_files": "Glob",
    "grep_search": "Grep", "search": "Grep", "search_files": "Grep", "ripgrep": "Grep",
    "fetch": "WebFetch", "fetch_url": "WebFetch", "fetchurl": "WebFetch", "web_fetch": "WebFetch",
    "search_web": "WebSearch", "searchweb": "WebSearch", "web_search": "WebSearch",
    "todo_write": "TodoWrite", "settodolist": "TodoWrite", "update_todos": "TodoWrite",
    "agent": "Task", "subagent": "Task", "dispatch_agent": "Task",
    "ask_user": "AskUser", "askuserquestion": "AskUser", "ask_followup_question": "AskUser",
    "exit_plan_mode": "ExitPlanMode",
}
