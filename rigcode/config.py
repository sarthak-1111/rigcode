"""Layered settings: defaults < ~/.rigcode/settings.json < .rigcode/settings.json
< .rigcode/settings.local.json < environment < CLI flags.

Permission rule lists and hook lists are unioned across layers; everything else
deep-merges with later layers winning.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rigcode.utils.paths import rigcode_home

DEFAULTS: dict[str, Any] = {
    "provider": {
        # auto = find a local server (llama.cpp, Ollama, LM Studio, vLLM) on its default port.
        "base_url": "auto",
        "api_key": "none",
        # auto = the server's only model, or the one it has loaded.
        "model": "auto",
        # None = auto-detect from the server, falling back to 131072.
        "context_window": None,
        # Cap on generated tokens per request (thinking + answer). None = server default.
        "max_output_tokens": 32768,
        "temperature": None,
        "top_p": None,
        # low | medium | high | max | none (thinking off) | null (don't send; server default)
        "reasoning_effort": "low",
        # auto | native | text  (text = tool calls written as <tool_call> JSON in the reply)
        "tool_mode": "auto",
        # all | turn | none: which past reasoning_content to send back to the server.
        "keep_reasoning": "all",
        # Optional hard cap on thinking tokens per request (llama.cpp reasoning_budget_tokens).
        "reasoning_budget_tokens": None,
        "extra_body": {},
        "extra_headers": {},
        # Seconds without any data before giving up. Large: a queued request gets no bytes until a slot frees.
        "request_timeout": 3600,
        "connect_timeout": 15,
        "max_retries": 3,
        # Pre-process the static prompt prefix in the background at startup (llama.cpp).
        "warmup": True,
        # llama.cpp-only knobs; ignored for other servers.
        "llama_cpp": {
            "cache_prompt": True,
            "return_progress": True,
            # Pin agents to fixed slots. Leave this off unless the server runs with --no-cache-idle-slots:
            # by default llama.cpp moves idle slots to its RAM cache, and only unpinned requests get
            # their saved state back (a request pinned to a cleared slot re-reads its whole prompt).
            "slot_pinning": False,
            # With pinning: None = main agent on the highest slot, subagents on the slots below it.
            "main_slot": None,
            "subagent_slots": None,
        },
    },
    "profile": None,
    "profiles": {},
    "permissions": {
        # default | acceptEdits | plan | bypass
        "mode": "default",
        "allow": [],
        "deny": [],
        "ask": [],
        "additional_directories": [],
    },
    "hooks": {},
    "mcp_servers": {},
    "context": {
        "auto_compact": True,
        # Summarize the conversation when the prompt reaches this many tokens or this fraction of
        # the context window, whichever comes first. With slow prompt processing a huge context is
        # expensive to rebuild if the server cache is ever lost, so the token cap is deliberately low.
        "compact_at_tokens": 100000,
        "compact_at": 0.85,
        # Recent user messages kept verbatim after compaction (char budget).
        "keep_recent_chars": 6000,
        "repo_map": True,
        "repo_map_max_lines": 80,
        "max_tool_output_chars": 12000,
        "read_max_chars": 30000,
        "max_memory_chars": 20000,
    },
    "tools": {
        "glob_max_results": 100,
        "background_shells": True,
        "web": True,
    },
    "agents": {
        # How a subagent starts. fork: from a copy of the main conversation (its brief arrives as the
        # Task result), which reuses the server's cache of that conversation. fresh: from just the
        # system prompt and its brief. auto: fork while the main conversation is under fork_max_tokens
        # (default: the compaction threshold or half the context window, whichever is smaller).
        "context": "auto",
        "fork_max_tokens": None,
        # Parallel subagents only apply to fresh ones (parallel forks would each re-read the conversation).
        "max_concurrency": 1,
        "max_turns": 40,
    },
    "ui": {
        # collapsed | full | hidden
        "thinking": "collapsed",
        # bell | none
        "notify": "bell",
        "markdown": True,
        "tool_output_lines": 6,
    },
    # auto | bash | powershell | cmd | sh | <path to shell executable>
    "shell": "auto",
    "memory_files": ["AGENTS.md", "CLAUDE.md", "RIGCODE.md"],
    "compat": {"claude_project_dirs": True, "claude_user_dirs": False},
    "web": {
        # auto | duckduckgo | searxng | brave | tavily
        "search_provider": "auto",
        "searxng_url": None,
        "brave_api_key": None,
        "tavily_api_key": None,
        "user_agent": "Mozilla/5.0 (compatible; rigcode)",
    },
    "max_steps_per_turn": 150,
    "require_read_before_edit": True,
    "debug": False,
}

_UNION_LIST_KEYS = {("permissions", "allow"), ("permissions", "deny"), ("permissions", "ask"),
                    ("permissions", "additional_directories")}

ENV_OVERRIDES = {
    "RIGCODE_BASE_URL": ("provider", "base_url", str),
    "RIGCODE_MODEL": ("provider", "model", str),
    "RIGCODE_API_KEY": ("provider", "api_key", str),
    "RIGCODE_EFFORT": ("provider", "reasoning_effort", str),
    "RIGCODE_CONTEXT_WINDOW": ("provider", "context_window", int),
    "RIGCODE_TOOL_MODE": ("provider", "tool_mode", str),
}


@dataclass
class ConfigSource:
    scope: str
    path: Path
    loaded: bool = False
    error: str | None = None


def _merge(base: dict, overlay: dict, path: tuple = ()) -> dict:
    for key, val in overlay.items():
        kp = path + (key,)
        if kp in _UNION_LIST_KEYS and isinstance(val, list):
            existing = base.get(key) or []
            base[key] = existing + [v for v in val if v not in existing]
        elif path == ("hooks",) and isinstance(val, list):
            base[key] = (base.get(key) or []) + val
        elif isinstance(val, dict) and isinstance(base.get(key), dict):
            _merge(base[key], val, kp)
        else:
            base[key] = copy.deepcopy(val)
    return base


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, None
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            return None, "top-level JSON value must be an object"
        return data, None
    except (OSError, json.JSONDecodeError) as e:
        return None, str(e)


def settings_paths(project_root: Path) -> dict[str, Path]:
    return {
        "user": rigcode_home() / "settings.json",
        "project": project_root / ".rigcode" / "settings.json",
        "local": project_root / ".rigcode" / "settings.local.json",
    }


@dataclass
class Settings:
    data: dict
    project_root: Path
    sources: list[ConfigSource] = field(default_factory=list)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        """In-memory override for this process only."""
        parts = dotted.split(".")
        cur = self.data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def provider(self) -> dict:
        return self.data["provider"]

    def apply_profile(self, name: str) -> bool:
        prof = (self.data.get("profiles") or {}).get(name)
        if not isinstance(prof, dict):
            return False
        _merge(self.data["provider"], prof)
        self.data["profile"] = name
        return True


def load_settings(project_root: Path, cli: dict | None = None) -> Settings:
    data = copy.deepcopy(DEFAULTS)
    sources: list[ConfigSource] = []
    for scope, path in settings_paths(project_root).items():
        src = ConfigSource(scope, path)
        layer, err = _read_json(path)
        if layer is not None:
            _merge(data, layer)
            src.loaded = True
        src.error = err
        sources.append(src)

    # Claude-Code-compatible project MCP file.
    mcp_path = project_root / ".mcp.json"
    layer, err = _read_json(mcp_path)
    if layer is not None:
        servers = layer.get("mcpServers") or layer.get("mcp_servers") or {}
        if isinstance(servers, dict):
            for name, spec in servers.items():
                data["mcp_servers"].setdefault(name, spec)
        sources.append(ConfigSource("mcp", mcp_path, True, None))
    elif err:
        sources.append(ConfigSource("mcp", mcp_path, False, err))

    for env, (section, key, typ) in ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            try:
                data[section][key] = typ(val)
            except ValueError:
                pass

    settings = Settings(data=data, project_root=project_root, sources=sources)
    profile = (cli or {}).pop("profile", None) or data.get("profile")
    if profile:
        settings.apply_profile(profile)
    for dotted, value in (cli or {}).items():
        if value is not None:
            settings.set(dotted, value)
    return settings


def update_settings_file(path: Path, mutate) -> dict:
    """Read-modify-write a settings JSON file (creating it if needed)."""
    data, _err = _read_json(path)
    data = data or {}
    mutate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data


def add_permission_rule(project_root: Path, rule: str, kind: str = "allow", scope: str = "local") -> Path:
    path = settings_paths(project_root)[scope]

    def mutate(d: dict) -> None:
        perms = d.setdefault("permissions", {})
        lst = perms.setdefault(kind, [])
        if rule not in lst:
            lst.append(rule)

    update_settings_file(path, mutate)
    return path
