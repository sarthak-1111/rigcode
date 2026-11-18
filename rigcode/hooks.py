"""User-configured hooks: shell commands run at lifecycle events.

Configuration (settings.json), Claude-Code-compatible shape:
  "hooks": {
    "PreToolUse": [{"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "python check.py", "timeout": 60}]}],
    "PostToolUse": [...], "UserPromptSubmit": [...], "Stop": [...], "SessionStart": [...], ...
  }
The hook receives a JSON payload on stdin. Exit code 0 = ok (stdout may be JSON
with decision fields), 2 = block (stderr is fed back to the model), other =
non-blocking error shown to the user.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rigcode.tools.shell import ShellSpec, _child_env, _kill_tree, build_invocation
from rigcode.utils.text import decode_output

EVENTS = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop", "SessionStart",
          "SessionEnd", "PreCompact", "Notification")


@dataclass
class HookOutcome:
    block: bool = False
    reason: str | None = None
    permission: str | None = None          # allow | deny | ask (PreToolUse)
    updated_input: dict | None = None
    additional_context: list[str] = field(default_factory=list)
    halt: bool = False                     # "continue": false -> stop the agent
    halt_reason: str | None = None
    notices: list[str] = field(default_factory=list)


class HookRunner:
    def __init__(self, config: dict, shell: ShellSpec, session_id: str, transcript_path: Path | None):
        self.config = config or {}
        self.shell = shell
        self.session_id = session_id
        self.transcript_path = transcript_path

    def has(self, event: str) -> bool:
        return bool(self.config.get(event))

    def _matching(self, event: str, match_value: str | None) -> list[dict]:
        out = []
        for group in self.config.get(event) or []:
            if not isinstance(group, dict):
                continue
            matcher = str(group.get("matcher") or "")
            if matcher and matcher != "*" and match_value is not None:
                try:
                    if not re.fullmatch(matcher, match_value) and not re.search(rf"^(?:{matcher})$", match_value):
                        continue
                except re.error:
                    if matcher != match_value:
                        continue
            for hook in group.get("hooks") or []:
                if isinstance(hook, dict) and hook.get("type", "command") == "command" and hook.get("command"):
                    out.append(hook)
        return out

    async def run(self, event: str, payload: dict, cwd: Path, match_value: str | None = None) -> HookOutcome:
        outcome = HookOutcome()
        hooks = self._matching(event, match_value)
        if not hooks:
            return outcome
        data = {"session_id": self.session_id, "transcript_path": str(self.transcript_path or ""),
                "cwd": str(cwd), "hook_event_name": event, **payload}
        stdin = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        results = await asyncio.gather(*(self._run_one(h, stdin, cwd) for h in hooks))
        for hook, (code, out, err) in zip(hooks, results):
            name = hook["command"][:60]
            if code is None:
                outcome.notices.append(f"hook '{name}' timed out")
                continue
            if code == 2:
                outcome.block = True
                outcome.reason = "\n".join(x for x in (outcome.reason, err.strip() or out.strip()) if x)
                continue
            if code != 0:
                outcome.notices.append(f"hook '{name}' failed (exit {code}): {(err or out).strip()[:300]}")
                continue
            parsed = None
            text = out.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                if text and event in ("UserPromptSubmit", "SessionStart"):
                    outcome.additional_context.append(text)
                continue
            self._apply_json(parsed, outcome)
        return outcome

    def _apply_json(self, data: dict, outcome: HookOutcome) -> None:
        if data.get("continue") is False:
            outcome.halt = True
            outcome.halt_reason = data.get("stopReason")
        decision = str(data.get("decision") or "").lower()
        if decision in ("block", "deny"):
            outcome.block = True
            outcome.reason = data.get("reason") or outcome.reason
        elif decision in ("approve", "allow"):
            outcome.permission = "allow"
        hso = data.get("hookSpecificOutput") or {}
        if isinstance(hso, dict):
            pd = str(hso.get("permissionDecision") or "").lower()
            if pd in ("allow", "deny", "ask"):
                outcome.permission = pd
                if pd == "deny":
                    outcome.block = True
                    outcome.reason = hso.get("permissionDecisionReason") or outcome.reason
            if isinstance(hso.get("updatedInput"), dict):
                outcome.updated_input = hso["updatedInput"]
            if hso.get("additionalContext"):
                outcome.additional_context.append(str(hso["additionalContext"]))
        if data.get("additionalContext"):
            outcome.additional_context.append(str(data["additionalContext"]))

    async def _run_one(self, hook: dict, stdin: bytes, cwd: Path) -> tuple[int | None, str, str]:
        timeout = float(hook.get("timeout") or 60)
        argv = build_invocation(self.shell, hook["command"], cwd, track_cwd=False)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(cwd), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env={**_child_env(), "RIGCODE_PROJECT_DIR": str(cwd),
                                                     "CLAUDE_PROJECT_DIR": str(cwd)})
        except OSError as e:
            return 1, "", f"could not start hook: {e}"
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_tree(proc)
            return None, "", ""
        return proc.returncode, decode_output(out), decode_output(err)
