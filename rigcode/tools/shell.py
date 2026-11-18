"""Shell tools: Bash (foreground/background), BashOutput, KillShell."""
from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.tools.bashsafety import is_read_only_command
from rigcode.utils.paths import IS_WINDOWS, rigcode_home
from rigcode.utils.text import decode_output, strip_ansi, truncate_middle

MARKER = "__RIGCODE_CWD__"
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_CAPTURE = 8 * 1024 * 1024


@dataclass
class ShellSpec:
    kind: str          # bash | powershell | cmd | sh
    exe: str
    name: str


def _git_bash_candidates() -> list[str]:
    cands = []
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent  # .../Git/cmd/git.exe -> .../Git
        cands += [str(root / "bin" / "bash.exe"), str(root / "usr" / "bin" / "bash.exe")]
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", ""),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            cands.append(os.path.join(base, "Git", "bin", "bash.exe"))
    return cands


def resolve_shell(setting: str = "auto") -> ShellSpec:
    setting = (setting or "auto").strip()
    low = setting.lower()
    if low in ("auto", "bash", "git-bash", "gitbash"):
        if IS_WINDOWS:
            for cand in _git_bash_candidates():
                if cand and os.path.exists(cand):
                    return ShellSpec("bash", cand, "Git Bash")
            if low != "auto":
                raise RuntimeError("Git Bash not found; install Git for Windows or set \"shell\": \"powershell\"")
        else:
            exe = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
            return ShellSpec("bash" if exe.endswith("bash") else "sh", exe, os.path.basename(exe))
    if low in ("auto", "powershell", "pwsh"):
        exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        return ShellSpec("powershell", exe, "PowerShell")
    if low == "cmd":
        return ShellSpec("cmd", os.environ.get("COMSPEC", "cmd.exe"), "cmd.exe")
    if low == "sh":
        return ShellSpec("sh", shutil.which("sh") or "/bin/sh", "sh")
    name = os.path.basename(setting).lower()
    kind = "powershell" if "pwsh" in name or "powershell" in name else "cmd" if name.startswith("cmd") else "bash"
    return ShellSpec(kind, setting, os.path.basename(setting))


def _child_env() -> dict:
    env = dict(os.environ)
    env.update({
        "PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true",
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "CLICOLOR": "0",
        "RIGCODE": "1",
    })
    return env


def build_invocation(spec: ShellSpec, command: str, cwd: Path, track_cwd: bool = True) -> list[str]:
    if spec.kind in ("bash", "sh"):
        cwd_s = str(cwd).replace("\\", "/")
        pwd_cmd = '"$(pwd -W 2>/dev/null || pwd)"' if spec.kind == "bash" else '"$(pwd)"'
        script = f"cd {shlex.quote(cwd_s)} || exit 1\n{command}\n"
        if track_cwd:
            script += f"__kc_ec=$?\nprintf '\\n{MARKER}%s\\n' {pwd_cmd}\nexit $__kc_ec\n"
        return [spec.exe, "-c", script]
    if spec.kind == "powershell":
        lit = str(cwd).replace("'", "''")
        script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8\n"
            "$ProgressPreference = 'SilentlyContinue'\n"
            f"Set-Location -LiteralPath '{lit}'\n$global:LASTEXITCODE = 0\n{command}\n"
            "$__kc_ok = $?\n$__kc_ec = if ($LASTEXITCODE) { $LASTEXITCODE } elseif ($__kc_ok) { 0 } else { 1 }\n"
        )
        if track_cwd:
            script += f"Write-Output \"`n{MARKER}$((Get-Location).Path)\"\n"
        script += "exit $__kc_ec\n"
        enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [spec.exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", enc]
    # cmd.exe
    tail = f" & echo. & echo {MARKER}%CD%" if track_cwd else ""
    return [spec.exe, "/d", "/s", "/c", f'cd /d "{cwd}" && ({command}){tail}']


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


def _clean(raw: str) -> str:
    text = strip_ansi(raw).replace("\r\n", "\n")
    if "\r" in text:  # progress bars: keep the final state of each line
        text = "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))
    return text


def _extract_cwd(text: str) -> tuple[str, str | None]:
    idx = text.rfind(MARKER)
    if idx == -1:
        return text, None
    line_end = text.find("\n", idx)
    new_cwd = text[idx + len(MARKER): line_end if line_end != -1 else None].strip()
    before = text[:idx]
    if before.endswith("\n"):
        before = before[:-1]
    after = text[line_end + 1:] if line_end != -1 else ""
    return (before + after).rstrip("\n"), (new_cwd or None)


def _normalize_cwd(raw: str) -> Path | None:
    if IS_WINDOWS:
        m = re.match(r"^/([a-zA-Z])(/.*)?$", raw)
        if m:
            raw = f"{m.group(1).upper()}:{m.group(2) or '/'}"
    p = Path(raw)
    return p if p.is_dir() else None


def save_full_output(text: str) -> Path | None:
    try:
        d = rigcode_home() / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"output-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.txt"
        p.write_text(text, encoding="utf-8")
        return p
    except OSError:
        return None


async def _spawn(argv: list[str], cwd: Path) -> asyncio.subprocess.Process:
    kwargs: dict = {}
    if not IS_WINDOWS:
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd), stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, env=_child_env(), **kwargs)


class BashTool(Tool):
    name = "Bash"
    kind = "exec"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout": {"type": "integer", "description": f"Seconds before the command is killed (default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT})"},
            "run_in_background": {"type": "boolean", "description": "Start without waiting (servers, watchers); read output with BashOutput"},
        },
        "required": ["command"],
    }

    def __init__(self, spec: ShellSpec):
        self.spec = spec
        shell_note = {
            "bash": f"{spec.name} (bash syntax{', running on Windows: use forward slashes, /c/... or C:/... paths' if IS_WINDOWS else ''})",
            "sh": "POSIX sh",
            "powershell": "PowerShell (use PowerShell syntax, not bash)",
            "cmd": "cmd.exe (Windows batch syntax)",
        }[spec.kind]
        self.description = (
            f"Run a shell command in {shell_note} and return combined stdout/stderr and the exit code. The working "
            "directory persists between calls (cd works); environment variables do not. Use Read/Grep/Glob/Edit "
            "instead of cat/grep/find/sed for files. Never run interactive programs (editors, pagers, prompts); "
            "pass non-interactive flags. For servers or watchers use run_in_background=true."
        )

    aliases = {"cmd": "command", "script": "command", "background": "run_in_background"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("command", ""))

    def is_read_only(self, args: dict) -> bool:
        return is_read_only_command(str(args.get("command", "")))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult.error("command is empty")
        if args.get("run_in_background"):
            if not ctx.rt.settings.get("tools.background_shells", True):
                return ToolResult.error("Background commands are disabled in settings (tools.background_shells).")
            return await ctx.rt.shells.start(self.spec, command, ctx.cwd)
        timeout = int(args.get("timeout") or DEFAULT_TIMEOUT)
        if timeout > MAX_TIMEOUT * 10:   # looks like milliseconds
            timeout //= 1000
        timeout = max(1, min(timeout, MAX_TIMEOUT))
        argv = build_invocation(self.spec, command, ctx.cwd)
        try:
            proc = await _spawn(argv, ctx.cwd)
        except OSError as e:
            return ToolResult.error(f"Failed to start {self.spec.name}: {e}")
        chunks: list[bytes] = []
        size = 0
        started = time.monotonic()
        on_output = getattr(ctx.agent, "on_tool_output", None)

        async def pump() -> None:
            nonlocal size
            assert proc.stdout is not None
            while True:
                data = await proc.stdout.read(65536)
                if not data:
                    break
                if size < MAX_CAPTURE:
                    chunks.append(data)
                    size += len(data)
                if on_output:
                    try:
                        on_output(ctx.call_id, decode_output(data))
                    except Exception:  # noqa: BLE001 - UI must not break tools
                        pass
            await proc.wait()

        timed_out = False
        try:
            await asyncio.wait_for(pump(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_tree(proc)
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                pass
        except asyncio.CancelledError:
            _kill_tree(proc)
            raise
        elapsed = time.monotonic() - started
        text, new_cwd = _extract_cwd(_clean(decode_output(b"".join(chunks))))
        if new_cwd:
            p = _normalize_cwd(new_cwd)
            if p is not None:
                ctx.rt.set_cwd(p)
        code = proc.returncode
        limit = int(ctx.rt.settings.get("context.max_tool_output_chars", 12000))
        shown, truncated = truncate_middle(text, limit)
        parts = []
        if truncated:
            saved = save_full_output(text)
            parts.append(f"[Output was {len(text):,} chars; showing the start and end."
                         + (f" Full output saved to {saved} - use Read/Grep on it if needed.]" if saved else "]"))
        if shown.strip():
            parts.append(shown)
        elif not timed_out:
            parts.append("(command completed with no output)")
        if timed_out:
            parts.append(f"[Command timed out after {timeout}s and was killed. Use a longer timeout or "
                         "run_in_background=true for long-running processes.]")
        elif code:
            parts.append(f"[Exit code: {code}]")
        n_lines = text.count("\n") + 1 if text else 0
        summary = (f"Timed out after {timeout}s" if timed_out else
                   f"Exit {code} · {n_lines} lines · {elapsed:.1f}s" if code else f"{n_lines} lines · {elapsed:.1f}s")
        return ToolResult(content="\n".join(parts), is_error=bool(code) or timed_out, summary=summary,
                          display={"type": "output", "text": text[-4000:]})


@dataclass
class BackgroundShell:
    id: str
    command: str
    proc: asyncio.subprocess.Process
    started: float
    buffer: list[str] = field(default_factory=list)
    read_pos: int = 0
    task: asyncio.Task | None = None


class BackgroundShells:
    def __init__(self) -> None:
        self.shells: dict[str, BackgroundShell] = {}
        self._n = 0

    async def start(self, spec: ShellSpec, command: str, cwd: Path) -> ToolResult:
        argv = build_invocation(spec, command, cwd, track_cwd=False)
        try:
            proc = await _spawn(argv, cwd)
        except OSError as e:
            return ToolResult.error(f"Failed to start background command: {e}")
        self._n += 1
        sid = f"bash_{self._n}"
        shell = BackgroundShell(sid, command, proc, time.monotonic())

        async def reader() -> None:
            assert proc.stdout is not None
            total = 0
            while True:
                data = await proc.stdout.read(65536)
                if not data:
                    break
                text = _clean(decode_output(data))
                total += len(text)
                shell.buffer.append(text)
                if total > MAX_CAPTURE:  # keep memory bounded
                    joined = "".join(shell.buffer)[-MAX_CAPTURE // 2:]
                    shell.read_pos = max(0, shell.read_pos - (len("".join(shell.buffer)) - len(joined)))
                    shell.buffer = [joined]
                    total = len(joined)
            await proc.wait()

        shell.task = asyncio.create_task(reader())
        self.shells[sid] = shell
        return ToolResult(content=f"Started in background as {sid} (pid {proc.pid}). Check output with "
                                  f"BashOutput(bash_id='{sid}'); stop it with KillShell.",
                          summary=f"Background {sid}")

    def read(self, sid: str, pattern: str | None = None) -> ToolResult:
        shell = self.shells.get(sid)
        if not shell:
            known = ", ".join(self.shells) or "none"
            return ToolResult.error(f"No background shell '{sid}'. Known: {known}")
        text = "".join(shell.buffer)
        new = text[shell.read_pos:]
        shell.read_pos = len(text)
        if pattern:
            try:
                rx = re.compile(pattern)
                new = "\n".join(l for l in new.split("\n") if rx.search(l))
            except re.error as e:
                return ToolResult.error(f"Invalid filter regex: {e}")
        code = shell.proc.returncode
        status = "running" if code is None else f"exited with code {code}"
        shown, _ = truncate_middle(new, 12000)
        return ToolResult(content=f"[{sid}: {status}]\n" + (shown if shown.strip() else "(no new output)"),
                          summary=f"{sid} {status}")

    async def kill(self, sid: str) -> ToolResult:
        shell = self.shells.get(sid)
        if not shell:
            return ToolResult.error(f"No background shell '{sid}'.")
        _kill_tree(shell.proc)
        try:
            await asyncio.wait_for(shell.proc.wait(), 5)
        except asyncio.TimeoutError:
            pass
        return ToolResult(content=f"Killed {sid}.", summary=f"Killed {sid}")

    async def kill_all(self) -> None:
        for sid in list(self.shells):
            if self.shells[sid].proc.returncode is None:
                _kill_tree(self.shells[sid].proc)

    def describe(self) -> list[str]:
        out = []
        for s in self.shells.values():
            status = "running" if s.proc.returncode is None else f"exit {s.proc.returncode}"
            out.append(f"{s.id} [{status}] {s.command[:60]}")
        return out


class BashOutputTool(Tool):
    name = "BashOutput"
    kind = "read"
    read_only = True
    description = "Get new output from a background command started with Bash(run_in_background=true)."
    parameters = {
        "type": "object",
        "properties": {
            "bash_id": {"type": "string", "description": "Id returned when the command was started"},
            "filter": {"type": "string", "description": "Optional regex; only matching lines are returned"},
        },
        "required": ["bash_id"],
    }
    aliases = {"id": "bash_id", "shell_id": "bash_id"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("bash_id", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ctx.rt.shells.read(str(args["bash_id"]), args.get("filter"))


class KillShellTool(Tool):
    name = "KillShell"
    kind = "exec"
    read_only = True  # only stops processes the agent started itself
    description = "Stop a background command started with Bash(run_in_background=true)."
    parameters = {
        "type": "object",
        "properties": {"bash_id": {"type": "string", "description": "Id of the background command"}},
        "required": ["bash_id"],
    }
    aliases = {"id": "bash_id", "shell_id": "bash_id"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("bash_id", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        return await ctx.rt.shells.kill(str(args["bash_id"]))
