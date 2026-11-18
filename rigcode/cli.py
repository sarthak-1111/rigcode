"""rigcode command-line entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from rigcode import __version__
from rigcode.agent.runtime import Runtime
from rigcode.agent.session import SessionStore
from rigcode.config import load_settings, settings_paths, update_settings_file
from rigcode.utils.paths import find_git_root, project_slug, rigcode_home

SUBCOMMANDS = ("doctor", "config", "sessions")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rigcode",
        description="A terminal coding agent for self-hosted models behind an OpenAI-compatible API.",
        epilog="Subcommands: rigcode doctor [--quick] · rigcode config [get|set ...] · rigcode sessions")
    p.add_argument("prompt", nargs="*", help="initial prompt (interactive) or the task to run (-p)")
    p.add_argument("-p", "--print", action="store_true", dest="print_mode",
                   help="run one task non-interactively and print the result (stdin is appended to the prompt)")
    p.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text",
                   help="output for -p (default: text)")
    p.add_argument("-c", "--continue", action="store_true", dest="cont",
                   help="continue the most recent session in this project")
    p.add_argument("-r", "--resume", nargs="?", const="", default=None, metavar="ID",
                   help="resume a session by id (the most recent if no id is given)")
    p.add_argument("-m", "--model", help="model name, or the name of a profile from settings")
    p.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1 (default: find a local server)")
    p.add_argument("--api-key", help="API key (most self-hosted servers accept anything)")
    p.add_argument("-e", "--effort", help="reasoning effort: low, medium, high, max or none")
    p.add_argument("--mode", "--permission-mode", dest="mode",
                   help="permission mode: default, acceptEdits, plan or bypass")
    p.add_argument("--allow", action="append", default=[], metavar="RULE",
                   help="extra allow rule for this run, e.g. 'Bash(npm test:*)' (repeatable)")
    p.add_argument("--deny", action="append", default=[], metavar="RULE", help="extra deny rule (repeatable)")
    p.add_argument("-C", "--cwd", help="working directory (default: current directory)")
    p.add_argument("--max-steps", type=int, help="maximum model calls per turn")
    p.add_argument("--no-warmup", action="store_true", help="don't pre-process the prompt prefix at startup")
    p.add_argument("--plain", action="store_true", help="use the simple line-based interface")
    p.add_argument("-v", "--verbose", action="store_true", help="with -p: show progress on stderr")
    p.add_argument("--debug", action="store_true", help="log requests to ~/.rigcode/logs and show tracebacks")
    p.add_argument("--version", action="version", version=f"rigcode {__version__}")
    return p


def _fix_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        if not s.isatty():
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def _console_out():
    from rich.console import Console
    from rich.text import Text
    console = Console(highlight=False, emoji=False)
    styles = {"class:error": "red", "class:warn": "yellow", "class:dim": "dim", "class:faint": "dim",
              "class:ok": "green", "bold": "bold"}

    def out(text: str, style: str = "") -> None:
        console.print(Text(text, style=styles.get(style, "")))
    return out


def _project(cwd: Path) -> Path:
    return find_git_root(cwd) or cwd


def _final_text(agent) -> str:
    for m in reversed(agent.messages):
        role = m.get("role")
        if role == "assistant" and (m.get("content") or "").strip():
            return m["content"].strip()
        if role == "user" and not m.get("_meta"):
            break
    return ""


# ---------------------------------------------------------------- modes

async def _run_interactive(settings, project_root: Path, cwd: Path, prompt: str, args, resume: Path | None) -> int:
    from rigcode.ui.plain import PlainUI
    ui = None
    if not args.plain and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from rigcode.ui.terminal import TerminalUI
            ui = TerminalUI(settings, history_path=rigcode_home() / "projects" / project_slug(project_root) / "history")
        except Exception as e:  # noqa: BLE001 - e.g. no console screen buffer under mintty
            print(f"(using the plain interface: {e})", file=sys.stderr)
    if ui is None:
        ui = PlainUI(settings, interactive=True)
    base = settings.get("provider.base_url")
    print(f"Connecting to {'a local model server' if base in (None, '', 'auto') else base} …", end="\r", flush=True)
    rt = Runtime(settings, project_root, cwd, ui)
    try:
        await rt.start(resume=resume)
        print(" " * 70, end="\r", flush=True)
        await ui.run(rt, initial=prompt or None)
    finally:
        await rt.close()
    return 0


async def _run_print(settings, project_root: Path, cwd: Path, prompt: str, args, resume: Path | None) -> int:
    from rigcode.ui.plain import PlainUI
    settings.set("provider.warmup", False)          # we send the real request right away
    ui = PlainUI(settings, interactive=False, output_format=args.output_format, verbose=args.verbose)
    rt = Runtime(settings, project_root, cwd, ui)
    started = time.monotonic()
    try:
        await rt.start(resume=resume)
        ui.rt = rt
        if not rt.server.reachable:
            return 1
        ui._event("system", subtype="init", session_id=rt.session.id, model=rt.client.model,
                  base_url=rt.client.base_url, cwd=str(cwd),
                  tools=rt.tools.names(), permission_mode=rt.permissions.mode)
        stats = None
        first = prompt.split()[0][1:] if prompt.startswith("/") and not prompt.startswith("//") else ""
        if first and (ui.commands.find(first) or first in rt.commands):
            await ui.handle(prompt)
        else:
            stats = await ui.run_turn(prompt)
        text = _final_text(rt.main_agent)
        reason = stats.stopped_reason if stats else "done"
        u = rt.usage
        summary = {"result": text, "stopped_reason": reason, "session_id": rt.session.id,
                   "steps": stats.steps if stats else 0, "tool_calls": stats.tool_calls if stats else 0,
                   "elapsed_s": round(time.monotonic() - started, 2),
                   "usage": {"prompt_tokens": u.prompt_tokens, "cached_tokens": u.cached_tokens,
                             "completion_tokens": u.completion_tokens}}
        if args.output_format == "text":
            if text:
                print(text)
        elif args.output_format == "json":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            ui._event("result", **summary)
        return 0 if reason in ("done", "denied") else 1
    finally:
        await rt.close()


# ---------------------------------------------------------------- subcommands

def _cmd_doctor(argv: list[str]) -> int:
    from rigcode.doctor import run_doctor
    cwd = Path.cwd()
    root = _project(cwd)
    settings = load_settings(root)
    ok = asyncio.run(run_doctor(settings, root, _console_out(), chat_test="--quick" not in argv))
    return 0 if ok else 1


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _cmd_config(argv: list[str]) -> int:
    root = _project(Path.cwd())
    paths = settings_paths(root)
    scope = "project"
    for flag, sc in (("--user", "user"), ("--project", "project"), ("--local", "local")):
        if flag in argv:
            scope = sc
            argv = [a for a in argv if a != flag]
    if not argv or argv[0] == "show":
        settings = load_settings(root)
        for src in settings.sources:
            state = "loaded" if src.loaded else (f"error: {src.error}" if src.error else "not present")
            print(f"{src.scope:<8} {src.path}  ({state})")
        print(json.dumps(settings.data, indent=2, ensure_ascii=False))
        return 0
    if argv[0] == "get" and len(argv) >= 2:
        print(json.dumps(load_settings(root).get(argv[1]), indent=2, ensure_ascii=False))
        return 0
    if argv[0] == "set" and len(argv) >= 3:
        key, value = argv[1], _parse_value(" ".join(argv[2:]))

        def mutate(d: dict) -> None:
            cur = d
            parts = key.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
        update_settings_file(paths[scope], mutate)
        print(f"Set {key} = {json.dumps(value, ensure_ascii=False)} in {paths[scope]}")
        return 0
    if argv[0] == "path":
        for k, v in paths.items():
            print(f"{k:<8} {v}")
        return 0
    print("usage: rigcode config [show | get KEY | set KEY VALUE [--user|--project|--local] | path]", file=sys.stderr)
    return 2


def _cmd_sessions(argv: list[str]) -> int:
    root = _project(Path.cwd())
    infos = SessionStore.list(root, limit=30)
    if not infos:
        print("No sessions for this project yet.")
        return 0
    for i in infos:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(i.mtime))
        print(f"{i.id}  {when}  {i.n_messages:>4} msgs  {i.title[:70]}")
    print("\nResume with: rigcode -r <id>   (or rigcode -c for the latest)")
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    _fix_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        return {"doctor": _cmd_doctor, "config": _cmd_config, "sessions": _cmd_sessions}[argv[0]](argv[1:])
    args = build_parser().parse_args(argv)
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else Path.cwd()
    if not cwd.is_dir():
        print(f"rigcode: not a directory: {cwd}", file=sys.stderr)
        return 2
    os.chdir(cwd)
    project_root = _project(cwd)
    overrides: dict = {}
    if args.base_url:
        overrides["provider.base_url"] = args.base_url
    if args.api_key:
        overrides["provider.api_key"] = args.api_key
    if args.effort:
        overrides["provider.reasoning_effort"] = args.effort
    if args.mode:
        overrides["permissions.mode"] = args.mode
    if args.max_steps:
        overrides["max_steps_per_turn"] = args.max_steps
    if args.no_warmup:
        overrides["provider.warmup"] = False
    if args.debug:
        overrides["debug"] = True
    settings = load_settings(project_root, overrides)
    if args.model:
        if not settings.apply_profile(args.model):
            settings.set("provider.model", args.model)
    perms = settings.data.setdefault("permissions", {})
    perms.setdefault("allow", []).extend(args.allow)
    perms.setdefault("deny", []).extend(args.deny)

    prompt = " ".join(args.prompt).strip()
    if args.print_mode and not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            prompt = f"{prompt}\n\n{piped}" if prompt else piped
    resume = None
    if args.cont or args.resume is not None:
        resume = SessionStore.resolve(project_root, args.resume or None)
        if resume is None:
            print("rigcode: no session found to resume in this project.", file=sys.stderr)
            return 1
    try:
        if args.print_mode:
            if not prompt.strip():
                print("rigcode: -p needs a prompt (argument or stdin).", file=sys.stderr)
                return 2
            return asyncio.run(_run_print(settings, project_root, cwd, prompt, args, resume))
        return asyncio.run(_run_interactive(settings, project_root, cwd, prompt, args, resume))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
