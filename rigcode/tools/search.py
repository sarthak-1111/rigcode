"""Search tools: Glob (find files) and Grep (search contents)."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path, PurePosixPath

from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.utils.files import is_binary_file, list_project_files
from rigcode.utils.globmatch import basename_match, glob_match
from rigcode.utils.paths import display_path, resolve_path
from rigcode.utils.text import truncate_line

TYPE_GLOBS = {
    "py": ["*.py", "*.pyi"], "python": ["*.py", "*.pyi"],
    "js": ["*.js", "*.mjs", "*.cjs", "*.jsx"], "ts": ["*.ts", "*.tsx", "*.mts", "*.cts"],
    "jsx": ["*.jsx"], "tsx": ["*.tsx"], "rust": ["*.rs"], "rs": ["*.rs"], "go": ["*.go"],
    "java": ["*.java"], "kotlin": ["*.kt", "*.kts"], "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx", "*.h"], "cs": ["*.cs"], "csharp": ["*.cs"],
    "rb": ["*.rb"], "ruby": ["*.rb"], "php": ["*.php"], "swift": ["*.swift"], "scala": ["*.scala"],
    "md": ["*.md", "*.markdown"], "markdown": ["*.md", "*.markdown"], "json": ["*.json"],
    "yaml": ["*.yaml", "*.yml"], "toml": ["*.toml"], "xml": ["*.xml"], "html": ["*.html", "*.htm"],
    "css": ["*.css", "*.scss", "*.sass", "*.less"], "sh": ["*.sh", "*.bash", "*.zsh"],
    "ps1": ["*.ps1", "*.psm1"], "sql": ["*.sql"], "lua": ["*.lua"], "r": ["*.r", "*.R"],
    "vue": ["*.vue"], "svelte": ["*.svelte"], "dart": ["*.dart"], "matlab": ["*.m"], "jl": ["*.jl"],
    "txt": ["*.txt"], "ipynb": ["*.ipynb"],
}

MAX_SCAN_FILES = 40_000
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_CONTEXT = 50


def _rel(path: str | Path, ctx: ToolContext) -> str:
    return display_path(Path(path), ctx.cwd)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _matches_filters(rel: str, glob: str | None, globs_for_type: list[str] | None) -> bool:
    if glob:
        g = glob.replace("\\", "/")
        ok = glob_match(g, rel) if "/" in g else basename_match(g, rel)
        if not ok:
            return False
    if globs_for_type:
        if not any(basename_match(g, rel) for g in globs_for_type):
            return False
    return True


class GlobTool(Tool):
    name = "Glob"
    kind = "read"
    read_only = True
    description = (
        "Find files and directories by glob pattern (e.g. '**/*.py', 'src/**/test_*.ts', '*.{json,yaml}', "
        "'**/migrations'). Respects .gitignore. Returns paths sorted by modification time, newest first; "
        "directories end with '/'. Use it to locate files by name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, relative to path"},
            "path": {"type": "string", "description": "Directory to search (default: working directory)"},
        },
        "required": ["pattern"],
    }
    aliases = {"glob": "pattern", "dir": "path", "directory": "path", "cwd": "path"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("pattern", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        base = resolve_path(args.get("path") or ".", ctx.cwd)
        if not base.is_dir():
            return ToolResult.error(f"Not a directory: {_rel(base, ctx)}")
        pattern = str(args["pattern"]).strip().replace("\\", "/")
        # Absolute pattern inside base -> make relative.
        if os.path.isabs(pattern):
            try:
                pattern = os.path.relpath(pattern, base).replace("\\", "/")
            except ValueError:
                return ToolResult.error("pattern must be relative to path")
        limit = int(ctx.rt.settings.get("tools.glob_max_results", 100))
        files = await asyncio.to_thread(list_project_files, base, MAX_SCAN_FILES)
        # Directories that hold at least one listed file, so ignored and empty ones stay out.
        dirs = sorted({str(PurePosixPath(f).parents[i]) for f in files
                       for i in range(len(PurePosixPath(f).parents) - 1)})
        entries = [(f, False) for f in files] + [(d, True) for d in dirs]
        pattern = pattern.rstrip("/") or pattern
        hits = [(e, is_dir) for e, is_dir in entries if glob_match(pattern, e)]
        note = ""
        if not hits and "/" not in pattern and not pattern.startswith("**"):
            hits = [(e, is_dir) for e, is_dir in entries if glob_match("**/" + pattern, e)]
            if hits:
                note = f"(No top-level matches for '{pattern}'; showing matches at any depth via '**/{pattern}'.)\n"
        if not hits:
            return ToolResult(content=f"Nothing matches '{pattern}' under {_rel(base, ctx)}.", summary="No matches")
        paths = sorted(((base / e, is_dir) for e, is_dir in hits), key=lambda h: _mtime(h[0]), reverse=True)
        shown = [_rel(p, ctx) + ("/" if is_dir else "") for p, is_dir in paths[:limit]]
        more = f"\n({len(paths) - limit} more not shown - narrow the pattern)" if len(paths) > limit else ""
        return ToolResult(content=note + "\n".join(shown) + more,
                          summary=f"Found {len(paths)} match{'es' if len(paths) != 1 else ''}")


class GrepTool(Tool):
    name = "Grep"
    kind = "read"
    read_only = True
    description = (
        "Search file contents with a regular expression (ripgrep syntax). Respects .gitignore. output_mode: "
        "'files_with_matches' (default; just paths), 'content' (matching lines with line numbers), or 'count'. "
        "Narrow with glob (e.g. '*.py') or type (e.g. 'py', 'ts'). Escape regex characters for literal text "
        "(e.g. 'foo\\(' ). Prefer this over running grep through Bash."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "File or directory to search (default: working directory)"},
            "glob": {"type": "string", "description": "Only search files matching this glob"},
            "type": {"type": "string", "description": "Only search this file type: py, js, ts, go, rust, java, md..."},
            "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search"},
            "context": {"type": "integer", "description": "Lines of context around each match (content mode)"},
            "multiline": {"type": "boolean", "description": "Let patterns span lines ('.' matches newline)"},
            "max_results": {"type": "integer", "description": "Maximum files/lines to return (default 100)"},
        },
        "required": ["pattern"],
    }
    aliases = {"query": "pattern", "regex": "pattern", "dir": "path", "directory": "path", "include": "glob",
               "file_pattern": "glob", "-i": "ignore_case", "case_insensitive": "ignore_case",
               "head_limit": "max_results", "-C": "context", "context_lines": "context"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("pattern", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        for alias in ("-A", "-B"):
            if alias in args and "context" not in args:
                try:
                    args["context"] = int(args[alias])
                except (TypeError, ValueError):
                    pass
        target = resolve_path(args.get("path") or ".", ctx.cwd)
        if not target.exists():
            return ToolResult.error(f"Path does not exist: {_rel(target, ctx)}")
        mode = args.get("output_mode") or "files_with_matches"
        if mode not in ("files_with_matches", "content", "count"):
            mode = "files_with_matches"
        max_results = int(args.get("max_results") or 100)
        ftype = args.get("type")
        type_globs = TYPE_GLOBS.get(str(ftype).lower().lstrip(".")) if ftype else None
        if ftype and type_globs is None:
            type_globs = [f"*.{str(ftype).lstrip('.')}"]
        pattern = str(args["pattern"])
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        if args.get("multiline"):
            flags |= re.DOTALL | re.MULTILINE
        note = ""
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            regex = re.compile(re.escape(pattern), flags)
            note = f"(Invalid regex ({e}); searched for the literal text instead.)\n"
        requested = int(args.get("context") or 0)
        ctxl = max(0, min(requested, MAX_CONTEXT))
        rg = shutil.which("rg")
        found = None
        if rg and not note:
            found = await self._ripgrep(rg, args, target, mode, type_globs, ctxl, max_results, ctx)
        if found is None:
            found = await asyncio.to_thread(self._python_search, regex, target, args.get("glob"), type_globs,
                                            mode, ctxl, max_results, bool(args.get("multiline")), note, ctx)
        result, seen = found
        for path in seen:       # the model now has these lines verbatim: enough for an exact-match Edit
            ctx.rt.file_state.record(path, partial=True)
        if requested > MAX_CONTEXT and seen:
            result.content = f"(context capped at {MAX_CONTEXT} lines; Read the file for more)\n" + result.content
        return result

    async def _ripgrep(self, rg, args, target: Path, mode, type_globs, ctxl, max_results,
                       ctx) -> tuple[ToolResult, list[Path]] | None:
        cmd = [rg, "--no-config", "--color", "never", "--no-messages", "--max-columns", "400",
               "--max-columns-preview", "--max-filesize", "4M"]
        if args.get("ignore_case"):
            cmd.append("-i")
        if args.get("multiline"):
            cmd += ["-U", "--multiline-dotall"]
        if mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif mode == "count":
            cmd.append("--count")
        else:
            cmd += ["--line-number", "--with-filename"]
            if ctxl:
                cmd += ["-C", str(ctxl)]
        if args.get("glob"):
            cmd += ["--glob", str(args["glob"])]
        for g in type_globs or []:
            cmd += ["--glob", g]
        cmd += ["--", str(args["pattern"]), str(target)]
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        except (OSError, asyncio.TimeoutError):
            return None
        if proc.returncode not in (0, 1):
            return None  # bad regex for rg etc. -> python fallback
        lines = [l for l in out.decode("utf-8", errors="replace").splitlines() if l]
        if not lines:
            return ToolResult(content="No matches found.", summary="No matches"), []
        if mode == "files_with_matches":
            paths = sorted((Path(l) for l in lines), key=_mtime, reverse=True)
            shown = [_rel(p, ctx) for p in paths[:max_results]]
            more = f"\n({len(paths) - max_results} more files not shown)" if len(paths) > max_results else ""
            return ToolResult(content="\n".join(shown) + more, summary=f"Found {len(paths)} files"), []
        rel_lines = []
        seen: dict[str, Path] = {}
        for l in lines[:max_results]:
            m = re.match(r"^(.*?)([:-])(\d+)([:-])(.*)$", l) if mode == "content" else re.match(r"^(.*):(\d+)$", l)
            if mode == "content" and m and os.path.exists(m.group(1)):
                seen.setdefault(m.group(1), Path(m.group(1)))
                rel_lines.append(f"{_rel(m.group(1), ctx)}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}")
            elif mode == "count" and m:
                rel_lines.append(f"{_rel(m.group(1), ctx)}:{m.group(2)}")
            else:
                rel_lines.append(l)
        more = f"\n({len(lines) - max_results} more lines not shown - narrow the search)" if len(lines) > max_results else ""
        unit = "lines" if mode == "content" else "files"
        return ToolResult(content="\n".join(rel_lines) + more, summary=f"Found {len(lines)} {unit}"), list(seen.values())

    def _python_search(self, regex: re.Pattern, target: Path, glob, type_globs, mode, ctxl, max_results,
                       multiline: bool, note: str, ctx: ToolContext) -> tuple[ToolResult, list[Path]]:
        if target.is_file():
            candidates = [target]
        else:
            rels = list_project_files(target, MAX_SCAN_FILES)
            candidates = [target / r for r in rels if _matches_filters(r, glob, type_globs)]
        file_hits: list[Path] = []
        count_hits: list[tuple[Path, int]] = []
        content_lines: list[str] = []
        seen: list[Path] = []
        total_lines = 0
        for path in candidates:
            try:
                if path.stat().st_size > MAX_FILE_BYTES or is_binary_file(path):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if mode == "files_with_matches":
                if regex.search(text):
                    file_hits.append(path)
                continue
            lines = text.split("\n")
            if multiline:
                matched = set()
                for m in regex.finditer(text):
                    a = text.count("\n", 0, m.start())
                    b = text.count("\n", 0, m.end())
                    matched.update(range(a, b + 1))
                hit_idx = sorted(matched)
            else:
                hit_idx = [i for i, line in enumerate(lines) if regex.search(line)]
            if not hit_idx:
                continue
            if mode == "count":
                count_hits.append((path, len(hit_idx)))
                continue
            rel = _rel(path, ctx)
            shown: set[int] = set()
            for i in hit_idx:
                for j in range(max(0, i - ctxl), min(len(lines), i + ctxl + 1)):
                    if j in shown:
                        continue
                    shown.add(j)
                    sep = ":" if j == i or j in hit_idx else "-"
                    total_lines += 1
                    if len(content_lines) < max_results:
                        content_lines.append(f"{rel}{sep}{j + 1}{sep}{truncate_line(lines[j], 400)}")
                        if not seen or seen[-1] != path:
                            seen.append(path)
        if mode == "files_with_matches":
            if not file_hits:
                return ToolResult(content=note + "No matches found.", summary="No matches"), []
            file_hits.sort(key=_mtime, reverse=True)
            shown_paths = [_rel(p, ctx) for p in file_hits[:max_results]]
            more = f"\n({len(file_hits) - max_results} more files not shown)" if len(file_hits) > max_results else ""
            return ToolResult(content=note + "\n".join(shown_paths) + more,
                              summary=f"Found {len(file_hits)} files"), []
        if mode == "count":
            if not count_hits:
                return ToolResult(content=note + "No matches found.", summary="No matches"), []
            out = [f"{_rel(p, ctx)}:{n}" for p, n in count_hits[:max_results]]
            return ToolResult(content=note + "\n".join(out), summary=f"Matches in {len(count_hits)} files"), []
        if not content_lines:
            return ToolResult(content=note + "No matches found.", summary="No matches"), []
        more = f"\n({total_lines - max_results} more lines not shown - narrow the search)" if total_lines > max_results else ""
        return ToolResult(content=note + "\n".join(content_lines) + more, summary=f"Found {total_lines} lines"), seen
