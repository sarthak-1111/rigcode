"""File tools: Read, Write, Edit, MultiEdit."""
from __future__ import annotations

import base64
import difflib
import json
import warnings
from pathlib import Path

from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.tools.editmatch import EditError, apply_edit, find_omission_placeholder
from rigcode.utils.files import (
    IMAGE_EXTENSIONS,
    LFMap,
    is_binary_file,
    list_project_files,
    read_text_file,
    write_text_file,
)
from rigcode.utils.paths import display_path, resolve_path
from rigcode.utils.text import number_lines, one_line

DEFAULT_READ_LINES = 1000
MAX_LINE_CHARS = 2000


def _suggest_paths(path: Path, ctx: ToolContext, limit: int = 5) -> list[str]:
    """Similar existing paths to help the model recover from a wrong path."""
    try:
        files = list_project_files(ctx.rt.project_root, max_files=50_000)
    except Exception:  # noqa: BLE001 - suggestions are best effort
        return []
    name = path.name.lower()
    same_name = [f for f in files if f.rsplit("/", 1)[-1].lower() == name]
    if same_name:
        return same_name[:limit]
    names = {f.rsplit("/", 1)[-1]: f for f in files}
    close = difflib.get_close_matches(path.name, list(names), n=limit, cutoff=0.7)
    return [names[c] for c in close]


def _not_found(path: Path, ctx: ToolContext) -> ToolResult:
    shown = display_path(path, ctx.cwd)
    msg = f"File does not exist: {shown} (working directory: {ctx.cwd})."
    sugg = _suggest_paths(path, ctx)
    if sugg:
        msg += " Did you mean: " + ", ".join(sugg) + "?"
    return ToolResult.error(msg, summary=f"Not found: {shown}")


def syntax_problem(path: Path, text: str) -> str | None:
    ext = path.suffix.lower()
    try:
        if ext in (".py", ".pyw"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compile(text, str(path), "exec", dont_inherit=True)
        elif ext == ".json" and text.strip():
            json.loads(text)
        elif ext in (".yaml", ".yml"):
            import yaml
            list(yaml.safe_load_all(text))
        elif ext == ".toml":
            try:
                import tomllib  # type: ignore[import-not-found]
            except ImportError:
                return None
            tomllib.loads(text)
    except SyntaxError as e:
        return f"Python syntax error at line {e.lineno}: {e.msg}"
    except json.JSONDecodeError as e:
        return f"JSON syntax error at line {e.lineno}, column {e.colno}: {e.msg}"
    except Exception as e:  # noqa: BLE001 - yaml/toml errors
        return f"{ext[1:].upper()} syntax error: {one_line(str(e), 200)}"
    return None


def _syntax_warning(path: Path, before: str | None, after: str) -> str:
    problem = syntax_problem(path, after)
    if not problem:
        return ""
    if before is not None and syntax_problem(path, before):
        return ""  # already broken before this edit; don't nag
    return f"\nWARNING: the file now has a syntax error - {problem}. Fix it before moving on."


def make_diff(old: str, new: str, path_label: str, context: int = 2) -> tuple[str, int, int]:
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{path_label}", f"b/{path_label}",
                                     n=context, lineterm=""))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return "\n".join(diff), added, removed


def _snippet(text: str, first_line: int, n_lines: int, pad: int = 3, cap: int = 30) -> str:
    lines = text.split("\n")
    start = max(1, first_line - pad)
    end = min(len(lines), first_line + max(n_lines, 1) - 1 + pad)
    if end - start + 1 > cap:
        end = start + cap - 1
    return number_lines(lines[start - 1:end], start)


def _check_editable(path: Path, ctx: ToolContext, *, must_exist: bool, partial_ok: bool) -> ToolResult | None:
    if path.exists() and path.is_dir():
        return ToolResult.error(f"{display_path(path, ctx.cwd)} is a directory, not a file.")
    if must_exist and not path.exists():
        return _not_found(path, ctx)
    if path.exists() and ctx.rt.settings.get("require_read_before_edit", True):
        tracker = ctx.rt.file_state
        if not tracker.was_read(path, partial_ok=partial_ok):
            why = ("in full before overwriting it (you have only seen some of its lines)"
                   if tracker.was_read(path, partial_ok=True) else
                   "before modifying it (so your edit is based on its current contents)")
            return ToolResult.error(f"You must Read {display_path(path, ctx.cwd)} {why}. Read it, then retry.",
                                    summary="File not read yet")
        if tracker.changed_since_read(path):
            return ToolResult.error(
                f"{display_path(path, ctx.cwd)} was modified since you last read it (by the user or another "
                "process). Read it again, then redo your change.", summary="File changed since read")
    return None


class ReadTool(Tool):
    name = "Read"
    kind = "read"
    read_only = True
    description = (
        "Read a text file. Output lines are prefixed with their line number and a tab (cat -n style); that "
        "prefix is not part of the file. Returns up to 1000 lines - for large files use Grep to find the "
        "relevant region, then pass offset (1-based start line) and limit. Reading a directory lists it. "
        "Issue several Read calls in one reply to read multiple files at once. Don't re-read a file you just "
        "edited: the edit result already shows the updated lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "offset": {"type": "integer", "description": "1-based line number to start reading from"},
            "limit": {"type": "integer", "description": "Maximum number of lines to return"},
        },
        "required": ["file_path"],
    }
    aliases = {"path": "file_path", "filepath": "file_path", "file": "file_path", "filename": "file_path",
               "line_offset": "offset", "start_line": "offset", "n_lines": "limit", "lines": "limit",
               "num_lines": "limit", "max_lines": "limit"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("file_path", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = resolve_path(args["file_path"], ctx.cwd)
        shown = display_path(path, ctx.cwd)
        if not path.exists():
            return _not_found(path, ctx)
        if path.is_dir():
            return self._list_dir(path, shown)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return self._image(path, shown, ctx)
        if suffix == ".ipynb":
            return self._notebook(path, shown, ctx)
        if suffix == ".pdf":
            return self._pdf(path, shown, ctx)
        if is_binary_file(path):
            size = path.stat().st_size
            return ToolResult.error(f"{shown} is a binary file ({size:,} bytes); it cannot be shown as text.",
                                    summary="Binary file")
        try:
            tf = read_text_file(path)
        except OSError as e:
            return ToolResult.error(f"Could not read {shown}: {e}")
        ctx.rt.file_state.record(path)
        text = tf.text
        if not text:
            return ToolResult(content=f"(file {shown} exists but is empty)", summary="Empty file")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or DEFAULT_READ_LINES)
        if offset > total:
            return ToolResult.error(f"offset {offset} is past the end of {shown} ({total} lines).")
        budget = int(ctx.rt.settings.get("context.read_max_chars", 30000))
        selected, used = [], 0
        for line in lines[offset - 1: offset - 1 + limit]:
            cost = min(len(line), MAX_LINE_CHARS) + 8
            if selected and used + cost > budget:
                break
            selected.append(line)
            used += cost
        end = offset + len(selected) - 1
        body = number_lines(selected, offset, MAX_LINE_CHARS)
        notes = []
        if end < total or offset > 1:
            notes.append(f"[Showing lines {offset}-{end} of {total} total."
                         + (f" Continue with offset={end + 1}.]" if end < total else "]"))
        content = body + ("\n" + "\n".join(notes) if notes else "")
        return ToolResult(content=content, summary=f"Read {len(selected)} line{'s' if len(selected) != 1 else ''}"
                          + (f" (of {total})" if len(selected) < total else ""))

    def _list_dir(self, path: Path, shown: str) -> ToolResult:
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            return ToolResult.error(f"Cannot list {shown}: {e}")
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:500]]
        extra = f"\n... and {len(entries) - 500} more" if len(entries) > 500 else ""
        return ToolResult(content=f"{shown} is a directory with {len(entries)} entries:\n" + "\n".join(lines) + extra,
                          summary=f"Listed {len(entries)} entries")

    def _image(self, path: Path, shown: str, ctx: ToolContext) -> ToolResult:
        size = path.stat().st_size
        if not ctx.rt.server.vision:
            return ToolResult.error(f"{shown} is an image ({size:,} bytes). The current model/server has no vision "
                                    "support, so it cannot be viewed.", summary="Image (no vision)")
        if size > 10 * 1024 * 1024:
            return ToolResult.error(f"{shown} is too large to attach ({size:,} bytes).")
        mime = {"jpg": "jpeg"}.get(path.suffix.lower()[1:], path.suffix.lower()[1:])
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return ToolResult(content=f"Image {shown} ({size:,} bytes) is attached below.",
                          images=[{"mime": f"image/{mime}", "data": data}], summary="Read image")

    def _notebook(self, path: Path, shown: str, ctx: ToolContext) -> ToolResult:
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return ToolResult.error(f"Could not parse notebook {shown}: {e}")
        ctx.rt.file_state.record(path)
        parts = []
        for i, cell in enumerate(nb.get("cells", []), 1):
            src = "".join(cell.get("source", []))
            parts.append(f"## Cell {i} [{cell.get('cell_type', '?')}]\n{src}")
            outs = []
            for out in cell.get("outputs", []) or []:
                if "text" in out:
                    outs.append("".join(out["text"]))
                elif "data" in out and "text/plain" in out["data"]:
                    outs.append("".join(out["data"]["text/plain"]))
            if outs:
                joined = "\n".join(outs)
                parts.append("Output:\n" + (joined[:2000] + "\n...[truncated]" if len(joined) > 2000 else joined))
        return ToolResult(content="\n\n".join(parts) or "(empty notebook)",
                          summary=f"Read notebook ({len(nb.get('cells', []))} cells)")

    def _pdf(self, path: Path, shown: str, ctx: ToolContext) -> ToolResult:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return ToolResult.error(f"{shown} is a PDF. Install 'pypdf' (pip install pypdf) to extract its text.")
        try:
            reader = PdfReader(str(path))
            texts = [f"--- page {i} ---\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages[:50], 1)]
        except Exception as e:  # noqa: BLE001
            return ToolResult.error(f"Could not read PDF {shown}: {e}")
        text = "\n".join(texts)
        budget = int(ctx.rt.settings.get("context.read_max_chars", 60000))
        if len(text) > budget:
            text = text[:budget] + "\n[truncated]"
        return ToolResult(content=text, summary=f"Read PDF ({len(reader.pages)} pages)")


class WriteTool(Tool):
    name = "Write"
    kind = "edit"
    description = (
        "Create a new file, or completely replace an existing file's contents. Prefer Edit for changing part of "
        "an existing file. An existing file must be Read first. Parent directories are created as needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "content": {"type": "string", "description": "The full content to write"},
        },
        "required": ["file_path", "content"],
    }
    aliases = {"path": "file_path", "filepath": "file_path", "file": "file_path", "filename": "file_path",
               "text": "content", "contents": "content", "data": "content", "file_text": "content"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("file_path", ""))

    def _plan(self, args: dict, ctx: ToolContext):
        path = resolve_path(args["file_path"], ctx.cwd)
        shown = display_path(path, ctx.cwd)
        problem = _check_editable(path, ctx, must_exist=False, partial_ok=False)
        if problem:
            return problem
        content = str(args.get("content", "")).replace("\r\n", "\n")
        before, encoding, newline = None, "utf-8", "\n"
        if path.exists():
            try:
                tf = read_text_file(path)
            except OSError as e:
                return ToolResult.error(f"Could not read existing {shown}: {e}")
            before, encoding, newline = tf.text, tf.encoding, tf.newline
            placeholder = find_omission_placeholder(content, before)
            if placeholder:
                return ToolResult.error(
                    f"content contains a placeholder ('{placeholder}') instead of real code. Writing it would "
                    "delete the code it stands for. Write the complete file, or use Edit for a partial change.",
                    summary="Rejected placeholder content")
        return path, shown, content, before, encoding, newline

    async def preview(self, args: dict, ctx: ToolContext):
        plan = self._plan(args, ctx)
        if isinstance(plan, ToolResult):
            return plan
        _path, shown, content, before, _enc, _nl = plan
        if before is None:
            return {"type": "new_file", "path": shown, "content": content}
        diff, _a, _r = make_diff(before, content, shown)
        return {"type": "diff", "path": shown, "diff": diff}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        plan = self._plan(args, ctx)
        if isinstance(plan, ToolResult):
            return plan
        path, shown, content, before, encoding, newline = plan
        ctx.rt.checkpoints.snapshot(path)
        try:
            write_text_file(path, content, encoding, newline)
        except (OSError, UnicodeEncodeError) as e:
            return ToolResult.error(f"Could not write {shown}: {e}")
        ctx.rt.file_state.record(path)
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        warning = _syntax_warning(path, before, content)
        if before is not None:
            diff, added, removed = make_diff(before, content, shown)
            return ToolResult(content=f"Overwrote {shown} ({n_lines} lines).{warning}",
                              summary=f"Wrote {shown} (+{added} -{removed})",
                              display={"type": "diff", "path": shown, "diff": diff})
        return ToolResult(content=f"Created {shown} ({n_lines} lines).{warning}",
                          summary=f"Created {shown} ({n_lines} lines)",
                          display={"type": "new_file", "path": shown, "content": content})


_EDIT_ALIASES = {"old": "old_string", "old_str": "old_string", "search": "old_string", "find": "old_string",
                 "new": "new_string", "new_str": "new_string", "replace": "new_string",
                 "replacement": "new_string", "replaceAll": "replace_all", "all": "replace_all"}


class EditTool(Tool):
    name = "Edit"
    kind = "edit"
    description = (
        "Replace text in an existing file: finds old_string and replaces it with new_string. Read the file first. "
        "old_string must reproduce the current text exactly, including indentation, without the line-number "
        "prefixes that Read shows. It must match exactly one place - include a few surrounding lines to make it "
        "unique - or set replace_all=true to change every occurrence (e.g. a rename). For several separate "
        "changes to one file, use MultiEdit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "old_string": {"type": "string", "description": "Exact text to find"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }
    aliases = {"path": "file_path", "filepath": "file_path", "file": "file_path", "filename": "file_path",
               **_EDIT_ALIASES}

    def permission_target(self, args: dict) -> str:
        return str(args.get("file_path", ""))

    @staticmethod
    def _edits(args: dict) -> list[dict]:
        return [{"old_string": args.get("old_string", ""), "new_string": args.get("new_string", ""),
                 "replace_all": bool(args.get("replace_all", False))}]

    async def preview(self, args: dict, ctx: ToolContext):
        return preview_edits(args["file_path"], self._edits(args), ctx)

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        return apply_edits(args["file_path"], self._edits(args), ctx)


class MultiEditTool(Tool):
    name = "MultiEdit"
    kind = "edit"
    description = (
        "Apply several Edit-style replacements to one file atomically: edits run in order, each on the result of "
        "the previous one, and nothing is written unless all succeed. Same matching rules as Edit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path, or relative to the working directory"},
            "edits": {
                "type": "array",
                "description": "Ordered list of replacements",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["file_path", "edits"],
    }
    aliases = {"path": "file_path", "filepath": "file_path", "file": "file_path", "filename": "file_path",
               "edit": "edits", "changes": "edits", "replacements": "edits"}

    def apply_aliases(self, args: dict) -> dict:
        super().apply_aliases(args)
        edits = args.get("edits")
        if isinstance(edits, dict):
            edits = [edits]
        if isinstance(edits, list):
            fixed = []
            for e in edits:
                if isinstance(e, dict):
                    e = dict(e)
                    for alias, canonical in _EDIT_ALIASES.items():
                        if alias in e and canonical not in e:
                            e[canonical] = e.pop(alias)
                fixed.append(e)
            args["edits"] = fixed
        return args

    def permission_target(self, args: dict) -> str:
        return str(args.get("file_path", ""))

    async def preview(self, args: dict, ctx: ToolContext):
        if not args.get("edits"):
            return ToolResult.error("edits is empty; provide at least one {old_string, new_string}.")
        return preview_edits(args["file_path"], args["edits"], ctx)

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not args.get("edits"):
            return ToolResult.error("edits is empty; provide at least one {old_string, new_string}.")
        return apply_edits(args["file_path"], args["edits"], ctx)


def _span_newline(raw: str, start: int, end: int, default: str) -> str:
    """Line ending to use for a replacement: the matched region's own style."""
    seg = raw[start:end]
    if "\n" in seg:
        return "\r\n" if seg.count("\r\n") * 2 >= seg.count("\n") else "\n"
    nxt = raw.find("\n", end)
    if nxt == -1:
        return default
    return "\r\n" if nxt > 0 and raw[nxt - 1] == "\r" else "\n"


def _apply_outcome_to_raw(raw: str, outcome, default_newline: str) -> str:
    """Apply LF-view spans to the raw text, keeping each region's line endings."""
    if "\r\n" not in raw:
        return outcome.content
    lfmap = LFMap(raw)
    out = raw
    for span in sorted(outcome.spans, key=lambda s: s.start, reverse=True):
        a, b = lfmap.start(span.start), lfmap.end(span.end)
        nl = _span_newline(raw, a, b, default_newline)
        repl = span.replacement.replace("\n", nl) if nl == "\r\n" else span.replacement
        out = out[:a] + repl + out[b:]
    return out


def _compute_edits(raw_path: str, edits: list[dict], ctx: ToolContext):
    """Dry-run a list of edits. Returns a ToolResult on failure, else the planned change."""
    path = resolve_path(raw_path, ctx.cwd)
    shown = display_path(path, ctx.cwd)
    problem = _check_editable(path, ctx, must_exist=True, partial_ok=True)
    if problem:
        return problem
    try:
        tf = read_text_file(path)
    except OSError as e:
        return ToolResult.error(f"Could not read {shown}: {e}")
    raw, content = tf.raw, tf.text
    notes: list[str] = []
    first_line, last_new = 1, ""
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            return ToolResult.error(f"Edit {i} is not an object with old_string/new_string.")
        old, new = str(edit.get("old_string", "")), str(edit.get("new_string", ""))
        try:
            outcome = apply_edit(content, old, new, bool(edit.get("replace_all", False)))
        except EditError as e:
            prefix = f"Edit {i} of {len(edits)} failed, so no changes were written: " if len(edits) > 1 else ""
            return ToolResult.error(prefix + str(e), summary=one_line(f"Edit failed: {e}", 100))
        raw = _apply_outcome_to_raw(raw, outcome, tf.newline)
        content = outcome.content
        if i == 1 or outcome.first_line < first_line:
            first_line = outcome.first_line
        last_new = outcome.new_text
        if outcome.strategy != "exact":
            notes.append(f"edit {i} matched after {outcome.strategy} normalization" if len(edits) > 1
                         else f"matched after {outcome.strategy} normalization")
        if outcome.count > 1:
            notes.append(f"{outcome.count} occurrences replaced" + (f" in edit {i}" if len(edits) > 1 else ""))
    return path, shown, tf, raw, content, notes, first_line, last_new


def preview_edits(raw_path: str, edits: list[dict], ctx: ToolContext):
    plan = _compute_edits(raw_path, edits, ctx)
    if isinstance(plan, ToolResult):
        return plan
    _path, shown, tf, _raw, content, _notes, _fl, _ln = plan
    diff, _a, _r = make_diff(tf.text, content, shown)
    return {"type": "diff", "path": shown, "diff": diff}


def apply_edits(raw_path: str, edits: list[dict], ctx: ToolContext) -> ToolResult:
    plan = _compute_edits(raw_path, edits, ctx)
    if isinstance(plan, ToolResult):
        return plan
    path, shown, tf, raw, content, notes, first_line, last_new = plan
    ctx.rt.checkpoints.snapshot(path)
    try:
        write_text_file(path, raw, tf.encoding, None)
    except (OSError, UnicodeEncodeError) as e:
        return ToolResult.error(f"Could not write {shown}: {e}")
    ctx.rt.file_state.record(path)
    diff, added, removed = make_diff(tf.text, content, shown)
    warning = _syntax_warning(path, tf.text, content)
    note = f" ({'; '.join(notes)})" if notes else ""
    if len(edits) == 1:
        snippet = _snippet(content, first_line, last_new.count("\n") + 1)
        body = f"Edited {shown}{note}. Updated region:\n{snippet}{warning}"
    else:
        body = f"Applied {len(edits)} edits to {shown}{note} (+{added} -{removed} lines).{warning}"
    return ToolResult(content=body, summary=f"Edited {shown} (+{added} -{removed})",
                      display={"type": "diff", "path": shown, "diff": diff})
