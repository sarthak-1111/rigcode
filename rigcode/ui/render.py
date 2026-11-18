"""Rendering helpers: rich renderables -> ANSI strings, streamed-markdown block
splitting, diffs and todo lists."""
from __future__ import annotations

import io
import re
import shutil

from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.text import Text
from rich.theme import Theme

THEME = Theme({
    "markdown.code": "bold color(117)",
    "markdown.h1": "bold underline",
    "markdown.h2": "bold",
    "markdown.h3": "bold",
    "markdown.h4": "bold italic",
    "markdown.link": "underline color(117)",
    "markdown.link_url": "underline color(117)",
    "markdown.block_quote": "italic color(245)",
    "markdown.item.bullet": "color(117)",
    "markdown.item.number": "color(117)",
    "markdown.hr": "color(240)",
    "markdown.table.border": "color(240)",
    "diff.add": "color(114) on color(22)",
    "diff.del": "color(217) on color(52)",
    "diff.ctx": "color(250)",
    "diff.lineno": "color(242)",
    "diff.hunk": "color(242)",
})

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_TRAILING_RE = re.compile(r"[ ]+((?:\x1b\[[0-9;]*m)*)$", re.M)


def term_width(default: int = 100) -> int:
    return max(40, shutil.get_terminal_size((default, 30)).columns)


class _LeftHeading(Heading):
    """Headings left-aligned and unboxed (rich centers them and boxes h1)."""

    def __rich_console__(self, console, options):
        text = self.text.copy()
        text.justify = "left"
        if self.tag in ("h1", "h2"):
            yield Text("")
        yield text


class KMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _LeftHeading}


class Renderer:
    def __init__(self, code_theme: str = "ansi_dark"):
        self.code_theme = code_theme
        self.console = Console(file=io.StringIO(), force_terminal=True, color_system="256", theme=THEME,
                               legacy_windows=False, highlight=False, emoji=False, width=term_width() - 1)

    def ansi(self, *renderables, width: int | None = None) -> str:
        self.console.width = (width or term_width()) - 1
        with self.console.capture() as cap:
            for r in renderables:
                self.console.print(r)
        return _TRAILING_RE.sub(r"\1", cap.get())

    def markdown(self, text: str, width: int | None = None, style: str | None = None) -> str:
        md = KMarkdown(text, code_theme=self.code_theme, hyperlinks=False, justify="default", style=style or "none")
        return self.ansi(md, width=width)

    def diff(self, diff_text: str, max_lines: int = 40, indent: int = 5) -> Text:
        return render_diff(diff_text, max_lines=max_lines, indent=indent)

    def new_file(self, content: str, max_lines: int = 12, indent: int = 5) -> Text:
        out = Text()
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        width = len(str(min(len(lines), max_lines)))
        for i, line in enumerate(lines[:max_lines], 1):
            out.append(" " * indent)
            out.append(f"{i:>{width}} ", style="diff.lineno")
            out.append(line.expandtabs(4) + "\n", style="diff.ctx")
        if len(lines) > max_lines:
            out.append(" " * indent + f"… {len(lines) - max_lines} more lines\n", style="diff.lineno")
        return out


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _diff_rows(diff_text: str) -> list[tuple[str, int, str]]:
    """Unified diff -> rows of (kind, line number, text); kind is + - space or @ (hunk break)."""
    rows: list[tuple[str, int, str]] = []
    old_no = new_no = 0
    for line in diff_text.split("\n"):
        if line.startswith(("--- ", "+++ ", "\\")):
            continue
        m = _HUNK_RE.match(line)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            if rows:
                rows.append(("@", 0, ""))
            continue
        text = line[1:].expandtabs(4)
        if line.startswith("+"):
            rows.append(("+", new_no, text))
            new_no += 1
        elif line.startswith("-"):
            rows.append(("-", old_no, text))
            old_no += 1
        else:
            rows.append((" ", new_no, text))
            old_no += 1
            new_no += 1
    return rows


def render_diff(diff_text: str, max_lines: int = 40, indent: int = 5) -> Text:
    out = Text()
    rows = _diff_rows(diff_text)
    width = len(str(max((n for _, n, _ in rows), default=1)))
    pad = " " * indent
    shown = 0
    for kind, no, text in rows:
        if shown >= max_lines:
            break
        if kind == "@":
            out.append(pad + " " * width + " ⋮\n", style="diff.hunk")
            continue
        out.append(pad)
        out.append(f"{no:>{width}} ", style="diff.lineno")
        style = {"+": "diff.add", "-": "diff.del"}.get(kind, "diff.ctx")
        out.append(f"{kind} {text}\n", style=style)
        shown += 1
    total = sum(1 for k, _, _ in rows if k != "@")
    if total > shown:
        out.append(pad + f"… {total - shown} more diff lines\n", style="diff.lineno")
    return out


def render_todos(items: list[dict], indent: int = 5) -> Text:
    out = Text()
    for t in items:
        status = t.get("status", "pending")
        out.append(" " * indent)
        if status == "completed":
            out.append("☒ ", style="color(114)")
            out.append(str(t.get("content", "")) + "\n", style="strike color(244)")
        elif status == "in_progress":
            out.append("☐ ", style="bold color(117)")
            out.append(str(t.get("content", "")) + "\n", style="bold")
        else:
            out.append("☐ ", style="color(250)")
            out.append(str(t.get("content", "")) + "\n")
    return out


class BlockSplitter:
    """Splits streamed markdown into complete blocks that are safe to render permanently.

    A block ends at a blank line outside a fenced code block, or at the closing fence."""

    def __init__(self) -> None:
        self.buf = ""

    @property
    def pending(self) -> str:
        return self.buf

    def feed(self, text: str) -> list[str]:
        self.buf += text
        blocks = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            block, self.buf = self.buf[:cut], self.buf[cut:]
            if block.strip():
                blocks.append(block.strip("\n"))
        return blocks

    def flush(self) -> str:
        rest, self.buf = self.buf, ""
        return rest.strip("\n")

    def _find_cut(self) -> int | None:
        fence: str | None = None
        offset = 0
        lines = self.buf.split("\n")
        for line in lines[:-1]:                  # complete lines only
            end = offset + len(line) + 1
            m = _FENCE_RE.match(line)
            if fence:
                if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                    return end
            elif m:
                if self.buf[:offset].strip():
                    return offset                # finish the text before the code block first
                fence = m.group(1)
            elif not line.strip() and self.buf[:offset].strip():
                return end
            offset = end
        return None
