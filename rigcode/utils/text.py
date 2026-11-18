"""Text helpers: truncation, ANSI stripping, decoding, token estimates."""
from __future__ import annotations

import codecs
import locale
import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def decode_output(data: bytes) -> str:
    """Decode subprocess output: UTF-8 first, then the console/locale codepage."""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in (locale.getpreferredencoding(False), "cp932", "cp1252"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def truncate_middle(text: str, max_chars: int, head_ratio: float = 0.35) -> tuple[str, bool]:
    """Keep the head and tail of long text, eliding the middle on a line boundary."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head_len = int(max_chars * head_ratio)
    tail_len = max_chars - head_len
    head = text[:head_len]
    tail = text[-tail_len:]
    nl = head.rfind("\n")
    if nl > head_len * 0.6:
        head = head[: nl + 1]
    nl = tail.find("\n")
    if 0 <= nl < tail_len * 0.4:
        tail = tail[nl + 1 :]
    omitted = len(text) - len(head) - len(tail)
    omitted_lines = text.count("\n", len(head), len(text) - len(tail))
    marker = f"\n... [{omitted} characters / ~{omitted_lines} lines omitted] ...\n"
    return head + marker + tail, True


def truncate_line(line: str, max_len: int) -> str:
    if len(line) <= max_len:
        return line
    return line[:max_len] + f"... [line truncated, {len(line) - max_len} more chars]"


def one_line(text: str, max_len: int = 80) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def est_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token for ASCII, ~1 token per CJK char."""
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return int((len(text) - non_ascii) / 3.8 + non_ascii * 0.9) + 1


def number_lines(lines: list[str], start: int = 1, max_line_len: int = 2000) -> str:
    """cat -n style: right-aligned line number, tab, content."""
    out = []
    for i, line in enumerate(lines, start):
        out.append(f"{i:>6}\t{truncate_line(line, max_line_len)}")
    return "\n".join(out)


def detect_bom_encoding(raw: bytes) -> str | None:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    return None


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def fmt_tokens(n: int | float | None) -> str:
    if not n:
        return "0"
    n = float(n)
    if n < 1000:
        return f"{n:.0f}"
    if n < 100_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1000:.0f}k"
