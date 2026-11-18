"""String-replacement engine for the Edit tools.

Design (informed by what broke in other harnesses): exact matching first, then
only *lossless* normalizations - Unicode punctuation look-alikes, trailing
whitespace, and a uniform indentation offset - plus recovery from two common
model slips (copied line-number prefixes, double-escaped strings). A match must
be unique; ambiguity at any tier stops the search instead of falling through to
a looser tier. Similarity scoring is used only to build error hints, never to
apply an edit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


class EditError(Exception):
    pass


@dataclass
class Span:
    start: int          # offsets into the LF-normalized content
    end: int
    replacement: str


@dataclass
class EditOutcome:
    content: str        # new LF-normalized content
    count: int
    strategy: str
    first_line: int     # 1-based line of the first replacement
    new_text: str       # replacement actually inserted at the first span
    spans: list[Span] = field(default_factory=list)


_UNICODE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
})


def _line_of(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _find_all(haystack: str, needle: str) -> list[int]:
    out, pos = [], haystack.find(needle)
    while pos != -1:
        out.append(pos)
        pos = haystack.find(needle, pos + len(needle))
    return out


def _lines_with_offsets(content: str) -> tuple[list[str], list[int]]:
    lines = content.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    return lines, offsets


# ------------------------------------------------------------------ tiers
# Each tier returns a list of Spans (empty = no match).

def _tier_exact(content: str, old: str, new: str) -> list[Span]:
    return [Span(p, p + len(old), new) for p in _find_all(content, old)]


def _tier_unicode(content: str, old: str, new: str) -> list[Span]:
    norm_old = old.translate(_UNICODE_MAP)
    norm_content = content.translate(_UNICODE_MAP)
    if norm_old == old and norm_content == content:
        return []
    # 1:1 character map, so offsets in the normalized text are valid in the original.
    return [Span(p, p + len(old), new) for p in _find_all(norm_content, norm_old)]


def _split_old(old: str) -> tuple[list[str], bool]:
    old_lines = old.split("\n")
    trailing_nl = len(old_lines) > 1 and old_lines[-1] == ""
    if trailing_nl:
        old_lines = old_lines[:-1]
    return old_lines, trailing_nl


def _window_span(content: str, lines: list[str], offsets: list[int], i: int, k: int, trailing_nl: bool,
                 replacement: str) -> Span | None:
    start = offsets[i]
    end = offsets[i + k - 1] + len(lines[i + k - 1])
    if trailing_nl:
        if end >= len(content):
            return None  # old_string ended with a newline but the window is the last line
        end += 1
        if replacement and not replacement.endswith("\n"):
            replacement += "\n"
    return Span(start, end, replacement)


def _tier_trailing_ws(content: str, old: str, new: str) -> list[Span]:
    old_lines, trailing_nl = _split_old(old)
    if not any(l.strip() for l in old_lines):
        return []
    lines, offsets = _lines_with_offsets(content)
    key = lambda s: s.translate(_UNICODE_MAP).rstrip()  # noqa: E731
    okeys = [key(l) for l in old_lines]
    k = len(okeys)
    spans: list[Span] = []
    last_end = -1
    for i in range(0, len(lines) - k + 1):
        if key(lines[i]) != okeys[0] or offsets[i] < last_end:
            continue
        if all(key(lines[i + j]) == okeys[j] for j in range(1, k)):
            span = _window_span(content, lines, offsets, i, k, trailing_nl, new)
            if span:
                spans.append(span)
                last_end = span.end
    return spans


def _uniform_delta(window: list[str], old_lines: list[str]) -> tuple[str, int] | None:
    """If every non-blank line differs only by the same leading-whitespace
    prefix, return (prefix, +1 if the file is more indented, -1 if less)."""
    delta: str | None = None
    direction = 0
    for w, o in zip(window, old_lines):
        ws, os_ = w.translate(_UNICODE_MAP).rstrip(), o.translate(_UNICODE_MAP).rstrip()
        if not ws.strip() and not os_.strip():
            continue
        if not ws.strip() or not os_.strip():
            return None
        if len(ws) >= len(os_) and ws.endswith(os_) and not ws[: len(ws) - len(os_)].strip():
            d, dirn = ws[: len(ws) - len(os_)], 1
        elif os_.endswith(ws) and not os_[: len(os_) - len(ws)].strip():
            d, dirn = os_[: len(os_) - len(ws)], -1
        else:
            return None
        if not d:
            dirn = 0
        if delta is None:
            delta, direction = d, dirn
        elif d != delta or (dirn != direction and d):
            return None
    if not delta or direction == 0:
        return None
    return delta, direction


def _reindent(new: str, delta: str, direction: int) -> str:
    out = []
    for line in new.split("\n"):
        if not line.strip():
            out.append(line)
        elif direction > 0:
            out.append(delta + line)
        elif line.startswith(delta):
            out.append(line[len(delta):])
        else:
            lead = len(line) - len(line.lstrip(" \t"))
            out.append(line[min(lead, len(delta)):])
    return "\n".join(out)


def _tier_indent(content: str, old: str, new: str) -> list[Span]:
    old_lines, trailing_nl = _split_old(old)
    nonblank = [l for l in old_lines if l.strip()]
    if not nonblank:
        return []
    lines, offsets = _lines_with_offsets(content)
    k = len(old_lines)
    idx0 = old_lines.index(nonblank[0])
    first_core = nonblank[0].translate(_UNICODE_MAP).strip()
    spans: list[Span] = []
    last_end = -1
    for i in range(0, len(lines) - k + 1):
        if offsets[i] < last_end:
            continue
        window = lines[i:i + k]
        if window[idx0].translate(_UNICODE_MAP).strip() != first_core:
            continue
        found = _uniform_delta(window, old_lines)
        if not found:
            continue
        delta, direction = found
        span = _window_span(content, lines, offsets, i, k, trailing_nl, _reindent(new, delta, direction))
        if span:
            spans.append(span)
            last_end = span.end
    return spans


TIERS = [
    ("exact", _tier_exact),
    ("unicode-punctuation", _tier_unicode),
    ("trailing-whitespace", _tier_trailing_ws),
    ("indentation-offset", _tier_indent),
]

# ------------------------------------------------------------------ input repairs

_LINE_NUM_RE = re.compile(r"^\s*\d+(?:\t|→| \| |: )")


def strip_line_numbers(text: str) -> str | None:
    """Undo copied Read-output prefixes ('    12\\tcode'). Only if every non-blank line has one."""
    lines = text.split("\n")
    nonblank = [l for l in lines if l.strip()]
    if not nonblank or not all(_LINE_NUM_RE.match(l) for l in nonblank):
        return None
    return "\n".join(_LINE_NUM_RE.sub("", l, count=1) for l in lines)


def unescape(text: str) -> str | None:
    """Undo double-escaping ('\\n' typed as two characters) when it looks systematic."""
    if "\\n" not in text and "\\t" not in text and '\\"' not in text:
        return None
    if "\n" in text and text.count("\\n") < 2:
        return None
    return (text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
            .replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\"))


# ------------------------------------------------------------------ omission guard

_COMMENT_RE = re.compile(r"^\s*(?:#+|//+|/\*+|\*+|<!--|--|;+|%+|rem\b|')\s*(.*?)\s*(?:\*+/|-->)?\s*$", re.I)
_KEYWORD_RE = re.compile(r"\b(rest|remaining|existing|unchanged|previous|other|same|omitted|elided|truncated|more)\b", re.I)
_PHRASE_RE = re.compile(
    r"\b(rest of (the )?(code|file|function|method|class|implementation|content|module|body)|"
    r"(existing|remaining|unchanged|previous|other) (code|content|methods|functions|implementation|lines|imports|logic|body))\b",
    re.I)
_TRAIL_RE = re.compile(r"\b(unchanged|remains?|here|as before|omitted|goes here|stays the same)\b", re.I)


def find_omission_placeholder(new: str, old: str = "") -> str | None:
    """Detect 'lazy' placeholders like '// ... rest of the code ...' that would
    delete real code. Lines that also appear in old_string are ignored."""
    for line in new.split("\n"):
        stripped = line.strip()
        if not stripped or (old and stripped in old):
            continue
        m = _COMMENT_RE.match(line)
        if m:
            body = m.group(1)
        elif stripped.startswith(("...", "…")):
            body = stripped
        else:
            continue
        has_ellipsis = "..." in body or "…" in body
        if (has_ellipsis and _KEYWORD_RE.search(body)) or (_PHRASE_RE.search(body) and (has_ellipsis or _TRAIL_RE.search(body))):
            return stripped
    return None


# ------------------------------------------------------------------ driver

def _apply_spans(content: str, spans: list[Span]) -> str:
    out = content
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        out = out[: span.start] + span.replacement + out[span.end:]
    return out


def _run_tiers(content: str, old: str, new: str, replace_all: bool, prefix: str = "") -> EditOutcome | None:
    for name, tier in TIERS:
        spans = tier(content, old, new)
        if not spans:
            continue
        if len(spans) > 1 and not replace_all:
            where = ", ".join(str(_line_of(content, s.start)) for s in spans[:12])
            how = "" if name == "exact" else f" (after {name} normalization)"
            raise EditError(
                f"old_string matches {len(spans)} locations{how}, at lines {where}. Include more surrounding lines "
                "so it matches exactly one place, or set replace_all=true to change every occurrence.")
        new_content = _apply_spans(content, spans)
        return EditOutcome(new_content, len(spans), prefix + name, _line_of(content, spans[0].start),
                           spans[0].replacement, spans)
    return None


def apply_edit(content: str, old: str, new: str, replace_all: bool = False) -> EditOutcome:
    """Apply one replacement to LF-normalized content. Raises EditError with a
    self-correction hint when there is no unique match."""
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")
    if old == new:
        raise EditError("old_string and new_string are identical; there is nothing to change.")
    if not old:
        raise EditError("old_string is empty. To create a file or replace all of it, use the Write tool.")
    placeholder = find_omission_placeholder(new, old)
    if placeholder:
        raise EditError(
            f"new_string contains a placeholder ('{placeholder}') standing in for code that is not in "
            "old_string. Applying it would delete that code. Write out the complete replacement text instead.")
    outcome = _run_tiers(content, old, new, replace_all)
    if outcome is None:
        stripped = strip_line_numbers(old)
        if stripped is not None:
            new_stripped = strip_line_numbers(new)
            outcome = _run_tiers(content, stripped, new_stripped if new_stripped is not None else new,
                                 replace_all, "line-number-prefixes+")
    if outcome is None:
        un_old = unescape(old)
        if un_old is not None:
            outcome = _run_tiers(content, un_old, unescape(new) or new, replace_all, "unescaped+")
    if outcome is None:
        raise EditError(no_match_message(content, old, new))
    if outcome.content == content:
        raise EditError("The edit would leave the file unchanged (new_string equals the matched text).")
    return outcome


def closest_region(content: str, old: str) -> tuple[int, int, float] | None:
    """(start_line, end_line, similarity) of the region most similar to `old`."""
    lines = content.split("\n")
    old_lines = old.strip("\n").split("\n")
    k = len(old_lines)
    if not lines or not old.strip():
        return None
    target = "\n".join(l.strip() for l in old_lines)
    first_idx = next((i for i, l in enumerate(old_lines) if l.strip()), 0)
    first = old_lines[first_idx].strip()
    scored = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        sm = SequenceMatcher(None, first, s, autojunk=False)
        if sm.real_quick_ratio() < 0.3 or sm.quick_ratio() < 0.3:
            continue
        scored.append((sm.ratio(), i))
    scored.sort(reverse=True)
    best = (0.0, 0)
    for _, i in scored[:60]:
        start = max(0, i - first_idx)
        window = "\n".join(l.strip() for l in lines[start:start + k])
        r = SequenceMatcher(None, target, window, autojunk=False).ratio()
        if r > best[0]:
            best = (r, start)
    if best[0] <= 0:
        return None
    start = best[1]
    return start + 1, min(len(lines), start + k), best[0]


def no_match_message(content: str, old: str, new: str = "") -> str:
    msg = ("old_string was not found in the file. It must match the current text exactly (whitespace and "
           "indentation included, without Read's line-number prefixes).")
    if new and len(new.strip()) > 10 and new.strip("\n") in content:
        line = _line_of(content, content.find(new.strip("\n")))
        msg += f" Note: new_string already appears in the file at line {line} - the edit may already be applied."
    region = closest_region(content, old)
    if region and region[2] >= 0.6:
        a, b, sim = region
        lines = content.split("\n")
        shown = lines[a - 1: min(b, a - 1 + 20)]
        snippet = "\n".join(f"{n:>6}\t{l}" for n, l in enumerate(shown, a))
        msg += (f"\nThe most similar text ({sim:.0%} similar) is at lines {a}-{b}:\n{snippet}\n"
                "Copy the exact text from there (without the line-number prefixes) and retry.")
    else:
        msg += " Re-read the relevant part of the file (it may have changed) and retry with its exact text."
    return msg
