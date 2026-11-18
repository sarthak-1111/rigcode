"""Lenient JSON parsing for model-generated tool arguments.

Open-weight models occasionally emit almost-JSON: code fences, trailing commas,
raw newlines inside strings, Python literals, or a missing closing brace.
We try progressively more aggressive repairs before giving up.
"""
from __future__ import annotations

import ast
import json
import re


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", s, re.S)
    return m.group(1).strip() if m else s


def _escape_control_chars_in_strings(s: str) -> str:
    """Escape raw newlines/tabs that appear inside JSON string literals."""
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
            elif ch == "\\":
                esc = True
                out.append(ch)
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _close_brackets(s: str) -> str:
    """Append missing closing quotes/brackets for truncated JSON."""
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = '"' if in_str else ""
    return s + tail + "".join(reversed(stack))


def loads_lenient(raw: str | None) -> tuple[object | None, str | None]:
    """Return (value, None) on success or (None, error_message)."""
    if raw is None:
        return {}, None
    s = raw.strip()
    if not s:
        return {}, None
    first_err = None
    candidates = []
    s1 = _strip_fences(s)
    candidates.append(s1)
    # Extract the outermost {...} if there is surrounding prose.
    start, end = s1.find("{"), s1.rfind("}")
    if start > 0 or (end != -1 and end < len(s1) - 1):
        if start != -1 and end > start:
            candidates.append(s1[start : end + 1])
    for c in list(candidates):
        fixed = _escape_control_chars_in_strings(c)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        candidates.append(fixed)
        candidates.append(_close_brackets(fixed))
    for c in candidates:
        try:
            return json.loads(c), None
        except json.JSONDecodeError as e:
            first_err = first_err or e
    # Python-literal style dicts: {'a': True, 'b': None}
    try:
        val = ast.literal_eval(s1)
        if isinstance(val, (dict, list)):
            return json.loads(json.dumps(val)), None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    msg = f"{first_err.msg} at line {first_err.lineno} column {first_err.colno}" if first_err else "unparseable"
    return None, msg
