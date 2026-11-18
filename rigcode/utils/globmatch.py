"""Glob pattern -> regex translation with **, {a,b}, [abc] support."""
from __future__ import annotations

import functools
import os
import re

_CASE_INSENSITIVE = os.name == "nt"


def _translate(pat: str) -> str:
    i, n = 0, len(pat)
    res: list[str] = []
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                if pat.startswith("**/", i):
                    res.append("(?:.*/)?")
                    i += 3
                    continue
                res.append(".*")
                i += 2
                continue
            res.append("[^/]*")
        elif c == "?":
            res.append("[^/]")
        elif c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                res.append(re.escape(c))
            else:
                body = pat[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                res.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "{":
            depth, j = 0, i
            while j < n:
                if pat[j] == "{":
                    depth += 1
                elif pat[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n:
                alts = _split_top_level(pat[i + 1 : j])
                res.append("(?:" + "|".join(_translate(a) for a in alts) + ")")
                i = j
            else:
                res.append(re.escape(c))
        else:
            res.append(re.escape(c))
        i += 1
    return "".join(res)


def _split_top_level(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


@functools.lru_cache(maxsize=512)
def compile_glob(pattern: str) -> re.Pattern:
    pattern = pattern.replace("\\", "/")
    if pattern.startswith("./"):
        pattern = pattern[2:]
    flags = re.IGNORECASE if _CASE_INSENSITIVE else 0
    return re.compile("^" + _translate(pattern) + "$", flags)


def glob_match(pattern: str, rel_path: str) -> bool:
    return bool(compile_glob(pattern).match(rel_path.replace("\\", "/")))


def basename_match(pattern: str, rel_path: str) -> bool:
    """Match a slash-less pattern against the file name only (gitignore style)."""
    return bool(compile_glob(pattern).match(rel_path.replace("\\", "/").rsplit("/", 1)[-1]))
