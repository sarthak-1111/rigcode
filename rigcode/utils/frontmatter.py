"""Minimal YAML front-matter parsing for agent, skill and command files."""
from __future__ import annotations

import re

import yaml

_FM_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.S)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split '---\\nkey: value\\n---\\nbody' into (meta, body). Never raises."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    try:
        meta = yaml.safe_load(raw) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
        for line in raw.splitlines():
            if ":" in line and not line.startswith((" ", "\t")):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body.lstrip("\r\n")


def as_list(value) -> list[str]:
    """Accept 'a, b', ['a', 'b'] or None."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]
