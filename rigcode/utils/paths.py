"""Path helpers: resolving model-supplied paths, display paths, rigcode home."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

IS_WINDOWS = os.name == "nt"


def rigcode_home() -> Path:
    """Global config/data directory (~/.rigcode, overridable with RIGCODE_HOME)."""
    env = os.environ.get("RIGCODE_HOME")
    return Path(env).expanduser() if env else Path.home() / ".rigcode"


def resolve_path(raw: str, cwd: Path) -> Path:
    """Turn a model-supplied path into an absolute, normalized Path.

    Accepts absolute or cwd-relative paths, ~, quotes, and on Windows also
    Git-Bash (/d/foo) and WSL (/mnt/d/foo) style paths.
    """
    p = (raw or "").strip().strip('"').strip("'")
    if IS_WINDOWS:
        m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", p) or re.match(r"^/([a-zA-Z])(/.*)?$", p)
        if m:
            p = f"{m.group(1).upper()}:{m.group(2) or '/'}"
    p = os.path.expanduser(p)
    path = Path(p)
    if not path.is_absolute():
        path = cwd / path
    return Path(os.path.normpath(str(path)))


def is_within(path: Path, root: Path) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return a == b or a.startswith(b.rstrip("\\/") + os.sep)
    except (ValueError, OSError):
        return False


def is_within_any(path: Path, roots: list[Path]) -> bool:
    return any(is_within(path, r) for r in roots)


def display_path(path: Path | str, cwd: Path) -> str:
    """Short path for display: relative to cwd when inside it, else absolute."""
    path = Path(path)
    if is_within(path, cwd):
        try:
            rel = os.path.relpath(str(path), str(cwd))
            return rel.replace("\\", "/") if rel != "." else "."
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def project_slug(path: Path) -> str:
    """Stable, filesystem-safe directory name for per-project data."""
    norm = os.path.normcase(str(path.resolve()))
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name or "root").strip("-")[:40] or "root"
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
    return f"{name}-{digest}"
