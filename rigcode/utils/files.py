"""File helpers: project file listing (gitignore-aware), text IO that preserves
encoding and line endings, binary detection."""
from __future__ import annotations

import locale
import os
import re
import subprocess
import tempfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from rigcode.utils.text import detect_bom_encoding

DEFAULT_IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache", ".gradle", ".idea",
    "target", "dist", "build", "out", ".eggs", "coverage", ".terraform",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff", ".psd",
    ".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".jar", ".war",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj", ".class",
    ".pyc", ".pyo", ".whl", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".flac",
    ".ogg", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".sqlite", ".db", ".npy", ".npz",
    ".pkl", ".pt", ".pth", ".onnx", ".safetensors", ".gguf", ".h5", ".parquet", ".xlsx",
    ".xls", ".docx", ".pptx", ".iso", ".dmg",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def git_ls_files(root: Path, timeout: float = 20) -> list[str] | None:
    """Tracked + untracked-but-not-ignored files under root, relative POSIX paths.
    Returns None if root is not inside a git work tree."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", errors="replace")
    seen, files = set(), []
    for f in out.split("\0"):
        if f and f not in seen:
            seen.add(f)
            files.append(f)
    return files


def walk_files(root: Path, max_files: int = 100_000, ignored_dirs: set[str] | None = None) -> list[str]:
    ignored = DEFAULT_IGNORED_DIRS if ignored_dirs is None else ignored_dirs
    files: list[str] = []
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        dirnames[:] = sorted(d for d in dirnames if d not in ignored and not d.endswith(".egg-info"))
        rel_dir = os.path.relpath(dirpath, root_str)
        for name in sorted(filenames):
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            files.append(rel.replace("\\", "/"))
            if len(files) >= max_files:
                return files
    return files


def list_project_files(root: Path, max_files: int = 100_000) -> list[str]:
    """Gitignore-aware listing when possible, falling back to a pruned walk."""
    files = git_ls_files(root)
    if files is None:
        return walk_files(root, max_files)
    # git lists deleted-but-tracked files too; filter lazily-cheap here.
    out = []
    for f in files:
        if (root / f).exists():
            out.append(f)
            if len(out) >= max_files:
                break
    return out


def is_binary_file(path: Path, sniff: int = 8192) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sniff)
    except OSError:
        return False
    if not chunk:
        return False
    if detect_bom_encoding(chunk):
        return False
    if b"\0" in chunk:
        return True
    # Lots of control characters -> binary.
    ctrl = sum(1 for b in chunk if b < 9 or (13 < b < 32))
    return ctrl / len(chunk) > 0.3


@dataclass
class TextFile:
    raw: str         # decoded text exactly as on disk (original line endings)
    text: str        # LF view: CRLF normalized to \n
    encoding: str
    newline: str     # dominant line ending: "\n" or "\r\n"


def read_text_file(path: Path) -> TextFile:
    data = path.read_bytes()
    enc = detect_bom_encoding(data)
    raw = None
    if enc:
        raw = data.decode(enc)
    else:
        for candidate in ("utf-8", locale.getpreferredencoding(False), "cp932", "latin-1"):
            try:
                raw = data.decode(candidate)
                enc = candidate
                break
            except (UnicodeDecodeError, LookupError):
                continue
    assert raw is not None and enc is not None
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    newline = "\r\n" if crlf > lf else "\n"
    text = raw.replace("\r\n", "\n") if crlf else raw
    return TextFile(raw=raw, text=text, encoding=enc, newline=newline)


class LFMap:
    """Maps offsets in the LF view of a text back to the raw (CRLF-containing) text."""

    def __init__(self, raw: str) -> None:
        cr = [m.start() for m in re.finditer("\r\n", raw)]
        self._cr_lf = [p - j for j, p in enumerate(cr)]   # LF-view index of each '\n' that had a '\r'

    def start(self, q: int) -> int:
        return q + bisect_left(self._cr_lf, q)

    def end(self, q: int) -> int:
        return q + bisect_right(self._cr_lf, q - 1) if q > 0 else 0


def write_text_file(path: Path, text: str, encoding: str = "utf-8", newline: str | None = "\n") -> None:
    """Atomic-ish write: temp file in the same directory, then replace.
    newline="\\r\\n" converts LF to CRLF; None writes the text untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if newline == "\r\n":
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    data = text.encode(encoding, errors="strict")
    fd, tmp = tempfile.mkstemp(prefix=".rigcode-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        try:
            mode = path.stat().st_mode
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    write_text_file(path, text, "utf-8", "\n")
