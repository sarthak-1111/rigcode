"""Mutable session state shared by tools: file read tracking, checkpoints, todos."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


class FileStateTracker:
    """Remembers which files the model has read (and their mtime at that point)
    so edits can require a prior read and detect external modifications."""

    def __init__(self) -> None:
        self._reads: dict[str, int] = {}
        self._partial: set[str] = set()      # seen only in part (Grep lines): enough for Edit, not Write
        self._order: list[str] = []
        self._paths: dict[str, Path] = {}

    def record(self, path: Path, partial: bool = False) -> None:
        k = _key(path)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            self._reads.pop(k, None)
            return
        if not partial:
            self._partial.discard(k)
        elif self._reads.get(k) != mtime or k in self._partial:
            self._partial.add(k)        # a partial view doesn't downgrade a full read of the same version
        self._reads[k] = mtime
        self._paths[k] = path
        if k in self._order:
            self._order.remove(k)
        self._order.append(k)

    def was_read(self, path: Path, partial_ok: bool = False) -> bool:
        k = _key(path)
        return k in self._reads and (partial_ok or k not in self._partial)

    def changed_since_read(self, path: Path) -> bool:
        k = _key(path)
        if k not in self._reads:
            return False
        try:
            return path.stat().st_mtime_ns != self._reads[k]
        except OSError:
            return True

    def forget(self, path: Path) -> None:
        """The model's copy of this file is stale (e.g. after /undo): require a fresh Read."""
        k = _key(path)
        self._reads.pop(k, None)
        self._partial.discard(k)
        self._paths.pop(k, None)
        if k in self._order:
            self._order.remove(k)

    def recent(self, n: int = 10) -> list[Path]:
        return [self._paths[k] for k in self._order[-n:] if k in self._paths]

    def clear(self) -> None:
        self._reads.clear()
        self._partial.clear()
        self._order.clear()
        self._paths.clear()


@dataclass
class TurnCheckpoint:
    label: str
    files: dict[str, bytes | None] = field(default_factory=dict)   # path -> original bytes (None = did not exist)


class Checkpoints:
    """Per-turn snapshots of files before the agent modified them (for /undo)."""

    def __init__(self) -> None:
        self.turns: list[TurnCheckpoint] = []

    def begin_turn(self, label: str) -> None:
        if self.turns and not self.turns[-1].files:
            self.turns[-1].label = label
            return
        self.turns.append(TurnCheckpoint(label))
        del self.turns[:-50]

    def snapshot(self, path: Path) -> None:
        if not self.turns:
            self.begin_turn("(start)")
        cp = self.turns[-1]
        k = str(path.resolve())
        if k in cp.files:
            return
        if path.exists():
            try:
                if path.stat().st_size > MAX_SNAPSHOT_BYTES:
                    return
                cp.files[k] = path.read_bytes()
            except OSError:
                return
        else:
            cp.files[k] = None

    def last_with_changes(self) -> TurnCheckpoint | None:
        for cp in reversed(self.turns):
            if cp.files:
                return cp
        return None

    def undo_last(self) -> list[tuple[str, str]]:
        """Restore files changed in the most recent turn that changed files."""
        while self.turns and not self.turns[-1].files:
            self.turns.pop()
        if not self.turns:
            return []
        cp = self.turns.pop()
        results = []
        for path_str, data in cp.files.items():
            path = Path(path_str)
            try:
                if data is None:
                    if path.exists():
                        path.unlink()
                    results.append((path_str, "deleted (was created)"))
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    results.append((path_str, "restored"))
            except OSError as e:
                results.append((path_str, f"failed: {e}"))
        return results


@dataclass
class TodoItem:
    content: str
    status: str = "pending"        # pending | in_progress | completed
    active_form: str | None = None


class TodoState:
    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    def set(self, items: list[TodoItem]) -> None:
        self.items = items

    def render_text(self) -> str:
        if not self.items:
            return "(no todos)"
        marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        return "\n".join(f"{i}. {marks.get(t.status, '[ ]')} {t.content}" for i, t in enumerate(self.items, 1))

    def open_items(self) -> list[TodoItem]:
        return [t for t in self.items if t.status != "completed"]
