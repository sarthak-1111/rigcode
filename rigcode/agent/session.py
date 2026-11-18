"""Session transcripts: append-only JSONL under ~/.rigcode/projects/<project>/<id>.jsonl.

Record types: meta, message, compact (replaces the message list), note.
Append-only writes mean a crash can lose at most the last line.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rigcode.utils.paths import project_slug, rigcode_home


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def sessions_dir(project_root: Path) -> Path:
    return rigcode_home() / "projects" / project_slug(project_root)


@dataclass
class SessionInfo:
    id: str
    path: Path
    mtime: float
    title: str
    n_messages: int


class SessionStore:
    def __init__(self, project_root: Path, session_id: str | None = None):
        self.project_root = project_root
        self.id = session_id or new_session_id()
        self.dir = sessions_dir(project_root)
        self.path = self.dir / f"{self.id}.jsonl"
        self.enabled = True

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"ts": time.time(), **record}, ensure_ascii=False, default=str) + "\n"
            with open(self.path, "ab") as fh:
                if fh.tell() > 0 and not self._ends_with_newline():
                    line = "\n" + line          # a crash left a torn last line: don't glue onto it
                fh.write(line.encode("utf-8"))
        except OSError:
            pass

    def _ends_with_newline(self) -> bool:
        with open(self.path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"

    def meta(self, **fields) -> None:
        self._write({"type": "meta", **fields})

    def message(self, msg: dict) -> None:
        self._write({"type": "message", "message": msg})

    def compact(self, messages: list[dict], summary_tokens: int | None = None) -> None:
        self._write({"type": "compact", "messages": messages, "summary_tokens": summary_tokens})

    def note(self, text: str) -> None:
        self._write({"type": "note", "text": text})

    # ------------------------------------------------------------------ loading
    @staticmethod
    def load(path: Path) -> tuple[dict, list[dict]]:
        meta: dict = {}
        messages: list[dict] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line
                    kind = rec.get("type")
                    if kind == "meta":
                        meta.update({k: v for k, v in rec.items() if k not in ("type", "ts")})
                    elif kind == "message" and isinstance(rec.get("message"), dict):
                        messages.append(rec["message"])
                    elif kind == "compact" and isinstance(rec.get("messages"), list):
                        messages = list(rec["messages"])
        except OSError:
            pass
        return meta, _repair_tail(messages)

    @staticmethod
    def list(project_root: Path, limit: int = 30) -> list[SessionInfo]:
        d = sessions_dir(project_root)
        if not d.is_dir():
            return []
        infos = []
        for p in sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            title, count = "", 0
            try:
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("type") == "message":
                            count += 1
                            m = rec["message"]
                            if not title and m.get("role") == "user" and not m.get("_meta"):
                                title = _first_text(m.get("content"))
                        elif rec.get("type") == "meta" and rec.get("title"):
                            title = rec["title"]
            except OSError:
                continue
            infos.append(SessionInfo(p.stem, p, p.stat().st_mtime, " ".join(title.split())[:90], count))
        return infos

    @staticmethod
    def resolve(project_root: Path, session_id: str | None) -> Path | None:
        d = sessions_dir(project_root)
        if session_id:
            p = d / f"{session_id}.jsonl"
            if p.exists():
                return p
            matches = sorted(d.glob(f"*{session_id}*.jsonl"), key=os.path.getmtime, reverse=True) if d.is_dir() else []
            return matches[0] if matches else None
        infos = SessionStore.list(project_root, limit=1)
        return infos[0].path if infos else None


def _first_text(content) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    else:
        text = ""
    # Strip injected reminders so titles show what the user typed.
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    return text.strip()


def _repair_tail(messages: list[dict]) -> list[dict]:
    """Ensure every assistant tool call has a tool result (e.g. after a crash)."""
    out = list(messages)
    pending: list[dict] = []
    answered = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("id") not in answered:
                    pending.append({"role": "tool", "tool_call_id": tc.get("id"), "name": tc.get("function", {}).get("name"),
                                    "content": "[No result: the session ended before this tool call completed.]"})
    if pending:
        out.extend(pending)
    return out
