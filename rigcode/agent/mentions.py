"""@path mentions in a user message: attach the file (or directory listing) to the
message up front, which saves the model a slow tool round trip."""
from __future__ import annotations

import re

from rigcode.utils.files import is_binary_file, read_text_file
from rigcode.utils.paths import display_path, resolve_path
from rigcode.utils.text import number_lines

MENTION_RE = re.compile(r"(?<![\w/\\@])@(\"[^\"\n]+\"|[^\s\"'`<>|]+)")
MAX_FILES = 8
MAX_DIR_ENTRIES = 200


def expand_mentions(text: str, rt) -> tuple[str, list[str]]:
    """Returns (message with attachments appended, notes for the UI)."""
    budget = int(rt.settings.get("context.read_max_chars", 30000))
    attachments: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()
    for m in MENTION_RE.finditer(text):
        raw = m.group(1).strip('"').rstrip(".,;:!?)]}")
        if not raw:
            continue
        path = resolve_path(raw, rt.cwd)
        key = str(path).lower()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        if len(attachments) >= MAX_FILES:
            notes.append(f"Skipped @{raw}: at most {MAX_FILES} attachments per message")
            continue
        shown = display_path(path, rt.cwd)
        if path.is_dir():
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError as e:
                notes.append(f"Could not list @{raw}: {e}")
                continue
            listing = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_DIR_ENTRIES]]
            more = f"\n... {len(entries) - MAX_DIR_ENTRIES} more" if len(entries) > MAX_DIR_ENTRIES else ""
            attachments.append(f'<directory path="{shown}">\n' + "\n".join(listing) + more + "\n</directory>")
            notes.append(f"Listed {shown}/ ({len(entries)} entries)")
            continue
        if is_binary_file(path):
            notes.append(f"Not attaching @{raw}: binary file")
            continue
        try:
            tf = read_text_file(path)
        except OSError as e:
            notes.append(f"Could not read @{raw}: {e}")
            continue
        lines = tf.text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        body = number_lines(lines)
        truncated = ""
        if len(body) > budget:
            body = body[:budget].rsplit("\n", 1)[0]
            shown_lines = body.count("\n") + 1
            truncated = f"\n[Truncated: showing lines 1-{shown_lines} of {len(lines)}. Use Read with offset for more.]"
        else:
            rt.file_state.record(path)    # the model has the full current content: allow direct edits
        attachments.append(f'<file path="{shown}">\n{body}{truncated}\n</file>')
        notes.append(f"Attached {shown} ({len(lines)} lines)")
        budget = max(2000, budget - len(body))
    if not attachments:
        return text, notes
    return text + "\n\nAttached by the user:\n" + "\n\n".join(attachments), notes
