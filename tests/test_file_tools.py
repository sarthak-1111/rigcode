import asyncio
import os
import time

import pytest
from helpers import make_runtime

from rigcode.agent.loop import Agent
from rigcode.tools.base import ToolContext
from rigcode.tools.files import EditTool, MultiEditTool, ReadTool, WriteTool
from rigcode.tools.search import GlobTool, GrepTool


@pytest.fixture
def ctx(project):
    rt = asyncio.run(make_runtime(project, "http://127.0.0.1:9/v1", start=False))
    return ToolContext(rt, Agent(rt), "t1")


def run(tool, args, ctx):
    return asyncio.run(tool.run(args, ctx))


def test_read_formats_and_pages(project, ctx):
    (project / "a.txt").write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")
    res = run(ReadTool(), {"file_path": "a.txt", "offset": 10, "limit": 5}, ctx)
    assert "    10\tline 10" in res.content and "line 15" not in res.content
    assert "offset=15" in res.content
    missing = run(ReadTool(), {"file_path": "a.tx"}, ctx)
    assert missing.is_error and "a.txt" in missing.content


def test_edit_requires_read_and_preserves_crlf(project, ctx):
    p = project / "w.py"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    res = run(EditTool(), {"file_path": "w.py", "old_string": "b = 2", "new_string": "b = 3"}, ctx)
    assert res.is_error and "Read" in res.content
    run(ReadTool(), {"file_path": "w.py"}, ctx)
    res = run(EditTool(), {"file_path": "w.py", "old_string": "b = 2", "new_string": "b = 3\nc = 4"}, ctx)
    assert not res.is_error, res.content
    assert p.read_bytes() == b"a = 1\r\nb = 3\r\nc = 4\r\n"


def test_mixed_line_endings_kept_per_line(project, ctx):
    p = project / "m.txt"
    p.write_bytes(b"one\r\ntwo\nthree\r\n")
    run(ReadTool(), {"file_path": "m.txt"}, ctx)
    res = run(EditTool(), {"file_path": "m.txt", "old_string": "two", "new_string": "TWO"}, ctx)
    assert not res.is_error, res.content
    assert p.read_bytes() == b"one\r\nTWO\nthree\r\n"


def test_bom_and_external_change_detection(project, ctx):
    p = project / "b.py"
    p.write_bytes(b"\xef\xbb\xbfx = 1\n")
    run(ReadTool(), {"file_path": "b.py"}, ctx)
    assert not run(EditTool(), {"file_path": "b.py", "old_string": "x = 1", "new_string": "x = 2"}, ctx).is_error
    assert p.read_bytes() == b"\xef\xbb\xbfx = 2\n"
    time.sleep(0.02)
    p.write_bytes(b"\xef\xbb\xbfx = 5\n")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    res = run(EditTool(), {"file_path": "b.py", "old_string": "x = 5", "new_string": "x = 6"}, ctx)
    assert res.is_error and "modified since" in res.content


def test_write_and_multiedit_atomic(project, ctx):
    res = run(WriteTool(), {"file_path": "new/dir/f.py", "content": "a = 1\nb = 2\n"}, ctx)
    assert not res.is_error and (project / "new/dir/f.py").read_text() == "a = 1\nb = 2\n"
    bad = run(MultiEditTool(), {"file_path": "new/dir/f.py", "edits": [
        {"old_string": "a = 1", "new_string": "a = 10"}, {"old_string": "zzz", "new_string": "y"}]}, ctx)
    assert bad.is_error and (project / "new/dir/f.py").read_text() == "a = 1\nb = 2\n"
    ok = run(MultiEditTool(), {"file_path": "new/dir/f.py", "edits": [
        {"old_string": "a = 1", "new_string": "a = 10"}, {"old_string": "b = 2", "new_string": "b = 20"}]}, ctx)
    assert not ok.is_error and (project / "new/dir/f.py").read_text() == "a = 10\nb = 20\n"
    overwrite = run(WriteTool(), {"file_path": "unread.py", "content": "x"}, ctx)
    assert not overwrite.is_error
    (project / "exists.py").write_text("old", encoding="utf-8")
    res = run(WriteTool(), {"file_path": "exists.py", "content": "new"}, ctx)
    assert res.is_error and "Read" in res.content


def test_glob_and_grep(project, ctx):
    (project / "src").mkdir()
    (project / "src" / "a.py").write_text("def alpha():\n    return 'TODO'\n", encoding="utf-8")
    (project / "src" / "b.txt").write_text("nothing here\n", encoding="utf-8")
    res = run(GlobTool(), {"pattern": "*.py"}, ctx)
    assert "src/a.py" in res.content
    res = run(GrepTool(), {"pattern": "TODO", "output_mode": "content"}, ctx)
    assert "a.py" in res.content and "TODO" in res.content
    res = run(GrepTool(), {"pattern": "alpha("}, ctx)       # invalid regex -> literal search
    assert "a.py" in res.content


def test_glob_finds_directories(project, ctx):
    (project / "app" / "migrations").mkdir(parents=True)
    (project / "app" / "migrations" / "0001.py").write_text("x = 1\n", encoding="utf-8")
    res = run(GlobTool(), {"pattern": "**/migrations"}, ctx)
    assert res.content.strip() == "app/migrations/", res.content
    res = run(GlobTool(), {"pattern": "migrations/"}, ctx)
    assert "app/migrations/" in res.content


def test_grep_lines_allow_edit_but_not_write(project, ctx):
    (project / "g.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    run(GrepTool(), {"pattern": "return 1", "output_mode": "content"}, ctx)
    res = run(EditTool(), {"file_path": "g.py", "old_string": "return 1", "new_string": "return 2"}, ctx)
    assert not res.is_error, res.content
    (project / "h.py").write_text("a = 1\n", encoding="utf-8")
    run(GrepTool(), {"pattern": "a = 1", "output_mode": "content"}, ctx)
    res = run(WriteTool(), {"file_path": "h.py", "content": "b = 2\n"}, ctx)
    assert res.is_error and "in full" in res.content
    run(ReadTool(), {"file_path": "h.py"}, ctx)
    assert not run(WriteTool(), {"file_path": "h.py", "content": "b = 2\n"}, ctx).is_error


def test_grep_files_mode_does_not_count_as_read(project, ctx):
    (project / "k.py").write_text("x = 1\n", encoding="utf-8")
    run(GrepTool(), {"pattern": "x = 1"}, ctx)
    res = run(EditTool(), {"file_path": "k.py", "old_string": "x = 1", "new_string": "x = 2"}, ctx)
    assert res.is_error


def test_grep_context_cap_is_reported(project, ctx):
    (project / "c.py").write_text("\n".join(f"l{i}" for i in range(200)), encoding="utf-8")
    res = run(GrepTool(), {"pattern": "l100", "output_mode": "content", "context": 80}, ctx)
    assert res.content.startswith("(context capped at 50 lines")
    assert "c.py-51-l50" in res.content and "c.py-50-l49" not in res.content
