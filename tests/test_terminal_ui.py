"""Drive the interactive TerminalUI with a pipe input and a captured VT100 output."""
import asyncio
import io
import re

from helpers import make_runtime
from mock_server import Reply
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from rigcode.config import load_settings
from rigcode.ui.terminal import TerminalUI

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


class Capture(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


async def wait_for(cond, timeout=10.0):
    for _ in range(int(timeout / 0.05)):
        if cond():
            return True
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


async def submit(ui, inp, text):
    """Type a line, wait until the prompt accepted it and any task it started has finished."""
    inp.send_text(text + "\r")
    await wait_for(lambda: ui.session.default_buffer.text == "")
    await asyncio.sleep(0.1)
    await wait_for(lambda: not ui.busy)


def drive(project, server, script):
    """script(ui, rt, inp) is an async function that sends keys and waits."""
    buf = Capture()
    # enable_cpr=False: a StringIO can't answer cursor-position requests (each would stall ~1 s).
    out = Vt100_Output(buf, lambda: Size(rows=40, columns=110), enable_cpr=False)

    async def go():
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=out):
                settings = load_settings(project, {"provider.base_url": server.base_url, "provider.warmup": False})
                ui = TerminalUI(settings, history_path=None)
                rt = await make_runtime(project, server.base_url, ui, start=False)
                rt.settings = settings
                await rt.start()
                task = asyncio.create_task(ui.run(rt))
                await asyncio.sleep(0.3)
                try:
                    await script(ui, rt, inp)
                finally:
                    inp.send_text("\x04")            # Ctrl+D exits
                    await asyncio.wait_for(task, 15)
                    await rt.close()

    asyncio.run(go())
    return ANSI.sub("", buf.getvalue())


def test_turn_renders_markdown_and_exit(server, project):
    server.add(Reply(reasoning="Let me think.", content="Hello **from** the model.\n\n- item one\n- item two"))

    async def script(ui, rt, inp):
        inp.send_text("hi there\r")
        await wait_for(lambda: server.requests and not ui.busy)

    text = drive(project, server, script)
    first = server.requests[0]["messages"][-1]["content"]
    assert first.startswith("<system-reminder>\nEnvironment") and first.endswith("\n\nhi there")
    assert "from" in text and "the model." in text and "item two" in text
    assert "Thought for" in text


def test_permission_dialog_and_tool_block(server, project):
    server.add(Reply(tool_calls=[("Write", {"file_path": "made.txt", "content": "x\n"})]), Reply(content="Made it."))

    async def script(ui, rt, inp):
        inp.send_text("make a file\r")
        await wait_for(lambda: ui.dialog is not None)
        assert "Allow creating this file?" in ui.dialog.title
        inp.send_text("1")
        await wait_for(lambda: len(server.requests) == 2 and not ui.busy)

    text = drive(project, server, script)
    assert (project / "made.txt").read_text() == "x\n"
    assert "Write(made.txt)" in text and "Created made.txt" in text


def test_reject_with_feedback(server, project):
    server.add(Reply(tool_calls=[("Bash", {"command": "npm install"})]), Reply(content="ok"))

    async def script(ui, rt, inp):
        inp.send_text("install\r")
        await wait_for(lambda: ui.dialog is not None)
        inp.send_text("3")                      # "No, and tell rigcode what to do differently"
        await wait_for(lambda: ui.dialog is not None and ui.dialog.typing)
        inp.send_text("use pnpm\r")
        await wait_for(lambda: len(server.requests) == 2 and not ui.busy)

    drive(project, server, script)
    tool_msg = [m for m in server.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "use pnpm" in tool_msg["content"]


def test_escape_interrupts_and_queue(server, project):
    server.add(Reply(content="word " * 200, delay=0.03), Reply(content="Second."))

    async def script(ui, rt, inp):
        inp.send_text("long answer please\r")
        await wait_for(lambda: ui.busy and server.requests)
        await asyncio.sleep(0.3)
        inp.send_text("\x1b")
        await wait_for(lambda: not ui.busy)
        inp.send_text("next\r")
        await wait_for(lambda: len(server.requests) == 2 and not ui.busy)

    text = drive(project, server, script)
    assert "Interrupted" in text
    assert "interrupted" in server.requests[1]["messages"][-1]["content"].lower()


def test_slash_commands(server, project):
    async def script(ui, rt, inp):
        for cmd in ("/help", "/status", "/context", "/mode plan", "/cost", "/tools", "/agents", "/nosuch"):
            await submit(ui, inp, cmd)
        assert rt.permissions.mode == "plan"
        inp.send_text("\x1b[Z")                  # Shift+Tab cycles plan -> default
        await wait_for(lambda: rt.permissions.mode == "default")

    text = drive(project, server, script)
    assert "/compact" in text and "Permission mode: plan" in text and "explore" in text
    assert "Tool definitions" in text
