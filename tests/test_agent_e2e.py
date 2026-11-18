"""End-to-end agent runs against the scripted mock server."""
import asyncio
import sys
from pathlib import Path

from helpers import RecordingUI, make_runtime
from mock_server import Reply

from rigcode.agent.session import SessionStore
from rigcode.permissions import Decision


def run(coro):
    return asyncio.run(coro)


def tool_messages(body):
    return [m for m in body["messages"] if m["role"] == "tool"]


def test_basic_loop_write_then_answer(server, project):
    server.add(Reply(reasoning="Need a file.", content="Creating it.",
                     tool_calls=[("Write", {"file_path": "hello.txt", "content": "hi\n"})]),
               Reply(content="Done: created hello.txt."))
    ui = RecordingUI()

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        stats = await rt.submit("make hello.txt")
        await rt.close()
        return rt, stats

    rt, stats = run(go())
    assert stats.stopped_reason == "done" and stats.tool_calls == 1
    assert (project / "hello.txt").read_text() == "hi\n"
    first, second = server.requests
    # Prompt-cache invariants: same tools, one fixed system message, append-only history.
    assert first["tools"] == second["tools"]
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert "# Project layout" in first["messages"][0]["content"]
    assert "Date:" not in first["messages"][0]["content"]          # volatile facts stay out of the prefix
    opening = first["messages"][1]["content"]
    assert opening.startswith("<system-reminder>\nEnvironment at the start") and opening.endswith("make hello.txt")
    assert second["messages"][:len(first["messages"])] == first["messages"]
    assistant = second["messages"][len(first["messages"])]
    assert assistant["reasoning_content"] == "Need a file." and assistant["content"] == "Creating it."
    result = tool_messages(second)[0]
    assert result["tool_call_id"] == assistant["tool_calls"][0]["id"] and "Created hello.txt" in result["content"]
    assert "id_slot" not in first       # unpinned: llama.cpp restores the cached state itself
    assert first["cache_prompt"] is True and first["reasoning_effort"] == "low"
    assert ui.permission_requests and ui.permission_requests[0].tool_name == "Write"
    # transcript persisted
    _, msgs = SessionStore.load(rt.session.path)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]


def test_parallel_reads_and_repeat_guard(server, project):
    (project / "a.py").write_text("A = 1\n", encoding="utf-8")
    (project / "b.py").write_text("B = 2\n", encoding="utf-8")
    same = ("Grep", {"pattern": "nothing-matches"})
    server.add(Reply(tool_calls=[("Read", {"file_path": "a.py"}), ("Read", {"file_path": "b.py"})]),
               Reply(tool_calls=[same]), Reply(tool_calls=[same]), Reply(tool_calls=[same]),
               Reply(content="A is 1 and B is 2."))
    ui = RecordingUI()

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        stats = await rt.submit("what are A and B?")
        await rt.close()
        return stats

    stats = run(go())
    assert stats.stopped_reason == "done"
    results = tool_messages(server.requests[1])
    assert "A = 1" in results[0]["content"] and "B = 2" in results[1]["content"]
    third_grep = tool_messages(server.requests[4])[-1]["content"]
    assert "3 times" in third_grep
    assert not ui.permission_requests      # reads and greps are auto-approved


def test_rejection_with_feedback_and_noninteractive_denial(server, project):
    server.add(Reply(tool_calls=[("Bash", {"command": "npm install left-pad"})]),
               Reply(content="OK, I won't install it."))
    ui = RecordingUI(decisions=[Decision(allow=False, feedback="use the stdlib instead")])

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        await rt.submit("add left-pad")
        await rt.close()

    run(go())
    result = tool_messages(server.requests[1])[0]["content"]
    assert "rejected" in result and "use the stdlib instead" in result


TASK = ("Task", {"description": "find config", "subagent_type": "explore", "prompt": "Find where config is loaded."})


def test_subagent_forks_the_main_conversation(server, project):
    server.add(Reply(tool_calls=[TASK]),
               Reply(content="Config is loaded in app/config.py:12."),
               Reply(content="The config loader is app/config.py:12."))
    ui = RecordingUI()

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        await rt.submit("where is config loaded?")
        await rt.close()

    run(go())
    main1, sub, main2 = server.requests
    n = len(main1["messages"])
    # The fork continues from the main agent's exact prompt, so the server's cache of it is reused...
    assert sub["tools"] == main1["tools"] and sub["messages"][:n] == main1["messages"]
    task_call = sub["messages"][n]
    assert task_call["role"] == "assistant" and task_call["tool_calls"][0]["function"]["name"] == "Task"
    brief = sub["messages"][n + 1]
    assert brief["role"] == "tool" and brief["tool_call_id"] == task_call["tool_calls"][0]["id"]
    assert '"explore" subagent, forked' in brief["content"] and "Find where config is loaded." in brief["content"]
    # ...and the main agent then continues from the fork's state, which starts with its own conversation.
    assert main2["messages"][:n + 1] == sub["messages"][:n + 1]
    assert "app/config.py:12" in tool_messages(main2)[-1]["content"]
    assert all("id_slot" not in r for r in server.requests)
    assert not server.flushes()
    assert ui.subagents and ui.subagents[0][0] == "explore"


def test_fresh_subagent_shares_prefix_then_flushes_its_slot(server, project):
    server.add(Reply(tool_calls=[TASK]),
               Reply(content="Config is loaded in app/config.py:12."),
               Reply(content="The config loader is app/config.py:12."))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), agents__context="fresh")
        await rt.submit("where is config loaded?")
        await rt.close()

    run(go())
    main1, sub, main2 = server.requests
    assert sub["messages"][0] == main1["messages"][0] and len(sub["messages"]) == 2
    first = sub["messages"][1]
    assert first["role"] == "user" and '"explore" subagent, working for' in first["content"]
    assert "Environment at the start" in first["content"]
    assert len(server.flushes()) == 1       # moves the subagent's slot to the RAM cache before main2
    assert "app/config.py:12" in tool_messages(main2)[0]["content"]


def test_large_conversation_uses_fresh_subagent(server, project):
    server.add(Reply(tool_calls=[TASK]), Reply(content="report"), Reply(content="done"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), agents__fork_max_tokens=100)
        await rt.submit("where is config loaded?")
        await rt.close()

    run(go())
    sub = server.requests[1]
    assert len(sub["messages"]) == 2 and sub["messages"][1]["role"] == "user"


def test_pinned_slots_when_enabled(server, project):
    server.add(Reply(tool_calls=[TASK]), Reply(content="fork report"), Reply(content="first done"),
               Reply(tool_calls=[TASK]), Reply(content="fresh report"), Reply(content="second done"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), provider__llama_cpp__slot_pinning=True)
        await rt.submit("look")
        rt.settings.set("agents.context", "fresh")
        await rt.submit("look again")
        await rt.close()

    run(go())
    slots = [r.get("id_slot") for r in server.requests]
    assert slots == [3, 3, 3, 3, 2, 3]      # the fork stays on the main slot; fresh subagents get their own
    assert not server.flushes()             # pinned slots keep their own cache: nothing to flush


def test_read_only_subagent_cannot_write(server, project):
    server.add(Reply(tool_calls=[("Task", {"description": "x", "subagent_type": "explore", "prompt": "write a file"})]),
               Reply(tool_calls=[("Write", {"file_path": "x.txt", "content": "nope"})]),
               Reply(content="I could not write (read-only)."),
               Reply(content="done"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI())
        await rt.submit("go")
        await rt.close()

    run(go())
    assert not (project / "x.txt").exists()
    assert "not available to the explore subagent" in tool_messages(server.requests[2])[-1]["content"]


def test_text_mode_fallback(server, project):
    server.add(Reply(status=500, error="tools param requires --jinja flag"),
               Reply(content='I will read it.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "n.txt"}}\n</tool_call>'),
               Reply(content="It says hello."))
    (project / "n.txt").write_text("hello\n", encoding="utf-8")
    ui = RecordingUI()

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        stats = await rt.submit("read n.txt")
        await rt.close()
        return stats

    stats = run(go())
    assert stats.stopped_reason == "done"
    assert "tools" in server.requests[0] and "tools" not in server.requests[1]
    assert "<tools>" in server.requests[1]["messages"][0]["content"]
    last = server.requests[2]["messages"][-1]
    assert last["role"] == "user" and "<tool_result" in last["content"] and "hello" in last["content"]


def test_auto_compaction_mid_turn(server, project):
    (project / "big.txt").write_text(("x" * 79 + "\n") * 500, encoding="utf-8")
    server.add(Reply(tool_calls=[("Read", {"file_path": "big.txt"})]),
               Reply(content="Summary: user asked to inspect big.txt; it is 40k x characters."),
               Reply(content="big.txt is just x characters."))
    ui = RecordingUI()

    async def go():
        rt = await make_runtime(project, server.base_url, ui, context__compact_at_tokens=6000,
                                context__read_max_chars=40000)
        stats = await rt.submit("what is in big.txt?")
        n = len(rt.main_agent.messages)
        await rt.close()
        return stats, n

    stats, _ = run(go())
    compact_req = server.requests[1]
    assert compact_req.get("tool_choice") == "none" and compact_req["tools"] == server.requests[0]["tools"]
    assert "summary" in compact_req["messages"][-1]["content"].lower()
    final_req = server.requests[2]
    assert len(final_req["messages"]) == 2      # system + compacted summary message
    assert final_req["messages"][0] == server.requests[0]["messages"][0]
    assert "conversation-summary" in final_req["messages"][1]["content"]
    assert ui.compactions


def test_interrupt_then_continue(server, project):
    server.add(Reply(content="This is a long answer " * 20, delay=0.05), Reply(content="Second turn ok."))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI())
        task = asyncio.create_task(rt.submit("talk a lot"))
        await asyncio.sleep(0.4)
        task.cancel()
        stats = await task
        stats2 = await rt.submit("again")
        await rt.close()
        return stats, stats2

    stats, stats2 = run(go())
    assert stats.stopped_reason == "interrupted" and stats2.stopped_reason == "done"
    assert "interrupted" in server.requests[1]["messages"][-1]["content"].lower()


def test_queued_message_steers_next_step(server, project):
    server.add(Reply(tool_calls=[("Glob", {"pattern": "*.md"})]), Reply(content="Switched to Python files."))
    ui = RecordingUI(queued=["actually look at .py files"])

    async def go():
        rt = await make_runtime(project, server.base_url, ui)
        await rt.submit("find markdown files")
        await rt.close()

    run(go())
    assert server.requests[1]["messages"][-1] == {"role": "user", "content": "actually look at .py files"}


def test_pre_tool_use_hook_blocks(server, project):
    hook = project / "hook.py"
    hook.write_text("import sys, json\nd = json.load(sys.stdin)\n"
                    "sys.exit(2 if 'forbidden' in d['tool_input'].get('command', '') else 0)\n"
                    "", encoding="utf-8")
    py = Path(sys.executable).as_posix()
    hooks = {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": f'"{py}" "{hook.as_posix()}"'}]}]}
    server.add(Reply(tool_calls=[("Bash", {"command": "echo forbidden"})]), Reply(content="blocked, fine"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), hooks=hooks)
        await rt.submit("run it")
        await rt.close()

    run(go())
    assert "Blocked by a PreToolUse hook" in tool_messages(server.requests[1])[0]["content"]


def test_resume_restores_history(server, project):
    server.add(Reply(content="first answer"), Reply(content="second answer"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI())
        await rt.submit("first question")
        path = rt.session.path
        await rt.close()
        rt2 = await make_runtime(project, server.base_url, RecordingUI(), start=False)
        await rt2.start(resume=path)
        await rt2.submit("second question")
        await rt2.close()

    run(go())
    first, resumed = server.requests
    msgs = resumed["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[:2] == first["messages"]        # the saved history renders exactly as before (cache hit)
    assert msgs[2]["content"] == "first answer"
    assert "This session was resumed" in msgs[3]["content"] and msgs[3]["content"].endswith("second question")


def test_resume_warms_up_history(server, project):
    server.add(Reply(content="first answer"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI())
        await rt.submit("first question")
        await rt.close()
        rt2 = await make_runtime(project, server.base_url, RecordingUI(), start=False, provider__warmup=True)
        await rt2.start(resume=rt.session.path)
        await asyncio.wait_for(rt2._warmup_task, 10)
        await rt2.close()

    run(go())
    w = server.warmups[0]["messages"]
    assert [m["role"] for m in w] == ["system", "user", "assistant", "user"] and w[-1]["content"] == "ok"
    assert w[:2] == server.requests[0]["messages"]


def test_warmup_request(server, project):
    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), provider__warmup=True)
        await asyncio.wait_for(rt._warmup_task, 10)
        await rt.close()
        return rt

    rt = run(go())
    assert rt.warmup_state == "done"
    w = server.warmups[0]
    assert w["max_tokens"] == 1 and "id_slot" not in w
    assert w["messages"][1:] == [{"role": "user", "content": "ok"}] and w["messages"][0]["role"] == "system"


def test_first_request_waits_for_warmup(server, project):
    server.warmup_delay = 0.1
    server.add(Reply(content="hi"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI(), provider__warmup=True)
        await rt.submit("hello")            # sent while the warm-up is still running
        await rt.close()

    run(go())
    ends = [t for kind, edge, t in server.events if kind == "warmup" and edge == "end"]
    starts = [t for kind, edge, t in server.events if kind == "chat" and edge == "start"]
    assert ends and starts and starts[0] >= ends[0]
    assert server.requests[0]["messages"][0] == server.warmups[0]["messages"][0]


def test_clear_keeps_prefix_and_adds_fresh_environment(server, project):
    server.add(Reply(content="one"), Reply(content="two"))

    async def go():
        rt = await make_runtime(project, server.base_url, RecordingUI())
        await rt.submit("first")
        rt.new_conversation()
        await rt.submit("second")
        await rt.close()

    run(go())
    a, b = server.requests
    assert a["messages"][0] == b["messages"][0] and len(b["messages"]) == 2
    assert "Environment at the start" in b["messages"][1]["content"]


def test_medium_effort_maps_to_high_for_k3_template(project):
    from mock_server import MockServer
    s = MockServer(template="{%- elif reasoning_effort in ['low', 'high', 'max'] -%}")
    base = s.start()
    s.add(Reply(content="ok"))

    async def go():
        rt = await make_runtime(project, base, RecordingUI(), provider__reasoning_effort="medium")
        await rt.submit("hi")
        await rt.close()

    try:
        run(go())
    finally:
        s.stop()
    assert s.requests[0]["reasoning_effort"] == "high"
