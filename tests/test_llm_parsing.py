import json

from rigcode.llm.client import classify_error
from rigcode.llm.stream import ChunkAssembler, SSEDecoder
from rigcode.llm.textmode import has_text_tool_markup, parse_text_tool_calls
from rigcode.llm.wire import to_wire


def chunk(delta=None, finish=None, **extra):
    c = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    c.update(extra)
    return c


def test_sse_decoder():
    d = SSEDecoder()
    assert d.feed_line(": keep-alive") is None
    assert d.feed_line('data: {"a": 1}') is None
    assert d.feed_line("") == (None, '{"a": 1}')
    d.feed_line("event: ping")
    d.feed_line("data: x")
    d.feed_line("data: y")
    assert d.feed_line("") == ("ping", "x\ny")
    d.feed_line("data: tail")
    assert d.flush() == (None, "tail")


def test_assembler_reasoning_content_tools_and_usage():
    asm = ChunkAssembler()
    kinds = []
    for c in [
        {"choices": [], "prompt_progress": {"total": 10, "cache": 5, "processed": 10, "time_ms": 3}},
        chunk({"reasoning_content": "think "}), chunk({"reasoning_content": "more"}),
        chunk({"content": "Hello"}),
        chunk({"tool_calls": [{"index": 0, "id": "abc", "type": "function", "function": {"name": "Read", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"file_path"'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ': "a.py"}'}}]}),
        chunk({"tool_calls": [{"index": 1, "id": "def", "function": {"name": "Glob", "arguments": '{"pattern": "*"}'}}]}),
        chunk({}, "tool_calls"),
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7,
                                  "prompt_tokens_details": {"cached_tokens": 60}},
         "timings": {"cache_n": 60, "prompt_n": 40, "prompt_ms": 400.0, "predicted_n": 7, "predicted_ms": 1400.0}},
    ]:
        kinds += [d.kind for d in asm.ingest(c)]
    turn, _ = asm.finish()
    assert kinds[0] == "progress"
    assert "reasoning" in kinds and "content" in kinds and kinds.count("tool_start") == 2
    assert turn.reasoning == "think more"
    assert turn.content == "Hello"
    assert [(t.id, t.name, json.loads(t.arguments)) for t in turn.tool_calls] == [
        ("abc", "Read", {"file_path": "a.py"}), ("def", "Glob", {"pattern": "*"})]
    assert turn.finish_reason == "tool_calls"
    assert turn.usage.cached_tokens == 60 and turn.usage.prompt_tokens == 100
    assert round(turn.usage.gen_tps) == 5
    assert round(turn.usage.prompt_tps) == 100


def test_assembler_think_tags_in_content():
    asm = ChunkAssembler()
    for text in ["<thi", "nk>pondering", "</think>\n\nAnswer"]:
        asm.ingest(chunk({"content": text}))
    turn, _ = asm.finish()
    assert turn.reasoning.strip() == "pondering"
    assert turn.content == "Answer"


def test_assembler_missing_index():
    asm = ChunkAssembler()
    asm.ingest(chunk({"tool_calls": [{"id": "1", "function": {"name": "Read", "arguments": '{"a":'}}]}))
    asm.ingest(chunk({"tool_calls": [{"function": {"arguments": "1}"}}]}))
    asm.ingest(chunk({"tool_calls": [{"id": "2", "function": {"name": "Glob", "arguments": "{}"}}]}))
    turn, _ = asm.finish()
    assert [(t.name, t.arguments) for t in turn.tool_calls] == [("Read", '{"a":1}'), ("Glob", "{}")]


def test_text_tool_call_formats():
    content, calls = parse_text_tool_calls(
        'Let me look.\n<tool_call>\n{"name": "Read", "arguments": {"file_path": "a.py"}}\n</tool_call>')
    assert content == "Let me look."
    assert calls[0].name == "Read" and calls[0].args == {"file_path": "a.py"}

    _, calls = parse_text_tool_calls("<function=Bash><parameter=command>ls -la</parameter></function>")
    assert calls[0].name == "Bash" and calls[0].args == {"command": "ls -la"}

    kimi = ('<|tool_calls_section_begin|><|tool_call_begin|>functions.Grep:0<|tool_call_argument_begin|>'
            '{"pattern": "TODO"}<|tool_call_end|><|tool_calls_section_end|>')
    _, calls = parse_text_tool_calls(kimi)
    assert calls[0].name == "Grep" and calls[0].args == {"pattern": "TODO"}
    assert has_text_tool_markup(kimi)
    assert not has_text_tool_markup("plain text")


def test_wire_native_and_text_modes():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi", "_meta": True},
        {"role": "assistant", "content": "", "reasoning_content": "r1",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "a"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "Read", "content": "data"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "ok", "reasoning_content": "r2"},
    ]
    native = to_wire(history)
    assert native[1] == {"role": "user", "content": "hi"}
    assert native[2]["reasoning_content"] == "r1" and native[2]["tool_calls"][0]["id"] == "c1"
    assert native[3] == {"role": "tool", "tool_call_id": "c1", "content": "data"}
    turn_only = to_wire(history, keep_reasoning="turn")
    assert "reasoning_content" not in turn_only[2] and turn_only[5]["reasoning_content"] == "r2"
    text = to_wire(history, tool_mode="text")
    assert "<tool_call>" in text[2]["content"] and "tool_calls" not in text[2]
    assert text[3]["role"] == "user" and "<tool_result" in text[3]["content"]


def test_classify_error():
    assert classify_error(400, '{"error": {"message": "the request exceeds the available context size"}}').kind == "context_overflow"
    assert classify_error(401, "nope").kind == "auth"
    assert classify_error(503, "busy").retryable
    assert classify_error(500, '{"error": {"message": "tools param requires --jinja flag"}}').kind == "tools_unsupported"
    assert classify_error(404, "not found").kind == "not_found"


def test_template_effort_levels():
    from rigcode.llm.probe import template_effort_levels
    k3 = "{%- elif reasoning_effort in ['low', 'high', 'max'] -%}{%- set rens.effort = reasoning_effort -%}"
    assert template_effort_levels(k3) == ("low", "high", "max")
    hosted = ("{%- if reasoning_effort in ['minimal', 'low'] -%}{%- elif reasoning_effort in ['medium', 'high'] -%}"
              "{%- elif reasoning_effort in ['xhigh', 'max'] -%}" + k3)
    assert template_effort_levels(hosted) == ("minimal", "low", "medium", "high", "xhigh", "max")
    assert template_effort_levels("{{ messages }}") is None and template_effort_levels(None) is None
