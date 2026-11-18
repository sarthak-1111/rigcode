"""SSE parsing and assembly of streamed chat-completion chunks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from rigcode.llm.types import AssistantTurn, Delta, ToolCall, Usage

THINK_TAGS = (("<think>", "</think>"), ("◁think▷", "◁/think▷"))


class SSEDecoder:
    """Incremental Server-Sent-Events decoder (line based)."""

    def __init__(self) -> None:
        self._data: list[str] = []
        self._event: str | None = None

    def feed_line(self, line: str) -> tuple[str | None, str] | None:
        """Feed one line (without newline). Returns (event, data) when a frame completes."""
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            if not self._data:
                self._event = None
                return None
            data = "\n".join(self._data)
            event = self._event
            self._data, self._event = [], None
            return event, data
        if line.startswith(":"):
            return None
        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            self._data.append(value)
        elif field_name == "event":
            self._event = value
        return None

    def flush(self) -> tuple[str | None, str] | None:
        if self._data:
            data = "\n".join(self._data)
            event = self._event
            self._data, self._event = [], None
            return event, data
        return None


class ThinkSplitter:
    """Routes <think>...</think> found at the start of `content` into reasoning.

    Some servers (or reasoning-format=none) leave the model's thinking inline in
    the content. We only treat a think block at the very start of the message as
    reasoning, so legitimate mentions of the tag later in an answer are untouched.
    """

    def __init__(self) -> None:
        self.state = "start"   # start | think | content
        self.buf = ""
        self.close_tag = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += text
        while self.buf:
            if self.state == "start":
                stripped = self.buf.lstrip()
                matched = False
                for open_tag, close_tag in THINK_TAGS:
                    if stripped.startswith(open_tag):
                        self.buf = stripped[len(open_tag):]
                        self.state, self.close_tag, matched = "think", close_tag, True
                        break
                    if open_tag.startswith(stripped):  # could still become a tag
                        return out
                if not matched:
                    self.state = "content"
            elif self.state == "think":
                idx = self.buf.find(self.close_tag)
                if idx >= 0:
                    if idx:
                        out.append(("reasoning", self.buf[:idx]))
                    self.buf = self.buf[idx + len(self.close_tag):].lstrip("\n")
                    self.state = "content"
                else:
                    # Hold back a possible partial closing tag.
                    keep = 0
                    for k in range(1, len(self.close_tag)):
                        if self.buf.endswith(self.close_tag[:k]):
                            keep = k
                    emit = self.buf[: len(self.buf) - keep]
                    if emit:
                        out.append(("reasoning", emit))
                    self.buf = self.buf[len(self.buf) - keep:]
                    return out
            else:
                out.append(("content", self.buf))
                self.buf = ""
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.buf:
            return []
        kind = "reasoning" if self.state == "think" else "content"
        text, self.buf = self.buf, ""
        return [(kind, text)]


@dataclass
class _ToolAcc:
    id: str | None = None
    name: str = ""
    args: list[str] = field(default_factory=list)


class ChunkAssembler:
    """Turns OpenAI-style chat.completion.chunk dicts into Deltas + a final turn."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turn = AssistantTurn()
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tools: dict[int, _ToolAcc] = {}
        self._splitter = ThinkSplitter()
        self.saw_reasoning_field = False

    @property
    def partial_text(self) -> str:
        return "".join(self._content)

    def ingest(self, chunk: dict) -> list[Delta]:
        deltas: list[Delta] = []
        if chunk.get("usage"):
            self._apply_usage(chunk["usage"])
        if chunk.get("timings"):
            self._apply_timings(chunk["timings"])
        if "prompt_progress" in chunk and isinstance(chunk["prompt_progress"], dict):
            deltas.append(Delta("progress", data=chunk["prompt_progress"]))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                self.saw_reasoning_field = True
                self._reasoning.append(reasoning)
                deltas.append(Delta("reasoning", reasoning))
            content = delta.get("content")
            if isinstance(content, str) and content:
                for kind, text in self._splitter.feed(content):
                    (self._reasoning if kind == "reasoning" else self._content).append(text)
                    deltas.append(Delta(kind, text))
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index")
                if idx is None:
                    # Servers that omit `index`: a new id starts a new call,
                    # otherwise the chunk continues the latest call.
                    tc_id = tc.get("id")
                    if tc_id and all(t.id != tc_id for t in self._tools.values()):
                        idx = len(self._tools)
                    else:
                        idx = max(self._tools) if self._tools else 0
                acc = self._tools.get(idx)
                if acc is None:
                    acc = self._tools[idx] = _ToolAcc()
                if tc.get("id") and not acc.id:
                    acc.id = tc["id"]
                fn = tc.get("function") or {}
                name = fn.get("name")
                if name and name != acc.name:
                    # llama.cpp may re-send a partially parsed name longer; overwrite, don't append.
                    first = not acc.name
                    acc.name = name
                    if first:
                        deltas.append(Delta("tool_start", index=idx, name=name))
                args = fn.get("arguments")
                if args:
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    acc.args.append(args)
                    deltas.append(Delta("tool_args", args, index=idx))
            if choice.get("finish_reason"):
                self.turn.finish_reason = choice["finish_reason"]
        return deltas

    def _apply_usage(self, u: dict) -> None:
        usage = self.turn.usage
        usage.prompt_tokens = int(u.get("prompt_tokens") or usage.prompt_tokens or 0)
        usage.completion_tokens = int(u.get("completion_tokens") or usage.completion_tokens or 0)
        details = u.get("prompt_tokens_details") or {}
        if details.get("cached_tokens"):
            usage.cached_tokens = int(details["cached_tokens"])

    def _apply_timings(self, t: dict) -> None:
        usage = self.turn.usage
        if t.get("cache_n") is not None:
            usage.cached_tokens = int(t["cache_n"])
        if t.get("prompt_n") is not None and not usage.prompt_tokens:
            usage.prompt_tokens = int(t["prompt_n"]) + int(t.get("cache_n") or 0)
        if t.get("prompt_ms") is not None:
            usage.prompt_ms = float(t["prompt_ms"])
        if t.get("predicted_n") is not None and not usage.completion_tokens:
            usage.completion_tokens = int(t["predicted_n"])
        if t.get("predicted_ms") is not None:
            usage.gen_ms = float(t["predicted_ms"])

    def finish(self) -> tuple[AssistantTurn, list[Delta]]:
        deltas: list[Delta] = []
        for kind, text in self._splitter.flush():
            (self._reasoning if kind == "reasoning" else self._content).append(text)
            deltas.append(Delta(kind, text))
        content = "".join(self._content)
        reasoning = "".join(self._reasoning)
        # Unopened </think>: the server's template opened the think block itself.
        if not reasoning and "</think>" in content:
            head, _, tail = content.partition("</think>")
            reasoning, content = head.strip(), tail.lstrip("\n")
        self.turn.content = content
        self.turn.reasoning = reasoning
        calls = []
        for n, idx in enumerate(sorted(self._tools)):
            acc = self._tools[idx]
            if not acc.name:
                continue
            calls.append(ToolCall(id=acc.id or f"call_{n}", name=acc.name, arguments="".join(acc.args)))
        self.turn.tool_calls = calls
        return self.turn, deltas


def parse_usage_only(u: dict | None) -> Usage:
    usage = Usage()
    if u:
        usage.prompt_tokens = int(u.get("prompt_tokens") or 0)
        usage.completion_tokens = int(u.get("completion_tokens") or 0)
    return usage
