"""Convert the stored conversation into request messages.

History is always stored in OpenAI chat format with structured tool calls.
This module renders it for the active tool mode. The conversion is
deterministic so the server's prompt (KV) cache keeps matching the prefix.
"""
from __future__ import annotations

import json

from rigcode.llm.textmode import render_tool_call_text, render_tool_result_text
from rigcode.llm.types import ToolCall


def _last_user_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and not messages[i].get("_tool_results"):
            return i
    return -1


def to_wire(messages: list[dict], *, tool_mode: str = "native", keep_reasoning: str = "all") -> list[dict]:
    last_user = _last_user_index(messages)
    out: list[dict] = []
    pending_text_results: list[str] = []

    def flush_text_results() -> None:
        if pending_text_results:
            out.append({"role": "user", "content": "\n\n".join(pending_text_results)})
            pending_text_results.clear()

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "tool":
            if tool_mode == "text":
                pending_text_results.append(
                    render_tool_result_text(msg.get("name", "tool"), msg.get("tool_call_id", ""), _text(msg.get("content")))
                )
            else:
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": msg.get("content") or ""})
            continue
        flush_text_results()
        if role == "assistant":
            wire: dict = {"role": "assistant", "content": msg.get("content") or ""}
            reasoning = msg.get("reasoning_content")
            if reasoning and (keep_reasoning == "all" or (keep_reasoning == "turn" and i > last_user)):
                wire["reasoning_content"] = reasoning
            calls = msg.get("tool_calls") or []
            if calls:
                if tool_mode == "text":
                    rendered = []
                    for c in calls:
                        fn = c.get("function", {})
                        raw = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            args = None
                        rendered.append(render_tool_call_text(ToolCall(id=c.get("id", ""), name=fn.get("name", ""),
                                                                       arguments=raw, args=args)))
                    wire["content"] = (wire["content"] + "\n\n" if wire["content"] else "") + "\n".join(rendered)
                else:
                    wire["tool_calls"] = calls
            out.append(wire)
        else:
            out.append({"role": role, "content": msg.get("content") if msg.get("content") is not None else ""})
    flush_text_results()
    return out


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")
