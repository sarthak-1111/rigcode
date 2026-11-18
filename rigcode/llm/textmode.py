"""Text-based tool calling, used when the server lacks native tool support, plus
lenient recovery of tool calls that leak into `content` in native mode.

Recognized formats inside assistant text:
  <tool_call>{"name": "Read", "arguments": {...}}</tool_call>          (Hermes / rigcode)
  <function=Read><parameter=file_path>a.py</parameter></function>       (Qwen3-Coder XML)
  <|tool_call_begin|>functions.Read:0<|tool_call_argument_begin|>{...}<|tool_call_end|>  (Kimi native)
"""
from __future__ import annotations

import json
import re

from rigcode.llm.types import ToolCall
from rigcode.utils.jsonrepair import loads_lenient

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
_QWEN_FN_RE = re.compile(r"<function=([\w.\-]+)>\s*(.*?)\s*(?:</function>|\Z)", re.S)
_QWEN_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.S)
_KIMI_SECTION_RE = re.compile(r"<\|tool_calls_section_begin\|>(.*?)(?:<\|tool_calls_section_end\|>|\Z)", re.S)
_KIMI_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?:functions\.)?([\w.\-]+?)(?::(\d+))?\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*(?:<\|tool_call_end\|>|\Z)",
    re.S,
)


def render_tools_prompt(tool_schemas: list[dict]) -> str:
    """System-prompt section that teaches the text tool protocol."""
    lines = [
        "# Tool calling protocol",
        "You call tools by writing one or more blocks exactly like this in your reply:",
        "<tool_call>",
        '{"name": "TOOL_NAME", "arguments": {"param": "value"}}',
        "</tool_call>",
        "Rules: the JSON must be valid (escape newlines in strings as \\n). You may emit several "
        "<tool_call> blocks in one reply to run independent calls together. After your tool calls, "
        "stop writing and wait: results arrive in the next message inside <tool_result> tags. "
        "Never write a <tool_result> yourself or guess a result.",
        "",
        "Available tools (JSON Schema):",
        "<tools>",
    ]
    for schema in tool_schemas:
        fn = schema.get("function", schema)
        lines.append(json.dumps({"name": fn["name"], "description": fn.get("description", ""),
                                 "parameters": fn.get("parameters", {})}, ensure_ascii=False))
    lines.append("</tools>")
    return "\n".join(lines)


def _call_from_obj(obj: object, n: int) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        obj = name
        name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        parsed, err = loads_lenient(args)
        raw = args
        args = parsed if isinstance(parsed, dict) else None
        call = ToolCall(id=f"call_t{n}", name=name, arguments=raw, args=args)
        if err or args is None:
            call.error = f"Could not parse arguments as JSON: {err or 'not an object'}"
        return call
    if not isinstance(args, dict):
        args = {}
    return ToolCall(id=f"call_t{n}", name=name, arguments=json.dumps(args, ensure_ascii=False), args=args)


def _coerce_param(value: str) -> object:
    v = value.strip()
    if v and v[0] in "[{" or v in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", v or "x"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    return value


def parse_text_tool_calls(content: str, start_index: int = 0) -> tuple[str, list[ToolCall]]:
    """Extract tool calls from assistant text. Returns (remaining_text, calls)."""
    calls: list[ToolCall] = []
    text = content

    def take(pattern: re.Pattern, handler) -> None:
        nonlocal text
        pieces, last = [], 0
        for m in pattern.finditer(text):
            produced = handler(m)
            if produced:
                calls.extend(produced)
                pieces.append(text[last:m.start()])
                last = m.end()
        if pieces or last:
            pieces.append(text[last:])
            text = "".join(pieces)

    def kimi_section(m: re.Match) -> list[ToolCall]:
        out = []
        for cm in _KIMI_CALL_RE.finditer(m.group(1)):
            name, raw = cm.group(1), cm.group(3)
            parsed, err = loads_lenient(raw)
            call = ToolCall(id=f"call_k{start_index + len(calls) + len(out)}", name=name, arguments=raw,
                            args=parsed if isinstance(parsed, dict) else None)
            if call.args is None:
                call.error = f"Could not parse arguments as JSON: {err or 'not an object'}"
            out.append(call)
        return out

    def hermes(m: re.Match) -> list[ToolCall]:
        body = m.group(1).strip()
        if not body:
            return []
        fn = _QWEN_FN_RE.search(body)
        if fn and not body.startswith("{"):
            return qwen(fn)
        parsed, err = loads_lenient(body)
        if isinstance(parsed, list):
            out = [c for i, o in enumerate(parsed) if (c := _call_from_obj(o, start_index + len(calls) + i))]
            return out
        call = _call_from_obj(parsed, start_index + len(calls))
        if call:
            return [call]
        # Unparseable block: surface it as a broken call so the model can retry.
        name_m = re.search(r'"name"\s*:\s*"([\w.\-]+)"', body)
        if name_m:
            return [ToolCall(id=f"call_t{start_index + len(calls)}", name=name_m.group(1), arguments=body,
                             args=None, error=f"Could not parse the tool call JSON: {err}")]
        return []

    def qwen(m: re.Match) -> list[ToolCall]:
        name, body = m.group(1), m.group(2)
        args = {pm.group(1): _coerce_param(pm.group(2)) for pm in _QWEN_PARAM_RE.finditer(body)}
        return [ToolCall(id=f"call_t{start_index + len(calls)}", name=name,
                         arguments=json.dumps(args, ensure_ascii=False), args=args)]

    take(_KIMI_SECTION_RE, kimi_section)
    if "<tool_call>" in text:
        take(_TOOL_CALL_RE, hermes)
    if "<function=" in text:
        take(_QWEN_FN_RE, qwen)
    # Strip stray protocol tokens.
    text = re.sub(r"<\|tool_calls?_section_(?:begin|end)\|>|</?tool_call>", "", text)
    return text.strip(), calls


def has_text_tool_markup(content: str) -> bool:
    return any(tag in content for tag in ("<tool_call>", "<|tool_call_begin|>", "<function="))


def render_tool_call_text(call: ToolCall) -> str:
    args = call.args if call.args is not None else call.arguments
    return "<tool_call>\n" + json.dumps({"name": call.name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def render_tool_result_text(name: str, call_id: str, content: str) -> str:
    return f'<tool_result name="{name}" id="{call_id}">\n{content}\n</tool_result>'
