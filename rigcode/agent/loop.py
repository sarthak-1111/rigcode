"""The agent loop: call the model, run the requested tools, repeat until done."""
from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field

from rigcode.agent.prompts import compact_instruction, post_compact_message, reminder
from rigcode.agent.state import TodoState
from rigcode.hooks import HookOutcome
from rigcode.llm.stream import ChunkAssembler
from rigcode.llm.textmode import has_text_tool_markup, parse_text_tool_calls
from rigcode.llm.types import AssistantTurn, LLMError, ToolCall, Usage
from rigcode.llm.wire import to_wire
from rigcode.tools.base import Tool, ToolContext, ToolResult, validate_args
from rigcode.utils.jsonrepair import loads_lenient
from rigcode.utils.paths import display_path
from rigcode.utils.text import est_tokens, one_line

REPEAT_WARN_AT = (3, 5, 8)
REPEAT_ABORT_AT = 12
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.S)


@dataclass
class TurnStats:
    steps: int = 0
    tool_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    elapsed: float = 0.0
    stopped_reason: str = "done"   # done | interrupted | error | max_steps | denied | hook | loop


@dataclass
class PreparedCall:
    call: ToolCall
    tool: Tool | None
    args: dict
    error: str | None
    label: str
    repeat_note: str | None = None
    abort: bool = False


class Agent:
    def __init__(self, rt, *, name: str = "main", depth: int = 0, agent_def=None, slot: int | None = None,
                 effort: str | None = None, max_steps: int | None = None, messages: list[dict] | None = None,
                 client=None):
        self.rt = rt
        self.name = name
        self.depth = depth
        self.agent_def = agent_def
        self.slot = slot
        self.effort = effort
        self.client = client
        self.max_steps = max_steps or int(rt.settings.get("max_steps_per_turn", 150))
        self.messages: list[dict] = messages if messages is not None else [dict(m) for m in rt.prefix_messages]
        self.todos = TodoState()
        self.last_total_tokens = 0
        self._counted_upto = 0
        self.pending_reminders: list[str] = []
        self.env_note: str | None = None  # date/git snapshot for the next user message (conversation start)
        self._interrupted = False
        self._repeat_counts: dict[str, int] = {}
        self._stop_hook_active = False
        self.persist = depth == 0
        self.parent_call_id = ""          # for subagents: the Task call that started them
        self.forked = False               # for subagents: started from a copy of the main conversation
        # Messages before this index were inherited (prompt prefix, or the conversation a fork started from).
        self.own_start = self.n_prefix() if messages is None or depth == 0 else len(self.messages)

    # ------------------------------------------------------------------ helpers
    @property
    def is_main(self) -> bool:
        return self.depth == 0

    @property
    def llm(self):
        return self.client or self.rt.client

    def current_effort(self) -> str | None:
        return self.effort if self.effort is not None else self.rt.effort

    def _append(self, msg: dict) -> None:
        self.messages.append(msg)
        if self.persist:
            self.rt.session.message(msg)

    def add_reminder(self, text: str) -> None:
        self.pending_reminders.append(text)

    def n_prefix(self) -> int:
        """Number of leading messages that form the shared prompt prefix (the system message)."""
        n = 0
        for m in self.messages:
            if m.get("role") != "system":
                break
            n += 1
        return n

    def fork_messages(self, call_id: str) -> list[dict] | None:
        """This conversation up to and including the assistant message that made `call_id`.
        A forked subagent starts from it, so its prompt begins with this agent's cached state."""
        if not call_id:
            return None
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages[i]
            if m.get("role") == "assistant" and any(tc.get("id") == call_id for tc in m.get("tool_calls") or []):
                return [dict(x) for x in self.messages[: i + 1]]
        return None

    def context_tokens(self) -> int:
        """Best estimate of the current context size (what the next request will send)."""
        if self.last_total_tokens and self._counted_upto <= len(self.messages):
            extra = sum(est_tokens(_msg_text(m)) for m in self.messages[self._counted_upto:])
            return self.last_total_tokens + extra
        tools = est_tokens(json.dumps(self.rt.tool_schemas)) if self.llm.tool_mode == "native" else 0
        return tools + sum(est_tokens(_msg_text(m)) for m in self.messages)

    def on_tool_output(self, call_id: str, text: str) -> None:
        self.rt.ui.on_tool_output(self, call_id, text)

    # ------------------------------------------------------------------ turn
    async def run_turn(self, text: str, *, tool_call_id: str | None = None) -> TurnStats:
        """Run one turn. `tool_call_id` starts a forked subagent: `text` (its brief) becomes the
        result of that Task call instead of a user message."""
        rt = self.rt
        stats = TurnStats()
        started = time.monotonic()
        self._repeat_counts = {}
        self._stop_hook_active = False
        try:
            if self.is_main:
                rt.checkpoints.begin_turn(one_line(text, 60))
                if rt.hooks.has("UserPromptSubmit"):
                    hook = await rt.hooks.run("UserPromptSubmit", {"prompt": text}, rt.cwd)
                    for n in hook.notices:
                        rt.ui.on_notice("warn", n)
                    if hook.block:
                        rt.ui.on_notice("error", f"Prompt blocked by a hook: {hook.reason or '(no reason)'}")
                        stats.stopped_reason = "hook"
                        return stats
                    self.pending_reminders.extend(hook.additional_context)
            if tool_call_id:
                self._append({"role": "tool", "tool_call_id": tool_call_id, "name": "Task", "content": text})
            else:
                self._append({"role": "user", "content": self._compose_user_content(text)})
            length_retries = 0
            while True:
                if stats.steps >= self.max_steps:
                    rt.ui.on_notice("warn", f"{self.name}: stopped after {self.max_steps} steps.")
                    stats.stopped_reason = "max_steps"
                    break
                if not self.is_main and stats.steps and self.context_tokens() > 0.9 * rt.context_limit():
                    rt.ui.on_notice("warn", f"{self.name}: stopped - its context is nearly full.")
                    stats.stopped_reason = "context"
                    break
                stats.steps += 1
                if self.is_main:
                    if stats.steps > 1:
                        queued = rt.ui.take_queued_input()   # typed by the user while we were working
                        if queued:
                            self._append({"role": "user", "content": self._compose_user_content(queued)})
                    await self.maybe_compact()
                turn = await self._call_model(stats)
                if turn is None:
                    stats.stopped_reason = "error"
                    break
                prepared = self._prepare_calls(turn)
                self._record_assistant(turn)
                if not prepared:
                    if turn.finish_reason == "length" and length_retries < 2:
                        length_retries += 1
                        self._append({"role": "user", "_meta": True, "content": reminder(
                            "Your reply hit the output token limit and was cut off. Continue from where you "
                            "stopped. Be more concise, and split large file writes or edits into smaller pieces.")})
                        continue
                    if self.is_main and rt.hooks.has("Stop"):
                        out = await rt.hooks.run("Stop", {"stop_hook_active": self._stop_hook_active}, rt.cwd)
                        if out.block and out.reason and not out.halt:
                            self._stop_hook_active = True
                            self._append({"role": "user", "_meta": True,
                                          "content": reminder(f"A Stop hook asked you to continue: {out.reason}")})
                            continue
                    break
                stats.tool_calls += len(prepared)
                if any(p.abort for p in prepared):
                    for p in prepared:
                        self._append(self._tool_message(p, ToolResult.error(
                            "Stopped: this exact call has been repeated too many times.")))
                    rt.ui.on_notice("error", f"{self.name}: stopped - the model kept repeating the same tool call.")
                    stats.stopped_reason = "loop"
                    break
                results = await self._execute_calls(prepared)
                stop = False
                for p, res in zip(prepared, results):
                    self._append(self._tool_message(p, res))
                    stop = stop or res.stop_turn
                if stop:
                    stats.stopped_reason = "denied"
                    break
        except asyncio.CancelledError:
            stats.stopped_reason = "interrupted"
            self._interrupted = True
            self._close_dangling_calls("[Interrupted by the user before this tool call finished.]")
            if self.depth > 0:
                raise
        except Exception as e:  # noqa: BLE001 - never crash the REPL on a bug
            stats.stopped_reason = "error"
            self._close_dangling_calls(f"[Tool call aborted by an internal error: {e}]")
            rt.ui.on_notice("error", f"Internal error: {type(e).__name__}: {e}")
            if rt.debug:
                rt.ui.on_notice("error", traceback.format_exc())
        finally:
            stats.elapsed = time.monotonic() - started
        return stats

    def _compose_user_content(self, text: str) -> str:
        parts = []
        if self.env_note:
            parts.append(reminder(self.env_note))
            self.env_note = None
        if self._interrupted:
            parts.append(reminder("The user interrupted your previous turn."))
            self._interrupted = False
        for r in self.pending_reminders:
            parts.append(r if r.lstrip().startswith("<system-reminder>") else reminder(r))
        self.pending_reminders = []
        parts.append(text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ model call
    async def _call_model(self, stats: TurnStats, *, tools_allowed: bool = True) -> AssistantTurn | None:
        rt, ui, client = self.rt, self.rt.ui, self.llm
        keep = rt.settings.get("provider.keep_reasoning", "all")
        allow_compact_retry = self.is_main
        while True:
            wire = to_wire(self.messages, tool_mode=client.tool_mode, keep_reasoning=keep)
            tools = rt.tool_schemas if client.tool_mode == "native" else None
            extra = None if tools_allowed or not tools else {"tool_choice": "none"}
            asm = ChunkAssembler()
            ui.on_model_start(self, {"est_prompt_tokens": self.context_tokens(), "step": stats.steps})
            try:
                if self.is_main:
                    await rt.wait_warmup()
                turn = await client.stream_chat(
                    wire, tools, on_delta=lambda d: ui.on_delta(self, d), assembler=asm,
                    effort=self.current_effort(), max_tokens=rt.settings.get("provider.max_output_tokens"),
                    slot=self.slot, extra=extra)
            except asyncio.CancelledError:
                partial, _ = asm.finish()
                partial.interrupted = True
                ui.on_model_end(self, partial)
                if partial.content.strip():
                    msg = {"role": "assistant", "content": partial.content}
                    if partial.reasoning:
                        msg["reasoning_content"] = partial.reasoning
                    self._append(msg)
                raise
            except LLMError as e:
                ui.on_model_end(self, None)
                if e.kind == "context_overflow" and allow_compact_retry and len(self.messages) > self.n_prefix() + 1:
                    allow_compact_retry = False
                    ui.on_notice("warn", "The prompt exceeded the context window - compacting and retrying.")
                    if await self.compact(reason="overflow"):
                        continue
                if e.kind == "tools_unsupported" and client.tool_mode == "native" and \
                        rt.settings.get("provider.tool_mode", "auto") == "auto":
                    ui.on_notice("warn", "The server rejected native tool calling; switching to text-based tool calls.")
                    rt.switch_to_text_tools()
                    self._replace_prefix()
                    continue
                ui.on_notice("error", f"Model request failed: {e}" + _hint_for(e, rt))
                return None
            ui.on_model_end(self, turn)
            if client.tool_mode == "text" or (not turn.tool_calls and has_text_tool_markup(turn.content)):
                content, calls = parse_text_tool_calls(turn.content)
                if calls:
                    turn.content, turn.tool_calls = content, calls
            u = turn.usage
            if not u.prompt_tokens:
                u.prompt_tokens = self.context_tokens()
            if not u.completion_tokens:
                u.completion_tokens = est_tokens(turn.content + turn.reasoning)
            self.last_total_tokens = u.prompt_tokens + u.completion_tokens
            self._counted_upto = len(self.messages) + 1   # +1: the assistant message about to be recorded
            rt.add_usage(u, self)
            stats.usage.add(u)
            return turn

    def _replace_prefix(self) -> None:
        n = self.n_prefix()
        self.messages = [dict(m) for m in self.rt.prefix_messages] + self.messages[n:]
        self.own_start += len(self.rt.prefix_messages) - n

    def _record_assistant(self, turn: AssistantTurn) -> None:
        for _i, tc in enumerate(turn.tool_calls):
            if not tc.id:
                tc.id = f"call_{uuid.uuid4().hex[:12]}"
        self._append(turn.to_message())

    # ------------------------------------------------------------------ tool calls
    def _prepare_calls(self, turn: AssistantTurn) -> list[PreparedCall]:
        out = []
        for tc in turn.tool_calls:
            if not tc.id:
                tc.id = f"call_{uuid.uuid4().hex[:12]}"
            tool = self.rt.tools.get(tc.name)
            if tool is None:
                names = ", ".join(self.rt.tools.names())
                out.append(PreparedCall(tc, None, {}, f"Unknown tool '{tc.name}'. Available tools: {names}.", tc.name))
                continue
            args = tc.args
            error = tc.error
            if args is None and error is None:
                parsed, err = loads_lenient(tc.arguments)
                if err is not None or not isinstance(parsed, dict):
                    error = (f"The arguments for {tool.name} are not a valid JSON object ({err or 'not an object'}). "
                             "Re-send the call with valid JSON.")
                else:
                    args = parsed
            args = dict(args or {})
            if error is None:
                tool.apply_aliases(args)
                args, verr = validate_args(tool, args)
                if verr:
                    expected = ", ".join(
                        f"{k}{'' if k in (tool.parameters.get('required') or []) else '?'}"
                        for k in (tool.parameters.get("properties") or {}))
                    error = (f"Invalid arguments for {tool.name}: {verr}. Parameters: {expected}. "
                             "Fix the arguments; do not re-send this call unchanged.")
            p = PreparedCall(tc, tool, args, error, tool.label(args) if error is None else tc.name)
            if error is None:
                sig = tool.name + json.dumps(args, sort_keys=True, ensure_ascii=False)[:4000]
                count = self._repeat_counts.get(sig, 0) + 1
                self._repeat_counts[sig] = count
                if count >= REPEAT_ABORT_AT:
                    p.abort = True
                elif count in REPEAT_WARN_AT:
                    p.repeat_note = (f"You have now made this exact {tool.name} call {count} times in this turn. "
                                     "Repeating it will not give a different result. Change your approach: "
                                     "re-read the relevant code, try a different command or search, or ask the user.")
            out.append(p)
        return out

    def _parallel_ok(self, p: PreparedCall) -> bool:
        if p.error or p.tool is None or not p.tool.is_read_only(p.args):
            return False
        if p.tool.name == "Task":
            return self.rt.parallel_subagents()
        if p.tool.kind in ("interact",):
            return False
        verdict, _ = self.rt.permissions.evaluate(p.tool, p.args, self.rt.cwd)
        return verdict == "allow"

    async def _execute_calls(self, prepared: list[PreparedCall]) -> list[ToolResult]:
        n = len(prepared)
        results: list[ToolResult | None] = [None] * n
        i = 0
        stopped = False
        while i < n and not stopped:
            batch = []
            while i < n and self._parallel_ok(prepared[i]):
                batch.append(i)
                i += 1
            if len(batch) == 1:
                results[batch[0]] = await self._execute_one(prepared[batch[0]])
            elif batch:
                outs = await asyncio.gather(*(self._execute_one(prepared[j]) for j in batch))
                for j, r in zip(batch, outs):
                    results[j] = r
            if any(results[j] and results[j].stop_turn for j in batch):
                break
            if i < n:
                res = await self._execute_one(prepared[i])
                results[i] = res
                i += 1
                stopped = res.stop_turn
        return [r if r is not None else ToolResult(
            content="[Skipped: the user stopped this turn before this tool call ran.]", is_error=True, summary="Skipped")
            for r in results]

    def _restriction(self, tool: Tool, args: dict) -> str | None:
        a = self.agent_def
        if self.depth > 0 and tool.name == "Task":
            return "Subagents cannot start other subagents. Do the work directly."
        if a is None:
            return None
        if a.tools is not None and tool.name not in a.tools and not (tool.name.startswith("mcp__") and "mcp" in a.tools):
            return (f"The {tool.name} tool is not available to the {a.name} subagent. "
                    f"Use only: {', '.join(a.tools)}.")
        if a.read_only and not tool.is_read_only(args):
            return (f"The {a.name} subagent is read-only and this {tool.name} call could modify state. "
                    "Use read-only operations only.")
        return None

    async def _execute_one(self, p: PreparedCall) -> ToolResult:
        rt, ui = self.rt, self.rt.ui
        if p.error or p.tool is None:
            res = ToolResult.error(p.error or "invalid call")
            ui.on_tool_end(self, p, res)
            return res
        tool, args = p.tool, p.args
        ui.on_tool_start(self, p, args)
        restriction = self._restriction(tool, args)
        if restriction:
            res = ToolResult.error(restriction, summary="Not allowed for this agent")
            ui.on_tool_end(self, p, res)
            return res
        hook = HookOutcome()
        if rt.hooks.has("PreToolUse"):
            hook = await rt.hooks.run("PreToolUse", {"tool_name": tool.name, "tool_input": args}, rt.cwd, tool.name)
            for n in hook.notices:
                ui.on_notice("warn", n)
            if hook.block:
                res = ToolResult.error(f"Blocked by a PreToolUse hook: {hook.reason or '(no reason given)'}",
                                       summary="Blocked by hook")
                res.stop_turn = hook.halt
                ui.on_tool_end(self, p, res)
                return res
            if hook.updated_input:
                args, verr = validate_args(tool, {**args, **hook.updated_input})
                if verr:
                    res = ToolResult.error(f"A hook rewrote the input into an invalid call: {verr}")
                    ui.on_tool_end(self, p, res)
                    return res
        verdict, reason = rt.permissions.evaluate(tool, args, rt.cwd)
        if verdict != "deny" and hook.permission in ("allow", "ask"):
            verdict = hook.permission
        if verdict == "deny":
            res = ToolResult.error(f"Permission denied: {reason}.", summary="Denied")
            ui.on_tool_end(self, p, res)
            return res
        ctx = ToolContext(rt, self, p.call.id)
        if verdict == "ask":
            preview = None
            if hasattr(tool, "preview"):
                pv = await tool.preview(args, ctx)
                if isinstance(pv, ToolResult):   # e.g. an edit that would fail anyway - don't bother the user
                    ui.on_tool_end(self, p, pv)
                    return pv
                preview = pv
            req = rt.permissions.build_request(tool, args, p.label, reason)
            req.preview = preview
            decision = await ui.request_permission(req, self)
            if decision is None:
                res = ToolResult.error(
                    "This action needs the user's approval, but nobody is available to approve it, so it was not "
                    "run. Continue with what you can do without it, and mention it in your summary.",
                    summary="Needs approval (non-interactive)")
                ui.on_tool_end(self, p, res)
                return res
            if decision.remember_rule:
                where = rt.permissions.remember(decision.remember_rule)
                ui.on_notice("info", f"Allowed {decision.remember_rule}" + (f" (saved to {where})" if where else ""))
            if decision.session_mode:
                rt.set_mode(decision.session_mode)
            if not decision.allow:
                msg = "The user rejected this tool call."
                msg += f" Their instructions: {decision.feedback}" if decision.feedback else \
                    " Do not retry it. If the way forward is unclear, ask the user."
                res = ToolResult(content=msg, is_error=True, summary="Rejected", stop_turn=decision.stop)
                ui.on_tool_end(self, p, res)
                return res
        try:
            res = await tool.run(args, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            res = ToolResult.error(f"{tool.name} failed with an internal error: {type(e).__name__}: {e}")
            if rt.debug:
                ui.on_notice("error", traceback.format_exc())
        if p.repeat_note:
            res.content += "\n\n" + reminder(p.repeat_note)
        if rt.hooks.has("PostToolUse"):
            post = await rt.hooks.run("PostToolUse", {"tool_name": tool.name, "tool_input": args,
                                                       "tool_response": {"content": res.content[:20000],
                                                                         "is_error": res.is_error}},
                                      rt.cwd, tool.name)
            for n in post.notices:
                ui.on_notice("warn", n)
            if post.block and post.reason:
                res.content += "\n\n" + reminder(f"PostToolUse hook feedback: {post.reason}")
            for c in post.additional_context:
                res.content += "\n\n" + reminder(c)
            if post.halt:
                res.stop_turn = True
        ui.on_tool_end(self, p, res)
        return res

    def _tool_message(self, p: PreparedCall, res: ToolResult) -> dict:
        content = res.content if res.content.strip() else "(no output)"
        return {"role": "tool", "tool_call_id": p.call.id, "name": p.call.name, "content": content}

    def _close_dangling_calls(self, text: str) -> None:
        last = next((m for m in reversed(self.messages[self.own_start:]) if m.get("role") == "assistant"), None)
        if not last or not last.get("tool_calls"):
            return
        idx = max(i for i, m in enumerate(self.messages) if m is last)
        answered = {m.get("tool_call_id") for m in self.messages[idx + 1:] if m.get("role") == "tool"}
        for tc in last["tool_calls"]:
            if tc.get("id") not in answered:
                self._append({"role": "tool", "tool_call_id": tc.get("id"),
                              "name": tc.get("function", {}).get("name"), "content": text})

    # ------------------------------------------------------------------ compaction
    async def maybe_compact(self) -> None:
        if not self.rt.settings.get("context.auto_compact", True):
            return
        if len(self.messages) <= self.n_prefix() + 2:
            return
        if self.context_tokens() >= self.rt.compact_threshold():
            await self.compact(reason="auto")

    async def compact(self, instructions: str | None = None, reason: str = "manual") -> bool:
        rt, ui, client = self.rt, self.rt.ui, self.llm
        n_sys = self.n_prefix()
        if len(self.messages) <= n_sys + 1:
            ui.on_notice("info", "Nothing to compact yet.")
            return False
        before = self.context_tokens()
        if self.is_main and rt.hooks.has("PreCompact"):
            await rt.hooks.run("PreCompact", {"trigger": reason, "custom_instructions": instructions or ""}, rt.cwd)
        ui.on_compact_start(self, before, reason)
        request = self.messages + [{"role": "user", "content": compact_instruction(instructions)}]
        wire = to_wire(request, tool_mode=client.tool_mode, keep_reasoning=rt.settings.get("provider.keep_reasoning", "all"))
        tools = rt.tool_schemas if client.tool_mode == "native" else None   # identical prefix -> cache hit
        asm = ChunkAssembler()
        try:
            turn = await client.stream_chat(wire, tools, on_delta=lambda d: ui.on_compact_delta(self, d), assembler=asm,
                                            effort=self.current_effort(), max_tokens=12000, slot=self.slot,
                                            extra={"tool_choice": "none"} if tools else None)
        except LLMError as e:
            ui.on_compact_end(self, before, before, None)
            ui.on_notice("error", f"Compaction failed: {e}")
            return False
        summary = _REMINDER_RE.sub("", turn.content).strip()
        if not summary and turn.tool_calls:
            summary = "\n".join(tc.arguments for tc in turn.tool_calls)[:8000]
        if not summary:
            ui.on_compact_end(self, before, before, None)
            ui.on_notice("error", "Compaction failed: the model returned an empty summary.")
            return False
        rt.add_usage(turn.usage, self)
        recent = self._recent_user_messages(int(rt.settings.get("context.keep_recent_chars", 6000)))
        files = [display_path(p, rt.cwd) for p in rt.file_state.recent(12)]
        todos_text = self.todos.render_text() if self.todos.items else None
        new_user = {"role": "user", "_compact": True,
                    "content": post_compact_message(summary, recent, todos_text, files)}
        self.messages = [dict(m) for m in self.messages[:n_sys]] + [new_user]
        self.own_start = n_sys
        self.last_total_tokens = 0
        self._counted_upto = 0
        if self.persist:
            rt.session.compact(self.messages, turn.usage.completion_tokens)
        ui.on_compact_end(self, before, self.context_tokens(), summary)
        return True

    def _recent_user_messages(self, budget: int) -> list[str]:
        recent: list[str] = []
        for m in reversed(self.messages):
            if m.get("role") != "user" or m.get("_meta") or m.get("_compact"):
                continue
            text = _REMINDER_RE.sub("", _msg_text(m)).strip()
            if not text:
                continue
            if len(text) > budget:
                if recent:
                    break
                text = text[:budget] + " [...]"
            recent.insert(0, text)
            budget -= len(text)
            if budget <= 0 or len(recent) >= 4:
                break
        return recent

    async def final_report(self) -> str:
        """Ask a subagent that ran out of steps for its report (one tool-less call)."""
        self._append({"role": "user", "_meta": True, "content": reminder(
            "You have reached your step limit. Do not call any more tools. Write your final report now with what "
            "you found so far, and note what remains unverified.")})
        turn = await self._call_model(TurnStats(), tools_allowed=False)
        if turn is None:
            return ""
        turn.tool_calls = []
        self._record_assistant(turn)
        return turn.content.strip()

    def last_assistant_text(self) -> str:
        for m in reversed(self.messages[self.own_start:]):
            if m.get("role") == "assistant":
                return (m.get("content") or "").strip()
            if m.get("role") == "user":
                break
        return ""


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        text = c
    elif isinstance(c, list):
        text = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    else:
        text = ""
    if m.get("reasoning_content"):
        text += m["reasoning_content"]
    for tc in m.get("tool_calls") or []:
        text += tc.get("function", {}).get("arguments", "") + tc.get("function", {}).get("name", "")
    return text


def _hint_for(e: LLMError, rt) -> str:
    if e.kind == "connection":
        return (f"\nCheck that the server is running and reachable at {rt.client.base_url} "
                "(firewall/port), or run `rigcode doctor`.")
    if e.kind == "auth":
        return "\nCheck provider.api_key in your settings."
    if e.kind == "not_found":
        return "\nCheck provider.base_url (should usually end in /v1) and provider.model."
    if e.kind == "context_overflow":
        return "\nTry /compact or /clear."
    return ""
