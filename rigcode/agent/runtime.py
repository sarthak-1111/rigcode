"""Runtime: what one rigcode session shares - settings, the server connection,
tools, permissions, hooks, the main agent and subagent orchestration.

Prompt-cache strategy (llama.cpp). These server facts drive the design:
  * Hybrid models like Kimi K3 can only rewind the cache to a checkpoint, and
    checkpoints are made at the start of user messages (plus two at the end of
    each prompt). So every agent sends the same tools and the same single
    system message, and the history is append-only.
  * With the default unified KV cache, starting a request saves every idle slot
    to a RAM cache and clears it. Only an unpinned request gets the best saved
    state loaded back; a request pinned to a cleared slot starts from scratch.
    So requests are not pinned (unless provider.llama_cpp.slot_pinning is set).
  * A RAM-cache load moves the entry out of the cache. A subagent that started
    from the main agent's cached state would take it away. So by default a
    subagent forks the main conversation instead: it continues from the main
    agent's exact state (the brief arrives as the Task call's result), and when
    it finishes the main agent continues from the fork's state, which still
    begins with the whole main conversation.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from pathlib import Path

from rigcode import __version__
from rigcode.agent.loop import Agent, TurnStats
from rigcode.agent.prompts import build_env_snapshot, build_project_context, load_prompt, subagent_brief
from rigcode.agent.session import SessionStore
from rigcode.agent.state import Checkpoints, FileStateTracker
from rigcode.extensions import load_agents, load_commands, load_memory, load_skills
from rigcode.hooks import HookRunner
from rigcode.llm.client import LLMClient
from rigcode.llm.probe import ServerInfo, discover, probe_server
from rigcode.llm.textmode import render_tools_prompt
from rigcode.llm.types import LLMError, Usage
from rigcode.llm.wire import to_wire
from rigcode.mcp import McpManager
from rigcode.permissions import PermissionManager
from rigcode.tools.base import ToolRegistry
from rigcode.tools.files import EditTool, MultiEditTool, ReadTool, WriteTool
from rigcode.tools.interact import AskUserTool, ExitPlanModeTool
from rigcode.tools.search import GlobTool, GrepTool
from rigcode.tools.shell import BackgroundShells, BashOutputTool, BashTool, KillShellTool, resolve_shell
from rigcode.tools.skill import SkillTool
from rigcode.tools.task import TaskTool
from rigcode.tools.todo import TodoWriteTool
from rigcode.tools.web import WebFetchTool, WebSearchTool
from rigcode.ui.base import BaseUI
from rigcode.utils.paths import rigcode_home
from rigcode.utils.text import fmt_duration

DEFAULT_CONTEXT_WINDOW = 131072
MIN_LOCAL_CONTEXT = 16384    # below this the system prompt and tools crowd out the task
EFFORTS = ("low", "medium", "high", "max", "none")

PLAN_MODE_ON = (
    "Plan mode is now active. Investigate with read-only tools only (Read, Glob, Grep, read-only Bash commands, "
    "web tools, read-only subagents). Do not edit files or run commands that change anything. When you have a "
    "concrete plan, present it with ExitPlanMode. If the user only asked a question, just answer it.")
PLAN_MODE_OFF = "Plan mode is now off. You may make changes again (normal permission rules apply)."


class Runtime:
    def __init__(self, settings, project_root: Path, cwd: Path, ui: BaseUI | None = None,
                 session_id: str | None = None):
        self.settings = settings
        self.project_root = project_root
        self.cwd = cwd
        self.ui: BaseUI = ui or BaseUI()
        self.debug = bool(settings.get("debug", False))
        self.shell = resolve_shell(settings.get("shell", "auto"))
        self.server = ServerInfo()
        self.session = SessionStore(project_root, session_id)
        log_path = None
        if self.debug:
            log_dir = rigcode_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{self.session.id}.jsonl"
        self.log_path = log_path
        self.client = LLMClient(settings.provider, self.server, log_path)
        self.permissions = PermissionManager(settings, project_root)
        self.file_state = FileStateTracker()
        self.checkpoints = Checkpoints()
        self.shells = BackgroundShells()
        self.agents = load_agents(settings, project_root)
        self.skills = load_skills(settings, project_root)
        self.memory = load_memory(settings, project_root, cwd)
        self.commands = load_commands(settings, project_root)
        self.mcp = McpManager(settings.get("mcp_servers", {}), project_root)
        self.hooks = HookRunner(settings.get("hooks", {}), self.shell, self.session.id, self.session.path)
        self.tools = ToolRegistry()
        self.tool_schemas: list[dict] = []
        self.prefix_messages: list[dict] = []      # [system: core prompt + project context], shared by all agents
        self.project_context = ""
        self.effort: str | None = self._normalize_effort(settings.get("provider.reasoning_effort", "low"))
        self.context_window = DEFAULT_CONTEXT_WINDOW
        self.main_slot: int | None = None
        self.sub_slots: list[int] = []
        self._free_slots: list[int] = []
        self.subagent_concurrency = 1
        self._sub_sem: asyncio.Semaphore | None = None
        self._profile_clients: dict[str, LLMClient] = {}
        self.main_agent: Agent | None = None
        self.usage = Usage()
        self.requests = 0
        self.last_usage: Usage | None = None
        self.started = time.time()
        self.warmup_state = "off"          # off | running | done | failed
        self._warmup_task: asyncio.Task | None = None
        self._turn_mode = self.permissions.mode
        self._mode_reminder: str | None = None
        self._meta_written = False
        self.running_subagents: list[Agent] = []

    # ------------------------------------------------------------------ startup
    async def start(self, resume: Path | None = None) -> None:
        results = await asyncio.gather(discover(self.settings.provider), self.mcp.start_all(),
                                       return_exceptions=True)
        if isinstance(results[0], BaseException):
            server, found = ServerInfo(error=f"{type(results[0]).__name__}: {results[0]}"), None
        else:
            server, found = results[0]
        self.server = server
        self.client.server = server
        if found and server.reachable:
            self.ui.on_notice("info", found)
        if server.kind in ("ollama", "lmstudio") and server.context_window and server.context_window < MIN_LOCAL_CONTEXT:
            fix = ("set OLLAMA_CONTEXT_LENGTH=32768 (or more) and restart Ollama" if server.kind == "ollama"
                   else "raise the model's context length in LM Studio")
            self.ui.on_notice("warn", f"The server gives {self.client.model} only {server.context_window:,} tokens of "
                                      f"context, so long prompts get cut off. To fix it, {fix}.")
        if isinstance(results[1], BaseException):
            self.ui.on_notice("warn", f"MCP startup failed: {results[1]}")
        for s in self.mcp.servers.values():
            if s.status == "failed":
                self.ui.on_notice("warn", f"MCP server '{s.name}' failed to start: {s.error}")

        self.context_window = int(self.settings.get("provider.context_window") or server.context_window
                                  or DEFAULT_CONTEXT_WINDOW)
        self._plan_slots()
        self._build_tools()
        self.project_context = build_project_context(self)
        self._build_prefix()

        history: list[dict] = []
        if resume is not None:
            meta, msgs = SessionStore.load(resume)
            self.session = SessionStore(self.project_root, meta.get("id") or resume.stem)
            self.hooks.session_id, self.hooks.transcript_path = self.session.id, self.session.path
            history = _history_only(msgs)
            self._meta_written = True
        self.main_agent = self._new_main_agent(history, resumed=resume is not None)
        if self.hooks.has("SessionStart"):
            out = await self.hooks.run("SessionStart", {"source": "resume" if resume else "startup"}, self.cwd)
            for n in out.notices:
                self.ui.on_notice("warn", n)
            self.main_agent.pending_reminders.extend(out.additional_context)
        if not server.reachable:
            self.ui.on_notice("error", f"Cannot reach the model server at {self.client.base_url}: {server.error}. "
                                       "Run `rigcode doctor` for help.")
        else:
            self.start_warmup()

    def _new_main_agent(self, history: list[dict] | None = None, resumed: bool = False) -> Agent:
        agent = Agent(self, slot=self.main_slot, messages=[dict(m) for m in self.prefix_messages] + (history or []))
        agent.env_note = build_env_snapshot(self, resumed=resumed)
        return agent

    def _plan_slots(self) -> None:
        lc = self.settings.get("provider.llama_cpp", {}) or {}
        total = self.server.total_slots or 0
        self.main_slot, self.sub_slots = None, []
        if self.server.kind == "llama.cpp" and lc.get("slot_pinning", False) and total >= 1:
            main = lc.get("main_slot")
            main = int(main) if main is not None and 0 <= int(main) < total else total - 1
            subs = lc.get("subagent_slots")
            if subs is None:
                subs = [s for s in range(total - 1, -1, -1) if s != main]
            self.main_slot = main
            self.sub_slots = [int(s) for s in subs if 0 <= int(s) < total and int(s) != main]
        self._free_slots = list(self.sub_slots)
        wanted = max(1, int(self.settings.get("agents.max_concurrency", 1) or 1))
        if self.main_slot is not None:
            self.subagent_concurrency = min(wanted, len(self.sub_slots)) if self.sub_slots else 1
        elif self.server.kind == "llama.cpp" and total:
            self.subagent_concurrency = max(1, min(wanted, total - 1))
        else:
            self.subagent_concurrency = wanted
        self._sub_sem = asyncio.Semaphore(self.subagent_concurrency)
        self._fork_lock = asyncio.Lock()

    def _build_tools(self) -> None:
        s = self.settings
        reg = ToolRegistry()
        tools = [ReadTool(), WriteTool(), EditTool(), MultiEditTool(), GlobTool(), GrepTool(), BashTool(self.shell)]
        if s.get("tools.background_shells", True):
            tools += [BashOutputTool(), KillShellTool()]
        if s.get("tools.web", True):
            tools += [WebFetchTool(), WebSearchTool()]
        tools.append(TodoWriteTool())
        if self.agents:
            tools.append(TaskTool(self.agents))
        if self.skills:
            tools.append(SkillTool())
        tools += [AskUserTool(), ExitPlanModeTool()]
        tools += sorted(self.mcp.tools(), key=lambda t: t.name)
        disabled = set(s.get("tools.disabled", []) or [])
        for t in tools:
            if t.name not in disabled:
                reg.register(t)
        self.tools = reg
        self.tool_schemas = reg.schemas()

    def _build_prefix(self) -> None:
        core = load_prompt("system").strip()
        if self.client.tool_mode == "text":
            core += "\n\n" + render_tools_prompt(self.tool_schemas)
        self.prefix_messages = [{"role": "system", "content": core + "\n\n" + self.project_context}]

    def start_warmup(self) -> None:
        """Pre-process the main agent's prompt (prefix, plus history when resuming) in the background."""
        if (self.settings.get("provider.warmup", True) and self.server.kind == "llama.cpp"
                and self.server.reachable and self.main_agent is not None
                and (self._warmup_task is None or self._warmup_task.done())):
            self.warmup_state = "running"
            self._warmup_task = asyncio.create_task(self._warmup(list(self.main_agent.messages)))

    async def _warmup(self, messages: list[dict]) -> None:
        """Process the prompt prefix while the user types their first message.

        The stub user message matters: llama.cpp can only reuse a cached prefix up to a
        checkpoint, and it creates one at the start of the last user message. The real
        first message starts at the same position, so its request reuses everything before it."""
        me = asyncio.current_task()

        def current() -> bool:      # False once a newer warm-up (after /resume) has replaced this one
            return self._warmup_task is me

        self.warmup_state = "running"
        self.ui.on_warmup("start")
        keep = self.settings.get("provider.keep_reasoning", "all")
        msgs = to_wire(messages + [{"role": "user", "content": "ok"}], tool_mode=self.client.tool_mode,
                       keep_reasoning=keep)
        tools = self.tool_schemas if self.client.tool_mode == "native" else None

        def on_delta(d) -> None:
            if d.kind == "progress" and d.data and current():
                self.ui.on_warmup("progress", d.data)

        try:
            turn = await self.client.complete(msgs, effort=self.effort, max_tokens=1, on_delta=on_delta,
                                              tools=tools, slot=self.main_slot)
        except asyncio.CancelledError:
            if current():
                self.warmup_state = "off"
                self.ui.on_warmup("cancelled")
            raise
        except LLMError as e:
            if current():
                self.warmup_state = "failed"
                self.ui.on_warmup("failed", {"error": str(e)})
            return
        if current():
            self.warmup_state = "done"
            self.ui.on_warmup("done", {"prompt_tokens": turn.usage.prompt_tokens,
                                       "cached": turn.usage.cached_tokens, "seconds": turn.elapsed})

    async def wait_warmup(self) -> None:
        """Hold the main agent's request until the warm-up finishes. Sent alongside it, an unpinned
        request would land on another slot and re-read the whole prompt in parallel."""
        task = self._warmup_task
        if task is None or task.done():
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise           # our own turn was interrupted; the warm-up carries on
        except Exception:  # noqa: BLE001 - a failed warm-up only costs speed
            pass

    # ------------------------------------------------------------------ used by the agent loop and tools
    def set_cwd(self, path: Path) -> None:
        if path.is_dir():
            self.cwd = path

    def compact_threshold(self) -> int:
        cw = self.context_window
        frac = float(self.settings.get("context.compact_at", 0.85) or 0.85)
        reserve = min(int(self.settings.get("provider.max_output_tokens") or 8192), cw // 4)
        limits = [int(cw * frac), cw - reserve]
        cap = self.settings.get("context.compact_at_tokens")
        if cap:
            limits.append(int(cap))
        return max(4096, min(limits))

    def context_limit(self) -> int:
        """Hard ceiling for a prompt: the context window minus room for the reply."""
        cw = self.context_window
        return max(4096, cw - min(int(self.settings.get("provider.max_output_tokens") or 8192), cw // 4))

    def add_usage(self, usage: Usage, agent: Agent) -> None:
        self.usage.add(usage)
        self.requests += 1
        if agent.is_main:
            self.last_usage = usage
        self.ui.on_usage(agent, usage)

    def set_mode(self, mode: str) -> str:
        new = self.permissions.set_mode(mode)
        if new == "plan" and self._turn_mode != "plan":
            self._mode_reminder = PLAN_MODE_ON
        elif new != "plan" and self._turn_mode == "plan":
            self._mode_reminder = PLAN_MODE_OFF
        else:
            self._mode_reminder = None
        return new

    def cycle_mode(self) -> str:
        current = self.permissions.mode
        order = ["default", "acceptEdits", "plan"]
        nxt = order[(order.index(current) + 1) % len(order)] if current in order else "default"
        return self.set_mode(nxt)

    @staticmethod
    def _normalize_effort(value) -> str | None:
        v = str(value or "").strip().lower()
        if v in ("", "null"):
            return None
        return {"minimal": "low", "xhigh": "max", "off": "none"}.get(v, v)

    def set_effort(self, value: str) -> str | None:
        effort = self._normalize_effort(value)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"unknown effort '{value}' (use {', '.join(EFFORTS)})")
        self.effort = effort
        return effort

    async def switch_model(self, name: str) -> str:
        """Switch to a settings profile (by name) or another model on the same server."""
        profiles = self.settings.get("profiles", {}) or {}
        old_url = self.client.base_url
        if name in profiles:
            self.settings.apply_profile(name)
        else:
            self.settings.provider["model"] = name
        await self.client.aclose()
        self.client = LLMClient(self.settings.provider, self.server, self.log_path)
        if self.client.base_url != old_url:
            self.server = await probe_server(self.client.base_url, self.settings.get("provider.api_key"))
            self.client.server = self.server
            self.context_window = int(self.settings.get("provider.context_window") or self.server.context_window
                                      or DEFAULT_CONTEXT_WINDOW)
            self._plan_slots()
            if self.main_agent is not None:
                self.main_agent.slot = self.main_slot
        if self.client.tool_mode == "text" or self.settings.get("provider.tool_mode") == "text":
            self.client.tool_mode = "text"
        self.project_context = build_project_context(self)     # it names the model
        self._build_prefix()
        if self.main_agent is not None:
            self.main_agent._replace_prefix()
        where = self.server.describe() if self.server.reachable else f"unreachable: {self.server.error}"
        return f"{self.client.model} @ {self.client.base_url} ({where})"

    def switch_to_text_tools(self) -> None:
        self.client.tool_mode = "text"
        for c in self._profile_clients.values():
            c.tool_mode = "text"
        self._build_prefix()

    # ------------------------------------------------------------------ main agent
    async def submit(self, text: str) -> TurnStats:
        agent = self.main_agent
        assert agent is not None, "Runtime.start() was not called"
        if not self._meta_written:
            self._meta_written = True
            self.session.meta(id=self.session.id, version=__version__, cwd=str(self.cwd), model=self.client.model,
                              base_url=self.client.base_url, effort=self.effort, started=self.started)
        if self._mode_reminder:
            agent.add_reminder(self._mode_reminder)
            self._mode_reminder = None
        self._turn_mode = self.permissions.mode
        return await agent.run_turn(text)

    def new_conversation(self) -> None:
        """/clear: fresh history and transcript; the shared prompt prefix (and its cache) stays."""
        self.session = SessionStore(self.project_root)
        self.hooks.session_id, self.hooks.transcript_path = self.session.id, self.session.path
        self._meta_written = False
        self.file_state.clear()
        self.main_agent = self._new_main_agent()

    def resume(self, path: Path) -> int:
        """Load a saved transcript into a new main agent; returns the number of messages."""
        meta, msgs = SessionStore.load(path)
        history = _history_only(msgs)
        self.session = SessionStore(self.project_root, meta.get("id") or path.stem)
        self.hooks.session_id, self.hooks.transcript_path = self.session.id, self.session.path
        self._meta_written = True
        self.file_state.clear()
        self.main_agent = self._new_main_agent(history, resumed=True)
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()          # it was warming the conversation we just replaced
        self._warmup_task = None
        self.start_warmup()
        return len(history)

    # ------------------------------------------------------------------ subagents
    def _client_for(self, agent_def) -> LLMClient | None:
        name = agent_def.model
        if not name:
            return None
        if name in self._profile_clients:
            return self._profile_clients[name]
        profile = (self.settings.get("profiles", {}) or {}).get(name)
        if not isinstance(profile, dict):
            self.ui.on_notice("warn", f"Agent '{agent_def.name}' asks for unknown profile '{name}'; using the main model.")
            return None
        provider = copy.deepcopy(self.settings.provider)
        provider.update(profile)
        same_server = str(provider.get("base_url", "")).rstrip("/") == self.client.base_url
        client = LLMClient(provider, self.server if same_server else ServerInfo(), self.log_path)
        client.tool_mode = self.client.tool_mode
        self._profile_clients[name] = client
        return client

    def subagent_mode(self) -> str:
        mode = str(self.settings.get("agents.context", "auto") or "auto").lower()
        return mode if mode in ("auto", "fork", "fresh") else "auto"

    def parallel_subagents(self) -> bool:
        """Task calls may run side by side only as fresh subagents: parallel forks of one
        conversation would each need their own copy of the main agent's cache."""
        return self.subagent_concurrency > 1 and self.subagent_mode() == "fresh"

    def _should_fork(self, parent, agent_def, client) -> bool:
        mode = self.subagent_mode()
        if mode == "fresh" or parent is None or client is not None:
            return False
        if agent_def.effort and agent_def.effort != parent.current_effort():
            return False    # the effort setting sits before the conversation: a fork would re-read all of it
        if mode == "fork":
            return True
        # A fork needs room on top of the conversation. With the defaults this is the compaction
        # threshold, so in practice subagents always fork; fresh ones rely on the server's RAM cache
        # holding the main conversation while they run.
        limit = self.settings.get("agents.fork_max_tokens") or min(self.compact_threshold(), self.context_limit() // 2)
        return parent.context_tokens() < int(limit)

    async def run_subagent(self, agent_def, description: str, prompt: str, ctx) -> tuple[str, str]:
        assert self._sub_sem is not None
        parent = getattr(ctx, "agent", None)
        call_id = getattr(ctx, "call_id", "") or ""
        client = self._client_for(agent_def)
        base = parent.fork_messages(call_id) if self._should_fork(parent, agent_def, client) else None
        fork = base is not None
        async with (self._fork_lock if fork else self._sub_sem):
            if fork:
                slot = parent.slot
            else:
                slot = self._free_slots.pop(0) if self._free_slots else None
                base = [dict(m) for m in self.prefix_messages]
            if client is not None and client.server is not self.server:
                slot = None
            max_steps = int(agent_def.max_turns or self.settings.get("agents.max_turns", 40) or 40)
            sub = Agent(self, name=agent_def.name, depth=getattr(parent, "depth", 0) + 1, agent_def=agent_def,
                        slot=slot, effort=agent_def.effort, max_steps=max_steps, client=client, messages=base)
            sub.parent_call_id = call_id
            sub.forked = fork
            if fork:    # same messages as the parent's last request, so its token count still applies
                sub.last_total_tokens, sub._counted_upto = parent.last_total_tokens, min(parent._counted_upto, len(base))
            self.running_subagents.append(sub)
            self.ui.on_subagent_start(sub, description)
            stats = TurnStats()
            report = ""
            try:
                if fork:
                    brief = subagent_brief(agent_def, description, prompt, forked=True)
                    stats = await sub.run_turn(brief, tool_call_id=call_id)
                else:
                    brief = subagent_brief(agent_def, description, prompt, env=build_env_snapshot(self))
                    stats = await sub.run_turn(brief)
                report = sub.last_assistant_text()
                if stats.stopped_reason in ("max_steps", "loop", "context") or not report:
                    report = (await sub.final_report()) or report
                if self.hooks.has("SubagentStop"):
                    await self.hooks.run("SubagentStop", {"agent": agent_def.name, "description": description},
                                         self.cwd)
            finally:
                self.running_subagents.remove(sub)
                if slot is not None and slot in self.sub_slots:
                    self._free_slots.insert(0, slot)   # LIFO: reuse the slot whose cache is warm
                self.ui.on_subagent_end(sub, stats)
            if not fork and client is None and self.main_slot is None:
                await self._flush_idle_slots()
        summary = (f"{stats.steps} steps, {stats.tool_calls} tool calls, {fmt_duration(stats.elapsed)}"
                   + ("" if stats.stopped_reason == "done" else f", stopped: {stats.stopped_reason}"))
        return report, summary

    async def _flush_idle_slots(self) -> None:
        """After a fresh subagent, start a tiny request so llama.cpp moves the subagent's idle
        slot into its RAM cache. Otherwise the main agent's next request could be matched to
        that slot (they share the prompt prefix) and re-read its own history instead of
        getting its saved state back."""
        if self.server.kind != "llama.cpp" or not self.server.reachable:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.client.complete([{"role": "user", "content": "."}], effort="none",
                                                        max_tokens=1), 60)

    # ------------------------------------------------------------------ shutdown
    async def close(self) -> None:
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warmup_task
        if self.hooks.has("SessionEnd"):
            with contextlib.suppress(Exception):
                await self.hooks.run("SessionEnd", {"reason": "exit"}, self.cwd)
        with contextlib.suppress(Exception):
            await self.shells.kill_all()
        with contextlib.suppress(Exception):
            await self.mcp.close_all()
        await self.client.aclose()
        for c in self._profile_clients.values():
            await c.aclose()


def _history_only(msgs: list[dict]) -> list[dict]:
    """A saved transcript without the prompt prefix (rebuilt fresh for every session)."""
    return [m for m in msgs if m.get("role") != "system" and not m.get("_prefix")]
