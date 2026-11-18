from __future__ import annotations

from rigcode.agent.runtime import Runtime
from rigcode.config import load_settings
from rigcode.permissions import Decision
from rigcode.ui.base import BaseUI


class RecordingUI(BaseUI):
    """Records engine events; approves permission requests unless told otherwise."""

    interactive = True

    def __init__(self, decisions=None, answers=None, queued=None):
        self.decisions = list(decisions or [])
        self.answers = list(answers or [])
        self.queued = list(queued or [])
        self.permission_requests = []
        self.notices = []
        self.tool_ends = []
        self.subagents = []
        self.compactions = []
        self.deltas = []

    def on_notice(self, level, text):
        self.notices.append((level, text))

    def on_delta(self, agent, delta):
        self.deltas.append((agent.name, delta.kind, delta.text))

    def on_tool_end(self, agent, prepared, result):
        self.tool_ends.append((agent.name, prepared.call.name, result))

    def on_subagent_start(self, agent, description):
        self.subagents.append((agent.name, description, agent.slot))

    def on_compact_end(self, agent, before, after, summary):
        self.compactions.append((before, after, summary))

    async def request_permission(self, req, agent):
        self.permission_requests.append(req)
        return self.decisions.pop(0) if self.decisions else Decision(allow=True)

    async def ask_user(self, question, options):
        return self.answers.pop(0) if self.answers else None

    def take_queued_input(self):
        return self.queued.pop(0) if self.queued else None

    def results(self, tool_name=None):
        return [r for _, name, r in self.tool_ends if tool_name is None or name == tool_name]


async def make_runtime(project, base_url, ui=None, start=True, **overrides) -> Runtime:
    cli = {"provider.base_url": base_url, "provider.warmup": False, "context.repo_map": True}
    cli.update({k.replace("__", "."): v for k, v in overrides.items()})
    settings = load_settings(project, cli)
    rt = Runtime(settings, project, project, ui or RecordingUI())
    if start:
        await rt.start()
    return rt
