"""Task: delegate a self-contained job to a subagent with its own context."""
from __future__ import annotations

from rigcode.tools.base import Tool, ToolContext, ToolResult
from rigcode.utils.text import truncate_middle


class TaskTool(Tool):
    name = "Task"
    kind = "agent"
    read_only = True   # the subagent's own tool calls go through permissions individually
    aliases = {"agent": "subagent_type", "agent_type": "subagent_type", "type": "subagent_type",
               "task": "prompt", "instructions": "prompt", "brief": "prompt", "title": "description"}

    def __init__(self, agents: dict):
        self.agents = agents
        lines = [f"- {name}: {a.description}" for name, a in agents.items()]
        self.description = (
            "Delegate a self-contained task to a subagent that works in its own context and returns one final "
            "report. Use it to keep your context small: broad exploration across many files, independent "
            "research, reviewing or verifying a change. Don't delegate a read or search you can do in one or two "
            "calls. The subagent may not see this conversation, so give a complete brief (goal, relevant paths, "
            "constraints, what to return). Only its final report comes back to you.\nAgents:\n"
            + "\n".join(lines)
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "3-6 word label for the task"},
                "prompt": {"type": "string", "description": "Complete, self-contained brief for the subagent"},
                "subagent_type": {"type": "string", "enum": list(agents) or ["general"]},
            },
            "required": ["description", "prompt", "subagent_type"],
        }

    def permission_target(self, args: dict) -> str:
        return f"{args.get('subagent_type', '')}: {args.get('description', '')}"

    def label(self, args: dict, ctx: ToolContext | None = None) -> str:
        return f"Task[{args.get('subagent_type', '?')}]({args.get('description', '')})"

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        name = str(args.get("subagent_type") or "general")
        agent_def = self.agents.get(name)
        if agent_def is None:
            return ToolResult.error(f"Unknown subagent_type '{name}'. Available: {', '.join(self.agents)}")
        if getattr(ctx.agent, "depth", 0) >= 1:
            return ToolResult.error("Subagents cannot start further subagents. Do the work directly.")
        report, stats = await ctx.rt.run_subagent(agent_def, str(args.get("description", name)),
                                                  str(args.get("prompt", "")), ctx)
        text, _ = truncate_middle(report or "(the subagent returned no report)", 16000)
        footer = f"\n\n[{name} subagent: {stats}]"
        return ToolResult(content=text + footer, summary=f"{name}: {stats}",
                          display={"type": "markdown", "text": report or ""})
