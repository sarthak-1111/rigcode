"""Skill: load a skill's full instructions on demand (progressive disclosure)."""
from __future__ import annotations

from rigcode.tools.base import Tool, ToolContext, ToolResult


class SkillTool(Tool):
    name = "Skill"
    kind = "read"
    read_only = True
    description = ("Load the full instructions of a skill listed in your context, by name. Use it when the task "
                   "matches a skill's description, then follow the loaded instructions.")
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name"},
            "args": {"type": "string", "description": "Optional arguments for the skill"},
        },
        "required": ["name"],
    }
    aliases = {"skill": "name", "skill_name": "name", "arguments": "args"}

    def permission_target(self, args: dict) -> str:
        return str(args.get("name", ""))

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        name = str(args.get("name") or args.get("skill") or "").strip()
        skill = ctx.rt.skills.get(name) or next(
            (s for k, s in ctx.rt.skills.items() if k.lower() == name.lower()), None)
        if skill is None:
            known = ", ".join(sorted(ctx.rt.skills)) or "none"
            return ToolResult.error(f"Unknown skill '{name}'. Available: {known}")
        body = skill.load_body()
        extra = f"\nArguments: {args['args']}" if args.get("args") else ""
        return ToolResult(
            content=f"Skill '{skill.name}' (files in {skill.directory}):{extra}\n\n{body}",
            summary=f"Loaded skill {skill.name}")
