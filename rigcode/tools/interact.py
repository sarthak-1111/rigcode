"""Tools that talk to the user: AskUser and ExitPlanMode."""
from __future__ import annotations

from rigcode.tools.base import Tool, ToolContext, ToolResult


class AskUserTool(Tool):
    name = "AskUser"
    kind = "interact"
    read_only = True
    description = (
        "Ask the user a question when you are blocked on a decision only they can make (ambiguous, high-impact "
        "requirements; choosing between valid approaches). Offer 2-4 concrete options when possible. Don't ask "
        "about things you can find out yourself, and don't ask for permission to proceed with the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Suggested answers"},
        },
        "required": ["question"],
    }
    aliases = {"choices": "options", "answers": "options", "text": "question", "prompt": "question"}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        question = str(args.get("question", "")).strip()
        options = [str(o) for o in (args.get("options") or []) if str(o).strip()][:6]
        answer = await ctx.rt.ui.ask_user(question, options)
        if answer is None:
            return ToolResult(content="The user is not available to answer. Proceed with your best judgment and "
                                      "state the assumption you made.", summary="No answer (non-interactive)")
        return ToolResult(content=f"User answered: {answer}", summary=f"Answered: {answer[:60]}")


class ExitPlanModeTool(Tool):
    name = "ExitPlanMode"
    kind = "interact"
    read_only = True
    description = (
        "Plan mode only: present your finished implementation plan for approval. If approved you will be "
        "allowed to make changes; if not, you get feedback to revise the plan."
    )
    parameters = {
        "type": "object",
        "properties": {"plan": {"type": "string", "description": "The plan, in concise Markdown"}},
        "required": ["plan"],
    }
    aliases = {"content": "plan", "text": "plan"}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        perms = ctx.rt.permissions
        if perms.mode != "plan":
            return ToolResult(content="Plan mode is not active - no approval needed. Continue with the task.",
                              summary="Not in plan mode")
        plan = str(args.get("plan", "")).strip()
        decision = await ctx.rt.ui.review_plan(plan)
        if decision is None:
            return ToolResult(content="Plan recorded, but nobody is available to approve it. Stay in plan mode "
                                      "and stop here.", summary="Plan (no reviewer)", stop_turn=True)
        approved, mode, feedback = decision
        if approved:
            perms.set_mode(mode or perms.previous_mode or "default")
            return ToolResult(content=f"The user approved the plan. Plan mode is off (permission mode: {perms.mode}). "
                                      "Implement the plan now, tracking progress with TodoWrite.",
                              summary="Plan approved", display={"type": "markdown", "text": plan})
        return ToolResult(content="The user did not approve the plan. Still in plan mode. Feedback: "
                                  + (feedback or "(none)") + "\nRevise the plan accordingly.",
                          summary="Plan rejected", stop_turn=not feedback)
