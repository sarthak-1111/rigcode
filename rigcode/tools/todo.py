"""TodoWrite: the agent's visible task list."""
from __future__ import annotations

from rigcode.agent.state import TodoItem
from rigcode.tools.base import Tool, ToolContext, ToolResult

_STATUS = {
    "pending": "pending", "todo": "pending", "not_started": "pending", "open": "pending",
    "in_progress": "in_progress", "in-progress": "in_progress", "doing": "in_progress", "active": "in_progress",
    "started": "in_progress", "working": "in_progress",
    "completed": "completed", "complete": "completed", "done": "completed", "finished": "completed",
}


class TodoWriteTool(Tool):
    name = "TodoWrite"
    kind = "read"
    read_only = True
    description = (
        "Maintain your task list for multi-step work (3+ steps). Send the complete list every time. Keep "
        "exactly one item in_progress while you work on it and mark items completed as soon as they are done. "
        "Skip it for simple one- or two-step tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The task, imperative form"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    aliases = {"items": "todos", "tasks": "todos", "list": "todos", "todo_list": "todos"}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        items = []
        for raw in args.get("todos") or []:
            if isinstance(raw, str):
                items.append(TodoItem(content=raw))
                continue
            if not isinstance(raw, dict):
                continue
            content = raw.get("content") or raw.get("title") or raw.get("task") or raw.get("description") or ""
            status = _STATUS.get(str(raw.get("status", "pending")).lower().strip(), "pending")
            if content:
                items.append(TodoItem(content=str(content), status=status, active_form=raw.get("activeForm")))
        todos = ctx.agent.todos
        todos.set(items)
        done = sum(1 for t in items if t.status == "completed")
        return ToolResult(
            content=f"Todo list updated ({done}/{len(items)} done):\n{todos.render_text()}",
            summary=f"Todos {done}/{len(items)}",
            display={"type": "todos", "items": [vars(t) for t in items]},
        )
