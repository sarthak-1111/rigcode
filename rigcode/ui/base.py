"""The interface between the engine (Runtime/Agent) and a front end.

Every method has a harmless default, so a front end only overrides what it shows.
Methods returning None from the interactive calls mean "nobody can answer".
"""
from __future__ import annotations


class BaseUI:
    interactive = False

    # ---------------------------------------------------------------- notices and model output
    def on_notice(self, level: str, text: str) -> None:            # level: info | warn | error
        pass

    def on_model_start(self, agent, info: dict) -> None:
        pass

    def on_delta(self, agent, delta) -> None:
        pass

    def on_model_end(self, agent, turn) -> None:                    # turn is None when the request failed
        pass

    def on_usage(self, agent, usage) -> None:
        pass

    # ---------------------------------------------------------------- tools
    def on_tool_start(self, agent, prepared, args: dict) -> None:
        pass

    def on_tool_output(self, agent, call_id: str, text: str) -> None:
        pass

    def on_tool_end(self, agent, prepared, result) -> None:
        pass

    # ---------------------------------------------------------------- questions for the user
    async def request_permission(self, req, agent):
        return None

    async def ask_user(self, question: str, options: list[str]) -> str | None:
        return None

    async def review_plan(self, plan: str):
        return None

    def take_queued_input(self) -> str | None:
        """Text the user typed while the agent was busy, delivered at the next step."""
        return None

    # ---------------------------------------------------------------- lifecycle
    def on_compact_start(self, agent, before: int, reason: str) -> None:
        pass

    def on_compact_delta(self, agent, delta) -> None:
        pass

    def on_compact_end(self, agent, before: int, after: int, summary: str | None) -> None:
        pass

    def on_subagent_start(self, agent, description: str) -> None:
        pass

    def on_subagent_end(self, agent, stats) -> None:
        pass

    def on_warmup(self, state: str, data: dict | None = None) -> None:   # start | progress | done | failed
        pass
