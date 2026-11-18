"""Core LLM data types shared by the client, agent loop and UI."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""            # raw JSON text as produced by the model
    args: dict | None = None       # parsed arguments (None if unparseable)
    error: str | None = None       # parse/validation problem to report back

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0         # prompt tokens served from the server's KV cache
    prompt_ms: float = 0.0
    gen_ms: float = 0.0

    @property
    def gen_tps(self) -> float | None:
        if self.gen_ms > 0 and self.completion_tokens:
            return self.completion_tokens / (self.gen_ms / 1000)
        return None

    @property
    def prompt_tps(self) -> float | None:
        processed = self.prompt_tokens - self.cached_tokens
        if self.prompt_ms > 0 and processed > 0:
            return processed / (self.prompt_ms / 1000)
        return None

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.prompt_ms += other.prompt_ms
        self.gen_ms += other.gen_ms


@dataclass
class AssistantTurn:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    interrupted: bool = False
    elapsed: float = 0.0
    first_token_latency: float | None = None

    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


@dataclass
class Delta:
    """A streaming event from the model.

    kind: reasoning | content | tool_start | tool_args | progress | reset | first_token
    """
    kind: str
    text: str = ""
    index: int = 0
    name: str | None = None
    data: dict | None = None


class LLMError(Exception):
    def __init__(self, message: str, *, status: int | None = None, kind: str = "error",
                 body: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.body = body
        self.retryable = retryable
