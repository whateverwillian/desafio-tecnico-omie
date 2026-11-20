from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model that returns queued AIMessages."""

    responses: List[AIMessage] = Field(default_factory=list)
    bound_tools: List[Any] = Field(default_factory=list)
    on_invoke: Optional[Callable[[Sequence[BaseMessage]], None]] = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        # Keep the same responses queue across bind_tools clones.
        clone = self.model_copy(deep=False)
        clone.bound_tools = list(tools)
        clone.responses = self.responses
        clone.on_invoke = self.on_invoke
        return clone

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.on_invoke is not None:
            self.on_invoke(messages)
        if not self.responses:
            message = AIMessage(content="")
        else:
            message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict | None = None, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )
