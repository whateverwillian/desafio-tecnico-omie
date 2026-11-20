from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


def build_end_conversation_tool():
    @tool("end_conversation")
    def end_conversation(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Encerra o atendimento quando o cliente pede para terminar a conversa.

        Não use quando o cliente só recusa uma oferta, como a entrevista.
        """
        return Command(
            goto="end",
            graph=Command.PARENT,
            update={
                "end_requested": True,
                "messages": [
                    ToolMessage(
                        content="Atendimento marcado para encerramento.",
                        tool_call_id=tool_call_id,
                    )
                ],
            },
        )

    return end_conversation
