from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agile_bank.agents.routing import ALLOWED_DESTINATIONS
from agile_bank.agents.state import AgentName


def build_handoff_tool(
    *,
    name: str,
    description: str,
    current: AgentName,
    destination: AgentName,
    require_auth: bool = True,
    guard: Optional[Callable[[dict], Optional[str]]] = None,
    extra: Optional[Callable[[dict], Dict[str, Any]]] = None,
):
    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        if require_auth and not state.get("authenticated"):
            return ToolMessage(
                content="Ainda não autenticado. Autentique o cliente antes.",
                tool_call_id=tool_call_id,
            )
        if destination not in ALLOWED_DESTINATIONS.get(current, set()):
            return ToolMessage(
                content=f"Destino {destination} não permitido a partir de {current}.",
                tool_call_id=tool_call_id,
            )
        if guard is not None:
            blocked = guard(state)
            if blocked:
                return ToolMessage(content=blocked, tool_call_id=tool_call_id)

        update: Dict[str, Any] = {
            "active_agent": destination,
            "messages": [
                ToolMessage(
                    content=f"ok:{destination}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
        if extra is not None:
            update.update(extra(state))
        return Command(
            goto=destination,
            graph=Command.PARENT,
            update=update,
        )

    return handoff_tool
