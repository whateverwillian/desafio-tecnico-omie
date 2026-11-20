from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command


def restart_authentication(services: Any, operation: str, tool_call_id: str) -> Command:
    """Volta à triagem quando a sessão está autenticada sem cliente no estado."""
    services.record_error(operation, "Sessão autenticada sem CPF no estado.")
    return Command(
        goto="triage",
        graph=Command.PARENT,
        update={
            "authenticated": False,
            "customer": None,
            "active_agent": "triage",
            "interview": {},
            "open_limit_request": None,
            "interview_accepted": False,
            "handoff_reason": None,
            "turn_closed": False,
            "required_notice": None,
            "messages": [
                ToolMessage(
                    content="Cliente sem identificação válida. Autenticação reiniciada.",
                    tool_call_id=tool_call_id,
                )
            ],
        },
    )
