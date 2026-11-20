from __future__ import annotations

from typing import Dict, Set

from langchain_core.messages import HumanMessage

from agile_bank.agents.state import AgentName, ConversationState

ALLOWED_DESTINATIONS: Dict[AgentName, Set[AgentName]] = {
    "triage": {"credit", "currency"},
    "credit": {"interview", "currency"},
    "interview": {"credit"},
    "currency": {"credit"},
}


def has_human_message(state: ConversationState) -> bool:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return True
        if isinstance(message, dict) and message.get("role") == "user":
            return True
        if isinstance(message, dict) and message.get("type") == "human":
            return True
        if getattr(message, "type", None) == "human":
            return True
    return False


def route_entry(state: ConversationState) -> str:
    if state.get("end_requested"):
        return "end"
    if state.get("auth_attempts", 0) >= 3 and not state.get("authenticated"):
        return "end"
    if not has_human_message(state):
        return "greet"
    if not state.get("authenticated"):
        return "triage"
    return state.get("active_agent", "triage")
