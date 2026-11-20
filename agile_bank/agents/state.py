from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

AgentName = Literal["triage", "credit", "interview", "currency"]


class ConversationState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    active_agent: AgentName
    authenticated: bool
    auth_attempts: int
    customer: Optional[Dict[str, Any]]
    interview: Dict[str, Any]
    open_limit_request: Optional[Dict[str, Any]]
    interview_accepted: bool
    handoff_reason: Optional[str]
    end_requested: bool
    turn_closed: bool
    required_notice: Optional[str]
