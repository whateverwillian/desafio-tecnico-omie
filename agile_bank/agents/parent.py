from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agile_bank.agents import copy
from agile_bank.agents.credit.graph import build_credit_graph
from agile_bank.agents.currency.graph import build_currency_graph
from agile_bank.agents.interview.graph import build_interview_graph
from agile_bank.agents.routing import route_entry
from agile_bank.agents.state import ConversationState
from agile_bank.agents.triage.graph import build_triage_graph


def greet_node(state: ConversationState) -> Dict[str, Any]:
    return {"messages": [AIMessage(content=copy.GREETING)]}


def end_node(state: ConversationState) -> Dict[str, Any]:
    if state.get("auth_attempts", 0) >= 3 and not state.get("authenticated"):
        text = copy.AUTH_BLOCKED
    else:
        text = copy.GOODBYE
    return {
        "messages": [AIMessage(content=text)],
        "end_requested": True,
    }


def build_conversation_graph(
    services: Any,
    model: Any,
    checkpointer: Optional[MemorySaver] = None,
):
    graph = StateGraph(ConversationState)
    graph.add_node("greet", greet_node)
    graph.add_node("triage", build_triage_graph(services, model))
    graph.add_node("credit", build_credit_graph(services, model))
    graph.add_node("interview", build_interview_graph(services, model))
    graph.add_node("currency", build_currency_graph(services, model))
    graph.add_node("end", end_node)

    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "greet": "greet",
            "triage": "triage",
            "credit": "credit",
            "interview": "interview",
            "currency": "currency",
            "end": "end",
        },
    )
    graph.add_edge("greet", END)
    graph.add_edge("end", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state() -> ConversationState:
    return {
        "messages": [],
        "active_agent": "triage",
        "authenticated": False,
        "auth_attempts": 0,
        "customer": None,
        "interview": {},
        "open_limit_request": None,
        "interview_accepted": False,
        "handoff_reason": None,
        "end_requested": False,
        "turn_closed": False,
        "required_notice": None,
    }
