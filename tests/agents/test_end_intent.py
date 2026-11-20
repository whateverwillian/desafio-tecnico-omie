from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agile_bank.agents import copy
from agile_bank.agents.parent import build_conversation_graph, initial_state
from agile_bank.agents.routing import route_entry
from tests.agents.fake_model import ScriptedChatModel, tool_call
from tests.agents.test_agent_rails import FakeServices, _thread, _visible_ai


def test_router_does_not_end_on_goodbye_phrase() -> None:
    state = initial_state()
    state.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "messages": [HumanMessage(content="quero encerrar")],
        }
    )
    assert route_entry(state) == "credit"


def test_agent_end_conversation_tool_closes_session() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[tool_call("end_conversation", {}, "e1")]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("end-tool")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero encerrar")]},
        config=thread,
    )
    assert result["end_requested"] is True
    assert copy.GOODBYE in result["messages"][-1].content
    assert any(
        isinstance(message, ToolMessage)
        and "encerramento" in str(message.content).lower()
        for message in result["messages"]
    )


def test_sair_da_entrevista_does_not_end_via_router() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("handoff_to_credit", {}, "h1"),
            AIMessage(content="Certo."),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("no-end-interview")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "interview",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
            "interview": {"monthly_income": "10000"},
            "open_limit_request": {
                "cpf": "11144477735",
                "requested_at": "2026-11-18T12:00:00-03:00",
                "requested_limit": "8000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero sair da entrevista")]},
        config=thread,
    )
    assert result.get("end_requested") is not True
    assert result["active_agent"] == "credit"
    assert copy.GOODBYE not in " ".join(_visible_ai(result["messages"]))
