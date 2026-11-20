from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState

from agile_bank.agents import copy
from agile_bank.agents.parent import initial_state
from agile_bank.agents.runtime import build_tool_agent
from agile_bank.agents.state import ConversationState
from tests.agents.fake_model import ScriptedChatModel, tool_call
from tests.agents.test_agent_rails import _thread, _visible_ai


def test_tool_exception_keeps_checkpoint_usable() -> None:
    errors: list[tuple[str, str]] = []

    @tool("broken_tool")
    def broken_tool(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Tool que falha de propósito no teste."""
        raise RuntimeError("boom")

    model = ScriptedChatModel(
        responses=[
            tool_call("broken_tool", {}, "b1"),
            AIMessage(content=""),
            AIMessage(content="segundo turno ok"),
        ]
    )
    agent = build_tool_agent(
        "broken",
        model,
        [broken_tool],
        lambda _state: "teste",
        on_error=lambda op, msg: errors.append((op, msg)),
    )

    parent = StateGraph(ConversationState)
    parent.add_node("broken", agent)
    parent.add_edge(START, "broken")
    parent.add_edge("broken", END)
    graph = parent.compile(checkpointer=MemorySaver())
    thread = _thread("tool-exc")
    seed = initial_state()
    seed.update({"authenticated": True, "active_agent": "credit"})
    graph.invoke(seed, config=thread)
    first = graph.invoke(
        {"messages": [HumanMessage(content="dispare")]},
        config=thread,
    )
    assert any(isinstance(message, ToolMessage) for message in first["messages"])
    assert copy.GENERIC_RETRY in _visible_ai(first["messages"])
    assert first.get("end_requested") is not True
    assert errors and errors[0][0] == "broken_tool"

    second = graph.invoke(
        {"messages": [HumanMessage(content="continue")]},
        config=thread,
    )
    assert "segundo turno ok" in _visible_ai(second["messages"])
