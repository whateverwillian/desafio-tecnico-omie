from __future__ import annotations

from typing import Any, List

from agile_bank.agents.runtime import build_tool_agent, tool_names
from agile_bank.agents.tools.triage import build_triage_tools
from agile_bank.agents.triage.prompt import triage_prompt

TRIAGE_TOOL_NAMES = [
    "authenticate_customer",
    "refuse_other_customer_data",
    "handoff_to_credit",
    "handoff_to_currency",
    "end_conversation",
]


def build_triage_graph(services: Any, model: Any):
    tools = build_triage_tools(services)
    assert tool_names(tools) == TRIAGE_TOOL_NAMES
    return build_tool_agent(
        "triage",
        model,
        tools,
        triage_prompt,
        on_error=services.record_error,
    )


def list_triage_tools(services: Any) -> List[Any]:
    return build_triage_tools(services)
