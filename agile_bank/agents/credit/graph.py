from __future__ import annotations

from typing import Any, List

from agile_bank.agents.actions import (
    CreditAction,
    aimessage_to_credit_action,
    credit_action_to_message,
)
from agile_bank.agents.credit.prompt import credit_prompt
from agile_bank.agents.runtime import build_tool_agent, tool_names
from agile_bank.agents.tools.credit import build_credit_prelude, build_credit_tools

CREDIT_TOOL_NAMES = [
    "get_credit_limit",
    "request_limit_increase",
    "handoff_to_interview",
    "handoff_to_currency",
    "end_conversation",
]


def build_credit_graph(services: Any, model: Any):
    tools = build_credit_tools(services)
    assert tool_names(tools) == CREDIT_TOOL_NAMES
    return build_tool_agent(
        "credit",
        model,
        tools,
        credit_prompt,
        prelude=build_credit_prelude(services),
        on_error=services.record_error,
        action_schema=CreditAction,
        action_mapper=aimessage_to_credit_action,
        action_to_message=credit_action_to_message,
    )


def list_credit_tools(services: Any) -> List[Any]:
    return build_credit_tools(services)
