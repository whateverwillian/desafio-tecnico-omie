from __future__ import annotations

from typing import Any, List

from agile_bank.agents.actions import (
    CurrencyAction,
    aimessage_to_currency_action,
    currency_action_to_message,
)
from agile_bank.agents.currency.prompt import currency_prompt
from agile_bank.agents.runtime import build_tool_agent, tool_names
from agile_bank.agents.tools.currency import build_currency_tools

CURRENCY_TOOL_NAMES = [
    "get_currency_quote",
    "handoff_to_credit",
    "end_conversation",
]


def build_currency_graph(services: Any, model: Any):
    tools = build_currency_tools(services)
    assert tool_names(tools) == CURRENCY_TOOL_NAMES
    return build_tool_agent(
        "currency",
        model,
        tools,
        currency_prompt,
        on_error=services.record_error,
        action_schema=CurrencyAction,
        action_mapper=aimessage_to_currency_action,
        action_to_message=currency_action_to_message,
    )


def list_currency_tools(services: Any) -> List[Any]:
    return build_currency_tools(services)
