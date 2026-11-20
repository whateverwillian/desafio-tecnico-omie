from __future__ import annotations

from typing import Any, List

from agile_bank.agents.interview.prompt import interview_prompt
from agile_bank.agents.runtime import build_tool_agent, tool_names
from agile_bank.agents.tools.interview import build_interview_tools

INTERVIEW_TOOL_NAMES = [
    "record_interview_answer",
    "submit_interview",
    "handoff_to_credit",
    "end_conversation",
]


def build_interview_graph(services: Any, model: Any):
    tools = build_interview_tools(services)
    assert tool_names(tools) == INTERVIEW_TOOL_NAMES
    return build_tool_agent(
        "interview",
        model,
        tools,
        interview_prompt,
        on_error=services.record_error,
    )


def list_interview_tools(services: Any) -> List[Any]:
    return build_interview_tools(services)
