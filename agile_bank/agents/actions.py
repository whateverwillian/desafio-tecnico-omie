"""Ações fechadas: o modelo escolhe, o grafo executa e escreve o número."""

from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from agile_bank.agents import copy


class CurrencyAction(BaseModel):
    """Decisão do turno de câmbio."""

    action: Literal[
        "quote",
        "ask_currency",
        "refuse_dictated",
        "handoff_credit",
        "end",
    ]
    currency_code: Optional[str] = Field(
        default=None,
        description="Código de três letras quando action=quote (USD, EUR, ...).",
    )


class CreditAction(BaseModel):
    """Decisão do turno de crédito."""

    action: Literal[
        "consult_limit",
        "request_increase",
        "ask_amount",
        "handoff_interview",
        "handoff_currency",
        "end",
        "refuse_other_data",
        "speak",
    ]
    requested_limit: Optional[str] = Field(
        default=None,
        description="Valor do aumento quando action=request_increase.",
    )
    reply: Optional[str] = Field(
        default=None,
        description="Texto curto ao cliente quando action=speak.",
    )


_NOTICE_TOOLS = frozenset(
    {
        "get_currency_quote",
        "get_credit_limit",
        "request_limit_increase",
        "refuse_other_customer_data",
    }
)


def is_notice_tool(name: str) -> bool:
    return name in _NOTICE_TOOLS


def currency_action_to_message(
    action: CurrencyAction, *, call_id: str = "currency-action"
) -> AIMessage:
    if action.action == "ask_currency":
        return AIMessage(content=copy.CURRENCY_ASK)
    if action.action == "refuse_dictated":
        return AIMessage(content=copy.CURRENCY_REFUSE_DICTATED)
    if action.action == "quote":
        code = (action.currency_code or "").strip().upper()
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_currency_quote",
                    "args": {"currency_code": code},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    if action.action == "handoff_credit":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_to_credit",
                    "args": {},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "end_conversation",
                "args": {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def credit_action_to_message(
    action: CreditAction, *, call_id: str = "credit-action"
) -> AIMessage:
    if action.action == "ask_amount":
        return AIMessage(
            content=(
                "Por favor, informe o valor que você deseja para o "
                "aumento do seu limite."
            )
        )
    if action.action == "refuse_other_data":
        return AIMessage(content=copy.DATA_OTHER_REFUSAL)
    if action.action == "speak":
        text = (action.reply or "").strip() or copy.GENERIC_RETRY
        return AIMessage(content=text)
    if action.action == "consult_limit":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_credit_limit",
                    "args": {},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    if action.action == "request_increase":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "request_limit_increase",
                    "args": {"requested_limit": action.requested_limit or ""},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    if action.action == "handoff_interview":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_to_interview",
                    "args": {},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    if action.action == "handoff_currency":
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_to_currency",
                    "args": {},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "end_conversation",
                "args": {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def aimessage_to_currency_action(message: AIMessage) -> CurrencyAction:
    """Mapeia AIMessage scriptada (testes) para CurrencyAction."""
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        name = calls[0].get("name")
        args = calls[0].get("args") or {}
        if name == "get_currency_quote":
            return CurrencyAction(
                action="quote",
                currency_code=str(args.get("currency_code") or ""),
            )
        if name == "handoff_to_credit":
            return CurrencyAction(action="handoff_credit")
        if name == "end_conversation":
            return CurrencyAction(action="end")
    return CurrencyAction(action="ask_currency")


def aimessage_to_credit_action(message: AIMessage) -> CreditAction:
    """Mapeia AIMessage scriptada (testes) para CreditAction."""
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        name = calls[0].get("name")
        args = calls[0].get("args") or {}
        if name == "get_credit_limit":
            return CreditAction(action="consult_limit")
        if name == "request_limit_increase":
            return CreditAction(
                action="request_increase",
                requested_limit=str(args.get("requested_limit") or ""),
            )
        if name == "handoff_to_interview":
            return CreditAction(action="handoff_interview")
        if name == "handoff_to_currency":
            return CreditAction(action="handoff_currency")
        if name == "end_conversation":
            return CreditAction(action="end")
    content = message.content if isinstance(message.content, str) else ""
    if content.strip():
        return CreditAction(action="speak", reply=content)
    return CreditAction(action="ask_amount")


def wrap_model_for_actions(model: Any, schema: type[BaseModel], mapper: Any) -> Any:
    """Garante with_structured_output mesmo no ScriptedChatModel de testes."""
    if hasattr(model, "with_structured_output") and not hasattr(
        model, "responses"
    ):
        return model

    parent = model

    class _Structured:
        def invoke(self, messages: Any, config: Any = None) -> Any:
            if hasattr(parent, "on_invoke") and parent.on_invoke is not None:
                parent.on_invoke(messages)
            if getattr(parent, "responses", None):
                message = parent.responses.pop(0)
            else:
                message = AIMessage(content="")
            if isinstance(message, schema):
                return message
            if isinstance(message, AIMessage):
                return mapper(message)
            return mapper(AIMessage(content=str(message)))

    class _Wrapper:
        def with_structured_output(self, _schema: Any) -> Any:
            return _Structured()

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return parent.bind_tools(tools, **kwargs) if hasattr(parent, "bind_tools") else parent

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return parent.invoke(*args, **kwargs)

    return _Wrapper()
