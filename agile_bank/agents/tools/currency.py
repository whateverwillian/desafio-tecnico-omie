from __future__ import annotations

from typing import Annotated, Any, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agile_bank.agents import copy
from agile_bank.agents.helpers import format_money
from agile_bank.agents.tools.end import build_end_conversation_tool
from agile_bank.agents.tools.handoff import build_handoff_tool
from agile_bank.domain.result import Outcome


def build_get_currency_quote_tool(services: Any):
    @tool("get_currency_quote")
    def get_currency_quote(
        currency_code: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Consulta a cotação de uma moeda em reais via API externa."""
        code = (currency_code or "").strip().upper()
        if len(code) != 3 or not code.isalpha():
            return ToolMessage(
                content="Código de moeda inválido: use três letras, como USD.",
                tool_call_id=tool_call_id,
            )

        result = services.get_currency_quote(code)
        if result.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
            services.record_error("get_currency_quote", result.message)
            return Command(
                update={
                    "required_notice": copy.GENERIC_RETRY,
                    "messages": [
                        ToolMessage(
                            content="Falha técnica ao consultar a cotação.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        if not result.is_success or result.value is None:
            return ToolMessage(
                content=result.message or "Não foi possível obter a cotação.",
                tool_call_id=tool_call_id,
            )

        quote = result.value
        line = copy.CURRENCY_QUOTE.format(
            code=quote.currency_code,
            rate=format_money(quote.rate_brl),
        )
        pending = state.get("required_notice") or ""
        if pending.endswith(copy.CURRENCY_CLOSE):
            earlier = pending[: -len(copy.CURRENCY_CLOSE)].strip()
            line = f"{earlier} {line}"
        return Command(
            update={
                "required_notice": f"{line} {copy.CURRENCY_CLOSE}",
                "messages": [
                    ToolMessage(
                        content=(
                            f"{line} A mensagem com o valor e o fechamento "
                            "já está preparada."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return get_currency_quote


def build_currency_tools(services: Any) -> List[Any]:
    return [
        build_get_currency_quote_tool(services),
        build_handoff_tool(
            name="handoff_to_credit",
            description=(
                "Continua o mesmo atendimento no assunto de limite de crédito, "
                "sem avisar o cliente de uma troca."
            ),
            current="currency",
            destination="credit",
            require_auth=True,
        ),
        build_end_conversation_tool(),
    ]
