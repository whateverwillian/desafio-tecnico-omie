from __future__ import annotations

from typing import Annotated, Any, List, Literal, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import Field

from agile_bank.agents import copy
from agile_bank.agents.helpers import customer_to_dict
from agile_bank.agents.tools.end import build_end_conversation_tool
from agile_bank.agents.tools.handoff import build_handoff_tool
from agile_bank.domain.result import Outcome


def build_authenticate_tool(services: Any):
    @tool("authenticate_customer")
    def authenticate_customer(
        cpf: str,
        birth_date: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        topic: Annotated[
            Optional[Literal["credit", "currency"]],
            Field(
                description=(
                    "Assunto que o cliente já pediu nesta conversa. "
                    "credit para limite, crédito ou aumento; "
                    "currency para cotação, câmbio ou moeda. "
                    "Preencha quando o pedido já estiver na mensagem."
                )
            ),
        ] = None,
    ) -> Any:
        """Autentica o cliente pelo CPF e data de nascimento contra clientes.csv.

        Com topic preenchido, o cliente autenticado segue direto para o assunto.
        """
        existing = state.get("customer") or {}
        if state.get("authenticated") and existing.get("cpf"):
            return ToolMessage(
                content="Cliente já autenticado. Não altere a identidade da sessão.",
                tool_call_id=tool_call_id,
            )

        result = services.authenticate_customer(cpf, birth_date)

        if result.outcome == Outcome.INVALID_INPUT:
            return Command(
                goto=END,
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "Formato inválido de CPF ou data. "
                                "Tentativa não contabilizada."
                            ),
                            tool_call_id=tool_call_id,
                        ),
                        AIMessage(content=copy.AUTH_INVALID),
                    ],
                },
            )

        if result.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
            services.record_error("authenticate_customer", result.message)
            return Command(
                goto=END,
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "Falha técnica ao consultar a base de clientes. "
                                "Tentativa não contabilizada."
                            ),
                            tool_call_id=tool_call_id,
                        ),
                        AIMessage(content=copy.GENERIC_RETRY),
                    ],
                },
            )

        if result.outcome == Outcome.NOT_FOUND:
            attempts = int(state.get("auth_attempts", 0)) + 1
            if attempts >= 3:
                return Command(
                    goto="end",
                    graph=Command.PARENT,
                    update={
                        "auth_attempts": attempts,
                        "authenticated": False,
                        "end_requested": True,
                        "messages": [
                            ToolMessage(
                                content="Terceira falha de autenticação.",
                                tool_call_id=tool_call_id,
                            )
                        ],
                    },
                )
            return Command(
                goto=END,
                update={
                    "auth_attempts": attempts,
                    "authenticated": False,
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Dados não conferem. Falha {attempts} de 3."
                            ),
                            tool_call_id=tool_call_id,
                        ),
                        AIMessage(
                            content=copy.AUTH_FAILED.format(attempt=attempts)
                        ),
                    ],
                },
            )

        assert result.value is not None
        authenticated = {
            "authenticated": True,
            "customer": customer_to_dict(result.value),
        }
        if topic in ("credit", "currency"):
            return Command(
                goto=topic,
                graph=Command.PARENT,
                update={
                    **authenticated,
                    "active_agent": topic,
                    "messages": [
                        ToolMessage(content=f"ok:{topic}", tool_call_id=tool_call_id)
                    ],
                },
            )
        return Command(
            update={
                **authenticated,
                "messages": [
                    ToolMessage(
                        content=(
                            "Cliente autenticado. O campo topic veio vazio. "
                            "Isso não significa que o cliente não pediu nada. "
                            "Leia a mensagem dele: se já pediu limite ou aumento, "
                            "chame handoff_to_credit; se pediu cotação, chame "
                            "handoff_to_currency. Não escreva texto junto do "
                            "handoff. Só pergunte o assunto se ele ainda não "
                            "tiver dito o que quer."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return authenticate_customer


def build_refuse_other_data_tool():
    @tool("refuse_other_customer_data")
    def refuse_other_customer_data(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Recusa pedido de CPF, score ou dados de outro cliente."""
        return Command(
            update={
                "required_notice": copy.DATA_OTHER_REFUSAL,
                "messages": [
                    ToolMessage(
                        content="Pedido de dado alheio recusado. Mensagem preparada.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return refuse_other_customer_data


def build_triage_tools(services: Any) -> List[Any]:
    return [
        build_authenticate_tool(services),
        build_refuse_other_data_tool(),
        build_handoff_tool(
            name="handoff_to_credit",
            description="Encaminha o cliente autenticado ao agente de crédito.",
            current="triage",
            destination="credit",
            require_auth=True,
        ),
        build_handoff_tool(
            name="handoff_to_currency",
            description="Encaminha o cliente autenticado ao agente de câmbio.",
            current="triage",
            destination="currency",
            require_auth=True,
        ),
        build_end_conversation_tool(),
    ]
