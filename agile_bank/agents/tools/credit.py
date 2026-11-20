from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agile_bank.agents import copy
from agile_bank.agents.helpers import (
    format_money,
    parse_brl_amount,
    parse_open_request_timestamp,
)
from agile_bank.agents.tools.end import build_end_conversation_tool
from agile_bank.agents.tools.handoff import build_handoff_tool
from agile_bank.agents.tools.session import restart_authentication
from agile_bank.domain.result import Outcome
from agile_bank.integrations.csv.limit_requests import format_timestamp


def _cpf_from_state(state: dict) -> Optional[str]:
    customer = state.get("customer") or {}
    return customer.get("cpf")


def _with_limit(state: dict, credit_limit: Any) -> Dict[str, Any]:
    return {**(state.get("customer") or {}), "credit_limit": str(credit_limit)}


def build_get_credit_limit_tool(services: Any):
    @tool("get_credit_limit")
    def get_credit_limit(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Consulta o limite de crédito do cliente autenticado."""
        cpf = _cpf_from_state(state)
        if not cpf:
            return restart_authentication(services, "get_credit_limit", tool_call_id)
        result = services.get_credit_limit(cpf)
        if result.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
            services.record_error("get_credit_limit", result.message)
            return Command(
                update={
                    "required_notice": copy.GENERIC_RETRY,
                    "messages": [
                        ToolMessage(
                            content="Falha técnica ao ler o limite.",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        if not result.is_success or result.value is None:
            return ToolMessage(
                content=result.message or "Não foi possível consultar o limite.",
                tool_call_id=tool_call_id,
            )
        notice = copy.LIMIT_CURRENT.format(limit=format_money(result.value))
        return Command(
            update={
                "required_notice": notice,
                "messages": [
                    ToolMessage(
                        content=f"Limite atual: R$ {format_money(result.value)}.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return get_credit_limit


def build_request_limit_increase_tool(services: Any):
    @tool("request_limit_increase")
    def request_limit_increase(
        requested_limit: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Registra e decide um pedido de aumento de limite."""
        cpf = _cpf_from_state(state)
        if not cpf:
            return restart_authentication(
                services, "request_limit_increase", tool_call_id
            )
        try:
            amount = parse_brl_amount(requested_limit)
        except ValueError:
            return ToolMessage(
                content="Valor de limite inválido.",
                tool_call_id=tool_call_id,
            )
        if amount <= 0:
            return ToolMessage(
                content="O limite solicitado precisa ser maior que zero.",
                tool_call_id=tool_call_id,
            )

        current = services.get_credit_limit(cpf)
        if (
            current.is_success
            and current.value is not None
            and amount <= current.value
        ):
            notice = copy.LIMIT_NOT_INCREASE.format(
                limit=format_money(current.value)
            )
            return Command(
                update={
                    "required_notice": notice,
                    "messages": [
                        ToolMessage(
                            content=(
                                "Pedido não é aumento: valor menor ou igual ao "
                                "limite atual. Mensagem ao cliente já preparada."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        result = services.request_limit_increase(cpf, amount)
        if result.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
            services.record_error("request_limit_increase", result.message)
            return Command(
                update={
                    "required_notice": copy.GENERIC_RETRY,
                    "messages": [
                        ToolMessage(
                            content=result.message
                            or (
                                "Falha técnica ao gravar o pedido. "
                                "O limite não foi alterado."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        if not result.is_success or result.value is None:
            return ToolMessage(
                content=result.message or "Não foi possível registrar o pedido.",
                tool_call_id=tool_call_id,
            )

        created = result.value
        if created.status == "aprovado":
            notice = copy.LIMIT_APPROVED.format(limit=format_money(amount))
            return Command(
                update={
                    "customer": _with_limit(state, created.requested_limit),
                    "open_limit_request": None,
                    "interview_accepted": False,
                    "required_notice": notice,
                    "messages": [
                        ToolMessage(
                            content=(
                                "Pedido de aumento aprovado. Novo limite de "
                                f"crédito: R$ {format_money(amount)}."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        notice = copy.LIMIT_REJECTED_OFFER.format(requested=format_money(amount))
        return Command(
            update={
                "open_limit_request": {
                    "cpf": created.cpf,
                    "requested_at": format_timestamp(created.requested_at),
                    "requested_limit": str(created.requested_limit),
                },
                "interview_accepted": False,
                "required_notice": notice,
                "messages": [
                    ToolMessage(
                        content=(
                            "Pedido rejeitado pelo score atual. A mensagem com a "
                            "oferta da entrevista já está preparada."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return request_limit_increase


def build_credit_tools(services: Any) -> List[Any]:
    def interview_guard(state: dict) -> Optional[str]:
        if not state.get("open_limit_request"):
            return (
                "Não há pedido de aumento rejeitado em aberto. "
                "A entrevista só é oferecida após uma rejeição."
            )
        return None

    def interview_extra(_state: dict) -> Dict[str, Any]:
        return {"interview": {}, "interview_accepted": False}

    return [
        build_get_credit_limit_tool(services),
        build_request_limit_increase_tool(services),
        build_handoff_tool(
            name="handoff_to_interview",
            description=(
                "Encaminha o cliente para a entrevista de crédito "
                "após um pedido rejeitado."
            ),
            current="credit",
            destination="interview",
            require_auth=True,
            guard=interview_guard,
            extra=interview_extra,
        ),
        build_handoff_tool(
            name="handoff_to_currency",
            description="Encaminha o cliente ao agente de câmbio.",
            current="credit",
            destination="currency",
            require_auth=True,
        ),
        build_end_conversation_tool(),
    ]


def build_credit_prelude(services: Any):
    def prelude(state: dict) -> Dict[str, Any]:
        update: Dict[str, Any] = {}

        if state.get("handoff_reason") != "score_updated":
            return update

        open_request = state.get("open_limit_request")
        update["handoff_reason"] = None
        if not open_request:
            update["open_limit_request"] = None
            update["interview_accepted"] = False
            return update

        reviewed = services.review_limit_request(
            open_request["cpf"],
            parse_open_request_timestamp(open_request["requested_at"]),
        )
        if reviewed.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
            services.record_error("review_limit_request", reviewed.message)
            update["required_notice"] = copy.GENERIC_RETRY
            return update
        if not reviewed.is_success or reviewed.value is None:
            update["required_notice"] = copy.GENERIC_RETRY
            return update

        update["open_limit_request"] = None
        update["interview_accepted"] = False
        if reviewed.value.status == "aprovado":
            update["customer"] = _with_limit(state, reviewed.value.requested_limit)
            update["required_notice"] = copy.LIMIT_REVIEWED_APPROVED.format(
                limit=format_money(reviewed.value.requested_limit)
            )
        else:
            update["required_notice"] = copy.LIMIT_REVIEWED.format(
                status=reviewed.value.status
            )
        return update

    return prelude
