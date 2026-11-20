from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from agile_bank.agents import copy
from agile_bank.agents.helpers import customer_to_dict, parse_brl_amount
from agile_bank.agents.tools.end import build_end_conversation_tool
from agile_bank.agents.tools.handoff import build_handoff_tool
from agile_bank.agents.tools.session import restart_authentication
from agile_bank.domain.result import Outcome

INTERVIEW_ORDER = [
    "monthly_income",
    "employment_type",
    "monthly_expenses",
    "dependents",
    "has_debt",
]

EMPLOYMENT_TYPES = {"formal", "autônomo", "desempregado"}
DEBT_VALUES = {"sim", "não"}


def _answers(interview: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in interview.items()
        if not str(key).startswith("_")
    }


def _next_missing(interview: Dict[str, Any]) -> Optional[str]:
    answers = _answers(interview)
    for field in INTERVIEW_ORDER:
        if field not in answers:
            return field
    return None


def _parse_answer(field: str, value: Any) -> Any:
    if field in {"monthly_income", "monthly_expenses"}:
        amount = parse_brl_amount(value)
        if amount < 0:
            raise ValueError("valor negativo")
        return str(amount)
    if field == "employment_type":
        text = str(value).strip().lower()
        if text not in EMPLOYMENT_TYPES:
            raise ValueError("emprego inválido")
        return text
    if field == "dependents":
        dependents = int(value)
        if dependents < 0:
            raise ValueError("dependentes negativos")
        return dependents
    if field == "has_debt":
        text = str(value).strip().lower()
        if text in {"nao"}:
            text = "não"
        if text not in DEBT_VALUES:
            raise ValueError("dívida inválida")
        return text
    raise ValueError("campo inválido")


def complete_interview(
    services: Any,
    state: dict,
    interview: Dict[str, Any],
    tool_call_id: str,
) -> Any:
    """Calculate score and hand back to credit when all answers are present."""
    missing = _next_missing(interview)
    if missing is not None:
        return ToolMessage(
            content=f"Entrevista incompleta. Ainda falta o campo {missing}.",
            tool_call_id=tool_call_id,
        )

    customer = state.get("customer") or {}
    cpf = customer.get("cpf")
    if not cpf:
        return restart_authentication(services, "submit_interview", tool_call_id)

    answers = _answers(interview)
    updated = services.update_credit_score(
        cpf,
        Decimal(str(answers["monthly_income"])),
        Decimal(str(answers["monthly_expenses"])),
        answers["employment_type"],
        int(answers["dependents"]),
        answers["has_debt"],
    )
    if updated.outcome in {Outcome.STORAGE_ERROR, Outcome.INTEGRATION_ERROR}:
        services.record_error("update_credit_score", updated.message)
        return Command(
            goto=END,
            update={
                "messages": [
                    ToolMessage(
                        content="Falha técnica ao gravar o score.",
                        tool_call_id=tool_call_id,
                    ),
                    AIMessage(content=copy.GENERIC_RETRY),
                ],
            },
        )
    if not updated.is_success or updated.value is None:
        return ToolMessage(
            content=updated.message or "Não foi possível atualizar o score.",
            tool_call_id=tool_call_id,
        )

    return Command(
        goto="credit",
        graph=Command.PARENT,
        update={
            "customer": customer_to_dict(updated.value),
            "interview": answers,
            "handoff_reason": "score_updated",
            "active_agent": "credit",
            "messages": [
                ToolMessage(
                    content="Score atualizado. Seguindo para reanálise no crédito.",
                    tool_call_id=tool_call_id,
                )
            ],
        },
    )


def build_record_interview_answer_tool(services: Any):
    @tool("record_interview_answer")
    def record_interview_answer(
        field: str,
        value: str,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Registra uma resposta da entrevista financeira."""
        if field not in INTERVIEW_ORDER:
            return ToolMessage(
                content=(
                    "Campo inválido. Use um de: " + ", ".join(INTERVIEW_ORDER)
                ),
                tool_call_id=tool_call_id,
            )
        interview = dict(state.get("interview") or {})
        answers = _answers(interview)
        next_field = _next_missing(interview)
        if field != next_field and field not in answers:
            missing = next_field or field
            return ToolMessage(
                content=(
                    f"Ainda não é a vez de {field}. "
                    f"Próximo campo: {missing}."
                ),
                tool_call_id=tool_call_id,
            )
        try:
            parsed = _parse_answer(field, value)
        except (ValueError, InvalidOperation, TypeError):
            missing = _next_missing(interview) or field
            return ToolMessage(
                content=(
                    f"Resposta inválida para {field}. "
                    f"Próximo campo necessário: {missing}."
                ),
                tool_call_id=tool_call_id,
            )

        interview[field] = parsed
        remaining = _next_missing(interview)
        if remaining is None:
            return complete_interview(services, state, interview, tool_call_id)

        return Command(
            update={
                "interview": interview,
                "messages": [
                    ToolMessage(
                        content=f"Resposta salva. Próximo campo: {remaining}.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return record_interview_answer


def build_submit_interview_tool(services: Any):
    @tool("submit_interview")
    def submit_interview(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Any:
        """Calcula o score com as cinco respostas e devolve o cliente ao crédito."""
        interview = dict(state.get("interview") or {})
        return complete_interview(services, state, interview, tool_call_id)

    return submit_interview


def build_interview_tools(services: Any) -> List[Any]:
    return [
        build_record_interview_answer_tool(services),
        build_submit_interview_tool(services),
        build_handoff_tool(
            name="handoff_to_credit",
            description=(
                "Devolve o cliente que desistiu da entrevista ao crédito, "
                "sem recalcular o score."
            ),
            current="interview",
            destination="credit",
            require_auth=True,
        ),
        build_end_conversation_tool(),
    ]
