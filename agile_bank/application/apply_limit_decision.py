from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.credit.repository import LimitRequestRepository
from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result


def apply_approved_increase(
    customer_repository: CustomerRepository,
    limit_request_repository: LimitRequestRepository,
    *,
    cpf: str,
    requested_at: datetime,
    current_limit: Decimal,
    requested_limit: Decimal,
) -> Result[LimitRequest]:
    """Raise the customer limit first, then append an approved request row."""
    raised = customer_repository.update_credit_limit(cpf, requested_limit)
    if not raised.is_success:
        return Result(outcome=raised.outcome, message=raised.message)

    created = limit_request_repository.append(
        cpf=cpf,
        requested_at=requested_at,
        current_limit=current_limit,
        requested_limit=requested_limit,
        status="aprovado",
    )
    if created.is_success:
        return created

    restored = customer_repository.update_credit_limit(cpf, current_limit)
    if not restored.is_success:
        return Result(
            outcome=restored.outcome,
            message=(
                f"Falha ao gravar o pedido após elevar o limite, e a "
                f"compensação também falhou: {restored.message}"
            ),
        )
    return Result(outcome=created.outcome, message=created.message)


def apply_approved_review(
    customer_repository: CustomerRepository,
    limit_request_repository: LimitRequestRepository,
    *,
    cpf: str,
    requested_at: datetime,
    previous_limit: Decimal,
    requested_limit: Decimal,
) -> Result[LimitRequest]:
    """Update limit first, then flip the existing request to approved."""
    raised = customer_repository.update_credit_limit(cpf, requested_limit)
    if not raised.is_success:
        return Result(outcome=raised.outcome, message=raised.message)

    decided = limit_request_repository.update_status(
        cpf, requested_at, "aprovado"
    )
    if decided.is_success:
        return decided

    restored = customer_repository.update_credit_limit(cpf, previous_limit)
    if not restored.is_success:
        return Result(
            outcome=restored.outcome,
            message=(
                f"Falha ao atualizar o status após elevar o limite, e a "
                f"compensação também falhou: {restored.message}"
            ),
        )
    return Result(outcome=decided.outcome, message=decided.message)
