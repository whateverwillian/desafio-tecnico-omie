from __future__ import annotations

from datetime import datetime

from agile_bank.application.apply_limit_decision import apply_approved_review
from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.credit.policy import limit_status
from agile_bank.domain.credit.repository import (
    LimitRequestRepository,
    ScoreBandRepository,
)
from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain


@traced_chain("application.review_limit_request")
def review_limit_request(
    customer_repository: CustomerRepository,
    score_band_repository: ScoreBandRepository,
    limit_request_repository: LimitRequestRepository,
    cpf: str,
    requested_at: datetime,
) -> Result[LimitRequest]:
    customer = customer_repository.find_by_cpf(cpf)
    if not customer.is_success:
        return Result(
            outcome=customer.outcome,
            message=customer.message,
        )
    assert customer.value is not None

    existing = limit_request_repository.find(cpf, requested_at)
    if not existing.is_success:
        return Result(
            outcome=existing.outcome,
            message=existing.message,
        )
    assert existing.value is not None

    bands = score_band_repository.list_bands()
    if not bands.is_success:
        return Result(
            outcome=bands.outcome,
            message=bands.message,
        )
    assert bands.value is not None

    status = limit_status(
        current_limit=existing.value.current_limit,
        requested_limit=existing.value.requested_limit,
        score=customer.value.credit_score,
        bands=bands.value,
    )
    if status == "aprovado":
        return apply_approved_review(
            customer_repository,
            limit_request_repository,
            cpf=cpf,
            requested_at=requested_at,
            previous_limit=customer.value.credit_limit,
            requested_limit=existing.value.requested_limit,
        )
    return limit_request_repository.update_status(cpf, requested_at, status)
