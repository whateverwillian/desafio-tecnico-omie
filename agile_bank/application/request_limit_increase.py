from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from agile_bank.application.apply_limit_decision import apply_approved_increase
from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.credit.policy import limit_status
from agile_bank.domain.credit.repository import (
    LimitRequestRepository,
    ScoreBandRepository,
)
from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain

BRAZIL_TZ = timezone(timedelta(hours=-3))


@traced_chain("application.request_limit_increase")
def request_limit_increase(
    customer_repository: CustomerRepository,
    score_band_repository: ScoreBandRepository,
    limit_request_repository: LimitRequestRepository,
    cpf: str,
    requested_limit: Decimal,
    requested_at: datetime | None = None,
) -> Result[LimitRequest]:
    customer = customer_repository.find_by_cpf(cpf)
    if not customer.is_success:
        return Result(
            outcome=customer.outcome,
            message=customer.message,
        )
    assert customer.value is not None

    bands = score_band_repository.list_bands()
    if not bands.is_success:
        return Result(
            outcome=bands.outcome,
            message=bands.message,
        )
    assert bands.value is not None

    stamp = requested_at or datetime.now(BRAZIL_TZ)
    status = limit_status(
        current_limit=customer.value.credit_limit,
        requested_limit=requested_limit,
        score=customer.value.credit_score,
        bands=bands.value,
    )

    if status == "aprovado":
        return apply_approved_increase(
            customer_repository,
            limit_request_repository,
            cpf=cpf,
            requested_at=stamp,
            current_limit=customer.value.credit_limit,
            requested_limit=requested_limit,
        )

    return limit_request_repository.append(
        cpf=cpf,
        requested_at=stamp,
        current_limit=customer.value.credit_limit,
        requested_limit=requested_limit,
        status=status,
    )
