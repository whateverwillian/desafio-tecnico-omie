from __future__ import annotations

from decimal import Decimal

from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain


@traced_chain("application.get_credit_limit")
def get_credit_limit(
    customer_repository: CustomerRepository, cpf: str
) -> Result[Decimal]:
    customer = customer_repository.find_by_cpf(cpf)
    if not customer.is_success:
        return Result(
            outcome=customer.outcome,
            message=customer.message,
        )
    assert customer.value is not None
    return Result.ok(customer.value.credit_limit)
