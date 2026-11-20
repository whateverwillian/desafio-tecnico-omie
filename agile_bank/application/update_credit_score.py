from __future__ import annotations

from decimal import Decimal

from agile_bank.domain.credit.scoring import calculate_score
from agile_bank.domain.customer.model import Customer
from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain


@traced_chain("application.update_credit_score")
def update_credit_score(
    customer_repository: CustomerRepository,
    cpf: str,
    monthly_income: Decimal,
    monthly_expenses: Decimal,
    employment_type: str,
    dependents: int,
    has_debt: str,
) -> Result[Customer]:
    try:
        score = calculate_score(
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            employment_type=employment_type,
            dependents=dependents,
            has_debt=has_debt,
        )
    except ValueError as exc:
        return Result.invalid(str(exc))

    return customer_repository.update_profile(
        cpf,
        credit_score=score,
        employment_type=employment_type,
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        dependents=dependents,
        has_debt=has_debt,
    )
