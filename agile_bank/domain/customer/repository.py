from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional, Protocol

from agile_bank.domain.customer.model import Customer
from agile_bank.domain.result import Result


class CustomerRepository(Protocol):
    def find_by_cpf_and_birth_date(
        self, cpf: str, birth_date: date
    ) -> Result[Customer]:
        ...

    def find_by_cpf(self, cpf: str) -> Result[Customer]:
        ...

    def update_profile(
        self,
        cpf: str,
        *,
        credit_score: int,
        employment_type: str,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        dependents: int,
        has_debt: str,
    ) -> Result[Customer]:
        ...

    def update_credit_limit(
        self, cpf: str, credit_limit: Decimal
    ) -> Result[Customer]:
        ...
