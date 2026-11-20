from __future__ import annotations

from agile_bank.domain.customer.model import BirthDate, Cpf, Customer
from agile_bank.domain.customer.repository import CustomerRepository
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain


@traced_chain("application.authenticate_customer")
def authenticate_customer(
    customer_repository: CustomerRepository,
    cpf_raw: str,
    birth_raw: str,
) -> Result[Customer]:
    cpf = Cpf.parse(cpf_raw)
    if not cpf.is_success:
        return Result.invalid(cpf.message)
    birth = BirthDate.parse(birth_raw)
    if not birth.is_success:
        return Result.invalid(birth.message)
    assert cpf.value is not None
    assert birth.value is not None
    return customer_repository.find_by_cpf_and_birth_date(
        cpf.value.value, birth.value.value
    )
