from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional

from agile_bank.domain.customer.model import Customer
from agile_bank.domain.result import Result
from agile_bank.integrations.csv.store import read_rows, write_rows
from agile_bank.observability import traced_integration

CUSTOMER_FIELDS = [
    "cpf",
    "nome_completo",
    "data_nascimento",
    "score_credito",
    "limite_credito",
    "tipo_emprego",
    "renda_mensal",
    "despesas_fixas",
    "dependentes",
    "possui_dividas",
]


def _optional_decimal(raw: str) -> Optional[Decimal]:
    text = (raw or "").strip()
    if not text:
        return None
    return Decimal(text)


def _optional_int(raw: str) -> Optional[int]:
    text = (raw or "").strip()
    if not text:
        return None
    return int(text)


def _optional_str(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    return text or None


def _row_to_customer(row: Dict[str, str]) -> Customer:
    return Customer(
        cpf=row["cpf"].strip(),
        full_name=row["nome_completo"].strip(),
        birth_date=datetime.strptime(
            row["data_nascimento"].strip(), "%Y-%m-%d"
        ).date(),
        credit_score=int(row["score_credito"]),
        credit_limit=Decimal(row["limite_credito"]),
        employment_type=_optional_str(row.get("tipo_emprego", "")),
        monthly_income=_optional_decimal(row.get("renda_mensal", "")),
        monthly_expenses=_optional_decimal(row.get("despesas_fixas", "")),
        dependents=_optional_int(row.get("dependentes", "")),
        has_debt=_optional_str(row.get("possui_dividas", "")),
    )


def _customer_to_row(customer: Customer) -> Dict[str, str]:
    return {
        "cpf": customer.cpf,
        "nome_completo": customer.full_name,
        "data_nascimento": customer.birth_date.isoformat(),
        "score_credito": str(customer.credit_score),
        "limite_credito": str(customer.credit_limit),
        "tipo_emprego": customer.employment_type or "",
        "renda_mensal": (
            str(customer.monthly_income)
            if customer.monthly_income is not None
            else ""
        ),
        "despesas_fixas": (
            str(customer.monthly_expenses)
            if customer.monthly_expenses is not None
            else ""
        ),
        "dependentes": (
            str(customer.dependents)
            if customer.dependents is not None
            else ""
        ),
        "possui_dividas": customer.has_debt or "",
    }


class CsvCustomerRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> Result[List[Dict[str, str]]]:
        try:
            return Result.ok(read_rows(self._path))
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))

    @traced_integration("customers.find_by_cpf_and_birth_date")
    def find_by_cpf_and_birth_date(
        self, cpf: str, birth_date: date
    ) -> Result[Customer]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        for row in rows.value:
            if row["cpf"].strip() != cpf:
                continue
            customer = _row_to_customer(row)
            if customer.birth_date == birth_date:
                return Result.ok(customer)
        return Result.missing("Cliente não encontrado.")

    @traced_integration("customers.find_by_cpf")
    def find_by_cpf(self, cpf: str) -> Result[Customer]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        for row in rows.value:
            if row["cpf"].strip() == cpf:
                return Result.ok(_row_to_customer(row))
        return Result.missing("Cliente não encontrado.")

    @traced_integration("customers.update_profile")
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
        return self._replace(
            cpf,
            lambda customer: replace(
                customer,
                credit_score=credit_score,
                employment_type=employment_type,
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                dependents=dependents,
                has_debt=has_debt,
            ),
        )

    @traced_integration("customers.update_credit_limit")
    def update_credit_limit(
        self, cpf: str, credit_limit: Decimal
    ) -> Result[Customer]:
        return self._replace(
            cpf, lambda customer: replace(customer, credit_limit=credit_limit)
        )

    def _replace(
        self, cpf: str, change: Callable[[Customer], Customer]
    ) -> Result[Customer]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        updated: Optional[Customer] = None
        new_rows: List[Dict[str, str]] = []
        for row in rows.value:
            if row["cpf"].strip() != cpf:
                new_rows.append(row)
                continue
            updated = change(_row_to_customer(row))
            new_rows.append(_customer_to_row(updated))
        if updated is None:
            return Result.missing("Cliente não encontrado.")
        try:
            write_rows(self._path, CUSTOMER_FIELDS, new_rows)
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))
        return Result.ok(updated)
