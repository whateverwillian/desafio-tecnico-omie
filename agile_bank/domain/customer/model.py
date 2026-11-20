from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from agile_bank.domain.result import Result


@dataclass(frozen=True)
class Cpf:
    value: str

    @classmethod
    def parse(cls, raw: str) -> Result[Cpf]:
        digits = re.sub(r"\D", "", raw or "")
        if len(digits) != 11:
            return Result.invalid("CPF deve ter 11 dígitos.")
        if digits == digits[0] * 11:
            return Result.invalid("CPF inválido.")
        if not cls._check_digits(digits):
            return Result.invalid("CPF inválido.")
        return Result.ok(cls(value=digits))

    @staticmethod
    def _check_digits(digits: str) -> bool:
        nums = [int(c) for c in digits]

        def verifier(slice_end: int, weights_start: int) -> int:
            total = sum(
                nums[i] * (weights_start - i) for i in range(slice_end)
            )
            remainder = total % 11
            return 0 if remainder < 2 else 11 - remainder

        return nums[9] == verifier(9, 10) and nums[10] == verifier(10, 11)


@dataclass(frozen=True)
class BirthDate:
    value: date

    @classmethod
    def parse(cls, raw: str) -> Result[BirthDate]:
        text = (raw or "").strip()
        formats = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d")
        for fmt in formats:
            try:
                parsed = datetime.strptime(text, fmt).date()
                return Result.ok(cls(value=parsed))
            except ValueError:
                continue
        return Result.invalid(
            "Data de nascimento inválida. Use DD/MM/AAAA."
        )


@dataclass(frozen=True)
class Customer:
    cpf: str
    full_name: str
    birth_date: date
    credit_score: int
    credit_limit: Decimal
    employment_type: Optional[str] = None
    monthly_income: Optional[Decimal] = None
    monthly_expenses: Optional[Decimal] = None
    dependents: Optional[int] = None
    has_debt: Optional[str] = None
