from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict

from agile_bank.domain.customer.model import Customer


def customer_to_dict(customer: Customer) -> Dict[str, Any]:
    return {
        "cpf": customer.cpf,
        "full_name": customer.full_name,
        "birth_date": customer.birth_date.isoformat(),
        "credit_score": customer.credit_score,
        "credit_limit": str(customer.credit_limit),
        "employment_type": customer.employment_type,
        "monthly_income": (
            str(customer.monthly_income)
            if customer.monthly_income is not None
            else None
        ),
        "monthly_expenses": (
            str(customer.monthly_expenses)
            if customer.monthly_expenses is not None
            else None
        ),
        "dependents": customer.dependents,
        "has_debt": customer.has_debt,
    }


def format_money(value: Decimal | str | float) -> str:
    amount = Decimal(str(value))
    return f"{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


_THOUSANDS_ONLY = re.compile(r"^\d{1,3}(\.\d{3})+$")


def parse_brl_amount(raw: Any) -> Decimal:
    """Lê valores como 8000, 8.000, 8.000,50, R$ 8000,50 ou 8000.50."""
    text = re.sub(r"[R$\s]", "", str(raw))
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif _THOUSANDS_ONLY.match(text):
        text = text.replace(".", "")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"valor inválido: {raw}") from exc
    if not amount.is_finite():
        raise ValueError(f"valor inválido: {raw}")
    return amount


def parse_open_request_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw)
