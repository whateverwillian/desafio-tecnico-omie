from __future__ import annotations

from decimal import Decimal

EMPLOYMENT_WEIGHTS = {
    "formal": 300,
    "autônomo": 200,
    "desempregado": 0,
}

DEPENDENT_WEIGHTS = {
    0: 100,
    1: 80,
    2: 60,
}

INCOME_WEIGHT = Decimal("30")
DEBT_WEIGHTS = {
    "sim": -100,
    "não": 100,
}


def calculate_score(
    monthly_income: Decimal,
    monthly_expenses: Decimal,
    employment_type: str,
    dependents: int,
    has_debt: str,
) -> int:
    if employment_type not in EMPLOYMENT_WEIGHTS:
        raise ValueError(f"Tipo de emprego inválido: {employment_type}")
    if has_debt not in DEBT_WEIGHTS:
        raise ValueError(f"Valor de dívidas inválido: {has_debt}")

    income_component = (
        (monthly_income / (monthly_expenses + Decimal("1"))) * INCOME_WEIGHT
    )
    employment_component = Decimal(EMPLOYMENT_WEIGHTS[employment_type])
    if dependents >= 3:
        dependents_component = Decimal("30")
    else:
        dependents_component = Decimal(DEPENDENT_WEIGHTS[dependents])
    debt_component = Decimal(DEBT_WEIGHTS[has_debt])

    raw = (
        income_component
        + employment_component
        + dependents_component
        + debt_component
    )
    score = int(raw)
    return max(0, min(1000, score))
