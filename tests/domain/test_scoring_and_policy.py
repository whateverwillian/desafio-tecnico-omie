from decimal import Decimal

from agile_bank.domain.credit.model import ScoreBand
from agile_bank.domain.credit.policy import limit_status
from agile_bank.domain.credit.scoring import calculate_score

BANDS = [
    ScoreBand(0, 299, Decimal("1000")),
    ScoreBand(300, 499, Decimal("5000")),
    ScoreBand(500, 699, Decimal("10000")),
    ScoreBand(700, 849, Decimal("20000")),
    ScoreBand(850, 1000, Decimal("50000")),
]


def test_interview_score_lands_in_middle_band() -> None:
    score = calculate_score(
        monthly_income=Decimal("10000"),
        monthly_expenses=Decimal("2000"),
        employment_type="formal",
        dependents=0,
        has_debt="não",
    )
    assert 500 <= score <= 699


def test_score_clamps_to_bounds() -> None:
    high = calculate_score(
        monthly_income=Decimal("1000000"),
        monthly_expenses=Decimal("1"),
        employment_type="formal",
        dependents=0,
        has_debt="não",
    )
    assert high == 1000

    low = calculate_score(
        monthly_income=Decimal("0"),
        monthly_expenses=Decimal("10000"),
        employment_type="desempregado",
        dependents=5,
        has_debt="sim",
    )
    assert low == 0


def test_dependents_three_or_more_use_weight_thirty() -> None:
    with_two = calculate_score(
        Decimal("5000"), Decimal("1000"), "formal", 2, "não"
    )
    with_three = calculate_score(
        Decimal("5000"), Decimal("1000"), "formal", 3, "não"
    )
    assert with_two - with_three == 30


def test_limit_status_approves_inside_band() -> None:
    assert (
        limit_status(Decimal("3000"), Decimal("8000"), 600, BANDS)
        == "aprovado"
    )


def test_limit_status_rejects_above_band_ceiling() -> None:
    assert (
        limit_status(Decimal("3000"), Decimal("8000"), 400, BANDS)
        == "rejeitado"
    )


def test_limit_status_rejects_not_higher_than_current() -> None:
    assert (
        limit_status(Decimal("8000"), Decimal("8000"), 600, BANDS)
        == "rejeitado"
    )
