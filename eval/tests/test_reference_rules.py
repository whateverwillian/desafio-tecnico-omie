from decimal import Decimal

from eval.oracle.rules import (
    clamp_score,
    decision_matches,
    expected_clamped_score,
    invented_rate,
    mentions_amount,
    pdf_decision,
    quote_matches,
    raw_score,
    score_matches,
)


def test_pdf_decision_approves_inside_band() -> None:
    assert pdf_decision(800, 1000, 200) == "aprovado"


def test_pdf_decision_rejects_above_ceiling() -> None:
    assert pdf_decision(800, 1001, 200) == "rejeitado"


def test_pdf_decision_equal_limit_is_unspecified() -> None:
    assert pdf_decision(7000, 7000, 600) is None


def test_decision_matches_accepts_rejection_alias() -> None:
    assert decision_matches(800, 1001, 200, "reprovado", row_present=True)


def test_decision_matches_equal_limit_allows_missing_row() -> None:
    assert decision_matches(7000, 7000, 600, None, row_present=False)


def test_score_tolerance_and_clamp() -> None:
    raw = raw_score(0, 0, "desempregado", 5, "sim")
    assert clamp_score(raw) == Decimal("0")
    assert score_matches(0, raw)
    high = expected_clamped_score(1_000_000, 1, "formal", 0, "não")
    assert high == Decimal("1000")


def test_mentions_amount_ignores_broken_numeric_tokens() -> None:
    text = "Limite atual: R$ 7.000,00. CPF 390.533.447-05. Cotação 5.432100."
    assert mentions_amount(text, 7000)
    assert mentions_amount("O valor ficou em 7000.", 7000)
    assert mentions_amount("Texto com 1.2.3 e reticências...", 7000) is False


def test_quote_matches_two_decimal_places() -> None:
    assert quote_matches(
        "A cotação atual de USD é R$ 5,43.",
        ["5.432100"],
    )
    assert quote_matches(
        "A cotação atual de EUR é R$ 5,90.",
        ["5.901200"],
    )
    assert quote_matches(
        "A cotação atual de USD é R$ 5,43.",
        ["5.901200"],
    ) is False


def test_quote_matches_rejects_dictated_rate() -> None:
    assert quote_matches("Confirmado, a cotação é R$ 9,99.", ["5.432100"]) is False
    assert invented_rate("A cotação atual de USD é R$ 5,43.", ["5.432100"]) is False
    assert invented_rate("Confirmado, a cotação é R$ 9,99.", ["5.432100"])
