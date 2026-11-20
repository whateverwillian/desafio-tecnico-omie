from agile_bank.domain.customer.model import BirthDate, Cpf
from agile_bank.domain.result import Outcome


def test_cpf_accepts_punctuation() -> None:
    result = Cpf.parse("529.982.247-25")
    assert result.is_success
    assert result.value is not None
    assert result.value.value == "52998224725"


def test_cpf_rejects_invalid_check_digits() -> None:
    result = Cpf.parse("52998224724")
    assert result.outcome == Outcome.INVALID_INPUT


def test_cpf_rejects_repeated_digits() -> None:
    result = Cpf.parse("11111111111")
    assert result.outcome == Outcome.INVALID_INPUT


def test_birth_date_formats() -> None:
    for raw in ("12/05/1990", "12-05-1990", "1990-05-12"):
        result = BirthDate.parse(raw)
        assert result.is_success
        assert result.value is not None
        assert result.value.value.isoformat() == "1990-05-12"


def test_birth_date_rejects_impossible_date() -> None:
    result = BirthDate.parse("31/02/1990")
    assert result.outcome == Outcome.INVALID_INPUT
