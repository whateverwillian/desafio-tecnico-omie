from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from agile_bank.domain.result import Outcome
from agile_bank.integrations.csv.customers import (
    CUSTOMER_FIELDS,
    CsvCustomerRepository,
)
from agile_bank.integrations.csv.limit_requests import (
    CsvLimitRequestRepository,
    format_timestamp,
)
from agile_bank.integrations.csv.score_bands import CsvScoreBandRepository
from agile_bank.integrations.csv.store import write_rows


def _seed_customers(path: Path) -> None:
    write_rows(
        path,
        CUSTOMER_FIELDS,
        [
            {
                "cpf": "11144477735",
                "nome_completo": "Bruno Lima",
                "data_nascimento": "1985-08-20",
                "score_credito": "400",
                "limite_credito": "3000",
                "tipo_emprego": "autônomo",
                "renda_mensal": "4000",
                "despesas_fixas": "2500",
                "dependentes": "1",
                "possui_dividas": "sim",
            }
        ],
    )


def _seed_bands(path: Path) -> None:
    write_rows(
        path,
        ["score_minimo", "score_maximo", "limite_maximo"],
        [
            {
                "score_minimo": "300",
                "score_maximo": "499",
                "limite_maximo": "5000",
            },
            {
                "score_minimo": "500",
                "score_maximo": "699",
                "limite_maximo": "10000",
            },
        ],
    )


def test_authenticate_customer(tmp_path: Path) -> None:
    customers_path = tmp_path / "clientes.csv"
    _seed_customers(customers_path)
    repo = CsvCustomerRepository(customers_path)
    result = repo.find_by_cpf_and_birth_date(
        "11144477735", date(1985, 8, 20)
    )
    assert result.is_success
    assert result.value is not None
    assert result.value.full_name == "Bruno Lima"


def test_missing_customer(tmp_path: Path) -> None:
    customers_path = tmp_path / "clientes.csv"
    _seed_customers(customers_path)
    repo = CsvCustomerRepository(customers_path)
    result = repo.find_by_cpf_and_birth_date(
        "11144477735", date(1999, 1, 1)
    )
    assert result.outcome == Outcome.NOT_FOUND


def test_storage_error_when_file_missing(tmp_path: Path) -> None:
    repo = CsvCustomerRepository(tmp_path / "missing.csv")
    result = repo.find_by_cpf("11144477735")
    assert result.outcome == Outcome.STORAGE_ERROR


def test_limit_request_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "solicitacoes.csv"
    repo = CsvLimitRequestRepository(path)
    requested_at = datetime.fromisoformat("2026-11-17T12:00:00-03:00")
    created = repo.append(
        cpf="11144477735",
        requested_at=requested_at,
        current_limit=Decimal("3000"),
        requested_limit=Decimal("8000"),
        status="pendente",
    )
    assert created.is_success
    updated = repo.update_status(
        "11144477735", requested_at, "rejeitado"
    )
    assert updated.is_success
    assert updated.value is not None
    assert updated.value.status == "rejeitado"
    found = repo.find("11144477735", requested_at)
    assert found.is_success
    assert found.value is not None
    assert found.value.status == "rejeitado"
    assert format_timestamp(requested_at).endswith("-03:00")


def test_score_band_lookup(tmp_path: Path) -> None:
    path = tmp_path / "score_limite.csv"
    _seed_bands(path)
    repo = CsvScoreBandRepository(path)
    result = repo.find_for_score(400)
    assert result.is_success
    assert result.value is not None
    assert result.value.max_credit_limit == Decimal("5000")
