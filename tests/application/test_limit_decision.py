from decimal import Decimal
from functools import partial
from pathlib import Path

from agile_bank.application.authenticate_customer import authenticate_customer
from agile_bank.application.get_credit_limit import get_credit_limit
from agile_bank.application.get_currency_quote import get_currency_quote
from agile_bank.application.record_error import record_error
from agile_bank.application.request_limit_increase import request_limit_increase
from agile_bank.application.review_limit_request import review_limit_request
from agile_bank.application.traced_services import TracedAppServices
from agile_bank.application.update_credit_score import update_credit_score
from agile_bank.composition import build_services
from agile_bank.domain.customer.model import Customer
from agile_bank.domain.result import Outcome, Result
from agile_bank.integrations.csv.customers import CsvCustomerRepository
from agile_bank.integrations.csv.limit_requests import CsvLimitRequestRepository
from agile_bank.integrations.csv.score_bands import CsvScoreBandRepository
from agile_bank.integrations.currency.awesome_api import AwesomeApiCurrencyGateway
from agile_bank.integrations.logging.error_log import ErrorLog


def _seed(tmp_path: Path, score: int = 600) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "clientes.csv").write_text(
        "cpf,nome_completo,data_nascimento,score_credito,limite_credito,"
        "tipo_emprego,renda_mensal,despesas_fixas,dependentes,possui_dividas\n"
        f"39053344705,Carla Mendes,1992-01-03,{score},7000,formal,8000,3000,0,não\n",
        encoding="utf-8",
    )
    (data / "score_limite.csv").write_text(
        "score_minimo,score_maximo,limite_maximo\n"
        "0,499,5000\n"
        "500,699,10000\n",
        encoding="utf-8",
    )
    (data / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,"
        "novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    return data


def _services(tmp_path: Path, score: int = 600):
    data = _seed(tmp_path, score=score)
    logs = tmp_path / "logs"
    logs.mkdir()
    return build_services(data_dir=data, log_dir=logs)


def test_approved_increase_raises_credit_limit(tmp_path: Path) -> None:
    services = _services(tmp_path)
    created = services.request_limit_increase("39053344705", Decimal("9000"))
    assert created.value is not None
    assert created.value.status == "aprovado"
    assert services.get_credit_limit("39053344705").value == Decimal("9000")


def test_rejected_increase_keeps_credit_limit(tmp_path: Path) -> None:
    services = _services(tmp_path)
    created = services.request_limit_increase("39053344705", Decimal("12000"))
    assert created.value is not None
    assert created.value.status == "rejeitado"
    assert services.get_credit_limit("39053344705").value == Decimal("7000")
    requests = (tmp_path / "data" / "solicitacoes_aumento_limite.csv").read_text(
        encoding="utf-8"
    )
    assert "pendente" not in requests
    assert "rejeitado" in requests


def test_approved_review_raises_credit_limit(tmp_path: Path) -> None:
    services = _services(tmp_path, score=400)
    created = services.request_limit_increase("39053344705", Decimal("9000"))
    assert created.value is not None
    assert created.value.status == "rejeitado"

    services.update_credit_score(
        "39053344705", Decimal("10000"), Decimal("2000"), "formal", 0, "não"
    )
    reviewed = services.review_limit_request(
        "39053344705", created.value.requested_at
    )
    assert reviewed.value is not None
    assert reviewed.value.status == "aprovado"
    assert services.get_credit_limit("39053344705").value == Decimal("9000")


class _FailingLimitCustomerRepository:
    def __init__(self, inner: CsvCustomerRepository) -> None:
        self._inner = inner

    def find_by_cpf_and_birth_date(self, cpf, birth_date):
        return self._inner.find_by_cpf_and_birth_date(cpf, birth_date)

    def find_by_cpf(self, cpf):
        return self._inner.find_by_cpf(cpf)

    def update_profile(self, cpf, **kwargs):
        return self._inner.update_profile(cpf, **kwargs)

    def update_credit_limit(self, cpf, credit_limit) -> Result[Customer]:
        return Result.storage_error("disco cheio")


def test_failed_limit_update_does_not_leave_approved_request(tmp_path: Path) -> None:
    data = _seed(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    customers = _FailingLimitCustomerRepository(
        CsvCustomerRepository(data / "clientes.csv")
    )
    bands = CsvScoreBandRepository(data / "score_limite.csv")
    requests = CsvLimitRequestRepository(data / "solicitacoes_aumento_limite.csv")
    currency = AwesomeApiCurrencyGateway()
    errors = ErrorLog(logs / "errors.jsonl")
    services = TracedAppServices(
        authenticate_customer=partial(authenticate_customer, customers),
        get_credit_limit=partial(get_credit_limit, customers),
        request_limit_increase=partial(
            request_limit_increase, customers, bands, requests
        ),
        review_limit_request=partial(
            review_limit_request, customers, bands, requests
        ),
        update_credit_score=partial(update_credit_score, customers),
        get_currency_quote=partial(get_currency_quote, currency),
        record_error=partial(record_error, errors),
    )

    created = services.request_limit_increase("39053344705", Decimal("9000"))
    assert created.outcome == Outcome.STORAGE_ERROR
    assert services.get_credit_limit("39053344705").value == Decimal("7000")
    content = (data / "solicitacoes_aumento_limite.csv").read_text(encoding="utf-8")
    assert "aprovado" not in content
