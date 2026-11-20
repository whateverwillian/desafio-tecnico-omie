from __future__ import annotations

import os
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
from agile_bank.integrations.csv.customers import CsvCustomerRepository
from agile_bank.integrations.csv.limit_requests import CsvLimitRequestRepository
from agile_bank.integrations.csv.score_bands import CsvScoreBandRepository
from agile_bank.integrations.currency.awesome_api import AwesomeApiCurrencyGateway
from agile_bank.integrations.logging.error_log import ErrorLog


def build_services(
    data_dir: Path | None = None,
    log_dir: Path | None = None,
    currency_gateway: object | None = None,
) -> TracedAppServices:
    root = Path(__file__).resolve().parent.parent
    data_path = Path(data_dir or os.environ.get("DATA_DIR", root / "data"))
    log_path = Path(log_dir or os.environ.get("LOG_DIR", root / "logs"))

    customers = CsvCustomerRepository(data_path / "clientes.csv")
    bands = CsvScoreBandRepository(data_path / "score_limite.csv")
    requests = CsvLimitRequestRepository(
        data_path / "solicitacoes_aumento_limite.csv"
    )
    currency = currency_gateway or AwesomeApiCurrencyGateway()
    errors = ErrorLog(log_path / "errors.jsonl")

    return TracedAppServices(
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
