from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import httpx

from agile_bank.domain.currency.model import Quote
from agile_bank.domain.result import Result
from agile_bank.observability import traced_integration


class AwesomeApiCurrencyGateway:
    def __init__(
        self,
        base_url: str = "https://economia.awesomeapi.com.br/json/last",
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @traced_integration("awesome_api.get_quote")
    def get_quote(self, currency_code: str) -> Result[Quote]:
        code = currency_code.upper()
        url = f"{self._base_url}/{code}-BRL"
        try:
            response = httpx.get(url, timeout=self._timeout)
        except httpx.HTTPError as exc:
            return Result.integration_error(str(exc))

        if response.status_code != 200:
            return Result.integration_error(
                f"AwesomeAPI retornou status {response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError:
            return Result.integration_error("Resposta inválida da AwesomeAPI.")

        key = f"{code}BRL"
        item = payload.get(key)
        if not item or "bid" not in item:
            return Result.integration_error(
                f"Cotação de {code} não encontrada."
            )

        try:
            rate = Decimal(str(item["bid"]))
        except (InvalidOperation, TypeError):
            return Result.integration_error("Valor de cotação inválido.")

        return Result.ok(
            Quote(
                currency_code=code,
                rate_brl=rate,
                quoted_at=datetime.now(timezone.utc),
            )
        )
