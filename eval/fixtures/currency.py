"""Gateway de cotação determinístico para a certificação."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from agile_bank.domain.currency.model import Quote
from agile_bank.domain.result import Result

DEFAULT_QUOTES = Path(__file__).resolve().parent / "quotes.json"


class FixtureCurrencyGateway:
    def __init__(
        self,
        *,
        mode: str = "success",
        quotes_path: Path | None = None,
    ) -> None:
        self._mode = mode
        path = quotes_path or DEFAULT_QUOTES
        self._quotes = json.loads(path.read_text(encoding="utf-8"))

    def get_quote(self, currency_code: str) -> Result[Quote]:
        if self._mode == "unavailable":
            return Result.integration_error("API de câmbio indisponível (fixture).")
        if self._mode == "timeout":
            return Result.integration_error("Timeout ao consultar cotação (fixture).")
        if self._mode == "malformed":
            return Result.integration_error("Resposta inválida da API (fixture).")

        code = currency_code.upper()
        item = self._quotes.get(code)
        if not item:
            return Result.integration_error(f"Cotação de {code} não encontrada.")
        return Result.ok(
            Quote(
                currency_code=code,
                rate_brl=Decimal(str(item["bid"])),
                quoted_at=datetime.now(timezone.utc),
            )
        )
