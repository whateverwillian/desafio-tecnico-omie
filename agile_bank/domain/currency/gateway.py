from __future__ import annotations

from typing import Protocol

from agile_bank.domain.currency.model import Quote
from agile_bank.domain.result import Result


class CurrencyGateway(Protocol):
    def get_quote(self, currency_code: str) -> Result[Quote]:
        ...
