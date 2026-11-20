from __future__ import annotations

from agile_bank.domain.currency.gateway import CurrencyGateway
from agile_bank.domain.currency.model import Quote
from agile_bank.domain.result import Result
from agile_bank.observability import traced_chain


@traced_chain("application.get_currency_quote")
def get_currency_quote(
    currency_gateway: CurrencyGateway, currency_code: str
) -> Result[Quote]:
    code = (currency_code or "").strip().upper()
    if len(code) != 3 or not code.isalpha():
        return Result.invalid("Informe um código de moeda com 3 letras.")
    return currency_gateway.get_quote(code)
