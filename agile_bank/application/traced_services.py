from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agile_bank.agents.tool_config import get_runnable_config
from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.currency.model import Quote
from agile_bank.domain.customer.model import Customer
from agile_bank.domain.result import Outcome, Result


def _invoke(tool: StructuredTool, payload: dict) -> Any:
    config = get_runnable_config()
    if config is not None:
        return tool.invoke(payload, config=config)
    return tool.invoke(payload)


def _result(
    payload: dict, *, parse_value: Callable[[Any], Any] | None = None
) -> Result:
    outcome = Outcome(payload["outcome"])
    message = payload.get("message") or ""
    if outcome != Outcome.SUCCESS:
        return Result(outcome=outcome, message=message)
    value = payload.get("value")
    if parse_value is not None and value is not None:
        value = parse_value(value)
    return Result.ok(value, message=message)


def _dump_result(result: Result, value: Any = None) -> dict:
    return {
        "outcome": result.outcome.value,
        "message": result.message,
        "value": value,
    }


class AuthenticateArgs(BaseModel):
    cpf: str = Field(description="CPF informado pelo cliente")
    birth_date: str = Field(description="Data de nascimento informada")


class CpfArgs(BaseModel):
    cpf: str = Field(description="CPF do cliente autenticado")


class LimitIncreaseArgs(BaseModel):
    cpf: str
    requested_limit: str = Field(description="Novo limite solicitado")


class ReviewLimitArgs(BaseModel):
    cpf: str
    requested_at: str = Field(description="Timestamp ISO da solicitação")


class UpdateScoreArgs(BaseModel):
    cpf: str
    monthly_income: str
    monthly_expenses: str
    employment_type: str
    dependents: int
    has_debt: str


class CurrencyArgs(BaseModel):
    currency_code: str = Field(description="Código ISO da moeda, ex: USD")


class RecordErrorArgs(BaseModel):
    operation: str
    message: str


class TracedAppServices:
    """Casos de uso invocados como StructuredTool para aparecer no LangSmith."""

    def __init__(
        self,
        *,
        authenticate_customer: Callable,
        get_credit_limit: Callable,
        request_limit_increase: Callable,
        review_limit_request: Callable,
        update_credit_score: Callable,
        get_currency_quote: Callable,
        record_error: Callable,
    ) -> None:
        self._authenticate = authenticate_customer
        self._get_limit = get_credit_limit
        self._request_increase = request_limit_increase
        self._review = review_limit_request
        self._update_score = update_credit_score
        self._get_quote = get_currency_quote
        self._record_error = record_error

        self._authenticate_tool = StructuredTool.from_function(
            name="services.authenticate_customer",
            description="Autentica o cliente pelo CPF e data de nascimento.",
            func=self._authenticate_impl,
            args_schema=AuthenticateArgs,
        )
        self._get_limit_tool = StructuredTool.from_function(
            name="services.get_credit_limit",
            description="Consulta o limite de crédito do cliente autenticado.",
            func=self._get_limit_impl,
            args_schema=CpfArgs,
        )
        self._request_increase_tool = StructuredTool.from_function(
            name="services.request_limit_increase",
            description="Registra e decide um pedido de aumento de limite.",
            func=self._request_increase_impl,
            args_schema=LimitIncreaseArgs,
        )
        self._review_tool = StructuredTool.from_function(
            name="services.review_limit_request",
            description="Reavalia um pedido de aumento após atualização de score.",
            func=self._review_impl,
            args_schema=ReviewLimitArgs,
        )
        self._update_score_tool = StructuredTool.from_function(
            name="services.update_credit_score",
            description="Recalcula e grava o score a partir da entrevista.",
            func=self._update_score_impl,
            args_schema=UpdateScoreArgs,
        )
        self._quote_tool = StructuredTool.from_function(
            name="services.get_currency_quote",
            description="Consulta a cotação de uma moeda em reais.",
            func=self._quote_impl,
            args_schema=CurrencyArgs,
        )
        self._record_error_tool = StructuredTool.from_function(
            name="services.record_error",
            description="Registra erro técnico para análise posterior.",
            func=self._record_error_impl,
            args_schema=RecordErrorArgs,
        )

    def _authenticate_impl(self, cpf: str, birth_date: str) -> dict:
        result = self._authenticate(cpf, birth_date)
        value = None
        if result.is_success and result.value is not None:
            customer = result.value
            value = {
                "cpf": customer.cpf,
                "full_name": customer.full_name,
                "birth_date": customer.birth_date.isoformat(),
                "credit_score": customer.credit_score,
                "credit_limit": str(customer.credit_limit),
                "employment_type": customer.employment_type,
                "monthly_income": (
                    str(customer.monthly_income)
                    if customer.monthly_income is not None
                    else None
                ),
                "monthly_expenses": (
                    str(customer.monthly_expenses)
                    if customer.monthly_expenses is not None
                    else None
                ),
                "dependents": customer.dependents,
                "has_debt": customer.has_debt,
            }
        return _dump_result(result, value)

    def _get_limit_impl(self, cpf: str) -> dict:
        result = self._get_limit(cpf)
        value = str(result.value) if result.value is not None else None
        return _dump_result(result, value)

    def _request_increase_impl(self, cpf: str, requested_limit: str) -> dict:
        result = self._request_increase(cpf, Decimal(requested_limit))
        value = None
        if result.is_success and result.value is not None:
            req = result.value
            value = {
                "cpf": req.cpf,
                "requested_at": req.requested_at.isoformat(),
                "current_limit": str(req.current_limit),
                "requested_limit": str(req.requested_limit),
                "status": req.status,
            }
        return _dump_result(result, value)

    def _review_impl(self, cpf: str, requested_at: str) -> dict:
        result = self._review(cpf, datetime.fromisoformat(requested_at))
        value = None
        if result.is_success and result.value is not None:
            req = result.value
            value = {
                "cpf": req.cpf,
                "requested_at": req.requested_at.isoformat(),
                "current_limit": str(req.current_limit),
                "requested_limit": str(req.requested_limit),
                "status": req.status,
            }
        return _dump_result(result, value)

    def _update_score_impl(
        self,
        cpf: str,
        monthly_income: str,
        monthly_expenses: str,
        employment_type: str,
        dependents: int,
        has_debt: str,
    ) -> dict:
        result = self._update_score(
            cpf,
            Decimal(monthly_income),
            Decimal(monthly_expenses),
            employment_type,
            dependents,
            has_debt,
        )
        value = None
        if result.is_success and result.value is not None:
            customer = result.value
            value = {
                "cpf": customer.cpf,
                "full_name": customer.full_name,
                "birth_date": customer.birth_date.isoformat(),
                "credit_score": customer.credit_score,
                "credit_limit": str(customer.credit_limit),
                "employment_type": customer.employment_type,
                "monthly_income": (
                    str(customer.monthly_income)
                    if customer.monthly_income is not None
                    else None
                ),
                "monthly_expenses": (
                    str(customer.monthly_expenses)
                    if customer.monthly_expenses is not None
                    else None
                ),
                "dependents": customer.dependents,
                "has_debt": customer.has_debt,
            }
        return _dump_result(result, value)

    def _quote_impl(self, currency_code: str) -> dict:
        result = self._get_quote(currency_code)
        value = None
        if result.is_success and result.value is not None:
            quote = result.value
            value = {
                "currency_code": quote.currency_code,
                "rate_brl": str(quote.rate_brl),
                "quoted_at": quote.quoted_at.isoformat(),
            }
        return _dump_result(result, value)

    def _record_error_impl(self, operation: str, message: str) -> dict:
        self._record_error(operation, message)
        return {"outcome": Outcome.SUCCESS.value, "message": "", "value": True}

    def authenticate_customer(self, cpf: str, birth_date: str) -> Result[Customer]:
        payload = _invoke(
            self._authenticate_tool, {"cpf": cpf, "birth_date": birth_date}
        )

        def parse(value: dict) -> Customer:
            return Customer(
                cpf=value["cpf"],
                full_name=value["full_name"],
                birth_date=datetime.fromisoformat(value["birth_date"]).date(),
                credit_score=int(value["credit_score"]),
                credit_limit=Decimal(value["credit_limit"]),
                employment_type=value.get("employment_type"),
                monthly_income=(
                    Decimal(value["monthly_income"])
                    if value.get("monthly_income")
                    else None
                ),
                monthly_expenses=(
                    Decimal(value["monthly_expenses"])
                    if value.get("monthly_expenses")
                    else None
                ),
                dependents=value.get("dependents"),
                has_debt=value.get("has_debt"),
            )

        return _result(payload, parse_value=parse)

    def get_credit_limit(self, cpf: str) -> Result[Decimal]:
        payload = _invoke(self._get_limit_tool, {"cpf": cpf})
        return _result(payload, parse_value=Decimal)

    def request_limit_increase(
        self, cpf: str, requested_limit: Decimal
    ) -> Result[LimitRequest]:
        payload = _invoke(
            self._request_increase_tool,
            {"cpf": cpf, "requested_limit": str(requested_limit)},
        )

        def parse(value: dict) -> LimitRequest:
            return LimitRequest(
                cpf=value["cpf"],
                requested_at=datetime.fromisoformat(value["requested_at"]),
                current_limit=Decimal(value["current_limit"]),
                requested_limit=Decimal(value["requested_limit"]),
                status=value["status"],
            )

        return _result(payload, parse_value=parse)

    def review_limit_request(
        self, cpf: str, requested_at: datetime
    ) -> Result[LimitRequest]:
        payload = _invoke(
            self._review_tool,
            {"cpf": cpf, "requested_at": requested_at.isoformat()},
        )

        def parse(value: dict) -> LimitRequest:
            return LimitRequest(
                cpf=value["cpf"],
                requested_at=datetime.fromisoformat(value["requested_at"]),
                current_limit=Decimal(value["current_limit"]),
                requested_limit=Decimal(value["requested_limit"]),
                status=value["status"],
            )

        return _result(payload, parse_value=parse)

    def update_credit_score(
        self,
        cpf: str,
        monthly_income: Decimal,
        monthly_expenses: Decimal,
        employment_type: str,
        dependents: int,
        has_debt: str,
    ) -> Result[Customer]:
        payload = _invoke(
            self._update_score_tool,
            {
                "cpf": cpf,
                "monthly_income": str(monthly_income),
                "monthly_expenses": str(monthly_expenses),
                "employment_type": employment_type,
                "dependents": dependents,
                "has_debt": has_debt,
            },
        )

        def parse(value: dict) -> Customer:
            return Customer(
                cpf=value["cpf"],
                full_name=value["full_name"],
                birth_date=datetime.fromisoformat(value["birth_date"]).date(),
                credit_score=int(value["credit_score"]),
                credit_limit=Decimal(value["credit_limit"]),
                employment_type=value.get("employment_type"),
                monthly_income=(
                    Decimal(value["monthly_income"])
                    if value.get("monthly_income")
                    else None
                ),
                monthly_expenses=(
                    Decimal(value["monthly_expenses"])
                    if value.get("monthly_expenses")
                    else None
                ),
                dependents=value.get("dependents"),
                has_debt=value.get("has_debt"),
            )

        return _result(payload, parse_value=parse)

    def get_currency_quote(self, currency_code: str) -> Result[Quote]:
        payload = _invoke(
            self._quote_tool, {"currency_code": currency_code}
        )

        def parse(value: dict) -> Quote:
            return Quote(
                currency_code=value["currency_code"],
                rate_brl=Decimal(value["rate_brl"]),
                quoted_at=datetime.fromisoformat(value["quoted_at"]),
            )

        return _result(payload, parse_value=parse)

    def record_error(self, operation: str, message: str) -> None:
        _invoke(
            self._record_error_tool,
            {"operation": operation, "message": message},
        )
