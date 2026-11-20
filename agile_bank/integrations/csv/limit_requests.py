from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.result import Result
from agile_bank.integrations.csv.store import read_rows, write_rows
from agile_bank.observability import traced_integration

LIMIT_REQUEST_FIELDS = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]

BRAZIL_TZ = timezone(timedelta(hours=-3))


def format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=BRAZIL_TZ)
    return value.astimezone(BRAZIL_TZ).isoformat(timespec="seconds")


def parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _row_to_request(row: Dict[str, str]) -> LimitRequest:
    return LimitRequest(
        cpf=row["cpf_cliente"].strip(),
        requested_at=parse_timestamp(row["data_hora_solicitacao"].strip()),
        current_limit=Decimal(row["limite_atual"]),
        requested_limit=Decimal(row["novo_limite_solicitado"]),
        status=row["status_pedido"].strip(),
    )


def _request_to_row(request: LimitRequest) -> Dict[str, str]:
    return {
        "cpf_cliente": request.cpf,
        "data_hora_solicitacao": format_timestamp(request.requested_at),
        "limite_atual": str(request.current_limit),
        "novo_limite_solicitado": str(request.requested_limit),
        "status_pedido": request.status,
    }


class CsvLimitRequestRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        if not self._path.exists():
            write_rows(self._path, LIMIT_REQUEST_FIELDS, [])

    def _read(self) -> Result[List[Dict[str, str]]]:
        try:
            return Result.ok(read_rows(self._path))
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))

    @traced_integration("limit_requests.append")
    def append(
        self,
        cpf: str,
        requested_at: datetime,
        current_limit: Decimal,
        requested_limit: Decimal,
        status: str = "pendente",
    ) -> Result[LimitRequest]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        request = LimitRequest(
            cpf=cpf,
            requested_at=requested_at,
            current_limit=current_limit,
            requested_limit=requested_limit,
            status=status,
        )
        new_rows = list(rows.value) + [_request_to_row(request)]
        try:
            write_rows(self._path, LIMIT_REQUEST_FIELDS, new_rows)
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))
        return Result.ok(request)

    @traced_integration("limit_requests.update_status")
    def update_status(
        self, cpf: str, requested_at: datetime, status: str
    ) -> Result[LimitRequest]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        stamp = format_timestamp(requested_at)
        updated: LimitRequest | None = None
        new_rows: List[Dict[str, str]] = []
        for row in rows.value:
            if (
                row["cpf_cliente"].strip() == cpf
                and row["data_hora_solicitacao"].strip() == stamp
            ):
                request = _row_to_request(row)
                updated = LimitRequest(
                    cpf=request.cpf,
                    requested_at=request.requested_at,
                    current_limit=request.current_limit,
                    requested_limit=request.requested_limit,
                    status=status,
                )
                new_rows.append(_request_to_row(updated))
            else:
                new_rows.append(row)
        if updated is None:
            return Result.missing("Solicitação não encontrada.")
        try:
            write_rows(self._path, LIMIT_REQUEST_FIELDS, new_rows)
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))
        return Result.ok(updated)

    @traced_integration("limit_requests.find")
    def find(
        self, cpf: str, requested_at: datetime
    ) -> Result[LimitRequest]:
        rows = self._read()
        if not rows.is_success:
            return Result.storage_error(rows.message)
        assert rows.value is not None
        stamp = format_timestamp(requested_at)
        for row in rows.value:
            if (
                row["cpf_cliente"].strip() == cpf
                and row["data_hora_solicitacao"].strip() == stamp
            ):
                return Result.ok(_row_to_request(row))
        return Result.missing("Solicitação não encontrada.")
