from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, Protocol, Sequence

from agile_bank.domain.credit.model import LimitRequest, ScoreBand
from agile_bank.domain.result import Result


class ScoreBandRepository(Protocol):
    def list_bands(self) -> Result[Sequence[ScoreBand]]:
        ...

    def find_for_score(self, score: int) -> Result[ScoreBand]:
        ...


class LimitRequestRepository(Protocol):
    def append(
        self,
        cpf: str,
        requested_at: datetime,
        current_limit: Decimal,
        requested_limit: Decimal,
        status: str = "pendente",
    ) -> Result[LimitRequest]:
        ...

    def update_status(
        self, cpf: str, requested_at: datetime, status: str
    ) -> Result[LimitRequest]:
        ...

    def find(
        self, cpf: str, requested_at: datetime
    ) -> Result[LimitRequest]:
        ...
