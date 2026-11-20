from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class ScoreBand:
    min_score: int
    max_score: int
    max_credit_limit: Decimal


@dataclass(frozen=True)
class LimitRequest:
    cpf: str
    requested_at: datetime
    current_limit: Decimal
    requested_limit: Decimal
    status: str
