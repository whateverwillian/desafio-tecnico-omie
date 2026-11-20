from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class Quote:
    currency_code: str
    rate_brl: Decimal
    quoted_at: datetime
