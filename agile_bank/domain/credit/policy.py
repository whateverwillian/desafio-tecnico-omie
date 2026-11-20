from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from agile_bank.domain.credit.model import ScoreBand


def find_band(score: int, bands: Sequence[ScoreBand]) -> ScoreBand | None:
    for band in bands:
        if band.min_score <= score <= band.max_score:
            return band
    return None


def limit_status(
    current_limit: Decimal,
    requested_limit: Decimal,
    score: int,
    bands: Sequence[ScoreBand],
) -> str:
    band = find_band(score, bands)
    if band is None:
        return "rejeitado"
    if requested_limit <= current_limit:
        return "rejeitado"
    if requested_limit > band.max_credit_limit:
        return "rejeitado"
    return "aprovado"
