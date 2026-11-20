from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import List, Sequence

from agile_bank.domain.credit.model import ScoreBand
from agile_bank.domain.credit.policy import find_band
from agile_bank.domain.result import Result
from agile_bank.integrations.csv.store import read_rows
from agile_bank.observability import traced_integration


class CsvScoreBandRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    @traced_integration("score_bands.list_bands")
    def list_bands(self) -> Result[Sequence[ScoreBand]]:
        try:
            rows = read_rows(self._path)
        except Exception as exc:  # noqa: BLE001
            return Result.storage_error(str(exc))
        bands: List[ScoreBand] = []
        for row in rows:
            bands.append(
                ScoreBand(
                    min_score=int(row["score_minimo"]),
                    max_score=int(row["score_maximo"]),
                    max_credit_limit=Decimal(row["limite_maximo"]),
                )
            )
        return Result.ok(bands)

    @traced_integration("score_bands.find_for_score")
    def find_for_score(self, score: int) -> Result[ScoreBand]:
        bands = self.list_bands()
        if not bands.is_success:
            return Result.storage_error(bands.message)
        assert bands.value is not None
        band = find_band(score, bands.value)
        if band is None:
            return Result.missing("Faixa de score não encontrada.")
        return Result.ok(band)
