from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agile_bank.observability import traced_step


class ErrorLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @traced_step("error_log.record")
    def record(self, operation: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "message": message,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
