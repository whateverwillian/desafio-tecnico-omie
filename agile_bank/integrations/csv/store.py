from __future__ import annotations

import csv
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Sequence

from langsmith import traceable

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


@traceable(name="csv.read_rows", run_type="tool")
def read_rows(path: Path) -> List[Dict[str, str]]:
    lock = _lock_for(path)
    with lock:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]


@traceable(name="csv.write_rows", run_type="tool")
def write_rows(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]
) -> None:
    lock = _lock_for(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            tmp_name = tmp.name
        os.replace(tmp_name, path)
