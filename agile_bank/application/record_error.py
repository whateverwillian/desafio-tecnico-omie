from __future__ import annotations

from typing import Protocol

from agile_bank.observability import traced_step


class ErrorRecorder(Protocol):
    def record(self, operation: str, message: str) -> None:
        ...


@traced_step("record_error")
def record_error(
    error_recorder: ErrorRecorder, operation: str, message: str
) -> None:
    error_recorder.record(operation, message)
