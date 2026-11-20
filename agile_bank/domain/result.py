from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Outcome(str, Enum):
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    STORAGE_ERROR = "storage_error"
    INTEGRATION_ERROR = "integration_error"


@dataclass(frozen=True)
class Result(Generic[T]):
    outcome: Outcome
    value: Optional[T] = None
    message: str = ""

    @classmethod
    def ok(cls, value: T, message: str = "") -> Result[T]:
        return cls(outcome=Outcome.SUCCESS, value=value, message=message)

    @classmethod
    def invalid(cls, message: str) -> Result[T]:
        return cls(outcome=Outcome.INVALID_INPUT, message=message)

    @classmethod
    def missing(cls, message: str) -> Result[T]:
        return cls(outcome=Outcome.NOT_FOUND, message=message)

    @classmethod
    def storage_error(cls, message: str) -> Result[T]:
        return cls(outcome=Outcome.STORAGE_ERROR, message=message)

    @classmethod
    def integration_error(cls, message: str) -> Result[T]:
        return cls(outcome=Outcome.INTEGRATION_ERROR, message=message)

    @property
    def is_success(self) -> bool:
        return self.outcome == Outcome.SUCCESS
