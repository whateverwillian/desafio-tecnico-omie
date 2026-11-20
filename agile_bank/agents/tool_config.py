from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_runnable_config: ContextVar[Optional[dict]] = ContextVar(
    "agent_runnable_config", default=None
)


def get_runnable_config() -> Optional[dict]:
    return _runnable_config.get()


@contextmanager
def runnable_config_scope(config: Any) -> Iterator[None]:
    token = _runnable_config.set(config)
    try:
        yield
    finally:
        _runnable_config.reset(token)
