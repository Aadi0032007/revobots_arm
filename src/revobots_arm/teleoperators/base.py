"""Structural input contract; integrating controllers may expose feedback hooks."""

from __future__ import annotations
from typing import Any, Protocol


class Teleoperator(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_action(self) -> dict[str, Any]: ...
