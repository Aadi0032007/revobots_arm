"""Lazy dataclass adapter registration, shared by robots and teleoperators."""

from __future__ import annotations
from dataclasses import fields
from importlib import import_module
from typing import Any


def instantiate(cls: type, spec: dict[str, Any], kind: str) -> Any:
    accepted = {f.name for f in fields(cls)}
    unknown = sorted(set(spec) - accepted)
    if unknown:
        raise ValueError(
            f"{kind}: '{cls.__name__}' does not accept {unknown}. Accepted keys: {sorted(accepted)}"
        )
    return cls(**spec)


class DeviceRegistry:
    """Register config/driver import paths without importing optional hardware SDKs.

    Registration is explicit at application startup. Duplicate names are rejected.
    Import errors from selected adapters propagate rather than silently falling back.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, tuple[str, str]] = {}
        self._adapters: dict[str, str] = {}

    def register(self, name: str, config: str, implementation: str) -> None:
        if not name or name in self:
            raise ValueError(f"Duplicate or empty {self.kind} registration: {name!r}")
        for path in (config, implementation):
            if ":" not in path:
                raise ValueError("Adapter paths must use module:attribute")
        self._entries[name] = (config, implementation)

    def register_adapter(self, name: str, adapter: str) -> None:
        if not name or name in self:
            raise ValueError(f"Duplicate or empty {self.kind} registration: {name!r}")
        if ":" not in adapter:
            raise ValueError("Adapter paths must use module:attribute")
        self._adapters[name] = adapter

    def __contains__(self, name: str) -> bool:
        return name in self._entries or name in self._adapters

    @staticmethod
    def _resolve(path: str) -> Any:
        module, attribute = path.split(":", 1)
        return getattr(import_module(module), attribute)

    def config(self, name: str, spec: dict[str, Any]) -> Any:
        if name in self._adapters:
            return self._resolve(self._adapters[name]).config(spec)
        cls = self._resolve(self._entries[name][0])
        return instantiate(
            cls, {k: v for k, v in spec.items() if k != "type"}, self.kind
        )

    def create(self, name: str, config: Any) -> Any:
        if name in self._adapters:
            return self._resolve(self._adapters[name]).create(config)
        return self._resolve(self._entries[name][1])(config)
