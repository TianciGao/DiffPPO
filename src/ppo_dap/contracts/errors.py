"""Uniform machine-readable contract failures."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class ContractViolation(ValueError):
    """A fail-closed violation with a stable code and structured context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("ContractViolation code must be a non-empty string")
        if not isinstance(message, str) or not message:
            raise ValueError("ContractViolation message must be a non-empty string")
        self.code = code
        self.context = MappingProxyType(dict(context or {}))
        super().__init__(f"{code}: {message}")
