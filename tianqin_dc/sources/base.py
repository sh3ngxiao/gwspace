from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult


class SourceFactory(ABC):
    kind: str
    family: str = "generic"
    default_engine: str = "unknown"
    default_implementation: str = "custom"
    domain: str = "time"
    required_parameters: tuple[str, ...] = tuple()
    notes: tuple[str, ...] = tuple()

    def prepare_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, Any]:
        prepared = deepcopy(parameters)
        missing = [name for name in self.required_parameters if name not in prepared]
        if missing:
            raise ValueError(f"Source '{self.kind}' is missing required parameters: {missing}.")
        return prepared

    def get_notes(self) -> list[str]:
        return list(self.notes)

    def make_result(
        self,
        channels: dict[str, Any],
        parameters: dict[str, Any],
        *,
        notes: list[str] | None = None,
        engine: str | None = None,
        implementation: str | None = None,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SourceGenerationResult:
        return SourceGenerationResult(
            channels=channels,
            parameters=deepcopy(parameters),
            family=self.family,
            engine=engine or self.default_engine,
            implementation=implementation or self.default_implementation,
            domain=domain or self.domain,
            notes=list(self.get_notes()) + list(notes or []),
            metadata=deepcopy(metadata or {}),
        )

    @abstractmethod
    def generate(self, parameters: dict[str, Any], observation: ObservationConfig) -> SourceGenerationResult:
        raise NotImplementedError
