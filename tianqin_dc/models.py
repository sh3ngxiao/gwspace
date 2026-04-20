from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceGenerationResult:
    channels: dict[str, np.ndarray]
    parameters: dict[str, Any]
    family: str
    engine: str
    implementation: str
    domain: str
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectionRecord:
    source_id: str
    kind: str
    family: str
    population_name: str
    population_role: str
    engine: str
    implementation: str
    domain: str
    parameters: dict[str, Any]
    seed: int
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "family": self.family,
            "population_name": self.population_name,
            "population_role": self.population_role,
            "engine": self.engine,
            "implementation": self.implementation,
            "domain": self.domain,
            "parameters": self.parameters,
            "seed": self.seed,
            "notes": list(self.notes),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DatasetBundle:
    time_s: np.ndarray
    frequency_hz: np.ndarray
    signal: dict[str, np.ndarray]
    noise: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]
    noise_psd: dict[str, np.ndarray]
    injections: list[InjectionRecord]
    run_config: dict[str, Any]
    seed_book: dict[str, Any]
    metadata: dict[str, Any]
    labels: dict[str, Any]
    per_source: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
