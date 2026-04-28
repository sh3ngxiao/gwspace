from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SUPPORTED_CHANNELS = ("A", "E", "T")
SUPPORTED_DETECTORS = ("TQ", "TianQin", "LISA", "Taiji")
SUPPORTED_TDI_GENERATIONS = (1, 2)


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _maybe_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    data = _mapping(value, field_name=field_name)
    return dict(data)


@dataclass(frozen=True)
class SamplerConfig:
    distribution: str
    value: Any | None = None
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    choices: list[Any] | None = None

    @classmethod
    def from_config(cls, value: Any) -> "SamplerConfig":
        if not isinstance(value, Mapping):
            return cls(distribution="fixed", value=value)

        distribution = str(value.get("distribution", "fixed")).lower()
        return cls(
            distribution=distribution,
            value=value.get("value"),
            low=value.get("low"),
            high=value.get("high"),
            mean=value.get("mean"),
            std=value.get("std"),
            choices=list(value["choices"]) if value.get("choices") is not None else None,
        )


@dataclass(frozen=True)
class SourcePopulationConfig:
    kind: str
    name: str | None = None
    role: str = "signal"
    enabled: bool = True
    count: int = 1
    fixed: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, SamplerConfig] = field(default_factory=dict)
    parameters: list[dict[str, Any]] | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "SourcePopulationConfig":
        kind = str(value["kind"]).lower()
        count = int(value.get("count", 1))
        if count < 0:
            raise ValueError(f"Population '{kind}' must have count >= 0.")

        parameters_raw = value.get("parameters")
        parameters: list[dict[str, Any]] | None = None
        if parameters_raw is not None:
            if not isinstance(parameters_raw, list):
                raise TypeError(f"Population '{kind}' field 'parameters' must be a list.")
            parameters = [dict(_mapping(item, field_name=f"{kind}.parameters")) for item in parameters_raw]
            if len(parameters) == 0:
                raise ValueError(f"Population '{kind}' field 'parameters' must not be empty.")

        sampler_raw = _maybe_mapping(value.get("sampler"), field_name=f"{kind}.sampler")
        sampler = {name: SamplerConfig.from_config(spec) for name, spec in sampler_raw.items()}

        return cls(
            kind=kind,
            name=(None if value.get("name") is None else str(value["name"])),
            role=str(value.get("role", "signal")),
            enabled=bool(value.get("enabled", True)),
            count=count,
            fixed=_maybe_mapping(value.get("fixed"), field_name=f"{kind}.fixed"),
            sampler=sampler,
            parameters=parameters,
        )

    @property
    def realized_count(self) -> int:
        if self.parameters is not None:
            return len(self.parameters)
        return self.count


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    description: str = ""
    format_name: str = "tianqin-dc-prototype"
    format_version: str = "0.1.0"

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "DatasetConfig":
        if value is None:
            return cls(name="tianqin_dc_dataset")
        data = _mapping(value, field_name="dataset")
        name = str(data.get("name", "tianqin_dc_dataset"))
        description = str(data.get("description", ""))
        format_name = str(data.get("format_name", "tianqin-dc-prototype"))
        format_version = str(data.get("format_version", "0.1.0"))
        return cls(
            name=name,
            description=description,
            format_name=format_name,
            format_version=format_version,
        )


@dataclass(frozen=True)
class ObservationConfig:
    duration_s: float
    sample_rate_hz: float
    detector: str = "TQ"
    tdi_generation: int = 1
    channels: tuple[str, ...] = SUPPORTED_CHANNELS
    use_gpu: bool = False

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ObservationConfig":
        data = _mapping(value, field_name="observation")
        duration_s = float(data["duration_s"])
        sample_rate_hz = float(data["sample_rate_hz"])
        detector = str(data.get("detector", "TQ"))
        tdi_generation = int(data.get("tdi_generation", 1))
        channels = tuple(str(channel) for channel in data.get("channels", SUPPORTED_CHANNELS))
        use_gpu = bool(data.get("use_gpu", False))

        if duration_s <= 0:
            raise ValueError("Observation duration must be positive.")
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        if detector not in SUPPORTED_DETECTORS:
            raise ValueError(f"Unsupported detector '{detector}'. Supported: {', '.join(SUPPORTED_DETECTORS)}.")
        if tdi_generation not in SUPPORTED_TDI_GENERATIONS:
            raise ValueError(
                f"Unsupported TDI generation {tdi_generation}. Supported: {SUPPORTED_TDI_GENERATIONS}."
            )
        if not channels:
            raise ValueError("At least one output channel must be requested.")
        unknown_channels = tuple(channel for channel in channels if channel not in SUPPORTED_CHANNELS)
        if unknown_channels:
            raise ValueError(
                f"Unsupported channels {unknown_channels}. Supported: {SUPPORTED_CHANNELS}."
            )

        return cls(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            detector=detector,
            tdi_generation=tdi_generation,
            channels=channels,
            use_gpu=use_gpu,
        )

    @property
    def sample_spacing_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def num_samples(self) -> int:
        samples = int(round(self.duration_s * self.sample_rate_hz))
        if samples < 2:
            raise ValueError("Observation must contain at least 2 samples.")
        return samples

    @property
    def effective_duration_s(self) -> float:
        return self.num_samples / self.sample_rate_hz

    def time_array(self) -> np.ndarray:
        return np.arange(self.num_samples, dtype=np.float64) * self.sample_spacing_s


@dataclass(frozen=True)
class NoiseConfig:
    enabled: bool = True
    model: str = "TQ"
    unit: str = "relative_frequency"
    include_confusion: bool = False
    confusion_duration_yr: float | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "NoiseConfig":
        if value is None:
            return cls()
        data = _mapping(value, field_name="noise")
        return cls(
            enabled=bool(data.get("enabled", True)),
            model=str(data.get("model", "TQ")),
            unit=str(data.get("unit", "relative_frequency")),
            include_confusion=bool(data.get("include_confusion", False)),
            confusion_duration_yr=(
                None if data.get("confusion_duration_yr") is None else float(data["confusion_duration_yr"])
            ),
        )


@dataclass(frozen=True)
class OutputConfig:
    path: str
    overwrite: bool = False
    compression: str | None = "gzip"
    compression_level: int = 4
    save_signal: bool = True
    save_noise: bool = True
    save_per_source: bool = True

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "OutputConfig":
        data = _mapping(value, field_name="output")
        return cls(
            path=str(data["path"]),
            overwrite=bool(data.get("overwrite", False)),
            compression=data.get("compression", "gzip"),
            compression_level=int(data.get("compression_level", 4)),
            save_signal=bool(data.get("save_signal", True)),
            save_noise=bool(data.get("save_noise", True)),
            save_per_source=bool(data.get("save_per_source", True)),
        )


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetConfig
    observation: ObservationConfig
    noise: NoiseConfig
    output: OutputConfig
    sources: tuple[SourcePopulationConfig, ...]
    seed: int

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "RunConfig":
        dataset = DatasetConfig.from_config(data.get("dataset"))
        observation = ObservationConfig.from_config(_mapping(data["observation"], field_name="observation"))
        noise = NoiseConfig.from_config(data.get("noise"))
        output = OutputConfig.from_config(_mapping(data["output"], field_name="output"))

        source_items = data.get("sources")
        if not isinstance(source_items, list) or not source_items:
            raise ValueError("Config field 'sources' must be a non-empty list.")
        sources = tuple(SourcePopulationConfig.from_config(_mapping(item, field_name="sources[]")) for item in source_items)

        seed = int(data.get("seed", 123456789))
        return cls(
            dataset=dataset,
            observation=observation,
            noise=noise,
            output=output,
            sources=sources,
            seed=seed,
        )


def load_run_config(path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return RunConfig.from_config(raw), raw
