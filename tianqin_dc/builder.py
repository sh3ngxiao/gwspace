from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import platform
from pathlib import Path
from typing import Any

import numpy as np

import gwspace

from tianqin_dc.config import RunConfig
from tianqin_dc.io import save_dataset_hdf5
from tianqin_dc.models import DatasetBundle, InjectionRecord
from tianqin_dc.noise import generate_noise
from tianqin_dc.sampling import sample_population_parameters
from tianqin_dc.sources import available_sources, get_source_factory


def _child_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _mixing_mode(source_types: list[str]) -> str:
    unique = sorted(set(source_types))
    if not unique:
        return "noise_only"
    if len(unique) == 1:
        return "single_class"
    if len(unique) == 2:
        return "dual_class"
    return "multi_class"


class TianQinDatasetBuilder:
    def __init__(self, config: RunConfig, raw_config: dict[str, Any]) -> None:
        self.config = config
        self.raw_config = deepcopy(raw_config)

    def build(self) -> DatasetBundle:
        observation = self.config.observation
        time_s = observation.time_array()
        signal = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in observation.channels}
        per_source: dict[str, dict[str, np.ndarray]] = {}
        injections: list[InjectionRecord] = []
        assumption_notes: list[str] = []
        source_counts: dict[str, int] = defaultdict(int)
        family_counts: dict[str, int] = defaultdict(int)
        engine_counts: dict[str, int] = defaultdict(int)
        population_summary: list[dict[str, Any]] = []

        active_populations = [population for population in self.config.sources if population.enabled]

        root_sequence = np.random.SeedSequence(self.config.seed)
        child_sequences = root_sequence.spawn(len(active_populations) + 1)
        population_sequences = child_sequences[:-1]
        noise_sequence = child_sequences[-1]

        seed_book: dict[str, Any] = {
            "root_seed": int(self.config.seed),
            "populations": [],
            "noise": {"seed": _child_seed(noise_sequence)},
        }

        source_index = 0
        for population, population_sequence in zip(active_populations, population_sequences, strict=True):
            population_rng = np.random.default_rng(_child_seed(population_sequence))
            sampled_parameters = sample_population_parameters(population, population_rng)
            source_sequences = population_sequence.spawn(len(sampled_parameters))

            population_seeds: list[int] = []
            realized_source_ids: list[str] = []
            for parameters, source_sequence in zip(sampled_parameters, source_sequences, strict=True):
                source_seed = _child_seed(source_sequence)
                population_seeds.append(source_seed)
                factory = get_source_factory(population.kind)

                try:
                    generated = factory.generate(parameters, observation)
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(
                        f"Failed to generate source '{population.kind}' with parameters {parameters}."
                    ) from exc

                source_id = f"{population.kind}_{source_index:04d}"
                source_index += 1
                realized_source_ids.append(source_id)
                source_counts[population.kind] += 1
                family_counts[generated.family] += 1
                engine_counts[generated.engine] += 1

                for channel, series in generated.channels.items():
                    signal[channel] += series

                if self.config.output.save_per_source:
                    per_source[source_id] = {
                        channel: np.asarray(series, dtype=np.float64) for channel, series in generated.channels.items()
                    }

                notes = generated.notes
                assumption_notes.extend(note for note in notes if note not in assumption_notes)
                injections.append(
                    InjectionRecord(
                        source_id=source_id,
                        kind=population.kind,
                        family=generated.family,
                        population_name=population.name or f"{population.kind}_population",
                        population_role=population.role,
                        engine=generated.engine,
                        implementation=generated.implementation,
                        domain=generated.domain,
                        parameters=generated.parameters,
                        seed=source_seed,
                        notes=notes,
                        metadata=generated.metadata,
                    )
                )

            population_name = population.name or f"{population.kind}_population"
            population_summary.append(
                {
                    "name": population_name,
                    "kind": population.kind,
                    "role": population.role,
                    "enabled": population.enabled,
                    "requested_count": population.realized_count,
                    "realized_count": len(sampled_parameters),
                    "seed": _child_seed(population_sequence),
                    "source_ids": realized_source_ids,
                }
            )
            seed_book["populations"].append(
                {
                    "name": population_name,
                    "kind": population.kind,
                    "role": population.role,
                    "seed": _child_seed(population_sequence),
                    "source_seeds": population_seeds,
                }
            )

        if self.config.noise.enabled:
            noise_result = generate_noise(observation, self.config.noise, seed=_child_seed(noise_sequence))
            noise = noise_result.series
            noise_psd = noise_result.psd
            assumption_notes.extend(note for note in noise_result.notes if note not in assumption_notes)
            frequency_hz = noise_result.frequency_hz
        else:
            frequency_hz = np.fft.rfftfreq(observation.num_samples, d=observation.sample_spacing_s)
            noise = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in observation.channels}
            noise_psd = {
                channel: np.zeros_like(frequency_hz, dtype=np.float64) for channel in observation.channels
            }

        observed = {channel: signal[channel] + noise[channel] for channel in observation.channels}
        source_types_present = sorted(source_counts)
        population_roles = sorted({summary["role"] for summary in population_summary})
        labels = {
            "present_source_types": source_types_present,
            "source_type_counts": dict(source_counts),
            "family_counts": dict(family_counts),
            "population_roles": population_roles,
            "population_summary": population_summary,
            "mixing_mode": _mixing_mode(source_types_present),
            "num_injections": len(injections),
        }

        metadata = {
            "generator": "tianqin_dc",
            "generator_version": "0.1.0",
            "python_version": platform.python_version(),
            "gwspace_module": getattr(gwspace, "__file__", "unknown"),
            "supported_source_types": list(available_sources()),
            "source_counts": dict(source_counts),
            "family_counts": dict(family_counts),
            "engine_counts": dict(engine_counts),
            "population_summary": population_summary,
            "interface_assumptions": assumption_notes,
        }

        return DatasetBundle(
            time_s=time_s,
            frequency_hz=frequency_hz,
            signal=signal if self.config.output.save_signal else {},
            noise=noise if self.config.output.save_noise else {},
            observed=observed,
            noise_psd=noise_psd,
            injections=injections,
            run_config=self.raw_config,
            seed_book=seed_book,
            metadata=metadata,
            labels=labels,
            per_source=per_source,
        )

    def build_and_save(self) -> Path:
        bundle = self.build()
        return save_dataset_hdf5(bundle, self.config)


def build_dataset(config: RunConfig, raw_config: dict[str, Any]) -> DatasetBundle:
    return TianQinDatasetBuilder(config, raw_config).build()
