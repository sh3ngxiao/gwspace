from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tianqin_dc.config import NoiseConfig, ObservationConfig
from tianqin_dc.minimal_aet_io import MinimalOutputConfig, read_minimal_aet_hdf5, save_minimal_aet_hdf5
from tianqin_dc.noise import generate_noise


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _input_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    data = _mapping(value, field_name="signal_inputs[]")
    return str(data["path"])


@dataclass(frozen=True)
class MergeMinimalAETConfig:
    signal_inputs: tuple[str, ...]
    seed: int
    observation: ObservationConfig
    noise: NoiseConfig
    output: MinimalOutputConfig
    noise_output: MinimalOutputConfig | None = None

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "MergeMinimalAETConfig":
        inputs_raw = data.get("signal_inputs", data.get("inputs"))
        if not isinstance(inputs_raw, list) or not inputs_raw:
            raise ValueError("Config field 'signal_inputs' must be a non-empty list.")
        observation_data = dict(_mapping(data["observation"], field_name="observation"))
        observation_data["channels"] = ["A", "E", "T"]
        noise_output = None
        if data.get("noise_output") is not None:
            noise_output = MinimalOutputConfig.from_config(_mapping(data["noise_output"], field_name="noise_output"))
        return cls(
            signal_inputs=tuple(_input_path(item) for item in inputs_raw),
            seed=int(data.get("seed", 123456789)),
            observation=ObservationConfig.from_config(observation_data),
            noise=NoiseConfig.from_config(data.get("noise")),
            output=MinimalOutputConfig.from_config(_mapping(data["output"], field_name="output")),
            noise_output=noise_output,
        )


def load_merge_minimal_aet_config(path: str | Path) -> tuple[MergeMinimalAETConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return MergeMinimalAETConfig.from_config(raw), raw


def _assert_same_time(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
        raise ValueError(f"Time column in '{label}' does not match the merge observation time grid.")


def build_merged_minimal_aet(config: MergeMinimalAETConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observation = config.observation
    time_s = observation.time_array()
    summed = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("A", "E", "T")}

    for input_path in config.signal_inputs:
        input_time_s, channels = read_minimal_aet_hdf5(input_path)
        _assert_same_time(time_s, input_time_s, label=input_path)
        for channel in ("A", "E", "T"):
            summed[channel] += np.asarray(channels[channel], dtype=np.float64)

    if config.noise.enabled:
        noise = generate_noise(observation, config.noise, seed=config.seed).series
    else:
        noise = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("A", "E", "T")}

    if config.noise_output is not None:
        save_minimal_aet_hdf5(
            config.noise_output,
            time_s=time_s,
            a=noise["A"],
            e=noise["E"],
            t=noise["T"],
        )

    for channel in ("A", "E", "T"):
        summed[channel] += noise[channel]
    return time_s, summed


def run_merge_minimal_aet(config: MergeMinimalAETConfig) -> Path:
    time_s, channels = build_merged_minimal_aet(config)
    output_path = save_minimal_aet_hdf5(
        config.output,
        time_s=time_s,
        a=channels["A"],
        e=channels["E"],
        t=channels["T"],
    )
    print(f"Wrote merged minimal A/E/T file to {output_path}")
    print(f"Signal inputs merged: {len(config.signal_inputs)}")
    print(f"Noise enabled: {config.noise.enabled}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge signal-only minimal A/E/T files and add one noise realization."
    )
    parser.add_argument("--config", required=True, help="Path to a merge minimal A/E/T JSON config.")
    parser.add_argument("--input", action="append", dest="inputs", help="Override signal_inputs from the config.")
    parser.add_argument("--output", help="Override output.path from the config.")
    parser.add_argument("--noise-output", help="Override noise_output.path from the config.")
    parser.add_argument("--no-noise-output", action="store_true", help="Do not write the optional noise-only file.")
    parser.add_argument("--dry-run", action="store_true", help="Parse the config and print what would be merged.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config, _raw = load_merge_minimal_aet_config(args.config)
    if args.inputs:
        config = replace(config, signal_inputs=tuple(args.inputs))
    if args.output:
        config = replace(config, output=replace(config.output, path=args.output))
    if args.no_noise_output:
        config = replace(config, noise_output=None)
    elif args.noise_output:
        if config.noise_output is None:
            config = replace(
                config,
                noise_output=MinimalOutputConfig(path=args.noise_output, overwrite=True),
            )
        else:
            config = replace(config, noise_output=replace(config.noise_output, path=args.noise_output))

    if args.dry_run:
        print(f"signal_inputs: {len(config.signal_inputs)}")
        for path in config.signal_inputs:
            print(f"- {path}")
        print(f"noise_enabled: {config.noise.enabled}")
        print(f"noise_include_confusion: {config.noise.include_confusion}")
        print(f"num_samples: {config.observation.num_samples}")
        print(f"sample_spacing_s: {config.observation.sample_spacing_s}")
        print(f"output: {config.output.path}")
        return 0

    run_merge_minimal_aet(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
