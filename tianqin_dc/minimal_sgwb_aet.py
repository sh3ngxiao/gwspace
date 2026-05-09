from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.constants import parsec

from tianqin_dc.config import ObservationConfig
from tianqin_dc.minimal_aet_io import MinimalOutputConfig, save_minimal_aet_hdf5


_MPC_M = 1.0e6 * parsec


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


@dataclass(frozen=True)
class SGWBModelConfig:
    model: str = "power_law"
    omega_ref: float = 6.0e-12
    f_ref_hz: float = 1.0e-3
    beta: float = 0.0
    h0_km_s_mpc: float = 67.7
    f_min_hz: float | None = 1.0e-4
    f_max_hz: float | None = None
    channel_response: Mapping[str, float] | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "SGWBModelConfig":
        data = _mapping(value, field_name="sgwb")
        model = str(data.get("model", "power_law")).lower()
        if model != "power_law":
            raise ValueError("Only sgwb.model='power_law' is currently supported.")
        channel_response = data.get("channel_response")
        if channel_response is not None:
            channel_response = {
                str(channel).upper(): float(scale)
                for channel, scale in _mapping(channel_response, field_name="sgwb.channel_response").items()
            }
        return cls(
            model=model,
            omega_ref=float(data.get("omega_ref", 6.0e-12)),
            f_ref_hz=float(data.get("f_ref_hz", 1.0e-3)),
            beta=float(data.get("beta", 0.0)),
            h0_km_s_mpc=float(data.get("h0_km_s_mpc", 67.7)),
            f_min_hz=(None if data.get("f_min_hz") is None else float(data["f_min_hz"])),
            f_max_hz=(None if data.get("f_max_hz") is None else float(data["f_max_hz"])),
            channel_response=channel_response,
        )

    @property
    def h0_s_inv(self) -> float:
        return self.h0_km_s_mpc * 1000.0 / _MPC_M

    def omega_gw(self, frequencies_hz: np.ndarray) -> np.ndarray:
        values = np.zeros_like(frequencies_hz, dtype=np.float64)
        mask = frequencies_hz > 0.0
        if self.f_min_hz is not None:
            mask &= frequencies_hz >= self.f_min_hz
        if self.f_max_hz is not None:
            mask &= frequencies_hz <= self.f_max_hz
        values[mask] = self.omega_ref * (frequencies_hz[mask] / self.f_ref_hz) ** self.beta
        return values

    def strain_psd(self, frequencies_hz: np.ndarray) -> np.ndarray:
        omega = self.omega_gw(frequencies_hz)
        psd = np.zeros_like(frequencies_hz, dtype=np.float64)
        mask = frequencies_hz > 0.0
        psd[mask] = (3.0 * self.h0_s_inv**2 / (2.0 * np.pi**2)) * omega[mask] / frequencies_hz[mask] ** 3
        return psd

    def response_for(self, channel: str) -> float:
        defaults = {"A": 1.0, "E": 1.0, "T": 0.0}
        if self.channel_response is None:
            return defaults[channel]
        return float(self.channel_response.get(channel, defaults[channel]))


@dataclass(frozen=True)
class MinimalSGWBAETConfig:
    seed: int
    observation: ObservationConfig
    sgwb: SGWBModelConfig
    output: MinimalOutputConfig

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "MinimalSGWBAETConfig":
        observation_data = dict(_mapping(data["observation"], field_name="observation"))
        observation_data["channels"] = ["A", "E", "T"]
        return cls(
            seed=int(data.get("seed", 20260509)),
            observation=ObservationConfig.from_config(observation_data),
            sgwb=SGWBModelConfig.from_config(_mapping(data["sgwb"], field_name="sgwb")),
            output=MinimalOutputConfig.from_config(_mapping(data["output"], field_name="output")),
        )


def load_minimal_sgwb_aet_config(path: str | Path) -> tuple[MinimalSGWBAETConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return MinimalSGWBAETConfig.from_config(raw), raw


def _draw_gaussian_series_from_one_sided_psd(
    psd: np.ndarray,
    *,
    sample_rate_hz: float,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coefficients = np.zeros(psd.shape, dtype=np.complex128)
    if len(psd) > 2:
        sigma = np.sqrt(psd[1:-1] * sample_rate_hz * num_samples / 4.0)
        coefficients[1:-1] = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)
    if num_samples % 2 == 0 and len(psd) > 1:
        coefficients[-1] = rng.normal(0.0, np.sqrt(psd[-1] * sample_rate_hz * num_samples))
    return np.fft.irfft(coefficients, n=num_samples)


def generate_minimal_sgwb_aet(config: MinimalSGWBAETConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observation = config.observation
    time_s = observation.time_array()
    frequencies = np.fft.rfftfreq(observation.num_samples, d=observation.sample_spacing_s)
    base_psd = config.sgwb.strain_psd(frequencies)
    channels: dict[str, np.ndarray] = {}

    for index, channel in enumerate(("A", "E", "T")):
        response = config.sgwb.response_for(channel)
        if response == 0.0:
            channels[channel] = np.zeros(observation.num_samples, dtype=np.float64)
            continue
        if response < 0.0:
            raise ValueError(f"sgwb.channel_response.{channel} must be non-negative.")
        rng = np.random.default_rng(config.seed + index)
        channels[channel] = _draw_gaussian_series_from_one_sided_psd(
            base_psd * response,
            sample_rate_hz=observation.sample_rate_hz,
            num_samples=observation.num_samples,
            rng=rng,
        )

    return time_s, channels


def run_minimal_sgwb_aet(config: MinimalSGWBAETConfig) -> Path:
    time_s, channels = generate_minimal_sgwb_aet(config)
    output_path = save_minimal_aet_hdf5(
        config.output,
        time_s=time_s,
        a=channels["A"],
        e=channels["E"],
        t=channels["T"],
    )
    print(f"Wrote SGWB minimal A/E/T file to {output_path}")
    print(f"num_samples: {config.observation.num_samples}")
    print(f"sample_spacing_s: {config.observation.sample_spacing_s}")
    print(f"omega_ref: {config.sgwb.omega_ref}")
    print(f"f_ref_hz: {config.sgwb.f_ref_hz}")
    print(f"beta: {config.sgwb.beta}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a signal-only minimal A/E/T SGWB realization.")
    parser.add_argument("--config", required=True, help="Path to a minimal SGWB A/E/T JSON config.")
    parser.add_argument("--output", help="Override output.path from the config.")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and print generation settings.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config, _raw = load_minimal_sgwb_aet_config(args.config)
    if args.output:
        config = replace(config, output=replace(config.output, path=args.output))

    if args.dry_run:
        print(f"num_samples: {config.observation.num_samples}")
        print(f"sample_spacing_s: {config.observation.sample_spacing_s}")
        print(f"output: {config.output.path}")
        print(f"model: {config.sgwb.model}")
        print(f"omega_ref: {config.sgwb.omega_ref}")
        print(f"f_ref_hz: {config.sgwb.f_ref_hz}")
        print(f"beta: {config.sgwb.beta}")
        print(f"f_min_hz: {config.sgwb.f_min_hz}")
        print(f"f_max_hz: {config.sgwb.f_max_hz}")
        print(
            "channel_response: "
            + ", ".join(f"{channel}={config.sgwb.response_for(channel)}" for channel in ("A", "E", "T"))
        )
        return 0

    run_minimal_sgwb_aet(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
