from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.constants import parsec

from gwspace.Noise import detector_noises

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
    response_model: str = "orf_aet"
    transfer_frequency_hz: float | None = None
    t_low_frequency_power: float = 2.0
    t_response_floor: float = 0.0
    response_nside: int = 6
    response_frequency_points: int = 256
    response_time_segments: int = 8
    response_chunk_size: int = 65536
    channel_response: Mapping[str, float] | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "SGWBModelConfig":
        data = _mapping(value, field_name="sgwb")
        model = str(data.get("model", "power_law")).lower()
        if model != "power_law":
            raise ValueError("Only sgwb.model='power_law' is currently supported.")
        response_model = str(data.get("response_model", "low_frequency_aet")).lower()
        if response_model not in ("constant", "low_frequency_aet", "orf_aet"):
            raise ValueError("sgwb.response_model must be 'constant', 'low_frequency_aet', or 'orf_aet'.")
        channel_response = data.get("channel_response")
        if channel_response is not None:
            channel_response = {
                str(channel).upper(): float(scale)
                for channel, scale in _mapping(channel_response, field_name="sgwb.channel_response").items()
            }
        t_response_floor = float(data.get("t_response_floor", 0.0))
        if not 0.0 <= t_response_floor <= 1.0:
            raise ValueError("sgwb.t_response_floor must be in [0, 1].")
        return cls(
            model=model,
            omega_ref=float(data.get("omega_ref", 6.0e-12)),
            f_ref_hz=float(data.get("f_ref_hz", 1.0e-3)),
            beta=float(data.get("beta", 0.0)),
            h0_km_s_mpc=float(data.get("h0_km_s_mpc", 67.7)),
            f_min_hz=(None if data.get("f_min_hz") is None else float(data["f_min_hz"])),
            f_max_hz=(None if data.get("f_max_hz") is None else float(data["f_max_hz"])),
            response_model=response_model,
            transfer_frequency_hz=(
                None if data.get("transfer_frequency_hz") is None else float(data["transfer_frequency_hz"])
            ),
            t_low_frequency_power=float(data.get("t_low_frequency_power", 2.0)),
            t_response_floor=t_response_floor,
            response_nside=int(data.get("response_nside", 6)),
            response_frequency_points=int(data.get("response_frequency_points", 256)),
            response_time_segments=int(data.get("response_time_segments", 8)),
            response_chunk_size=int(data.get("response_chunk_size", 65536)),
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

    def transfer_frequency_for(self, observation: ObservationConfig) -> float:
        if self.transfer_frequency_hz is not None:
            return self.transfer_frequency_hz
        detector = observation.detector
        if detector not in detector_noises:
            supported = ", ".join(sorted(detector_noises))
            raise ValueError(f"Unsupported detector '{detector}' for SGWB response. Supported: {supported}.")
        return float(detector_noises[detector]().f_star)

    def response_psd_for(
        self,
        channel: str,
        frequencies_hz: np.ndarray,
        observation: ObservationConfig,
    ) -> np.ndarray:
        """Return the PSD response multiplier for an A/E/T SGWB realization.

        The low_frequency_aet model is still an isotropic Gaussian-background
        approximation. It keeps A/E normalized to the input strain PSD while
        applying a smooth low-frequency null-channel suppression to T.
        """

        scale = self.response_for(channel)
        if scale < 0.0:
            raise ValueError(f"sgwb.channel_response.{channel} must be non-negative.")
        response = np.full_like(frequencies_hz, scale, dtype=np.float64)
        if self.response_model == "constant" or channel != "T" or scale == 0.0:
            return response

        if self.t_low_frequency_power < 0.0:
            raise ValueError("sgwb.t_low_frequency_power must be non-negative.")

        transfer_frequency_hz = self.transfer_frequency_for(observation)
        if transfer_frequency_hz <= 0.0:
            raise ValueError("SGWB transfer frequency must be positive.")

        ratio = np.clip(frequencies_hz / transfer_frequency_hz, a_min=0.0, a_max=None)
        t_suppression = np.minimum(1.0, ratio**self.t_low_frequency_power)
        t_suppression = self.t_response_floor + (1.0 - self.t_response_floor) * t_suppression
        return response * t_suppression


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


def _detector_orf_params(config: MinimalSGWBAETConfig) -> dict[str, float | int | str]:
    if config.sgwb.response_nside <= 0:
        raise ValueError("sgwb.response_nside must be positive.")
    return {
        "nside": config.sgwb.response_nside,
        "TQarmlength": 3.0**0.5 * 1.0e8,
        "LSarmlength": 2.5e9,
        "Tjarmlength": 3.0e9,
        "cspeed": 3.0e8,
        "response type": "aet",
    }


def _response_time_grid(config: MinimalSGWBAETConfig) -> np.ndarray:
    n_segments = int(config.sgwb.response_time_segments)
    if n_segments <= 0:
        raise ValueError("sgwb.response_time_segments must be positive.")
    duration_s = config.observation.effective_duration_s
    if n_segments == 1:
        return np.array([0.5 * duration_s], dtype=np.float64)
    step_s = duration_s / n_segments
    return (np.arange(n_segments, dtype=np.float64) + 0.5) * step_s


def _response_frequency_grid(frequencies_hz: np.ndarray, active: np.ndarray, n_points: int) -> np.ndarray:
    if n_points <= 1:
        raise ValueError("sgwb.response_frequency_points must be greater than 1.")
    active_freqs = frequencies_hz[active & (frequencies_hz > 0.0)]
    if active_freqs.size == 0:
        return np.array([], dtype=np.float64)
    f_min = float(active_freqs[0])
    f_max = float(active_freqs[-1])
    if active_freqs.size <= n_points:
        return active_freqs.astype(np.float64, copy=True)
    grid = np.geomspace(f_min, f_max, n_points)
    grid[0] = f_min
    grid[-1] = f_max
    return np.unique(grid.astype(np.float64))


def _auto_response_function(detector: str):
    from tianqin_dc.sgwb import ORF

    if detector in ("TQ", "TianQin"):
        return ORF.TQ_auto_response_pix
    if detector == "LISA":
        return ORF.LS_auto_response_pix
    if detector == "Taiji":
        return ORF.Tj_auto_response_pix
    raise ValueError(f"Unsupported detector '{detector}' for SGWB ORF response.")


def _build_orf_response_grid(
    config: MinimalSGWBAETConfig,
    frequencies_hz: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grid_frequencies = _response_frequency_grid(
        frequencies_hz,
        active,
        config.sgwb.response_frequency_points,
    )
    if grid_frequencies.size == 0:
        return grid_frequencies, np.zeros((3, 3, 0), dtype=np.complex128)

    response_fn = _auto_response_function(config.observation.detector)
    params = _detector_orf_params(config)
    response = response_fn(grid_frequencies, _response_time_grid(config), params)
    # ORF.py returns [channel, channel, frequency, time, sky_pixel].  The
    # isotropic response is the sky integral of the pixel response, then
    # averaged over the sampled orbit times.
    response_grid = 4.0 * np.pi * np.mean(response, axis=(3, 4))
    response_grid = 0.5 * (response_grid + np.conj(np.swapaxes(response_grid, 0, 1)))

    scales = np.array([config.sgwb.response_for(channel) for channel in ("A", "E", "T")], dtype=np.float64)
    if np.any(scales < 0.0):
        raise ValueError("sgwb.channel_response values must be non-negative.")
    scale_matrix = np.sqrt(np.outer(scales, scales))[:, :, None]
    return grid_frequencies, response_grid * scale_matrix


def _interpolate_response_chunk(
    grid_frequencies: np.ndarray,
    response_grid: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    matrices = np.zeros((frequencies_hz.size, 3, 3), dtype=np.complex128)
    if frequencies_hz.size == 0 or grid_frequencies.size == 0:
        return matrices
    if grid_frequencies.size == 1:
        matrices[:] = np.moveaxis(response_grid[:, :, :1], 2, 0)[0]
        return matrices

    log_grid = np.log(grid_frequencies)
    log_freq = np.log(frequencies_hz)
    for row in range(3):
        for col in range(3):
            values = response_grid[row, col]
            real = np.interp(log_freq, log_grid, values.real)
            imag = np.interp(log_freq, log_grid, values.imag)
            matrices[:, row, col] = real + 1j * imag
    return 0.5 * (matrices + np.conj(np.swapaxes(matrices, 1, 2)))


def _sample_complex_gaussian_from_csd(
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    standard = (
        rng.normal(0.0, 1.0, size=eigenvalues.shape)
        + 1j * rng.normal(0.0, 1.0, size=eigenvalues.shape)
    ) / np.sqrt(2.0)
    weighted = np.sqrt(eigenvalues) * standard
    return np.einsum("fij,fj->fi", eigenvectors, weighted)


def _draw_correlated_gaussian_series_from_one_sided_csd(
    base_psd: np.ndarray,
    response_grid_frequencies: np.ndarray,
    response_grid: np.ndarray,
    *,
    sample_rate_hz: float,
    num_samples: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("sgwb.response_chunk_size must be positive.")

    frequencies = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate_hz)
    spectra = np.zeros((3, frequencies.size), dtype=np.complex128)
    covariance_scale = 0.5 * num_samples * sample_rate_hz

    interior_stop = frequencies.size - 1 if num_samples % 2 == 0 else frequencies.size
    for start in range(1, interior_stop, chunk_size):
        stop = min(start + chunk_size, interior_stop)
        psd_chunk = base_psd[start:stop]
        active = psd_chunk > 0.0
        if not np.any(active):
            continue

        chunk_freqs = frequencies[start:stop]
        matrices = _interpolate_response_chunk(
            response_grid_frequencies,
            response_grid,
            chunk_freqs[active],
        )
        covariance = matrices * psd_chunk[active, None, None] * covariance_scale
        samples = _sample_complex_gaussian_from_csd(covariance, rng)
        spectra[:, start + np.nonzero(active)[0]] = samples.T

    if num_samples % 2 == 0 and base_psd[-1] > 0.0:
        matrix = _interpolate_response_chunk(
            response_grid_frequencies,
            response_grid,
            frequencies[-1:],
        )[0]
        covariance = (matrix * base_psd[-1] * num_samples * sample_rate_hz).real
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        standard = rng.normal(0.0, 1.0, size=3)
        spectra[:, -1] = eigenvectors @ (np.sqrt(eigenvalues) * standard)

    return {
        channel: np.fft.irfft(spectra[index], n=num_samples).astype(np.float64, copy=False)
        for index, channel in enumerate(("A", "E", "T"))
    }


def generate_minimal_sgwb_aet(config: MinimalSGWBAETConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observation = config.observation
    time_s = observation.time_array()
    frequencies = np.fft.rfftfreq(observation.num_samples, d=observation.sample_spacing_s)
    base_psd = config.sgwb.strain_psd(frequencies)
    if config.sgwb.response_model == "orf_aet":
        response_grid_frequencies, response_grid = _build_orf_response_grid(config, frequencies, base_psd > 0.0)
        channels = _draw_correlated_gaussian_series_from_one_sided_csd(
            base_psd,
            response_grid_frequencies,
            response_grid,
            sample_rate_hz=observation.sample_rate_hz,
            num_samples=observation.num_samples,
            chunk_size=config.sgwb.response_chunk_size,
            rng=np.random.default_rng(config.seed),
        )
        return time_s, channels

    channels: dict[str, np.ndarray] = {}
    for index, channel in enumerate(("A", "E", "T")):
        response_psd = config.sgwb.response_psd_for(channel, frequencies, observation)
        if not np.any(response_psd):
            channels[channel] = np.zeros(observation.num_samples, dtype=np.float64)
            continue
        rng = np.random.default_rng(config.seed + index)
        channels[channel] = _draw_gaussian_series_from_one_sided_psd(
            base_psd * response_psd,
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
    print(f"response_model: {config.sgwb.response_model}")
    if config.sgwb.response_model == "low_frequency_aet":
        print(f"transfer_frequency_hz: {config.sgwb.transfer_frequency_for(config.observation)}")
        print(f"t_low_frequency_power: {config.sgwb.t_low_frequency_power}")
        print(f"t_response_floor: {config.sgwb.t_response_floor}")
    elif config.sgwb.response_model == "orf_aet":
        print(f"response_nside: {config.sgwb.response_nside}")
        print(f"response_frequency_points: {config.sgwb.response_frequency_points}")
        print(f"response_time_segments: {config.sgwb.response_time_segments}")
        print(f"response_chunk_size: {config.sgwb.response_chunk_size}")
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
        print(f"response_model: {config.sgwb.response_model}")
        if config.sgwb.response_model == "low_frequency_aet":
            print(f"transfer_frequency_hz: {config.sgwb.transfer_frequency_for(config.observation)}")
            print(f"t_low_frequency_power: {config.sgwb.t_low_frequency_power}")
            print(f"t_response_floor: {config.sgwb.t_response_floor}")
        elif config.sgwb.response_model == "orf_aet":
            print(f"response_nside: {config.sgwb.response_nside}")
            print(f"response_frequency_points: {config.sgwb.response_frequency_points}")
            print(f"response_time_segments: {config.sgwb.response_time_segments}")
            print(f"response_chunk_size: {config.sgwb.response_chunk_size}")
        print(
            "channel_response: "
            + ", ".join(f"{channel}={config.sgwb.response_for(channel)}" for channel in ("A", "E", "T"))
        )
        return 0

    run_minimal_sgwb_aet(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
