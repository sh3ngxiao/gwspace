from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gwspace.Noise import detector_noises
from gwspace.constants import YRSID_SI

from tianqin_dc.config import NoiseConfig, ObservationConfig


@dataclass(frozen=True)
class NoiseResult:
    series: dict[str, np.ndarray]
    psd: dict[str, np.ndarray]
    frequency_hz: np.ndarray
    notes: list[str]


def synthesize_real_noise_from_psd(
    psd: np.ndarray,
    sample_rate_hz: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a real-valued stationary Gaussian time series from a one-sided PSD."""

    if psd.shape[0] != (n_samples // 2 + 1):
        raise ValueError("PSD length does not match the requested number of time samples.")

    spectrum = np.zeros_like(psd, dtype=np.complex128)
    psd = np.clip(np.asarray(psd, dtype=np.float64), a_min=0.0, a_max=None)

    if n_samples % 2 == 0:
        interior = slice(1, -1)
        if psd.shape[0] > 2:
            std = np.sqrt(0.25 * n_samples * sample_rate_hz * psd[interior])
            spectrum[interior] = rng.normal(0.0, std) + 1j * rng.normal(0.0, std)
        spectrum[-1] = rng.normal(0.0, np.sqrt(n_samples * sample_rate_hz * psd[-1]))
    else:
        interior = slice(1, None)
        if psd.shape[0] > 1:
            std = np.sqrt(0.25 * n_samples * sample_rate_hz * psd[interior])
            spectrum[interior] = rng.normal(0.0, std) + 1j * rng.normal(0.0, std)

    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=n_samples).astype(np.float64, copy=False)


def _confusion_duration_years(config: NoiseConfig, observation: ObservationConfig) -> float:
    if config.confusion_duration_yr is not None:
        return float(config.confusion_duration_yr)
    return observation.effective_duration_s / YRSID_SI


def generate_noise(
    observation: ObservationConfig,
    config: NoiseConfig,
    seed: int,
) -> NoiseResult:
    """Generate TianQin noise in the A/E/T basis.

    Assumption: GWspace.Noise.TianQinNoise.noise_AET() returns one-sided A/E/T
    PSDs that are compatible with the time-domain A/E/T response normalization.
    If a later challenge version adopts a different TDI convention, this wrapper
    is the only place that needs to change.
    """

    if observation.channels and tuple(observation.channels) not in (
        ("A",),
        ("E",),
        ("T",),
        ("A", "E"),
        ("A", "T"),
        ("E", "T"),
        ("A", "E", "T"),
    ):
        raise NotImplementedError("Noise generation currently supports the A/E/T basis only.")

    model_name = config.model
    if model_name not in detector_noises:
        supported = ", ".join(sorted(detector_noises))
        raise ValueError(f"Unsupported noise model '{model_name}'. Supported: {supported}.")

    model = detector_noises[model_name]()
    freqs = np.fft.rfftfreq(observation.num_samples, d=observation.sample_spacing_s)
    safe_freqs = freqs.copy()
    if safe_freqs.shape[0] > 1:
        safe_freqs[0] = safe_freqs[1]
    else:
        safe_freqs[0] = 1.0 / observation.effective_duration_s

    wd_foreground = _confusion_duration_years(config, observation) if config.include_confusion else 0.0
    psd_ae, psd_t = model.noise_AET(
        safe_freqs,
        unit=config.unit,
        TDIgen=observation.tdi_generation,
        wd_foreground=wd_foreground,
    )
    psd_ae = np.asarray(psd_ae, dtype=np.float64)
    psd_t = np.asarray(psd_t, dtype=np.float64)
    psd_ae[0] = 0.0
    psd_t[0] = 0.0

    rng = np.random.default_rng(seed)
    psd_by_channel = {"A": psd_ae, "E": psd_ae, "T": psd_t}
    series = {
        channel: synthesize_real_noise_from_psd(
            psd_by_channel[channel],
            observation.sample_rate_hz,
            observation.num_samples,
            rng,
        )
        for channel in observation.channels
    }

    notes = [
        "Noise is sampled as stationary Gaussian noise in the A/E/T basis from GWspace PSD models.",
    ]
    if config.include_confusion:
        notes.append("Confusion noise was requested through GWspace's wd_foreground model.")

    return NoiseResult(
        series=series,
        psd={channel: psd_by_channel[channel] for channel in observation.channels},
        frequency_hz=freqs,
        notes=notes,
    )
