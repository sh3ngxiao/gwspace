from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

from gwspace.response_b import get_AET_td, get_XYZ_td

from tianqin_dc.config import ObservationConfig


def _to_numpy(value: Any) -> np.ndarray:
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _select_channels(full: dict[str, np.ndarray], observation: ObservationConfig) -> dict[str, np.ndarray]:
    return {channel: full[channel] for channel in observation.channels}


def generate_tdi_channels_td(
    waveform: Any,
    time_s: np.ndarray,
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    """Generate TianQin A/E/T channels for a single source from a TD waveform."""

    a_channel, e_channel, t_channel = get_AET_td(
        waveform,
        time_s,
        det=observation.detector,
        TDIgen=observation.tdi_generation,
        use_gpu=observation.use_gpu,
    )
    full = {
        "A": _to_numpy(a_channel).astype(np.float64, copy=False),
        "E": _to_numpy(e_channel).astype(np.float64, copy=False),
        "T": _to_numpy(t_channel).astype(np.float64, copy=False),
    }
    return _select_channels(full, observation)


def generate_tdi_xyz_td(
    waveform: Any,
    time_s: np.ndarray,
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    """Generate TianQin X/Y/Z channels for a single source from a TD waveform."""

    x_channel, y_channel, z_channel = get_XYZ_td(
        waveform,
        time_s,
        det=observation.detector,
        TDIgen=observation.tdi_generation,
        use_gpu=observation.use_gpu,
    )
    return {
        "X": _to_numpy(x_channel).astype(np.float64, copy=False),
        "Y": _to_numpy(y_channel).astype(np.float64, copy=False),
        "Z": _to_numpy(z_channel).astype(np.float64, copy=False),
    }


def response_frequency_grid(observation: ObservationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the full rFFT grid and the strictly positive frequency grid used for FD sources."""

    full = np.fft.rfftfreq(observation.num_samples, d=observation.sample_spacing_s)
    positive = full[1:]
    if positive.size == 0:
        raise ValueError("Observation is too short to build a non-zero frequency grid.")
    return full, positive


def rfft_spectrum_to_time_series(
    spectrum: np.ndarray,
    observation: ObservationConfig,
) -> np.ndarray:
    """Convert a one-sided continuous-frequency spectrum to a real TD series.

    Assumption: GWspace FD responses are samples of the continuous Fourier-domain
    signal on the observation rFFT grid. The conversion therefore multiplies by
    the sampling frequency before applying irfft so that the discrete inverse FFT
    approximates the inverse Fourier integral.
    """

    array = np.asarray(spectrum, dtype=np.complex128)
    if array.shape[-1] != observation.num_samples // 2 + 1:
        raise ValueError("FD spectrum length does not match the observation rFFT grid.")
    return np.fft.irfft(array * observation.sample_rate_hz, n=observation.num_samples, axis=-1).astype(
        np.float64,
        copy=False,
    )


def generate_tdi_channels_fd(
    waveform: Any,
    observation: ObservationConfig,
    *,
    response_kwargs: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Generate TianQin A/E/T channels for a single source from a FD response."""

    kwargs = dict(response_kwargs or {})
    full_freq, positive_freq = response_frequency_grid(observation)
    response = waveform.get_tdi_response(
        f_series=positive_freq,
        channel="AET",
        det=observation.detector,
        TDIgen=observation.tdi_generation,
        **kwargs,
    )
    if isinstance(response, tuple):
        fd_channels, freq_out = response
        freq_out = np.asarray(freq_out, dtype=np.float64)
    else:
        fd_channels = response
        freq_out = positive_freq

    fd_channels = np.asarray(fd_channels, dtype=np.complex128)
    if fd_channels.shape[0] != 3:
        raise ValueError(f"Expected three A/E/T FD channels, got shape {fd_channels.shape}.")
    if fd_channels.shape[1] != freq_out.shape[0]:
        raise ValueError("FD response and frequency grid lengths do not match.")

    full_spectrum = np.zeros((3, full_freq.shape[0]), dtype=np.complex128)
    df = 1.0 / observation.effective_duration_s
    freq_bins = np.rint(freq_out / df).astype(int)
    valid = (freq_bins >= 1) & (freq_bins < full_freq.shape[0])
    if np.any(valid):
        full_spectrum[:, freq_bins[valid]] = fd_channels[:, valid]

    td_channels = rfft_spectrum_to_time_series(full_spectrum, observation)
    full = {
        "A": td_channels[0],
        "E": td_channels[1],
        "T": td_channels[2],
    }
    return _select_channels(full, observation)


def generate_tdi_channels(
    waveform: Any,
    time_s: np.ndarray,
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    """Backward-compatible alias for the TD response path."""

    return generate_tdi_channels_td(waveform, time_s, observation)
