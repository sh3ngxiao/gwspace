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


def _numerical_detector_time_window(detector: str) -> tuple[float, float] | None:
    try:
        from gwspace.Orbit import detectors
        from tianqin_dc.numerical_tq_orbit import NumericalTianQinOrbit, numerical_orbit_time_window_s
    except Exception:
        return None

    if detectors.get(detector) is not NumericalTianQinOrbit:
        return None
    return numerical_orbit_time_window_s()


def _is_eccfd_waveform(waveform: Any) -> bool:
    return callable(getattr(waveform, "get_ori_waveform", None))


def _generate_eccfd_tdi_response(
    waveform: Any,
    f_series: np.ndarray,
    *,
    channel: str,
    detector: str,
    tdi_generation: int,
    response_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    from gwspace.Waveform import check_detector_and_channel

    trans_func, det_class = check_detector_and_channel(detector, channel)
    waveform_components, freq = waveform.get_ori_waveform(
        f_series=f_series,
        space_cutoff=True,
    )
    freq = np.asarray(freq, dtype=np.float64)
    gw_tdi = np.zeros((3, len(freq)), dtype=np.complex128)
    t_delay = np.exp(2j * np.pi * freq * waveform.tc)
    p_p, p_c = waveform.polarization()
    detector_time_window = _numerical_detector_time_window(detector)

    for h_p, h_c, tf_vec in waveform_components:
        h_p = np.asarray(h_p)
        h_c = np.asarray(h_c)
        tf_vec = np.asarray(tf_vec, dtype=np.float64)
        active = (h_p != 0.0) | (h_c != 0.0)
        if detector_time_window is not None:
            start_s, stop_s = detector_time_window
            active &= (tf_vec >= start_s) & (tf_vec <= stop_s)

        indices = np.flatnonzero(active)
        if indices.size == 0:
            continue

        det_obj = det_class(tf_vec[indices], **response_kwargs)
        gw_tdi_p, gw_tdi_c = trans_func(
            waveform.vec_k,
            (p_p, p_c),
            det_obj,
            freq[indices],
            tdi_generation,
        )
        gw_tdi[:, indices] += gw_tdi_p * h_p[None, indices] + gw_tdi_c * h_c[None, indices]

    return gw_tdi * t_delay, freq


def _can_generate_amp_phase_tdi_response(waveform: Any) -> bool:
    return callable(getattr(waveform, "get_amp_phase", None)) and callable(getattr(waveform, "p_lm", None))


def _generate_amp_phase_tdi_response(
    waveform: Any,
    f_series: np.ndarray,
    *,
    channel: str,
    detector: str,
    tdi_generation: int,
    response_kwargs: dict[str, Any],
) -> np.ndarray:
    from gwspace.Waveform import check_detector_and_channel

    trans_func, det_class = check_detector_and_channel(detector, channel)
    amp, phase, tf = waveform.get_amp_phase(f_series=f_series)
    gw_tdi = np.zeros((3, len(f_series)), dtype=np.complex128)
    t_delay = np.exp(2j * np.pi * f_series * waveform.tc)
    detector_time_window = _numerical_detector_time_window(detector)

    for mode in amp.keys():
        h_lm = np.asarray(amp[mode]) * np.exp(1j * np.asarray(phase[mode]))
        tf_vec = np.asarray(tf[mode], dtype=np.float64)
        active = h_lm != 0.0
        if detector_time_window is not None:
            start_s, stop_s = detector_time_window
            active &= (tf_vec >= start_s) & (tf_vec <= stop_s)

        indices = np.flatnonzero(active)
        if indices.size == 0:
            continue

        det_obj = det_class(tf_vec[indices], **response_kwargs)
        gw_tdi_lm = trans_func(
            waveform.vec_k,
            waveform.p_lm(*mode),
            det_obj,
            f_series[indices],
            tdi_generation,
        )[0]
        gw_tdi[:, indices] += gw_tdi_lm * h_lm[None, indices]

    return gw_tdi * t_delay


def _generate_tdi_fd(
    waveform: Any,
    observation: ObservationConfig,
    *,
    channel: str,
    response_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs = dict(response_kwargs or {})
    full_freq, positive_freq = response_frequency_grid(observation)

    response_f_min = kwargs.pop("response_f_min_hz", None)
    if response_f_min is None:
        response_f_min = kwargs.pop("f_min_hz", None)
    if response_f_min is None:
        response_f_min = getattr(waveform, "add_para", {}).get("response_f_min_hz")
    if response_f_min is not None:
        response_f_min = float(response_f_min)
        if response_f_min < 0.0:
            raise ValueError("response_f_min_hz must be non-negative.")
        positive_freq = positive_freq[positive_freq >= response_f_min]
        if positive_freq.size == 0:
            return np.zeros((3, observation.num_samples), dtype=np.float64), positive_freq

    if _is_eccfd_waveform(waveform):
        response = _generate_eccfd_tdi_response(
            waveform,
            positive_freq,
            channel=channel,
            detector=observation.detector,
            tdi_generation=observation.tdi_generation,
            response_kwargs=kwargs,
        )
    elif (
        _numerical_detector_time_window(observation.detector) is not None
        and _can_generate_amp_phase_tdi_response(waveform)
    ):
        response = _generate_amp_phase_tdi_response(
            waveform,
            positive_freq,
            channel=channel,
            detector=observation.detector,
            tdi_generation=observation.tdi_generation,
            response_kwargs=kwargs,
        )
    else:
        response = waveform.get_tdi_response(
            f_series=positive_freq,
            channel=channel,
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
        raise ValueError(f"Expected three {channel} FD channels, got shape {fd_channels.shape}.")
    if fd_channels.shape[1] != freq_out.shape[0]:
        raise ValueError("FD response and frequency grid lengths do not match.")

    full_spectrum = np.zeros((3, full_freq.shape[0]), dtype=np.complex128)
    df = 1.0 / observation.effective_duration_s
    freq_bins = np.rint(freq_out / df).astype(int)
    valid = (freq_bins >= 1) & (freq_bins < full_freq.shape[0])
    if np.any(valid):
        full_spectrum[:, freq_bins[valid]] = fd_channels[:, valid]

    return rfft_spectrum_to_time_series(full_spectrum, observation), freq_out


def generate_tdi_channels_fd(
    waveform: Any,
    observation: ObservationConfig,
    *,
    response_kwargs: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Generate TianQin A/E/T channels for a single source from a FD response."""

    td_channels, _ = _generate_tdi_fd(waveform, observation, channel="AET", response_kwargs=response_kwargs)
    full = {
        "A": td_channels[0],
        "E": td_channels[1],
        "T": td_channels[2],
    }
    return _select_channels(full, observation)


def generate_tdi_xyz_fd(
    waveform: Any,
    observation: ObservationConfig,
    *,
    response_kwargs: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Generate TianQin X/Y/Z channels for a single source from a FD response."""

    td_channels, _ = _generate_tdi_fd(waveform, observation, channel="XYZ", response_kwargs=response_kwargs)
    return {
        "X": td_channels[0],
        "Y": td_channels[1],
        "Z": td_channels[2],
    }


def generate_tdi_channels(
    waveform: Any,
    time_s: np.ndarray,
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    """Backward-compatible alias for the TD response path."""

    return generate_tdi_channels_td(waveform, time_s, observation)
