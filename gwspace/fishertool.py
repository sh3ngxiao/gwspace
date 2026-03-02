#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fisher information matrix (FIM) utilities for GW sources.

This module provides a lightweight, waveform-agnostic way to compute
SNR and the Fisher matrix for frequency-domain TDI responses.

Example
-------
>>> from gwspace.Waveform import BHBWaveform
>>> from gwspace.fishertool import fisher_matrix
>>> wf = BHBWaveform(
...     mass1=30, mass2=30, T_obs=4*365*24*3600,
...     DL=1e3, Lambda=1.0, Beta=0.5, phi_c=0.1,
...     tc=0.0, iota=0.3, var_phi=0.0, psi=0.2, chi1=0.0, chi2=0.0
... )
>>> result = fisher_matrix(
...     wf,
...     params=("Mc", "eta", "DL", "tc", "Lambda", "Beta", "psi"),
...     det="TQ",
...     channel="AET",
...     f_min=wf.f_min,
...     f_max=1.0,
...     delta_f=1.0 / wf.T_obs,
... )
>>> result["fisher"]
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from gwspace.Noise import BasicNoise, detector_noises
from gwspace.utils import to_m1m2

ArrayLike = Union[np.ndarray, Sequence[float]]

_ANGLE_PARAMS = {
    "Lambda", "Beta", "psi", "iota", "phi_c", "var_phi", "phi0",
    "Phi_phi0", "Phi_theta0", "Phi_r0", "qS", "phiS", "qK", "phiK",
}
_POSITIVE_PARAMS = {
    "mass1", "mass2", "Mc", "DL", "f0", "fdot", "fddot", "T_obs", "dist",
}


def make_frequency_series(
    wf,
    f_series: Optional[ArrayLike] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    delta_f: Optional[float] = None,
) -> np.ndarray:
    """Construct a frequency series for FD calculations."""
    if f_series is not None:
        freq = np.asarray(f_series, dtype=float)
        if freq.ndim != 1:
            raise ValueError("f_series must be a 1D array.")
        return freq

    if delta_f is None:
        T_obs = getattr(wf, "T_obs", None)
        if T_obs is None:
            raise ValueError("delta_f is required when wf.T_obs is unavailable.")
        delta_f = 1.0 / T_obs

    if f_min is None:
        f_min = getattr(wf, "f_min", None)
    if f_min is None:
        f_min = delta_f

    if f_max is None:
        f_max = 1.0

    if f_min <= 0:
        raise ValueError("f_min must be positive for one-sided PSD usage.")
    if f_max <= f_min:
        raise ValueError("f_max must be larger than f_min.")

    return np.arange(f_min, f_max, delta_f, dtype=float)


def get_noise_psd(
    det: str,
    channel: str,
    freq: np.ndarray,
    noise: Optional[Union[str, BasicNoise]] = None,
    unit: str = "relative_frequency",
    TDIgen: int = 1,
    wd_foreground: float = 0.0,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Get PSD(s) for the given detector/channel."""
    if noise is None:
        if not isinstance(det, str):
            raise ValueError("det must be a detector name string when noise is None.")
        noise_cls = detector_noises.get(det)
        if noise_cls is None:
            raise ValueError(f"Unknown detector noise for '{det}'.")
        noise_obj = noise_cls()
    elif isinstance(noise, str):
        noise_cls = detector_noises.get(noise)
        if noise_cls is None:
            raise ValueError(f"Unknown detector noise '{noise}'.")
        noise_obj = noise_cls()
    else:
        noise_obj = noise

    channel = channel.upper()
    if channel == "AET":
        s_ae, s_t = noise_obj.noise_AET(
            freq, unit=unit, TDIgen=TDIgen, wd_foreground=wd_foreground
        )
        return np.vstack([s_ae, s_ae, s_t])
    if channel == "XYZ":
        s_x, s_xy = noise_obj.noise_XYZ(
            freq, unit=unit, TDIgen=TDIgen, wd_foreground=wd_foreground
        )
        return s_x, s_xy

    raise ValueError(f"Unknown channel '{channel}'. Use 'AET' or 'XYZ'.")


def inner_product(
    a: np.ndarray,
    b: np.ndarray,
    psd: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    df: float,
    channel: str = "AET",
    use_T: bool = True,
) -> float:
    """Compute the noise-weighted inner product (a|b)."""
    channel = channel.upper()
    if channel == "AET":
        psd_arr = np.asarray(psd, dtype=float)
        if psd_arr.shape[0] != 3:
            raise ValueError("AET PSD should have shape (3, N).")
        idx = [0, 1, 2] if use_T else [0, 1]
        a_sel = a[idx]
        b_sel = b[idx]
        psd_sel = psd_arr[idx]
        safe_psd = np.where((psd_sel > 0) & np.isfinite(psd_sel), psd_sel, np.inf)
        integrand = a_sel.conj() * b_sel / safe_psd
        return 4.0 * np.real(np.sum(integrand, axis=0).sum() * df)

    if channel == "XYZ":
        s_x, s_xy = psd
        s_x = np.asarray(s_x, dtype=float)
        s_xy = np.asarray(s_xy, dtype=float)
        denom = (s_x - s_xy)
        denom2 = (s_x + 2.0 * s_xy)
        safe = (denom != 0) & (denom2 != 0) & np.isfinite(denom) & np.isfinite(denom2)
        alpha = np.where(safe, 1.0 / denom, 0.0)
        beta = np.where(safe, -s_xy / (denom * denom2), 0.0)
        ab = np.sum(a.conj() * b, axis=0)
        a1 = np.sum(a.conj(), axis=0)
        b1 = np.sum(b, axis=0)
        integrand = alpha * ab + beta * a1 * b1
        return 4.0 * np.real(np.sum(integrand) * df)

    raise ValueError(f"Unknown channel '{channel}'. Use 'AET' or 'XYZ'.")


def snr_from_response(
    h: np.ndarray,
    psd: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    df: float,
    channel: str = "AET",
    use_T: bool = True,
) -> float:
    """Compute SNR from a frequency-domain response."""
    return float(np.sqrt(inner_product(h, h, psd, df, channel=channel, use_T=use_T)))


def covariance_matrix(fisher: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Compute covariance matrix via (pseudo-)inverse."""
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("Fisher matrix must be square.")
    return np.linalg.pinv(fisher, rcond=rcond)


def _get_param_value(wf, name: str) -> float:
    if name == "Mc":
        return wf.Mc
    if name == "eta":
        return wf.eta
    if hasattr(wf, name):
        return getattr(wf, name)
    if hasattr(wf, "add_para") and name in wf.add_para:
        return wf.add_para[name]
    raise AttributeError(f"Waveform has no parameter '{name}'.")


def _capture_state(wf, name: str) -> Dict[str, float]:
    if name in ("Mc", "eta"):
        return {"mass1": wf.mass1, "mass2": wf.mass2}
    if hasattr(wf, name):
        return {name: getattr(wf, name)}
    if hasattr(wf, "add_para") and name in wf.add_para:
        return {f"add_para::{name}": wf.add_para[name]}
    raise AttributeError(f"Waveform has no parameter '{name}'.")


def _restore_state(wf, state: Dict[str, float]) -> None:
    for key, val in state.items():
        if key.startswith("add_para::"):
            name = key.split("::", 1)[1]
            wf.add_para[name] = val
        else:
            setattr(wf, key, val)


def _set_param_value(wf, name: str, value: float) -> None:
    if name == "Mc":
        eta = wf.eta
        m1, m2 = to_m1m2(value, eta)
        wf.mass1 = m1
        wf.mass2 = m2
        return
    if name == "eta":
        mc = wf.Mc
        m1, m2 = to_m1m2(mc, value)
        wf.mass1 = m1
        wf.mass2 = m2
        return
    if hasattr(wf, name):
        setattr(wf, name, value)
        return
    if hasattr(wf, "add_para") and name in wf.add_para:
        wf.add_para[name] = value
        return
    raise AttributeError(f"Waveform has no parameter '{name}'.")


def _step_size(
    name: str,
    value: float,
    step: Optional[Union[float, Dict[str, float]]],
    rel_step: float,
) -> float:
    if isinstance(step, dict) and name in step:
        return float(step[name])
    if isinstance(step, (int, float)):
        return float(step)
    if name in _ANGLE_PARAMS:
        return float(rel_step)
    v = abs(float(value))
    return float(rel_step * v if v > 0 else rel_step)


def _get_response_fd(
    wf,
    f_series: np.ndarray,
    channel: str,
    det: str,
    TDIgen: int,
    **kwargs,
) -> np.ndarray:
    if not hasattr(wf, "get_tdi_response"):
        raise AttributeError("Waveform object does not provide get_tdi_response().")
    resp = wf.get_tdi_response(
        f_series=f_series,
        channel=channel,
        det=det,
        TDIgen=TDIgen,
        **kwargs,
    )
    if isinstance(resp, tuple):
        resp = resp[0]
    return np.asarray(resp)


def fisher_matrix(
    wf,
    params: Sequence[str],
    *,
    det: str = "TQ",
    channel: str = "AET",
    TDIgen: int = 1,
    f_series: Optional[ArrayLike] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    delta_f: Optional[float] = None,
    noise: Optional[Union[str, BasicNoise]] = None,
    unit: str = "relative_frequency",
    wd_foreground: float = 0.0,
    use_T: bool = True,
    step: Optional[Union[float, Dict[str, float]]] = None,
    rel_step: float = 1e-6,
    return_derivatives: bool = False,
    **resp_kwargs,
) -> Dict[str, object]:
    """Compute Fisher matrix for a waveform object.

    Parameters
    ----------
    wf : waveform instance
        Must provide get_tdi_response().
    params : sequence of str
        Parameter names (e.g. "Mc", "eta", "DL", "tc", "Lambda", "Beta", "psi").
    det, channel, TDIgen : str/int
        Detector and TDI configuration.
    f_series / f_min / f_max / delta_f : float or array
        Frequency grid specification.
    noise : str or BasicNoise
        Noise model or name; defaults to detector mapping.
    step / rel_step : float or dict
        Finite-difference step size.
    return_derivatives : bool
        If True, include derivatives in the result dict.
    resp_kwargs : dict
        Passed through to waveform.get_tdi_response().
    """
    freq = make_frequency_series(wf, f_series, f_min, f_max, delta_f)
    df = float(freq[1] - freq[0])
    psd = get_noise_psd(
        det,
        channel,
        freq,
        noise=noise,
        unit=unit,
        TDIgen=TDIgen,
        wd_foreground=wd_foreground,
    )

    h0 = _get_response_fd(wf, freq, channel, det, TDIgen, **resp_kwargs)
    snr = snr_from_response(h0, psd, df, channel=channel, use_T=use_T)

    n = len(params)
    fisher = np.zeros((n, n), dtype=float)
    derivs: Dict[str, np.ndarray] = {}

    for i, p in enumerate(params):
        base = _get_param_value(wf, p)
        step_i = _step_size(p, base, step, rel_step)
        state = _capture_state(wf, p)

        try:
            if p in _POSITIVE_PARAMS and base - step_i <= 0:
                _set_param_value(wf, p, base + step_i)
                h_plus = _get_response_fd(wf, freq, channel, det, TDIgen, **resp_kwargs)
                deriv = (h_plus - h0) / step_i
            else:
                _set_param_value(wf, p, base + step_i)
                h_plus = _get_response_fd(wf, freq, channel, det, TDIgen, **resp_kwargs)
                _set_param_value(wf, p, base - step_i)
                h_minus = _get_response_fd(wf, freq, channel, det, TDIgen, **resp_kwargs)
                deriv = (h_plus - h_minus) / (2.0 * step_i)
        finally:
            _restore_state(wf, state)

        derivs[p] = deriv

    for i, p in enumerate(params):
        for j in range(i, n):
            q = params[j]
            val = inner_product(
                derivs[p],
                derivs[q],
                psd,
                df,
                channel=channel,
                use_T=use_T,
            )
            fisher[i, j] = val
            fisher[j, i] = val

    cov = covariance_matrix(fisher)
    sigma = {params[i]: float(np.sqrt(cov[i, i])) for i in range(n)}

    result = {
        "params": list(params),
        "fisher": fisher,
        "cov": cov,
        "sigma": sigma,
        "snr": snr,
        "frequency": freq,
    }
    if return_derivatives:
        result["derivatives"] = derivs
    return result


__all__ = [
    "make_frequency_series",
    "get_noise_psd",
    "inner_product",
    "snr_from_response",
    "covariance_matrix",
    "fisher_matrix",
]
