#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==================================
# File Name: response_m.py
# Author: Unified NumPy/CuPy version (CPU/GPU selectable)
# ==================================
"""
Unified response module with CPU/GPU selectable backend and robust shape handling.

Key features:
- use_gpu=True & CuPy available -> GPU path; otherwise CPU/NumPy (optionally Numba).
- Time-domain path batches all TDI delays to minimize CPU<->GPU transfers.
- Robust inner-products: supports orbits shaped (3, T) or (T, 3); k/n as (3,), (3,1), or (1,3).
- Frequency-domain transfer functions compatible with both backends.

Public API:
    get_y_slr_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True)
    get_XYZ_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True)
    get_AET_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True)
    tdi_XYZ2AET(X, Y, Z, use_gpu=True)
    trans_y_slr_fd(vec_k, p, det, f, use_gpu=True)
    trans_XYZ_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True)
    trans_AET_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True)
"""

from typing import Tuple, Dict
import numpy as np
from gwspace.Orbit import detectors

# Optional Numba for CPU fast path
try:
    from numba import njit
    _has_numba = True
except Exception:  # pragma: no cover
    njit = None
    _has_numba = False

# Optional CuPy for GPU path
try:
    import cupy as cp  # noqa
    _has_cupy = True
except Exception:  # pragma: no cover
    cp = None
    _has_cupy = False

# Host-side constants (broadcast-safe)
SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
SQRT6 = np.sqrt(6.0)


# ---------------- Backend helpers ----------------

def _get_xp(use_gpu: bool):
    return cp if (use_gpu and _has_cupy) else np


def _to_xp(arr, xp):
    """Convert array to xp backend without unnecessary copies."""
    if xp is np:
        if _has_cupy and isinstance(arr, cp.ndarray):
            return arr.get()
        return arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    # xp is CuPy
    if _has_cupy and isinstance(arr, cp.ndarray):
        return arr
    return xp.asarray(arr)


def _sinc(x, xp):
    """sinc(x) = sin(pi x)/(pi x) with stable x≈0 handling."""
    if hasattr(xp, "sinc"):
        return xp.sinc(x)
    pi_x = xp.pi * x
    y = xp.sin(pi_x) / pi_x
    return xp.where(xp.isclose(x, 0, atol=1e-12), xp.ones_like(x), y)


# Robust inner-product helpers (handle shapes (3,T) / (T,3) and 3-vectors)

def _ensure_vec1d(v, xp):
    v = _to_xp(v, xp).reshape(-1)
    if v.size != 3:
        raise ValueError(f"Expected 3-vector, got shape {tuple(v.shape)} with size {v.size}")
    return v


def _normalize_vec_k(vec_k, xp):
    """Normalize vec_k to shape (3,) for einsum signatures."""
    return _ensure_vec1d(vec_k, xp)


def _k_dot_orbit(k, P, xp):
    """
    k·P with P shaped (3,T) or (T,3).
    Returns shape (T,).
    """
    k = _ensure_vec1d(k, xp)
    P = _to_xp(P, xp)
    if P.ndim != 2 or 3 not in P.shape:
        raise ValueError(f"Orbit matrix must be 2D with one dim=3, got {P.shape}")
    if P.shape[0] == 3:
        # (3,) · (3, T) -> (T,)
        return xp.tensordot(k, P, axes=(0, 0))
    else:
        # (T,3) · (3,) -> (T,)
        return xp.dot(P, k)


def _k_dot_vec(k, v, xp):
    """k·v where v is a 3-vector; returns scalar."""
    k = _ensure_vec1d(k, xp)
    v = _ensure_vec1d(v, xp)
    return xp.dot(k, v)


# Matrix contraction n^T P n

if _has_numba:
    @njit
    def _matrix_res_pro_cpu(n, p):
        return (n[0] * p[0, 0] * n[0] + n[0] * p[0, 1] * n[1] + n[0] * p[0, 2] * n[2]
                + n[1] * p[1, 0] * n[0] + n[1] * p[1, 1] * n[1] + n[1] * p[1, 2] * n[2]
                + n[2] * p[2, 0] * n[0] + n[2] * p[2, 1] * n[1] + n[2] * p[2, 2] * n[2])
else:
    def _matrix_res_pro_cpu(n, p):
        return (n[0] * p[0, 0] * n[0] + n[0] * p[0, 1] * n[1] + n[0] * p[0, 2] * n[2]
                + n[1] * p[1, 0] * n[0] + n[1] * p[1, 1] * n[1] + n[1] * p[1, 2] * n[2]
                + n[2] * p[2, 0] * n[0] + n[2] * p[2, 1] * n[1] + n[2] * p[2, 2] * n[2])


def _matrix_res_pro(n, p, xp):
    """Backend-agnostic n^T P n."""
    if xp is np:
        return _matrix_res_pro_cpu(n, p)
    n = _to_xp(n, xp)
    p = _to_xp(p, xp)
    if n.ndim == 1:
        return xp.einsum("i,ij,j->", n, p, n)
    if n.ndim == 2 and n.shape[0] == 3:
        return xp.einsum("it,ij,jt->t", n, p, n)
    if n.ndim == 2 and n.shape[1] == 3:
        return xp.einsum("ti,ij,tj->t", n, p, n)
    raise ValueError(f"Unexpected n shape for matrix contraction: {n.shape}")


# ---------------- Time-domain (TD) ----------------

def get_y_slr_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True) -> Dict[Tuple[int, int], list]:
    """
    Compute single-link y_ij in time domain with batched delays to minimize transfers.
    """
    if TDIgen == 1:
        TDI_delay = 4
    elif TDIgen == 2:
        TDI_delay = 8
    else:
        raise NotImplementedError

    xp = _get_xp(use_gpu)

    # Detector expects NumPy tf
    tf_cpu = np.asarray(tf)
    det_obj = detectors[det](tf_cpu)

    # Orbits & unit vectors (NumPy) -> xp
    p1, p2, p3 = det_obj.orbits
    L = det_obj.L_T
    n1 = det_obj.uni_vec_ij(3, 2)
    n2 = det_obj.uni_vec_ij(1, 3)
    n3 = det_obj.uni_vec_ij(2, 1)

    # Normalize k to shape (3,) to satisfy einsum signatures
    k = _normalize_vec_k(wf.vec_k, xp)
    p_p, p_c = wf.polarization()
    p_p, p_c = _to_xp(p_p, xp), _to_xp(p_c, xp)
    n1, n2, n3 = (_to_xp(n, xp) for n in (n1, n2, n3))
    p1, p2, p3 = (_to_xp(p, xp) for p in (p1, p2, p3))
    tf_xp = _to_xp(tf_cpu, xp)

    # Projection scalars
    xi1 = (_matrix_res_pro(n1, p_p, xp), _matrix_res_pro(n1, p_c, xp))
    xi2 = (_matrix_res_pro(n2, p_p, xp), _matrix_res_pro(n2, p_c, xp))
    xi3 = (_matrix_res_pro(n3, p_p, xp), _matrix_res_pro(n3, p_c, xp))

    # Robust k·p and k·n
    tf_kp1 = tf_xp - _k_dot_orbit(k, p1, xp)
    tf_kp2 = tf_xp - _k_dot_orbit(k, p2, xp)
    tf_kp3 = tf_xp - _k_dot_orbit(k, p3, xp)
    # n1/n2/n3 are time-series vectors shaped (3, T) or (T, 3)
    kn1 = _k_dot_orbit(k, n1, xp)
    kn2 = _k_dot_orbit(k, n2, xp)
    kn3 = _k_dot_orbit(k, n3, xp)

    def h_tdi_delay(tf_s, xi_p, xi_c):
        """
        Return [h(t - iL)*xi] (i=0..D) on xp backend.
        GPU path stays on GPU; CPU path stays on CPU.
        """
        D = TDI_delay
        if xp is np:
            tf_stack = np.stack([tf_s - i_ * L for i_ in range(D + 1)], axis=0)
            hp_stack, hc_stack = wf.get_hphc(tf_stack)
            h_stack = hp_stack * xi_p + hc_stack * xi_c
            return [h_stack[i] for i in range(D + 1)]

        tf_stack = xp.stack([tf_s - i_ * L for i_ in range(D + 1)], axis=0)
        hp_stack, hc_stack = wf.get_hphc(tf_stack)
        hp_stack = _to_xp(hp_stack, xp)
        hc_stack = _to_xp(hc_stack, xp)
        h_stack = hp_stack * xi_p + hc_stack * xi_c
        return [h_stack[i] for i in range(D + 1)]

    # Build needed delayed waveforms
    h3_p2 = h_tdi_delay(tf_kp2, *xi3)
    h3_p1 = h_tdi_delay(tf_kp1, *xi3)
    h2_p3 = h_tdi_delay(tf_kp3, *xi2)
    h2_p1 = h_tdi_delay(tf_kp1, *xi2)
    h1_p3 = h_tdi_delay(tf_kp3, *xi1)
    h1_p2 = h_tdi_delay(tf_kp2, *xi1)

    def get_y(hi_pj, hi_pk, denominator):
        return [(hi_pj[i + 1] - hi_pk[i]) / denominator for i in range(TDI_delay)]

    y_slr = {
        (1, 2): get_y(h3_p1, h3_p2, 2 * (1 + kn3)),
        (2, 1): get_y(h3_p2, h3_p1, 2 * (1 - kn3)),
        (1, 3): get_y(h2_p1, h2_p3, 2 * (1 - kn2)),
        (3, 1): get_y(h2_p3, h2_p1, 2 * (1 + kn2)),
        (2, 3): get_y(h1_p2, h1_p3, 2 * (1 + kn1)),
        (3, 2): get_y(h1_p3, h1_p2, 2 * (1 - kn1)),
    }
    return y_slr


def tdi_XYZ2AET(X, Y, Z, use_gpu=True):
    xp = _get_xp(use_gpu)
    inv_s2 = 1.0 / SQRT2
    inv_s6 = 1.0 / SQRT6
    inv_s3 = 1.0 / SQRT3
    A = inv_s2 * (Z - X)
    E = inv_s6 * (X - 2 * Y + Z)
    T = inv_s3 * (X + Y + Z)
    return A, E, T


def get_XYZ_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True):
    xp = _get_xp(use_gpu)
    y_slr = get_y_slr_td(wf, tf, det, TDIgen, use_gpu)
    y31, y13 = y_slr[(3, 1)], y_slr[(1, 3)]
    y12, y21 = y_slr[(1, 2)], y_slr[(2, 1)]
    y23, y32 = y_slr[(2, 3)], y_slr[(3, 2)]

    if TDIgen == 1:
        X = (y31[0] + y13[1] + y21[2] + y12[3] - y21[0] - y12[1] - y31[2] - y13[3])
        Y = (y12[0] + y21[1] + y32[2] + y23[3] - y32[0] - y23[1] - y12[2] - y21[3])
        Z = (y23[0] + y32[1] + y13[2] + y31[3] - y13[0] - y31[1] - y23[2] - y32[3])
    elif TDIgen == 2:
        X = (y31[0] + y13[1] + y21[2] + y12[3] + y21[4] + y12[5] + y31[6] + y13[7]
             - y21[0] - y12[1] - y31[2] - y13[3] - y31[4] - y13[5] - y21[6] - y12[7])
        Y = (y12[0] + y21[1] + y32[2] + y23[3] + y32[4] + y23[5] + y12[6] + y21[7]
             - y32[0] - y23[1] - y12[2] - y21[3] - y12[4] - y21[5] - y32[6] - y23[7])
        Z = (y23[0] + y32[1] + y13[2] + y31[3] + y13[4] + y31[5] + y23[6] + y32[7]
             - y13[0] - y31[1] - y23[2] - y32[3] - y23[4] - y32[5] - y13[6] - y31[7])
    else:
        raise NotImplementedError

    return xp.asarray(X), xp.asarray(Y), xp.asarray(Z)


def get_AET_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True):
    X, Y, Z = get_XYZ_td(wf, tf, det, TDIgen, use_gpu)
    return tdi_XYZ2AET(X, Y, Z, use_gpu)


# ---------------- Frequency-domain (FD) ----------------

def trans_y_slr_fd(vec_k, p, det, f, use_gpu=True):
    """
    Frequency-domain y_slr transfer functions (backend-aware).
    Returns transfer dicts to be multiplied with FD waveform.
    """
    xp = _get_xp(use_gpu)
    vec_k = _normalize_vec_k(vec_k, xp)
    f = _to_xp(f, xp)

    u12 = _to_xp(det.uni_vec_ij(1, 2), xp)
    u23 = _to_xp(det.uni_vec_ij(2, 3), xp)
    u13 = _to_xp(det.uni_vec_ij(1, 3), xp)
    ls = det.L_T
    p_1, p_2, p_3 = (_to_xp(p_, xp) for p_ in det.orbits)

    com_f = 1j * xp.pi * f * ls

    vk12 = _k_dot_vec(vec_k, u12, xp)
    vk23 = _k_dot_vec(vec_k, u23, xp)
    vk13 = _k_dot_vec(vec_k, u13, xp)

    exp12 = xp.exp(1j * xp.pi * f * (ls + _k_dot_orbit(vec_k, p_1 + p_2, xp)))
    exp23 = xp.exp(1j * xp.pi * f * (ls + _k_dot_orbit(vec_k, p_2 + p_3, xp)))
    exp31 = xp.exp(1j * xp.pi * f * (ls + _k_dot_orbit(vec_k, p_3 + p_1, xp)))

    y12_pre = com_f * _sinc(f * ls * (1 - vk12), xp) * exp12
    y21_pre = com_f * _sinc(f * ls * (1 + vk12), xp) * exp12
    y13_pre = com_f * _sinc(f * ls * (1 - vk13), xp) * exp31
    y31_pre = com_f * _sinc(f * ls * (1 + vk13), xp) * exp31
    y23_pre = com_f * _sinc(f * ls * (1 - vk23), xp) * exp23
    y32_pre = com_f * _sinc(f * ls * (1 + vk23), xp) * exp23

    def trans_response(p_):
        p_ = _to_xp(p_, xp)
        n12pn12 = _matrix_res_pro(u12, p_, xp)
        n23pn23 = _matrix_res_pro(u23, p_, xp)
        n13pn13 = _matrix_res_pro(u13, p_, xp)

        y_slr = {
            (1, 2): y12_pre * n12pn12,
            (2, 1): y21_pre * n12pn12,
            (1, 3): y13_pre * n13pn13,
            (3, 1): y31_pre * n13pn13,
            (2, 3): y23_pre * n23pn23,
            (3, 2): y32_pre * n23pn23,
        }
        return y_slr

    if not isinstance(p, tuple):
        p = (p,)
    return tuple(trans_response(p_0) for p_0 in p)


def trans_XYZ_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True):
    """
    Calculate XYZ transfer functions in frequency domain (backend-aware).
    Returns transfer functions; multiply with FD waveform to get response.
    """
    xp = _get_xp(use_gpu)
    y_slr_list = trans_y_slr_fd(vec_k, p, det, f, use_gpu)
    Dt = xp.exp(2j * xp.pi * f * det.L_T)
    Dt2 = Dt * Dt

    def trans_xyz(y_slr):
        X = y_slr[(3, 1)] + Dt * y_slr[(1, 3)] - y_slr[(2, 1)] - Dt * y_slr[(1, 2)]
        Y = y_slr[(1, 2)] + Dt * y_slr[(2, 1)] - y_slr[(3, 2)] - Dt * y_slr[(2, 3)]
        Z = y_slr[(2, 3)] + Dt * y_slr[(3, 2)] - y_slr[(1, 3)] - Dt * y_slr[(3, 1)]
        return xp.array([X, Y, Z]) * (1.0 - Dt2)

    if TDIgen == 1:
        return tuple(trans_xyz(y) for y in y_slr_list)
    elif TDIgen == 2:
        fact = 1.0 - Dt2 * Dt2
        return tuple(trans_xyz(y) * fact for y in y_slr_list)
    else:
        raise NotImplementedError


def trans_AET_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True):
    """
    Calculate AET transfer functions in frequency domain (backend-aware).
    Returns transfer functions; multiply with FD waveform to get response.
    """
    xp = _get_xp(use_gpu)
    y_slr_list = trans_y_slr_fd(vec_k, p, det, f, use_gpu)
    Dt = xp.exp(2j * xp.pi * f * det.L_T)
    Dt2 = Dt * Dt

    inv_s2 = 1.0 / SQRT2
    inv_s6 = 1.0 / SQRT6
    inv_s3 = 1.0 / SQRT3

    def trans_aet(y_slr):
        A = ((1 + Dt) * (y_slr[(3, 1)] + y_slr[(1, 3)])
             - y_slr[(2, 3)] - Dt * y_slr[(3, 2)]
             - y_slr[(2, 1)] - Dt * y_slr[(1, 2)])
        E = ((1 - Dt) * (y_slr[(1, 3)] - y_slr[(3, 1)])
             + (1 + 2 * Dt) * (y_slr[(2, 1)] - y_slr[(2, 3)])
             + (2 + Dt) * (y_slr[(1, 2)] - y_slr[(3, 2)]))
        T = (1 - Dt) * (y_slr[(1, 3)] - y_slr[(3, 1)]
                        + y_slr[(2, 1)] - y_slr[(1, 2)]
                        + y_slr[(3, 2)] - y_slr[(2, 3)])
        A *= inv_s2 * (Dt2 - 1.0)
        E *= inv_s6 * (Dt2 - 1.0)
        T *= inv_s3 * (Dt2 - 1.0)
        return xp.array([A, E, T])

    if TDIgen == 1:
        return tuple(trans_aet(y) for y in y_slr_list)
    elif TDIgen == 2:
        fact = 1.0 - Dt2 * Dt2
        return tuple(trans_aet(y) * fact for y in y_slr_list)
    else:
        raise NotImplementedError


# Backward-compatible aliases
get_y_slr_td_backend = get_y_slr_td
get_XYZ_td_backend = get_XYZ_td
get_AET_td_backend = get_AET_td

__all__ = [
    "get_y_slr_td", "get_XYZ_td", "get_AET_td",
    "tdi_XYZ2AET",
    "trans_y_slr_fd", "trans_XYZ_fd", "trans_AET_fd",
    "get_y_slr_td_backend", "get_XYZ_td_backend", "get_AET_td_backend",
]
