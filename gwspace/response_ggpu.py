#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU-first 响应模块（独立实现，不依赖 response_b）。

默认使用 GPU（CuPy）进行计算；若未安装 CuPy，会自动回退至 NumPy，功能与 CPU 版保持一致。
"""

import numpy as np
from numba import njit
from gwspace.Orbit import detectors

try:
    import cupy as cp
    _has_cupy = True
except Exception:  # pragma: no cover
    cp = None
    _has_cupy = False


def _get_xp(use_gpu: bool):
    return cp if (use_gpu and _has_cupy) else np


def _to_xp(arr, xp):
    if xp is np:
        if _has_cupy and isinstance(arr, cp.ndarray):
            return arr.get()
        return np.asarray(arr)
    return xp.asarray(arr)


def _sinc(x, xp):
    pi_x = xp.pi * x
    return xp.where(x == 0, xp.ones_like(x), xp.sin(pi_x) / pi_x)


@njit
def _matrix_res_pro_cpu(n, p):
    return (n[0] * p[0, 0] * n[0] + n[0] * p[0, 1] * n[1] + n[0] * p[0, 2] * n[2]
            + n[1] * p[1, 0] * n[0] + n[1] * p[1, 1] * n[1] + n[1] * p[1, 2] * n[2]
            + n[2] * p[2, 0] * n[0] + n[2] * p[2, 1] * n[1] + n[2] * p[2, 2] * n[2])


def _matrix_res_pro(n, p, xp):
    if xp is np:
        return _matrix_res_pro_cpu(n, p)
    return (n[0] * p[0, 0] * n[0] + n[0] * p[0, 1] * n[1] + n[0] * p[0, 2] * n[2]
            + n[1] * p[1, 0] * n[0] + n[1] * p[1, 1] * n[1] + n[1] * p[1, 2] * n[2]
            + n[2] * p[2, 0] * n[0] + n[2] * p[2, 1] * n[1] + n[2] * p[2, 2] * n[2])


def get_y_slr_td(wf, tf, det="TQ", TDIgen=1, use_gpu=True):
    if TDIgen == 1:
        TDI_delay = 4
    elif TDIgen == 2:
        TDI_delay = 8
    else:
        raise NotImplementedError

    xp = _get_xp(use_gpu)
    tf_cpu = np.asarray(tf)
    det_obj = detectors[det](tf_cpu)
    p1, p2, p3 = det_obj.orbits
    L = det_obj.L_T
    n1 = det_obj.uni_vec_ij(3, 2)
    n2 = det_obj.uni_vec_ij(1, 3)
    n3 = det_obj.uni_vec_ij(2, 1)

    k = _to_xp(wf.vec_k, xp)
    p_p, p_c = wf.polarization()
    p_p, p_c = _to_xp(p_p, xp), _to_xp(p_c, xp)
    n1, n2, n3 = (_to_xp(n, xp) for n in (n1, n2, n3))
    p1, p2, p3 = (_to_xp(p, xp) for p in (p1, p2, p3))
    tf_xp = _to_xp(tf_cpu, xp)

    xi1 = (_matrix_res_pro(n1, p_p, xp), _matrix_res_pro(n1, p_c, xp))
    xi2 = (_matrix_res_pro(n2, p_p, xp), _matrix_res_pro(n2, p_c, xp))
    xi3 = (_matrix_res_pro(n3, p_p, xp), _matrix_res_pro(n3, p_c, xp))

    tf_kp1 = tf_xp - xp.dot(k, p1)
    tf_kp2 = tf_xp - xp.dot(k, p2)
    tf_kp3 = tf_xp - xp.dot(k, p3)
    kn1 = xp.dot(k, n1)
    kn2 = xp.dot(k, n2)
    kn3 = xp.dot(k, n3)

    def h_tdi_delay(tf_s, xi_p, xi_c):
        def to_cpu(arr):
            return arr.get() if _has_cupy and isinstance(arr, cp.ndarray) else np.asarray(arr)
        h_list = [wf.get_hphc(to_cpu(tf_s - i_ * L)) for i_ in range(TDI_delay + 1)]
        return [_to_xp(hp, xp) * xi_p + _to_xp(hc, xp) * xi_c for (hp, hc) in h_list]

    h3_p2 = h_tdi_delay(tf_kp2, *xi3)
    h3_p1 = h_tdi_delay(tf_kp1, *xi3)
    h2_p3 = h_tdi_delay(tf_kp3, *xi2)
    h2_p1 = h_tdi_delay(tf_kp1, *xi2)
    h1_p3 = h_tdi_delay(tf_kp3, *xi1)
    h1_p2 = h_tdi_delay(tf_kp2, *xi1)

    def get_y(hi_pj, hi_pk, denominator):
        return [(hi_pj[i + 1] - hi_pk[i]) / denominator for i in range(TDI_delay)]

    y_slr = {(1, 2): get_y(h3_p1, h3_p2, 2 * (1 + kn3)),
             (2, 1): get_y(h3_p2, h3_p1, 2 * (1 - kn3)),
             (1, 3): get_y(h2_p1, h2_p3, 2 * (1 - kn2)),
             (3, 1): get_y(h2_p3, h2_p1, 2 * (1 + kn2)),
             (2, 3): get_y(h1_p2, h1_p3, 2 * (1 + kn1)),
             (3, 2): get_y(h1_p3, h1_p2, 2 * (1 - kn1))}
    return y_slr


def tdi_XYZ2AET(X, Y, Z, use_gpu=True):
    xp = _get_xp(use_gpu)
    A = 1 / xp.sqrt(2) * (Z - X)
    E = 1 / xp.sqrt(6) * (X - 2 * Y + Z)
    T = 1 / xp.sqrt(3) * (X + Y + Z)
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


def trans_y_slr_fd(vec_k, p, det, f, use_gpu=True):
    xp = _get_xp(use_gpu)
    vec_k = _to_xp(vec_k, xp)

    u12 = _to_xp(det.uni_vec_ij(1, 2), xp)
    u23 = _to_xp(det.uni_vec_ij(2, 3), xp)
    u13 = _to_xp(det.uni_vec_ij(1, 3), xp)
    ls = det.L_T
    p_1, p_2, p_3 = (_to_xp(p_, xp) for p_ in det.orbits)

    com_f = 1j * xp.pi * f * ls

    vk12 = xp.dot(vec_k, u12)
    vk23 = xp.dot(vec_k, u23)
    vk13 = xp.dot(vec_k, u13)

    exp12 = xp.exp(1j * xp.pi * f * (ls + xp.dot(vec_k, p_1 + p_2)))
    exp23 = xp.exp(1j * xp.pi * f * (ls + xp.dot(vec_k, p_2 + p_3)))
    exp31 = xp.exp(1j * xp.pi * f * (ls + xp.dot(vec_k, p_3 + p_1)))

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
        n31pn31 = _matrix_res_pro(u13, p_, xp)

        y_slr = {(1, 2): y12_pre * n12pn12,
                 (2, 1): y21_pre * n12pn12,
                 (1, 3): y13_pre * n31pn31,
                 (3, 1): y31_pre * n31pn31,
                 (2, 3): y23_pre * n23pn23,
                 (3, 2): y32_pre * n23pn23}
        return y_slr

    if not isinstance(p, tuple):
        p = (p,)
    return tuple(trans_response(p_0) for p_0 in p)


def trans_XYZ_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True):
    xp = _get_xp(use_gpu)
    y_slr_list = trans_y_slr_fd(vec_k, p, det, f, use_gpu)
    Dt = xp.exp(2j * xp.pi * f * det.L_T)
    Dt2 = Dt * Dt

    def trans_xyz(y_slr):
        X = y_slr[(3, 1)] + Dt * y_slr[(1, 3)] - y_slr[(2, 1)] - Dt * y_slr[(1, 2)]
        Y = y_slr[(1, 2)] + Dt * y_slr[(2, 1)] - y_slr[(3, 2)] - Dt * y_slr[(2, 3)]
        Z = y_slr[(2, 3)] + Dt * y_slr[(3, 2)] - y_slr[(1, 3)] - Dt * y_slr[(3, 1)]
        return xp.array([X, Y, Z]) * (1. - Dt2)

    if TDIgen == 1:
        return tuple(trans_xyz(y) for y in y_slr_list)
    elif TDIgen == 2:
        fact = 1 - Dt2 * Dt2
        return tuple(trans_xyz(y) * fact for y in y_slr_list)
    else:
        raise NotImplementedError


def trans_AET_fd(vec_k, p, det, f, TDIgen=1, use_gpu=True):
    xp = _get_xp(use_gpu)
    y_slr_list = trans_y_slr_fd(vec_k, p, det, f, use_gpu)
    Dt = xp.exp(2j * xp.pi * f * det.L_T)
    Dt2 = Dt * Dt

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
        A *= 1 / xp.sqrt(2) * (Dt2 - 1)
        E *= 1 / xp.sqrt(6) * (Dt2 - 1)
        T *= 1 / xp.sqrt(3) * (Dt2 - 1)
        return xp.array([A, E, T])

    if TDIgen == 1:
        return tuple(trans_aet(y) for y in y_slr_list)
    elif TDIgen == 2:
        fact = 1 - Dt2 * Dt2
        return tuple(trans_aet(y) * fact for y in y_slr_list)
    else:
        raise NotImplementedError


get_y_slr_td_backend = get_y_slr_td
get_XYZ_td_backend = get_XYZ_td
get_AET_td_backend = get_AET_td
