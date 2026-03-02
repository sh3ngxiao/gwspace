import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except Exception:
    cp = None

from gwspace.Waveform import waveforms
from gwspace.constants import DAY
from gwspace.response_b import (
    get_y_slr_td,
    get_XYZ_td,
    get_AET_td,
    tdi_XYZ2AET,
)

def to_cpu(x):
    return cp.asnumpy(x) if (cp is not None and isinstance(x, cp.ndarray)) else np.asarray(x)

def generate_td_data(
    pars,
    t_array,
    s_type="gcb",
    det="TQ",
    show_y_slr=False,
    use_gpu=False,
    warmup=True,
):
    print(f"Backend: {'GPU (CuPy)' if use_gpu and cp is not None else 'CPU (NumPy)'}")
    print(f"Generating {s_type} waveforms")
    wf = waveforms[s_type](**pars)

    def sync_gpu():
        if use_gpu and cp is not None:
            cp.cuda.Device().synchronize()

    if show_y_slr:
        sync_gpu()
        st = time.perf_counter()
        y_slr = get_y_slr_td(wf, t_array, det=det, use_gpu=use_gpu)
        sync_gpu()
        ed = time.perf_counter()
        print(f"Time cost: {ed - st:.3f}s for {t_array.shape[0]} points")
        tags = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
        for tag in tags:
            plt.figure()
            for j in range(4):
                plt.subplot(4, 1, j + 1)
                plt.plot(t_array, to_cpu(y_slr[tag][j]))
                plt.title(f"y_{tag} [{j}]L")

    if warmup:
        _X, _Y, _Z = get_XYZ_td(wf, t_array, det=det, TDIgen=1, use_gpu=use_gpu)
        _A, _E, _T = tdi_XYZ2AET(_X, _Y, _Z, use_gpu=use_gpu)
        sync_gpu()

    sync_gpu()
    st = time.perf_counter()
    X, Y, Z = get_XYZ_td(wf, t_array, det=det, TDIgen=1, use_gpu=use_gpu)
    A, E, T = tdi_XYZ2AET(X, Y, Z, use_gpu=use_gpu)  # or get_AET_td(wf, t_array, det=det, use_gpu=use_gpu)
    sync_gpu()
    ed = time.perf_counter()
    print(f"Time cost of calculating XYZ and AET: {ed - st:.3f}s")

    plt.subplots(2, 3, sharex="all", sharey="all")
    for i, (d, label) in enumerate(zip((X, Y, Z, A, E, T), "XYZAET")):
        plt.subplot(2, 3, i + 1)
        plt.plot(t_array[:-5], to_cpu(d)[:-5])
        plt.xlabel("Time (s)")
        plt.ylabel("h")
        plt.title(label)
    plt.tight_layout()

# ------------------ GCB 示例 ------------------
GCBpars = {
    "mass1": 0.5,
    "mass2": 0.5,
    "DL": 0.3,
    "phi0": 0.0,
    "f0": 0.001,
    "psi": 0.2,
    "iota": 0.3,
    "Lambda": 0.4,
    "Beta": 1.2,
    "T_obs": 10 * DAY,
}

delta_t = 0.5  # seconds
tf = np.arange(0, GCBpars["T_obs"], delta_t)

GCBwf = waveforms["gcb"](**GCBpars)
hp, hc = GCBwf.get_hphc(tf)
hp_cpu, hc_cpu = to_cpu(hp), to_cpu(hc)

# Quick waveform plot (optional)
# plt.figure()
# plt.plot(tf[:4000], hp_cpu[:4000], label="h+")
# plt.plot(tf[:4000], hc_cpu[:4000], label="hx")
# plt.xlabel("Time (s)")
# plt.ylabel("h")
# plt.legend()
# plt.tight_layout()

# GPU-aware response (set use_gpu=True if you want to try CuPy)
generate_td_data(GCBpars, tf, s_type="gcb", det="TQ", show_y_slr=False, use_gpu=False, warmup=True)
