import time
import resource
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from gwspace.Waveform import waveforms as waveforms_cpu
from gwspace.waveformgpu import waveforms as waveforms_gpu
from gwspace.constants import DAY
from gwspace.response_m import (
    get_y_slr_td,
    get_XYZ_td,
    get_AET_td,
    tdi_XYZ2AET,
)


def time_block(func, *args, **kwargs):
    st = time.perf_counter()
    out = func(*args, **kwargs)
    ed = time.perf_counter()
    return ed - st, out


def median_time(func, repeats, *args, **kwargs):
    times = []
    last_out = None
    for _ in range(repeats):
        t, last_out = time_block(func, *args, **kwargs)
        times.append(t)
    return float(np.median(times)), last_out


def get_rss_mb():
    # ru_maxrss is KB on Linux
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024.0


def get_meminfo_mb():
    mem_total = None
    mem_free = None
    mem_avail = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / 1024.0
                elif line.startswith("MemFree:"):
                    mem_free = int(line.split()[1]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return mem_total, mem_free, mem_avail


def sync_gpu(gpu_ok):
    if gpu_ok and cp is not None:
        cp.cuda.Device().synchronize()


def run_case(delta_t, pars, det="TQ", tdi_gen=1, use_gpu=False, warmup=True, repeats=3):
    tf = np.arange(0, pars["T_obs"], delta_t)
    npts = tf.size
    gpu_ok = use_gpu and (cp is not None)

    print("\n" + "=" * 60)
    print(f"delta_t = {delta_t} s, N = {npts}")
    print(f"Backend: {'GPU (CuPy)' if gpu_ok else 'CPU (NumPy)'}")
    meminfo = get_meminfo_mb()
    if meminfo is not None:
        mem_total, mem_free, mem_avail = meminfo
        print(f"Mem(MB): total={mem_total:.0f} free={mem_free:.0f} avail={mem_avail:.0f}")
    print(f"RSS(MB) start: {get_rss_mb():.0f}")

    wf_factory = waveforms_gpu if gpu_ok else waveforms_cpu
    t_wf, wf = time_block(wf_factory["gcb"], **pars)
    print(f"RSS(MB) after waveform: {get_rss_mb():.0f}")

    if warmup:
        _ = get_y_slr_td(wf, tf, det, tdi_gen, gpu_ok)
        _ = get_XYZ_td(wf, tf, det=det, TDIgen=tdi_gen, use_gpu=gpu_ok)
        _ = tdi_XYZ2AET(*_, use_gpu=gpu_ok)
        sync_gpu(gpu_ok)

    sync_gpu(gpu_ok)
    t_y, _y_slr = median_time(get_y_slr_td, repeats, wf, tf, det, tdi_gen, gpu_ok)
    sync_gpu(gpu_ok)
    rss_after_y = get_rss_mb()

    t_xyz, (X, Y, Z) = median_time(get_XYZ_td, repeats, wf, tf, det, tdi_gen, gpu_ok)
    sync_gpu(gpu_ok)
    rss_after_xyz = get_rss_mb()

    t_aet, _ = median_time(tdi_XYZ2AET, repeats, X, Y, Z, gpu_ok)
    sync_gpu(gpu_ok)
    rss_after_aet = get_rss_mb()

    t_all, _ = median_time(get_AET_td, repeats, wf, tf, det, tdi_gen, gpu_ok)
    sync_gpu(gpu_ok)
    rss_after_all = get_rss_mb()

    print(f"waveform_init: {t_wf:.4f}s")
    print(f"get_y_slr_td: {t_y:.4f}s")
    print(f"RSS(MB) after y_slr: {rss_after_y:.0f}")
    print(f"get_XYZ_td:   {t_xyz:.4f}s")
    print(f"RSS(MB) after XYZ:  {rss_after_xyz:.0f}")
    print(f"tdi_XYZ2AET:  {t_aet:.4f}s")
    print(f"RSS(MB) after AET:  {rss_after_aet:.0f}")
    print(f"get_AET_td:   {t_all:.4f}s")
    print(f"RSS(MB) after AETall: {rss_after_all:.0f}")

    return {
        "delta_t": delta_t,
        "npts": npts,
        "waveform_init": t_wf,
        "get_y_slr_td": t_y,
        "get_XYZ_td": t_xyz,
        "tdi_XYZ2AET": t_aet,
        "get_AET_td": t_all,
    }


def print_table(results_cpu, results_gpu):
    print("\n" + "=" * 60)
    print(
        f"{'delta_t(s)':>10} {'N':>12} "
        f"{'CPU_wf':>10} {'GPU_wf':>10} {'speed_wf':>10} "
        f"{'CPU_y':>10} {'GPU_y':>10} {'speed_y':>10} "
        f"{'CPU_XYZ':>10} {'GPU_XYZ':>10} {'speed_XYZ':>10} "
        f"{'CPU_AET':>10} {'GPU_AET':>10} {'speed_AET':>10} "
        f"{'CPU_AETall':>12} {'GPU_AETall':>12} {'speed_AETall':>12}"
    )
    for r_cpu, r_gpu in zip(results_cpu, results_gpu):
        speed_wf = r_cpu["waveform_init"] / r_gpu["waveform_init"] if r_gpu["waveform_init"] > 0 else np.nan
        speed_y = r_cpu["get_y_slr_td"] / r_gpu["get_y_slr_td"] if r_gpu["get_y_slr_td"] > 0 else np.nan
        speed_xyz = r_cpu["get_XYZ_td"] / r_gpu["get_XYZ_td"] if r_gpu["get_XYZ_td"] > 0 else np.nan
        speed_aet = r_cpu["tdi_XYZ2AET"] / r_gpu["tdi_XYZ2AET"] if r_gpu["tdi_XYZ2AET"] > 0 else np.nan
        speed_aet_all = r_cpu["get_AET_td"] / r_gpu["get_AET_td"] if r_gpu["get_AET_td"] > 0 else np.nan
        print(
            f"{r_cpu['delta_t']:10.3f} {r_cpu['npts']:12d} "
            f"{r_cpu['waveform_init']:10.3f} {r_gpu['waveform_init']:10.3f} {speed_wf:10.2f} "
            f"{r_cpu['get_y_slr_td']:10.3f} {r_gpu['get_y_slr_td']:10.3f} {speed_y:10.2f} "
            f"{r_cpu['get_XYZ_td']:10.3f} {r_gpu['get_XYZ_td']:10.3f} {speed_xyz:10.2f} "
            f"{r_cpu['tdi_XYZ2AET']:10.3f} {r_gpu['tdi_XYZ2AET']:10.3f} {speed_aet:10.2f} "
            f"{r_cpu['get_AET_td']:12.3f} {r_gpu['get_AET_td']:12.3f} {speed_aet_all:12.2f}"
        )


if __name__ == "__main__":
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

    delta_t_list = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]

    results_cpu = []
    results_gpu = []

    repeats = 5
    for dt in delta_t_list:
        results_cpu.append(run_case(dt, GCBpars, use_gpu=False, repeats=repeats))
        results_gpu.append(run_case(dt, GCBpars, use_gpu=True, repeats=repeats))

    print_table(results_cpu, results_gpu)
