import time
import numpy as np
import matplotlib.pyplot as plt

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

def to_cpu(x):
    return cp.asnumpy(x) if (cp is not None and isinstance(x, cp.ndarray)) else np.asarray(x)


# def median_time(func, repeats, *args, **kwargs):
#     times = []
#     last_out = None
#     for _ in range(repeats):
#         st = time.perf_counter()
#         last_out = func(*args, **kwargs)
#         ed = time.perf_counter()
#         times.append(ed - st)
#     return float(np.median(times)), last_out

def median_time(func, repeats, synchronizer=None, *args, **kwargs):
    times = []
    last_out = None
    
    # 预热一次（可选，防止第一次启动慢影响平均值）
    func(*args, **kwargs)
    if synchronizer: synchronizer()
    
    for _ in range(repeats):
        # 1. 确保在开始计时前，GPU是空闲的
        if synchronizer: synchronizer() 
        
        st = time.perf_counter()
        
        # 2. 执行任务
        last_out = func(*args, **kwargs)
        
        # 3. 关键点：在停止计时前，必须强制等待 GPU 完成！
        if synchronizer: synchronizer() 
        
        ed = time.perf_counter()
        times.append(ed - st)
        
    return float(np.median(times)), last_out

def generate_td_data(
    pars,
    t_array,
    s_type="gcb",
    det="TQ",
    show_y_slr=False,
    use_gpu=False,
    warmup=True,
    do_plot=True,  # 新增：是否画图
    repeats=5,
):
    # 如果 cupy 不可用，则自动退回 CPU
    gpu_ok = (use_gpu and (cp is not None))

    print(f"Backend: {'GPU (CuPy)' if gpu_ok else 'CPU (NumPy)'}")
    print(f"Generating {s_type} waveforms")
    wf_factory = waveforms_gpu if gpu_ok else waveforms_cpu
    wf = wf_factory[s_type](**pars)

    def sync_gpu():
        if gpu_ok:
            cp.cuda.Device().synchronize()

    # 可选：显示 y_slr
    if show_y_slr:
        sync_gpu()
        st = time.perf_counter()
        y_slr = get_y_slr_td(wf, t_array, det=det, use_gpu=gpu_ok)
        sync_gpu()
        ed = time.perf_counter()
        print(f"Time cost (y_slr): {ed - st:.3f}s for {t_array.shape[0]} points")

        if do_plot:
            tags = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
            for tag in tags:
                plt.figure()
                for j in range(4):
                    plt.subplot(4, 1, j + 1)
                    plt.plot(t_array, to_cpu(y_slr[tag][j]))
                    plt.title(f"y_{tag} [{j}]L")

    # warmup（GPU 对比必开）
    if warmup:
        _ = get_y_slr_td(wf, t_array, det=det, use_gpu=gpu_ok)
        _X, _Y, _Z = get_XYZ_td(wf, t_array, det=det, TDIgen=1, use_gpu=gpu_ok)
        _A, _E, _T = tdi_XYZ2AET(_X, _Y, _Z, use_gpu=gpu_ok)
        sync_gpu()

    # 正式计时：XYZ + AET（取中位数）
    sync_gpu()
    cost, _ = median_time(
        lambda: tdi_XYZ2AET(*get_XYZ_td(wf, t_array, det=det, TDIgen=1, use_gpu=gpu_ok), use_gpu=gpu_ok),
        repeats,
    )
    sync_gpu()
    print(f"Time cost (XYZ + AET): {cost:.3f}s (median of {repeats})")

    # 取一次结果用于后续绘图
    X, Y, Z = get_XYZ_td(wf, t_array, det=det, TDIgen=1, use_gpu=gpu_ok)
    A, E, T = tdi_XYZ2AET(X, Y, Z, use_gpu=gpu_ok)

    # 可选：画 X/Y/Z/A/E/T
    if do_plot:
        plt.figure(figsize=(12, 6))
        plt.subplots(2, 3, sharex="all", sharey="all")
        for i, (d, label) in enumerate(zip((X, Y, Z, A, E, T), "XYZAET")):
            plt.subplot(2, 3, i + 1)
            plt.plot(t_array[:-5], to_cpu(d)[:-5])
            plt.xlabel("Time (s)")
            plt.ylabel("h")
            plt.title(label)
        plt.tight_layout()

    return cost

# ------------------ GCB 示例参数 ------------------
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

# ------------------ 扫 delta_t：CPU vs GPU ------------------

delta_t_list = [8.0, 4.0, 2.0, 1.0, 0.5]

results = []  # (delta_t, npts, cpu_s, gpu_s, speedup)

for delta_t in delta_t_list:
    tf = np.arange(0, GCBpars["T_obs"], delta_t)
    npts = tf.size
    print("\n" + "=" * 60)
    print(f"delta_t = {delta_t} s, N = {npts}")

    # CPU
    cpu_s = generate_td_data(
        GCBpars,
        tf,
        s_type="gcb",
        det="TQ",
        show_y_slr=False,
        use_gpu=False,
        warmup=True,
        do_plot=False,   
        repeats=5,
    )

    # GPU（若 cupy 不存在会自动退回 CPU）
    gpu_s = generate_td_data(
        GCBpars,
        tf,
        s_type="gcb",
        det="TQ",
        show_y_slr=False,
        use_gpu=True,
        warmup=True,
        do_plot=False,
        repeats=5,
    )

    speedup = (cpu_s / gpu_s) if gpu_s > 0 else np.nan
    results.append((delta_t, npts, cpu_s, gpu_s, speedup))

# ------------------ 打印结果 ------------------
print("\n" + "=" * 60)
print(f"{'delta_t(s)':>10} {'N':>12} {'CPU(s)':>10} {'GPU(s)':>10} {'speedup':>10}")
for dt, n, cs, gs, sp in results:
    print(f"{dt:10.3f} {n:12d} {cs:10.3f} {gs:10.3f} {sp:10.2f}")

# ------------------ 可选：画耗时与加速比 ------------------
dts = np.array([r[0] for r in results], dtype=float)
cpu_t = np.array([r[2] for r in results], dtype=float)
gpu_t = np.array([r[3] for r in results], dtype=float)

plt.figure()
plt.plot(dts, cpu_t, marker="o", label="CPU")
plt.plot(dts, gpu_t, marker="o", label="GPU")
plt.gca().invert_xaxis()  
plt.yscale("log")
plt.xlabel("delta_t (s)")
plt.ylabel("time (s) [log]")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(dts, cpu_t / gpu_t, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("delta_t (s)")
plt.ylabel("CPU/GPU speedup")
plt.tight_layout()

plt.show()
