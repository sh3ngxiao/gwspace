# --- Cell 1: 强制用本地仓库版本（可选，但建议） ---
import sys
import importlib

sys.path.insert(0, "/home/sh3ng/projects/GWSpace")
import gwspace.fishertool as ft

importlib.reload(ft)

# --- Cell 2: 基本导入 ---
import numpy as np
try:
    from scipy.signal.windows import tukey
except ImportError:  # fallback for older scipy
    def tukey(M, alpha=0.5):
        if M <= 0:
            return np.array([])
        if alpha <= 0:
            return np.ones(M)
        if alpha >= 1:
            return np.hanning(M)
        n = np.arange(M)
        per = alpha * (M - 1) / 2.0
        w = np.ones(M)
        first = n < per
        last = n >= (M - per)
        w[first] = 0.5 * (1 + np.cos(np.pi * (2 * n[first] / (alpha * (M - 1)) - 1)))
        w[last] = 0.5 * (1 + np.cos(np.pi * (2 * n[last] / (alpha * (M - 1)) - 2 / alpha + 1)))
        return w

from gwspace.Waveform import EMRIWaveform
from gwspace.response import get_AET_td
from gwspace.Noise import TianQinNoise
from gwspace.constants import YRSID_SI
from gwspace.fishertool import fisher_matrix

# --- Cell 3: EMRI 参数 ---
EMRIpars = {
    "M": 4.3e6,
    "mu": 40.0,
    "a": 0.1,
    "p0": 9.0,
    "e0": 0.0,
    "x0": 1.0,
    "qS": 1.66,
    "phiS": 4.71,
    "qK": 0.2,
    "phiK": 0.2,
    "dist": 8.0e-6,
    "Phi_phi0": 1.0,
    "Phi_theta0": 2.0,
    "Phi_r0": 3.0,
    "T_obs": 5 * YRSID_SI,
}

# --- Cell 4: EMRIWaveform 实例 ---
wf = EMRIWaveform(**EMRIpars)


# --- Cell 5: EMRI -> AET 频域 wrapper（带参数转发） ---
class EMRIAETAdapter:
    _local = {"wf", "dt", "det", "TDIgen"}

    def __init__(self, wf, dt, det="TQ", TDIgen=1):
        object.__setattr__(self, "wf", wf)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "det", det)
        object.__setattr__(self, "TDIgen", TDIgen)

    def __getattr__(self, name):
        return getattr(self.wf, name)

    def __setattr__(self, name, value):
        if name in self._local:
            object.__setattr__(self, name, value)
        else:
            setattr(self.wf, name, value)

    def get_tdi_response(self, f_series=None, channel="AET", det=None, TDIgen=None, **kwargs):
        if channel.upper() != "AET":
            raise ValueError("EMRIAETAdapter 只支持 AET 通道。")
        det = det or self.det
        TDIgen = TDIgen or self.TDIgen

        t = np.arange(0, self.wf.T_obs, self.dt)
        A, E, T = get_AET_td(self.wf, t, det=det, TDIgen=TDIgen)
        win = tukey(len(t), 0.1)

        N = len(t)
        f_fft = np.fft.rfftfreq(N, self.dt)
        A_f = np.fft.rfft(A * win) * self.dt
        E_f = np.fft.rfft(E * win) * self.dt
        T_f = np.fft.rfft(T * win) * self.dt
        h = np.vstack([A_f, E_f, T_f])

        if f_series is None:
            return h

        f_series = np.asarray(f_series)
        if f_series.shape == f_fft.shape and np.allclose(f_series, f_fft):
            return h

        def interp_complex(y):
            return np.interp(f_series, f_fft, y.real) + 1j * np.interp(f_series, f_fft, y.imag)

        return np.vstack([interp_complex(A_f), interp_complex(E_f), interp_complex(T_f)])


# --- Cell 6: 频率轴与 FIM ---
dt = 20.0  # 秒，先用粗一点调试
adapter = EMRIAETAdapter(wf, dt=dt, det="TQ", TDIgen=1)

# 和 FFT 完全一致的频率轴，避免插值
t = np.arange(0, wf.T_obs, dt)
f_series = np.fft.rfftfreq(len(t), dt)
fmin, fmax = 1e-4, 1e-2
mask = (f_series >= fmin) & (f_series <= fmax)
f_series = f_series[mask]

# 参数列表（psi/iota 对 EMRIWaveform 不生效，建议先别加）
params = [
    "M",
    "mu",
    "p0",
]

result = fisher_matrix(
    adapter,
    params=params,
    det="TQ",
    channel="AET",
    TDIgen=1,
    f_series=f_series,
    noise=TianQinNoise(),
    use_T=False,
    rel_step=1e-6,
)

print("SNR:", result["snr"])
print("sigma:", result["sigma"])
