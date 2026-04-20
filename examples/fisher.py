import importlib
import os
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.signal.windows import tukey
except ImportError:  # 兼容较旧的 scipy 版本
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

sys.path.insert(0, "/home/sh3ng/projects/GWSpace")
import gwspace.fishertool as ft
from gwspace.Noise import TianQinNoise
from gwspace.Orbit import detectors
from gwspace.Waveform import EMRIWaveform
from gwspace.constants import YRSID_SI
from gwspace.fishertool import fisher_matrix
from gwspace.response import get_AET_td, tdi_XYZ2AET

importlib.reload(ft)


def _parse_csv_floats(text):
    """将环境变量中的逗号分隔字符串解析为浮点数列表。

    例如: "1e-6,1e-5,1e-4" -> [1e-6, 1e-5, 1e-4]
    """
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty float csv string")
    return [float(x) for x in values]


def _parse_csv_strings(text):
    """将逗号分隔字符串解析为字符串列表。"""
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty string csv")
    return values


def _env_flag(name, default=True):
    """读取环境变量中的布尔开关。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ====== EMRI 参数 ======
T_OBS_YR = float(os.getenv("FISHER_TOBS_YR", "5.0"))

EMRIpars = {
    "M": 4.3e6,
    "mu": 40.0,
    "a": 0.5,
    "p0": 9.0,
    "e0": 0.0,
    "x0": 1.0,
    "qS": 1.66,
    "phiS": 4.71,
    "qK": 0.2,
    "phiK": 0.2,
    "dist": 8.0e-6,
    "Phi_phi0": 0.0,
    "Phi_theta0": 0.0,
    "Phi_r0": 0.0,
    "T_obs": T_OBS_YR * YRSID_SI,
}

# ====== 分析配置 ======
DT = float(os.getenv("FISHER_DT", "20.0"))
FMIN = 1e-4
FMAX = 1e-2
PARAMS = _parse_csv_strings(os.getenv("FISHER_PARAMS", "M,mu"))
REL_STEPS = _parse_csv_floats(os.getenv("FISHER_REL_STEPS", "1e-6,1e-5,1e-4"))  # 用于收敛性检查
USE_T = False
WINDOW_ALPHA = 0.1
MAX_FREQ_BINS = int(os.getenv("FISHER_MAX_BINS", "120000"))  # 降低频点数，避免 5 年数据直接 OOM/超时
USE_INTERP_RESPONSE = _env_flag("FISHER_USE_INTERP_RESPONSE", True)
PINV_RCOND = float(os.getenv("FISHER_PINV_RCOND", "1e-12"))  # 伪逆截断阈值
ENABLE_SCALED_FISHER = _env_flag("FISHER_ENABLE_SCALED", True)  # 是否输出参数缩放后的 Fisher 诊断
REPORT_PER_PARAM = _env_flag("FISHER_REPORT_PER_PARAM", True)  # 是否逐参数打印敏感性
SCALE_MODE = os.getenv("FISHER_SCALE_MODE", "value").strip().lower()  # value/ones
PRIOR_REL = float(os.getenv("FISHER_PRIOR_REL", "0.0"))  # 相对先验(0=关闭): sigma_prior_i = PRIOR_REL * |theta_i|
PLOT_CORNER = _env_flag("FISHER_PLOT_CORNER", True)  # 是否绘制 corner 风格图
CORNER_COV_MODE = os.getenv("FISHER_CORNER_COV_MODE", "scaled").strip().lower()  # raw/scaled
CORNER_MAX_MODELS = int(os.getenv("FISHER_CORNER_MAX_MODELS", "3"))  # 叠加步长曲线数量上限
# 可选自定义图例标签（默认留空，自动使用 rel_step 标签，避免“多模型”误解）。
_CORNER_LABELS_RAW = os.getenv("FISHER_CORNER_LABELS", "").strip()
CORNER_LABELS = _parse_csv_strings(_CORNER_LABELS_RAW) if _CORNER_LABELS_RAW else []
CORNER_DIR = os.getenv("FISHER_CORNER_DIR", "examples/fisher_results")
CORNER_TAG = os.getenv("FISHER_CORNER_TAG", "")


class EMRIAETAdapter:
    """EMRI -> AET 频域适配器。

    主要职责:
    - 生成时域 A/E/T 响应并做 FFT 转到频域；
    - 对相同波形参数状态启用缓存，减少重复计算；
    - 应用 Tukey 窗与可选 RMS 功率校正；
    - 将复频谱插值到 fisher_matrix 需要的频率网格。
    """

    _local = {
        "wf", "dt", "det", "TDIgen", "window_alpha",
        "window_power_correction", "use_interp_response", "_cache", "_cache_size",
    }
    _state_params = (
        "M", "mu", "a", "p0", "e0", "x0", "dist",
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0", "T_obs",
    )

    def __init__(
        self,
        wf,
        dt,
        det="TQ",
        TDIgen=1,
        window_alpha=0.1,
        window_power_correction=True,
        use_interp_response=True,
        cache_size=16,
    ):
        object.__setattr__(self, "wf", wf)
        object.__setattr__(self, "dt", float(dt))
        object.__setattr__(self, "det", det)
        object.__setattr__(self, "TDIgen", TDIgen)
        object.__setattr__(self, "window_alpha", float(window_alpha))
        object.__setattr__(self, "window_power_correction", bool(window_power_correction))
        object.__setattr__(self, "use_interp_response", bool(use_interp_response))
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_cache_size", int(cache_size))

    def __getattr__(self, name):
        """将未知属性访问转发到内部 waveform 对象。"""
        return getattr(self.wf, name)

    def __setattr__(self, name, value):
        if name in self._local:
            object.__setattr__(self, name, value)
        else:
            setattr(self.wf, name, value)

    def _state_key(self):
        """根据当前波形参数构建可哈希的缓存键。"""
        return tuple(float(getattr(self.wf, key)) for key in self._state_params)

    @staticmethod
    def _matrix_res_pro(n_vec, pol):
        """计算每个采样点的张量缩并 (n_i n_j : P_ij)。"""
        return np.einsum("in,ij,jn->n", n_vec, pol, n_vec)

    def _get_aet_td_interp(self, det, TDIgen):
        """使用“延迟时间插值”方式构建时域 A/E/T。

        这样做的原因:
        - `EMRIWaveform.get_hphc` 更适合均匀采样时间序列；
        - 直接把延迟后的非均匀时间数组传入 get_AET_td，在较长观测时长下
          可能出现明显数值伪影。

        处理流程:
        1) 先在统一均匀网格上生成源波形 hp/hc；
        2) 对各条 TDI 延迟时刻用插值取样；
        3) 组装 y_ij，再构造 X/Y/Z，最后转换为 A/E/T。
        """
        if TDIgen == 1:
            tdi_delay = 4
        elif TDIgen == 2:
            tdi_delay = 8
        else:
            raise NotImplementedError(f"Unsupported TDI generation: {TDIgen}")

        t = np.arange(0, self.wf.T_obs, self.dt)
        hp_src, hc_src = self.wf.get_hphc(t)

        det_obj = detectors[det](t)
        p1, p2, p3 = det_obj.orbits
        L = det_obj.L_T
        n1 = det_obj.uni_vec_ij(3, 2)
        n2 = det_obj.uni_vec_ij(1, 3)
        n3 = det_obj.uni_vec_ij(2, 1)

        k = self.wf.vec_k
        p_plus, p_cross = self.wf.polarization()

        xi1 = (self._matrix_res_pro(n1, p_plus), self._matrix_res_pro(n1, p_cross))
        xi2 = (self._matrix_res_pro(n2, p_plus), self._matrix_res_pro(n2, p_cross))
        xi3 = (self._matrix_res_pro(n3, p_plus), self._matrix_res_pro(n3, p_cross))

        tf_kp1 = t - np.dot(k, p1)
        tf_kp2 = t - np.dot(k, p2)
        tf_kp3 = t - np.dot(k, p3)
        kn1 = np.dot(k, n1)
        kn2 = np.dot(k, n2)
        kn3 = np.dot(k, n3)

        def h_tdi_delay(tf_s, xi_p, xi_c):
            h_list = []
            for i_ in range(tdi_delay + 1):
                tau = tf_s - i_ * L
                hp = np.interp(tau, t, hp_src, left=0.0, right=0.0)
                hc = np.interp(tau, t, hc_src, left=0.0, right=0.0)
                h_list.append(hp * xi_p + hc * xi_c)
            return h_list

        h3_p2 = h_tdi_delay(tf_kp2, *xi3)
        h3_p1 = h_tdi_delay(tf_kp1, *xi3)
        h2_p3 = h_tdi_delay(tf_kp3, *xi2)
        h2_p1 = h_tdi_delay(tf_kp1, *xi2)
        h1_p3 = h_tdi_delay(tf_kp3, *xi1)
        h1_p2 = h_tdi_delay(tf_kp2, *xi1)

        def get_y(hi_pj, hi_pk, denominator):
            return [(hi_pj[i + 1] - hi_pk[i]) / denominator for i in range(tdi_delay)]

        y31 = get_y(h2_p3, h2_p1, 2 * (1 + kn2))
        y13 = get_y(h2_p1, h2_p3, 2 * (1 - kn2))
        y12 = get_y(h3_p1, h3_p2, 2 * (1 + kn3))
        y21 = get_y(h3_p2, h3_p1, 2 * (1 - kn3))
        y23 = get_y(h1_p2, h1_p3, 2 * (1 + kn1))
        y32 = get_y(h1_p3, h1_p2, 2 * (1 - kn1))

        if TDIgen == 1:
            X = (y31[0] + y13[1] + y21[2] + y12[3] - y21[0] - y12[1] - y31[2] - y13[3])
            Y = (y12[0] + y21[1] + y32[2] + y23[3] - y32[0] - y23[1] - y12[2] - y21[3])
            Z = (y23[0] + y32[1] + y13[2] + y31[3] - y13[0] - y31[1] - y23[2] - y32[3])
        else:  # TDIgen == 2 的二代组合
            X = (
                y31[0] + y13[1] + y21[2] + y12[3] + y21[4] + y12[5] + y31[6] + y13[7]
                - y21[0] - y12[1] - y31[2] - y13[3] - y31[4] - y13[5] - y21[6] - y12[7]
            )
            Y = (
                y12[0] + y21[1] + y32[2] + y23[3] + y32[4] + y23[5] + y12[6] + y21[7]
                - y32[0] - y23[1] - y12[2] - y21[3] - y12[4] - y21[5] - y32[6] - y23[7]
            )
            Z = (
                y23[0] + y32[1] + y13[2] + y31[3] + y13[4] + y31[5] + y23[6] + y32[7]
                - y13[0] - y31[1] - y23[2] - y32[3] - y23[4] - y32[5] - y13[6] - y31[7]
            )

        return tdi_XYZ2AET(X, Y, Z)

    def _build_response_fft(self, det, TDIgen):
        """计算并返回缓存的频域响应 h(f) = [A_f, E_f, T_f]。"""
        t = np.arange(0, self.wf.T_obs, self.dt)
        if self.use_interp_response:
            A, E, T = self._get_aet_td_interp(det=det, TDIgen=TDIgen)
        else:
            A, E, T = get_AET_td(self.wf, t, det=det, TDIgen=TDIgen)

        win = tukey(len(t), self.window_alpha)
        if self.window_power_correction:
            win_rms = np.sqrt(np.mean(win ** 2))
            scale = self.dt / win_rms if win_rms > 0 else self.dt
        else:
            scale = self.dt

        N = len(t)
        f_fft = np.fft.rfftfreq(N, self.dt)
        A_f = np.fft.rfft(A * win) * scale
        E_f = np.fft.rfft(E * win) * scale
        T_f = np.fft.rfft(T * win) * scale
        h = np.vstack([A_f, E_f, T_f])
        return f_fft, h

    def get_tdi_response(self, f_series=None, channel="AET", det=None, TDIgen=None, **kwargs):
        """实现 fisher_matrix 所需接口：返回 f_series 上的频域响应。"""
        if channel.upper() != "AET":
            raise ValueError("EMRIAETAdapter only supports AET channel.")

        det = det or self.det
        TDIgen = TDIgen or self.TDIgen

        key = (det, TDIgen, self._state_key())
        if key not in self._cache:
            self._cache[key] = self._build_response_fft(det=det, TDIgen=TDIgen)
            if len(self._cache) > self._cache_size:
                self._cache.pop(next(iter(self._cache)))

        f_fft, h = self._cache[key]
        if f_series is None:
            return h

        f_series = np.asarray(f_series, dtype=float)
        if f_series.shape == f_fft.shape and np.allclose(f_series, f_fft):
            return h

        def interp_complex(y):
            return np.interp(f_series, f_fft, y.real, left=0.0, right=0.0) + 1j * np.interp(
                f_series, f_fft, y.imag, left=0.0, right=0.0
            )

        return np.vstack([interp_complex(h[0]), interp_complex(h[1]), interp_complex(h[2])])


def build_frequency_series(T_obs, dt, fmin, fmax, max_bins=None):
    """构建 [fmin, fmax] 内的 rFFT 频率序列，可选降采样。"""
    t = np.arange(0, T_obs, dt)
    f_full = np.fft.rfftfreq(len(t), dt)
    mask = (f_full >= fmin) & (f_full <= fmax)
    f_band = f_full[mask]
    if f_band.size < 2:
        raise RuntimeError("Frequency bins in selected band are insufficient.")

    stride = 1
    if max_bins is not None and f_band.size > max_bins:
        stride = int(np.ceil(f_band.size / max_bins))
        f_band = f_band[::stride]

    return f_band, stride, f_full.size


def fisher_diagnostics(fisher):
    """返回 Fisher 矩阵的条件数与特征值范围。"""
    eigvals = np.linalg.eigvalsh(fisher)
    cond = np.linalg.cond(fisher)
    return {
        "cond": float(cond),
        "min_eig": float(eigvals[0]),
        "max_eig": float(eigvals[-1]),
    }


def relative_sigma_change(curr_sigma, ref_sigma):
    """计算各参数 sigma 的最大相对变化。"""
    keys = sorted(curr_sigma.keys())
    rel = []
    for k in keys:
        denom = abs(ref_sigma[k]) if ref_sigma[k] != 0 else 1.0
        rel.append(abs(curr_sigma[k] - ref_sigma[k]) / denom)
    return float(max(rel))


def relative_fisher_change(curr_fisher, ref_fisher):
    """计算两个 Fisher 矩阵的相对 Frobenius 范数变化。"""
    denom = np.linalg.norm(ref_fisher)
    if denom == 0:
        return np.inf
    return float(np.linalg.norm(curr_fisher - ref_fisher) / denom)


def per_param_relative_change(curr_sigma, ref_sigma):
    """返回每个参数的相对 sigma 变化。"""
    out = {}
    for k in sorted(curr_sigma.keys()):
        denom = abs(ref_sigma[k]) if ref_sigma[k] != 0 else 1.0
        out[k] = float(abs(curr_sigma[k] - ref_sigma[k]) / denom)
    return out


def get_param_value(wf, name):
    """从 waveform 实例读取参数值。"""
    if hasattr(wf, name):
        return float(getattr(wf, name))
    if hasattr(wf, "add_para") and name in wf.add_para:
        return float(wf.add_para[name])
    raise AttributeError(f"Waveform has no parameter '{name}'.")


def sigma_from_cov(cov, params):
    """由协方差矩阵提取每个参数的 sigma。"""
    sigma = {}
    for i, p in enumerate(params):
        sigma[p] = float(np.sqrt(max(cov[i, i], 0.0)))
    return sigma


def corr_from_cov(cov):
    """由协方差矩阵计算相关系数矩阵。"""
    d = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(d, d)
    corr = np.zeros_like(cov)
    mask = denom > 0
    corr[mask] = cov[mask] / denom[mask]
    np.fill_diagonal(corr, 1.0)
    return corr


def effective_rank_from_svals(svals, rcond):
    """按 rcond 阈值估计数值有效秩。"""
    if svals.size == 0:
        return 0
    thresh = float(rcond * svals[0])
    return int(np.sum(svals > thresh))


def build_scale_vector(param_values, mode):
    """构建参数缩放向量。"""
    values = np.asarray(param_values, dtype=float)
    if mode == "value":
        return np.where(np.abs(values) > 0, np.abs(values), 1.0)
    if mode == "ones":
        return np.ones_like(values)
    raise ValueError(f"Unknown SCALE_MODE='{mode}', expected 'value' or 'ones'.")


def build_relative_prior_sigmas(param_values, prior_rel):
    """将统一相对先验转成每个参数的绝对先验 sigma。"""
    if prior_rel <= 0:
        return None
    values = np.asarray(param_values, dtype=float)
    base = np.where(np.abs(values) > 0, np.abs(values), 1.0)
    return prior_rel * base


def build_param_value_map(params, param_values):
    """将参数名和对应数值打包成字典。"""
    return {p: float(v) for p, v in zip(params, param_values)}


def relative_sigma_from_values(sigma_map, value_map):
    """根据注入值计算相对误差 sigma / |theta|。"""
    out = {}
    for name, sigma in sigma_map.items():
        denom = abs(value_map[name])
        out[name] = np.nan if denom == 0 else float(sigma / denom)
    return out


def top_correlation_pairs(corr, params, top_k=5):
    """返回绝对值最大的相关系数参数对。"""
    corr = np.asarray(corr, dtype=float)
    pairs = []
    n = len(params)
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(corr[i, j])
            pairs.append((abs(rho), rho, params[i], params[j]))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs[:top_k]


def active_metric_suffix(enable_scaled):
    """返回当前报告所使用的指标后缀。"""
    return "scaled" if enable_scaled else "raw"


def choose_preferred_record(records, enable_scaled):
    """从步长扫描结果中选取用于最终报告的记录。

    策略:
    1) 优先 full-rank；
    2) 优先条件数有限；
    3) 优先与相邻步长/参考步长更一致；
    4) 若仍并列，取更小的 rel_step。
    """
    if not records:
        raise ValueError("records must not be empty.")

    suffix = active_metric_suffix(enable_scaled)
    cond_key = f"cond_{suffix}"
    rank_key = f"rank_{suffix}"
    sigma_key = f"sigma_{suffix}"
    delta_ref_key = f"delta_sigma_{suffix}_ref"
    delta_prev_key = f"delta_sigma_{suffix}_prev"
    n_params = len(records[0]["result"]["params"])

    best_rec = None
    best_key = None
    for idx, rec in enumerate(records):
        metrics = rec["metrics"]
        if metrics[sigma_key] is None:
            continue
        stability_prev = rec.get(delta_prev_key, np.nan)
        stability = stability_prev if np.isfinite(stability_prev) else rec.get(delta_ref_key, np.inf)
        key = (
            0 if metrics[rank_key] == n_params else 1,
            0 if np.isfinite(metrics[cond_key]) else 1,
            float(stability),
            float(rec["rel_step"]),
            idx,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_rec = rec

    return best_rec if best_rec is not None else records[0]


def format_corr_matrix(corr, params):
    """将相关系数矩阵格式化为便于终端阅读的文本。"""
    corr = np.asarray(corr, dtype=float)
    width = max(12, max(len(p) for p in params) + 2)
    lines = []
    header = " " * width + "".join(f"{p:>{width}}" for p in params)
    lines.append(header)
    for i, name in enumerate(params):
        row = f"{name:>{width}}" + "".join(f"{corr[i, j]:>{width}.3f}" for j in range(len(params)))
        lines.append(row)
    return lines


def build_verdict(rec, enable_scaled, n_params):
    """根据条件数/秩/收敛性生成一句简短结论。"""
    suffix = active_metric_suffix(enable_scaled)
    metrics = rec["metrics"]
    cond = metrics[f"cond_{suffix}"]
    rank = metrics[f"rank_{suffix}"]
    delta_ref = rec[f"delta_sigma_{suffix}_ref"]
    delta_prev = rec[f"delta_sigma_{suffix}_prev"]
    stable_delta = delta_prev if np.isfinite(delta_prev) else delta_ref

    level = "OK"
    reason = "step scan looks numerically stable"
    cond_limit = 1e10 if enable_scaled else 1e12

    if rank < n_params:
        level = "Warning"
        reason = f"{suffix} Fisher is rank-deficient"
    elif not np.isfinite(cond):
        level = "Warning"
        reason = f"{suffix} Fisher condition number is not finite"
    elif cond > cond_limit:
        level = "Caution"
        reason = f"{suffix} Fisher is ill-conditioned"
    elif np.isfinite(stable_delta) and stable_delta > 1e-2:
        level = "Caution"
        reason = f"{suffix} sigma has only marginal rel_step convergence"

    delta_text = "n/a" if not np.isfinite(stable_delta) else f"{stable_delta:.3e}"
    return (
        f"[{level}] Preferred report uses rel_step={rec['rel_step']:.1e}; "
        f"{reason} (cond_{suffix}={cond:.3e}, rank_{suffix}={rank}/{n_params}, "
        f"dSigma_{suffix}={delta_text})."
    )


def analyze_fisher_matrix(
    fisher,
    params,
    param_values,
    rcond,
    enable_scaled=True,
    scale_mode="value",
    prior_rel=0.0,
):
    """输出 raw/scaled Fisher 的反演稳定性与 sigma。

    支持可选相对先验:
    sigma_prior_i = prior_rel * |theta_i|
    并在 Fisher 上加对角先验信息 1/sigma_prior_i^2。
    """
    metrics = {}

    # 记录未加先验前的病态程度，便于对照。
    eigvals_input = np.linalg.eigvalsh(fisher)
    metrics["cond_input"] = float(np.linalg.cond(fisher))
    metrics["min_eig_input"] = float(eigvals_input[0])
    metrics["max_eig_input"] = float(eigvals_input[-1])

    fisher_eff = np.array(fisher, dtype=float, copy=True)
    metrics["prior_rel"] = float(prior_rel)
    metrics["prior_sigmas"] = None
    if prior_rel > 0:
        prior_sigmas = build_relative_prior_sigmas(param_values, prior_rel)
        fisher_eff = fisher_eff + np.diag(1.0 / np.square(prior_sigmas))
        metrics["prior_sigmas"] = {p: float(s) for p, s in zip(params, prior_sigmas)}

    eigvals = np.linalg.eigvalsh(fisher_eff)
    svals = np.linalg.svd(fisher_eff, compute_uv=False)
    cov_raw = np.linalg.pinv(fisher_eff, rcond=rcond)
    metrics["cond_raw"] = float(np.linalg.cond(fisher_eff))
    metrics["min_eig_raw"] = float(eigvals[0])
    metrics["max_eig_raw"] = float(eigvals[-1])
    metrics["rank_raw"] = effective_rank_from_svals(svals, rcond)
    metrics["sigma_raw"] = sigma_from_cov(cov_raw, params)
    metrics["corr_raw"] = corr_from_cov(cov_raw)
    metrics["cov_raw"] = cov_raw

    metrics["cond_scaled"] = np.nan
    metrics["rank_scaled"] = 0
    metrics["sigma_scaled"] = None
    metrics["corr_scaled"] = None
    metrics["cov_scaled"] = None

    if enable_scaled:
        scales = build_scale_vector(param_values, scale_mode)
        S = np.diag(scales)
        fisher_scaled = S @ fisher_eff @ S
        svals_scaled = np.linalg.svd(fisher_scaled, compute_uv=False)
        cov_scaled = np.linalg.pinv(fisher_scaled, rcond=rcond)
        cov_back = S @ cov_scaled @ S
        metrics["cond_scaled"] = float(np.linalg.cond(fisher_scaled))
        metrics["rank_scaled"] = effective_rank_from_svals(svals_scaled, rcond)
        metrics["sigma_scaled"] = sigma_from_cov(cov_back, params)
        metrics["corr_scaled"] = corr_from_cov(cov_back)
        metrics["cov_scaled"] = cov_back

    return metrics


def _format_param_title(name, value, sigma):
    """生成对角线子图标题字符串。"""
    if not np.isfinite(sigma):
        return f"{name} = {value:.6g}"
    return f"{name} = {value:.6g} ± {sigma:.2g}"


def _gaussian_pdf(x, mu, sigma):
    """返回一维高斯密度。"""
    if sigma <= 0:
        y = np.zeros_like(x)
        idx = np.argmin(np.abs(x - mu))
        y[idx] = 1.0
        return y
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def _select_corner_models(records, max_models):
    """按均匀索引从扫描记录中选取要叠加绘制的模型。"""
    if not records:
        return []
    if max_models <= 0 or max_models >= len(records):
        return list(records)
    idx = np.linspace(0, len(records) - 1, max_models, dtype=int)
    out = []
    seen = set()
    for i in idx:
        if int(i) not in seen:
            out.append(records[int(i)])
            seen.add(int(i))
    return out


def plot_corner_from_records(
    records,
    params,
    param_values,
    outdir,
    tag="",
    cov_mode="scaled",
    max_models=3,
    model_labels=None,
    axis_labels=None,
    info_lines=None,
    title_text=None,
):
    """根据 Fisher 扫描结果绘制 corner 风格图。

    图形含义:
    - 对角线: 每个参数的一维高斯近似后验；
    - 下三角: 参数两两的 1σ/2σ 误差椭圆；
    - 叠加多条曲线用于比较不同 rel_step 的结果。
    """
    try:
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
    except Exception as exc:  # pragma: no cover - 运行时依赖
        print(f"[Warning] Corner 绘图失败: 无法导入 matplotlib ({exc})")
        return None

    if cov_mode not in ("raw", "scaled"):
        raise ValueError("FISHER_CORNER_COV_MODE 仅支持 raw 或 scaled。")

    chosen = _select_corner_models(records, max_models)
    if not chosen:
        print("[Warning] Corner 绘图失败: 没有可用记录。")
        return None

    # 组装要绘制的协方差矩阵。
    use_scaled = (cov_mode == "scaled")
    series = []
    for rec in chosen:
        cov = rec["metrics"]["cov_scaled"] if use_scaled else rec["metrics"]["cov_raw"]
        if cov is None:
            # 当 scaled 不可用时自动回退到 raw，保证主流程不中断。
            cov = rec["metrics"]["cov_raw"]
        series.append({"rec": rec, "cov": np.asarray(cov, dtype=float)})

    n = len(params)
    axis_labels = list(axis_labels) if axis_labels is not None else list(params)
    if len(axis_labels) != n:
        raise ValueError("axis_labels must have the same length as params.")
    info_lines = [line for line in (info_lines or []) if str(line).strip()]
    extra_info_height = 0.28 * len(info_lines) + (0.5 if info_lines else 0.0)
    fig, axes = plt.subplots(n, n, figsize=(2.0 * n + 1.5, 2.0 * n + 1.0 + extra_info_height))
    if n == 1:
        axes = np.array([[axes]])

    # 类似示例图的三种主色：青绿、蓝、深蓝。
    color_cycle = ["#63c3a5", "#2f94c8", "#29327f", "#f08b3e", "#b04bb3"]
    level_specs = [
        (np.sqrt(2.30), 0.55),  # 2D ~68%
        (np.sqrt(6.17), 0.95),  # 2D ~95%
    ]

    # 按“最大 sigma”设置全局范围，避免不同模型坐标不一致。
    sigma_max = np.zeros(n, dtype=float)
    for i in range(n):
        vals = []
        for s in series:
            cii = float(s["cov"][i, i])
            vals.append(np.sqrt(max(cii, 0.0)))
        sigma_max[i] = max(vals) if vals else 0.0
        if not np.isfinite(sigma_max[i]) or sigma_max[i] <= 0:
            sigma_max[i] = max(abs(param_values[i]) * 1e-3, 1e-6)

    x_lims = []
    for i in range(n):
        span = 4.0 * sigma_max[i]
        x_lims.append((param_values[i] - span, param_values[i] + span))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            if i == j:
                x0, x1 = x_lims[i]
                x = np.linspace(x0, x1, 320)
                for k, s in enumerate(series):
                    color = color_cycle[k % len(color_cycle)]
                    sigma = np.sqrt(max(float(s["cov"][i, i]), 0.0))
                    y = _gaussian_pdf(x, param_values[i], sigma)
                    ax.plot(x, y, color=color, lw=1.8)
                    ax.fill_between(x, 0.0, y, color=color, alpha=0.12)
                ax.axvline(param_values[i], color="#d5368d", lw=1.1, alpha=0.85)
                if series:
                    sigma0 = np.sqrt(max(float(series[0]["cov"][i, i]), 0.0))
                    ax.set_title(_format_param_title(axis_labels[i], param_values[i], sigma0), fontsize=10)
                ax.set_xlim(x0, x1)
                ax.set_yticks([])
            else:
                x0, x1 = x_lims[j]
                y0, y1 = x_lims[i]
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                ax.axvline(param_values[j], color="#d5368d", lw=1.0, alpha=0.8)
                ax.axhline(param_values[i], color="#d5368d", lw=1.0, alpha=0.8)

                for k, s in enumerate(series):
                    cov2 = np.array(
                        [
                            [s["cov"][j, j], s["cov"][j, i]],
                            [s["cov"][i, j], s["cov"][i, i]],
                        ],
                        dtype=float,
                    )
                    vals, vecs = np.linalg.eigh(cov2)
                    vals = np.clip(vals, 0.0, None)
                    order = np.argsort(vals)[::-1]
                    vals = vals[order]
                    vecs = vecs[:, order]
                    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
                    color = color_cycle[k % len(color_cycle)]
                    for nsig, lw in level_specs:
                        width = 2.0 * nsig * np.sqrt(vals[0])
                        height = 2.0 * nsig * np.sqrt(vals[1])
                        ell = Ellipse(
                            xy=(param_values[j], param_values[i]),
                            width=width,
                            height=height,
                            angle=angle,
                            edgecolor=color,
                            facecolor="none",
                            lw=lw,
                            alpha=0.95,
                        )
                        ax.add_patch(ell)
                    ax.scatter(param_values[j], param_values[i], color=color, s=8, alpha=0.75)

            if i == n - 1:
                ax.set_xlabel(axis_labels[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(axis_labels[i], fontsize=10)
            elif j != 0:
                ax.set_yticklabels([])

    # 图例：优先用环境变量标签，不足时自动补 rel_step。
    labels = list(model_labels or [])
    while len(labels) < len(series):
        idx = len(labels)
        labels.append(f"rel_step={series[idx]['rec']['rel_step']:.1e}")
    labels = labels[:len(series)]

    from matplotlib.lines import Line2D
    handles = []
    for k, label in enumerate(labels):
        handles.append(Line2D([0], [0], color=color_cycle[k % len(color_cycle)], lw=2.2, label=label))

    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(0.995, 0.995))
    if title_text is None:
        title_text = f"EMRI Fisher Corner ({cov_mode}) | params={','.join(params)}"
    fig.suptitle(title_text, fontsize=12, y=0.995)
    if info_lines:
        info_text = "\n".join(info_lines)
        fig.text(
            0.02,
            0.945,
            info_text,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f9", "edgecolor": "#d6d6db", "alpha": 0.95},
        )
    top_rect = 0.975 if not info_lines else max(0.76, 0.90 - 0.035 * len(info_lines))
    fig.tight_layout(rect=[0.02, 0.02, 0.94, top_rect])

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)
    tag = tag.strip() if tag is not None else ""
    if not tag:
        tag = f"T{T_OBS_YR:.3f}yr_{'_'.join(params)}_{cov_mode}".replace(".", "p")
    out_png = outdir_path / f"fisher_corner_{tag}.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


def main():
    """执行 EMRI Fisher 步长扫描并输出稳定性诊断信息。"""
    wf = EMRIWaveform(**EMRIpars)
    adapter = EMRIAETAdapter(
        wf,
        dt=DT,
        det="TQ",
        TDIgen=1,
        window_alpha=WINDOW_ALPHA,
        window_power_correction=True,
        use_interp_response=USE_INTERP_RESPONSE,
    )

    f_series, stride, n_full_fft = build_frequency_series(
        T_obs=wf.T_obs,
        dt=DT,
        fmin=FMIN,
        fmax=FMAX,
        max_bins=MAX_FREQ_BINS,
    )

    n_time = np.arange(0, wf.T_obs, DT).size
    print("=== Run Configuration ===")
    print("Injected waveform parameters:")
    for key, value in EMRIpars.items():
        if key == "T_obs":
            print(f"  {key}: {float(value):.6e} s ({float(value) / YRSID_SI:.4f} yr)")
        else:
            print(f"  {key}: {float(value):.6e}")
    print("Estimated parameters (PARAMS):")
    for name in PARAMS:
        print(f"  {name}")
    print("Frequency / solver setup:")
    print(
        f"  T_obs={wf.T_obs / YRSID_SI:.4f} yr | DT={DT:.3f} s | "
        f"FMIN={FMIN:.3e} Hz | FMAX={FMAX:.3e} Hz"
    )
    print(
        f"  N_time={n_time}, N_fft={n_full_fft}, N_band_used={f_series.size}, stride={stride}, "
        f"max_bins={MAX_FREQ_BINS}, use_interp_response={USE_INTERP_RESPONSE}"
    )
    print(
        f"  PINV_RCOND={PINV_RCOND:.1e}, ENABLE_SCALED_FISHER={ENABLE_SCALED_FISHER}, "
        f"SCALE_MODE={SCALE_MODE}, PRIOR_REL={PRIOR_REL:.3e}"
    )
    print(f"  REL_STEPS={REL_STEPS}")
    if n_time > 2_000_000:
        print("[Warning] Time samples are very large; runtime may still be long.")

    records = []
    param_values = np.array([get_param_value(wf, p) for p in PARAMS], dtype=float)
    param_value_map = build_param_value_map(PARAMS, param_values)
    # 有限差分步长扫描：评估 Fisher/sigma 的数值稳定性。
    for rel_step in REL_STEPS:
        result = fisher_matrix(
            adapter,
            params=PARAMS,
            det="TQ",
            channel="AET",
            TDIgen=1,
            f_series=f_series,
            noise=TianQinNoise(),
            use_T=USE_T,
            rel_step=rel_step,
        )
        metrics = analyze_fisher_matrix(
            result["fisher"],
            params=PARAMS,
            param_values=param_values,
            rcond=PINV_RCOND,
            enable_scaled=ENABLE_SCALED_FISHER,
            scale_mode=SCALE_MODE,
            prior_rel=PRIOR_REL,
        )
        records.append({"rel_step": rel_step, "result": result, "metrics": metrics})

    reference = records[0]  # 以最小步长作为参考
    # 同时比较“相对参考步长”与“相对前一档步长”的变化。
    for i, rec in enumerate(records):
        rec["delta_fisher_ref"] = relative_fisher_change(rec["result"]["fisher"], reference["result"]["fisher"])
        rec["delta_sigma_raw_ref"] = relative_sigma_change(rec["metrics"]["sigma_raw"], reference["metrics"]["sigma_raw"])
        rec["delta_sigma_scaled_ref"] = (
            relative_sigma_change(rec["metrics"]["sigma_scaled"], reference["metrics"]["sigma_scaled"])
            if ENABLE_SCALED_FISHER else np.nan
        )
        rec["delta_sigma_raw_ref_map"] = per_param_relative_change(
            rec["metrics"]["sigma_raw"], reference["metrics"]["sigma_raw"]
        )
        rec["delta_sigma_scaled_ref_map"] = (
            per_param_relative_change(rec["metrics"]["sigma_scaled"], reference["metrics"]["sigma_scaled"])
            if ENABLE_SCALED_FISHER else None
        )
        if i == 0:
            rec["delta_fisher_prev"] = np.nan
            rec["delta_sigma_raw_prev"] = np.nan
            rec["delta_sigma_scaled_prev"] = np.nan
            rec["delta_sigma_raw_prev_map"] = None
            rec["delta_sigma_scaled_prev_map"] = None
        else:
            prev = records[i - 1]
            rec["delta_fisher_prev"] = relative_fisher_change(rec["result"]["fisher"], prev["result"]["fisher"])
            rec["delta_sigma_raw_prev"] = relative_sigma_change(rec["metrics"]["sigma_raw"], prev["metrics"]["sigma_raw"])
            rec["delta_sigma_scaled_prev"] = (
                relative_sigma_change(rec["metrics"]["sigma_scaled"], prev["metrics"]["sigma_scaled"])
                if ENABLE_SCALED_FISHER else np.nan
            )
            rec["delta_sigma_raw_prev_map"] = per_param_relative_change(
                rec["metrics"]["sigma_raw"], prev["metrics"]["sigma_raw"]
            )
            rec["delta_sigma_scaled_prev_map"] = (
                per_param_relative_change(rec["metrics"]["sigma_scaled"], prev["metrics"]["sigma_scaled"])
                if ENABLE_SCALED_FISHER else None
            )

    suffix = active_metric_suffix(ENABLE_SCALED_FISHER)
    cond_key = f"cond_{suffix}"
    rank_key = f"rank_{suffix}"
    sigma_key = f"sigma_{suffix}"
    corr_key = f"corr_{suffix}"
    delta_ref_key = f"delta_sigma_{suffix}_ref"
    delta_prev_key = f"delta_sigma_{suffix}_prev"
    delta_ref_map_key = f"delta_sigma_{suffix}_ref_map"
    delta_prev_map_key = f"delta_sigma_{suffix}_prev_map"

    print("\n=== Step-Scan Summary ===")
    print(f"Report mode: {suffix} | reference_rel_step={reference['rel_step']:.1e}")
    for rec in records:
        res = rec["result"]
        met = rec["metrics"]
        print(
            f"rel_step={rec['rel_step']:.1e} | "
            f"SNR={res['snr']:.6e} | "
            f"cond_{suffix}={met[cond_key]:.3e} | "
            f"rank_{suffix}={met[rank_key]}/{len(PARAMS)} | "
            f"lambda_min_raw={met['min_eig_raw']:.3e} | "
            f"dF_ref={rec['delta_fisher_ref']:.3e} | "
            f"dSigma_{suffix}_ref={rec[delta_ref_key]:.3e} | "
            f"dF_prev={rec['delta_fisher_prev']:.3e} | "
            f"dSigma_{suffix}_prev={rec[delta_prev_key]:.3e}"
        )
        if met["prior_sigmas"] is not None:
            print(f"  prior_sigmas={met['prior_sigmas']}")
        sigma_active = met[sigma_key]
        if sigma_active is not None:
            sigma_text = ", ".join(f"{p}={sigma_active[p]:.3e}" for p in PARAMS)
            print(f"  sigma_{suffix}: {sigma_text}")
        if REPORT_PER_PARAM:
            ref_map = rec[delta_ref_map_key]
            prev_map = rec[delta_prev_map_key]
            ref_text = ", ".join(f"{p}={ref_map[p]:.3e}" for p in PARAMS)
            if prev_map is None:
                prev_text = "n/a"
            else:
                prev_text = ", ".join(f"{p}={prev_map[p]:.3e}" for p in PARAMS)
            print(f"  dSigma_{suffix}_ref_by_param: {ref_text}")
            print(f"  dSigma_{suffix}_prev_by_param: {prev_text}")

    preferred = choose_preferred_record(records, ENABLE_SCALED_FISHER)
    preferred_result = preferred["result"]
    preferred_metrics = preferred["metrics"]
    sigma_active = preferred_metrics[sigma_key]
    corr_active = preferred_metrics[corr_key]
    rel_sigma = relative_sigma_from_values(sigma_active, param_value_map)
    frequency_size = preferred_result.get("frequency_size")
    if frequency_size is None:
        freq = preferred_result.get("frequency")
        frequency_size = len(freq) if freq is not None else "n/a"

    print("\n=== Preferred Report ===")
    print(
        f"rel_step={preferred['rel_step']:.1e} | mode={suffix} | "
        f"SNR={preferred_result['snr']:.6e} | frequency_size={frequency_size}"
    )
    print("Parameter constraints:")
    for name in PARAMS:
        rel_val = rel_sigma[name]
        rel_text = "n/a" if not np.isfinite(rel_val) else f"{rel_val:.3e}"
        print(
            f"  {name}: theta={param_value_map[name]:.6e} | "
            f"sigma_{suffix}={sigma_active[name]:.6e} | rel_sigma={rel_text}"
        )

    print("Numerical reliability:")
    print(
        f"  cond_input={preferred_metrics['cond_input']:.3e}, "
        f"cond_{suffix}={preferred_metrics[cond_key]:.3e}, "
        f"rank_{suffix}={preferred_metrics[rank_key]}/{len(PARAMS)}, "
        f"lambda_min_raw={preferred_metrics['min_eig_raw']:.3e}, "
        f"lambda_max_raw={preferred_metrics['max_eig_raw']:.3e}"
    )
    print(
        f"  dF_ref={preferred['delta_fisher_ref']:.3e}, "
        f"dSigma_{suffix}_ref={preferred[delta_ref_key]:.3e}, "
        f"dF_prev={preferred['delta_fisher_prev']:.3e}, "
        f"dSigma_{suffix}_prev={preferred[delta_prev_key]:.3e}"
    )
    if corr_active is not None:
        top_pairs = top_correlation_pairs(corr_active, PARAMS, top_k=min(5, max(0, len(PARAMS) * (len(PARAMS) - 1) // 2)))
        if top_pairs:
            print("Top correlations:")
            for _, rho, p0, p1 in top_pairs:
                print(f"  rho({p0}, {p1}) = {rho:.3f}")
        if len(PARAMS) <= 6:
            print(f"Correlation matrix ({suffix}):")
            for line in format_corr_matrix(corr_active, PARAMS):
                print(f"  {line}")

    print("\n=== Verdict ===")
    print(build_verdict(preferred, ENABLE_SCALED_FISHER, len(PARAMS)))

    last = records[-1]["metrics"]
    if PRIOR_REL > 0:
        print(
            f"\n[Info] Prior regularization: cond_input={last['cond_input']:.3e} "
            f"-> cond_raw={last['cond_raw']:.3e}"
        )
    if last["cond_raw"] > 1e12:
        print("\n[Warning] Raw Fisher is ill-conditioned (cond_raw > 1e12).")
    if ENABLE_SCALED_FISHER and last["cond_scaled"] > 1e10:
        print("[Warning] Scaled Fisher is still ill-conditioned (cond_scaled > 1e10).")
    if last["min_eig_raw"] <= 0:
        print("[Warning] Raw Fisher has non-positive eigenvalue(s).")
    if last["rank_raw"] < len(PARAMS):
        print("[Warning] Raw Fisher effective rank is deficient under current rcond.")
    if ENABLE_SCALED_FISHER and last["rank_scaled"] < len(PARAMS):
        print("[Warning] Scaled Fisher effective rank is deficient under current rcond.")

    # 可选输出 corner 风格图，便于直接用于报告/论文展示。
    if PLOT_CORNER:
        cov_mode = CORNER_COV_MODE
        if cov_mode == "scaled" and not ENABLE_SCALED_FISHER:
            print("[Warning] ENABLE_SCALED_FISHER=False，corner 图将自动使用 raw 协方差。")
            cov_mode = "raw"
        out_png = plot_corner_from_records(
            records=records,
            params=PARAMS,
            param_values=param_values,
            outdir=CORNER_DIR,
            tag=CORNER_TAG,
            cov_mode=cov_mode,
            max_models=CORNER_MAX_MODELS,
            model_labels=CORNER_LABELS,
        )
        if out_png is not None:
            print(f"\n[OK] Corner 图已保存: {out_png}")


if __name__ == "__main__":
    main()
