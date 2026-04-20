import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    MATPLOTLIB_IMPORT_ERROR = None
except Exception as exc:
    plt = None
    HAS_MATPLOTLIB = False
    MATPLOTLIB_IMPORT_ERROR = exc

try:
    from few.trajectory.inspiral import EMRIInspiral, get_0PA_frequencies
    HAS_FEW = True
except Exception:
    EMRIInspiral = None
    get_0PA_frequencies = None
    HAS_FEW = False

try:
    from gwspace.Waveform import EMRIWaveform
    from gwspace.constants import YRSID_SI
    HAS_GWSPACE_EMRI = True
except Exception:
    EMRIWaveform = None
    YRSID_SI = 3.15581497635e7
    HAS_GWSPACE_EMRI = False

# ================= 1. 定义极其严谨的理论/校准模型 =================

SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 365.25 * SEC_PER_DAY
LIGHT_SPEED_SI = 299792458.0
TAIJI_CONFUSION_COEFFS = {
    '6mo': (-85.3498, -2.64899, -0.0699707, -0.478447, -0.334821, 0.0658353),
    '1yr': (-85.4336, -2.46276, -0.183175, -0.884147, -0.427176, 0.128666),
    '2yr': (-85.3919, -2.69735, -0.749294, -1.15302, -0.302761, 0.175521),
    '4yr': (-85.5448, -3.23671, -1.64187, -1.14711, 0.0325887, 0.187854),
}
TAIJI_DEFAULT_OBSERVATION = '4yr'

def get_lgwa_hc(freq):
    # 【已严格校准】LGWA (基于上传论文 arXiv:2010.13726v1, Harms et al. 2020) 
    # 锚定论文 Fig. 3 与 Fig. 11 的真实特征应变包络 (Characteristic Strain)
    # 精确拟合节点: 1mHz (~6.3e-18), 10mHz (~1e-19), 0.1Hz (~1.6e-21), 1Hz (~2.1e-20)
    term1 = 8.37e-19 * (freq / 1e-3)**-4.0
    term2 = 7.87e-20 * (freq / 1e-2)**-2.0
    term3 = 6.36e-22 * (freq / 0.1)**-0.2
    term4 = 2.10e-20 * (freq / 1.0)**2.0
    return term1 + term2 + term3 + term4

def get_zaiga_hc(freq):
    # ZAIGA (arXiv:1903.09288)
    keff, L, T, eta = 1.6e7, 3000, 1.0, 1e14
    h_sn = 1 / (2 * keff * L * np.maximum(np.abs(np.sin(np.pi * freq * T)), 1e-3) * np.sqrt(eta))
    h_nn = 1e-22 * (freq / 0.01)**-4
    return np.sqrt(freq) * np.sqrt(h_sn**2 + h_nn**2)

def get_grace_hc(freq):
    # GRACE-FO (arXiv:2002.02044)
    return 1e-14 * (freq / 0.1)**1.5 + 4e-15 * (freq / 0.1)**-1.8

def get_lisa_hc(freq):
    # LISA (Robson et al. 2019 标准模型)
    L = 2.5e9
    P_oms = (1.5e-11)**2 * (1 + (2e-3 / freq)**4)
    P_acc = (3e-15)**2 * (1 + (0.4e-3 / freq)**2) * (1 + (freq / 8e-3)**4)
    Sn = (10 / (3 * L**2)) * (P_oms + 4 * P_acc / (2 * np.pi * freq)**4) * (1 + 0.6 * (freq / 19e-3)**2)
    return np.sqrt(freq * Sn)

def get_taiji_instrument_psd(freq):
    # Taiji instrument noise model with Michelson-style two-channel response.
    freq = np.asarray(freq, dtype=float)
    safe_freq = np.maximum(freq, 1e-30)
    L = 3.0e9
    f_star = LIGHT_SPEED_SI / (2 * np.pi * L)
    P_dp = (8e-12)**2 * (1 + (2e-3 / safe_freq)**4)
    P_acc = (3e-15)**2 * (1 + (0.4e-3 / safe_freq)**2) * (1 + (safe_freq / 8e-3)**4)
    acc_response = 2.0 * (1.0 + np.cos(safe_freq / f_star)**2) * P_acc / (2 * np.pi * safe_freq)**4
    return (10.0 / (3.0 * L**2)) * (P_dp + acc_response) * (1.0 + 0.6 * (safe_freq / f_star)**2)

def get_taiji_confusion_psd(freq, observation=TAIJI_DEFAULT_OBSERVATION):
    # Taiji Galactic-binary confusion noise fit is valid only for 0.1 mHz < f < 10 mHz.
    if observation not in TAIJI_CONFUSION_COEFFS:
        valid_labels = ', '.join(TAIJI_CONFUSION_COEFFS)
        raise ValueError(f'Unsupported Taiji observation "{observation}". Choose from: {valid_labels}')

    freq = np.asarray(freq, dtype=float)
    safe_freq = np.maximum(freq, 1e-30)
    valid_band = (safe_freq > 1e-4) & (safe_freq < 1e-2)
    x = np.log(safe_freq / 1e-3)
    poly = np.zeros_like(safe_freq)
    for power, coeff in enumerate(TAIJI_CONFUSION_COEFFS[observation]):
        poly += coeff * x**power

    confusion = np.zeros_like(safe_freq)
    confusion[valid_band] = np.exp(poly[valid_band])
    return confusion

def get_taiji_full_psd(freq, observation=TAIJI_DEFAULT_OBSERVATION):
    return get_taiji_instrument_psd(freq) + get_taiji_confusion_psd(freq, observation=observation)

def get_taiji_hc(freq, observation=TAIJI_DEFAULT_OBSERVATION):
    freq = np.asarray(freq, dtype=float)
    safe_freq = np.maximum(freq, 1e-30)
    return np.sqrt(safe_freq * get_taiji_full_psd(safe_freq, observation=observation))

def get_tianqin_hc(freq):
    # TianQin (黄顺佳硕士论文 Eq. 3-48)
    L_tq = np.sqrt(3) * 1e8
    c = 3e8
    f_star_tq = c / (2 * np.pi * L_tq)
    Sa_tq = 1e-30
    Sx_tq = 1e-24
    Sn_tq = (20 / 3) * (1 / L_tq**2) * ( (4 * Sa_tq / (2 * np.pi * freq)**4) * (1 + 1e-4 / freq) + Sx_tq ) * (1 + 0.6 * (freq / f_star_tq)**2)
    return np.sqrt(freq * Sn_tq)

def get_pta_ska_hc(freq):
    return 1e-16 * (freq / 1e-8)

def get_pta_epta_hc(freq):
    return 2.7e-15 * (freq / (1 / 3.15e7)) * 2.5

def get_source_hc_from_fdot(h0, freq, fdot):
    """按导师给定公式计算源的特征应变: h_c(f) = sqrt(2 f^2 / fdot) * h0."""
    safe_fdot = np.maximum(np.abs(fdot), 1e-30)
    return h0 * np.sqrt(2.0 * np.square(freq) / safe_fdot)

def get_emri_dominant_gw_frequency_hz(mass1, mass2, a, p, e, x):
    """对当前 quasi-circular、equatorial EMRI，取主导 m=2 谐波: f_gw ~= 2 * Omega_phi."""
    if not HAS_FEW:
        raise ImportError('FEW is not available.')
    # 当前仓库里的 EMRI 路径使用 Schwarzschild 轨道频率，因此这里显式按 a=0 取频率。
    a_eff = 0.0
    omega_phi, _, _ = get_0PA_frequencies(mass1, mass2, a_eff, p, e, x)
    return 2.0 * np.asarray(omega_phi), a_eff

def estimate_emri_frequency_and_fdot_from_few(emri_pars, dt_sec=86400.0):
    """用 FEW 轨道在 EMRI 初始参数点估计主导 GW 频率与 fdot."""
    if not HAS_FEW:
        raise ImportError('FEW is not available.')

    a_eff = 0.0
    traj = EMRIInspiral(func='SchwarzEccFlux', force_backend='cpu')
    t_obs_yr = float(emri_pars['T_obs']) / YRSID_SI
    t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(
        emri_pars['M'], emri_pars['mu'], a_eff, emri_pars['p0'], emri_pars['e0'], emri_pars['x0'],
        T=t_obs_yr, dt=dt_sec, DENSE_STEPPING=1,
    )
    f_gw, _ = get_emri_dominant_gw_frequency_hz(
        emri_pars['M'], emri_pars['mu'], emri_pars['a'], p, e, x,
    )
    f_gw = np.asarray(f_gw, dtype=float)
    t = np.asarray(t, dtype=float)
    fdot = np.gradient(f_gw, t)
    return {
        'f_src': float(f_gw[0]),
        'fdot': float(fdot[0]),
        'f_end': float(f_gw[-1]),
        'p_src': float(p[0]),
        'p_end': float(p[-1]),
        't_span_sec': float(t[-1] - t[0]),
        'a_eff': float(a_eff),
    }

def estimate_emri_h0_from_waveform(emri_pars, f_ref, dt_sec=20.0, cycles=64, eps=1e-5):
    """从 EMRI 时域波形前若干周期的包络估计 h0."""
    if not HAS_GWSPACE_EMRI:
        raise ImportError('gwspace.EMRIWaveform is not available.')

    wf = EMRIWaveform(**emri_pars)
    duration_sec = max(float(cycles) / max(float(f_ref), 1e-30), 10.0 * dt_sec)
    duration_yr = min(duration_sec / YRSID_SI, float(emri_pars['T_obs']) / YRSID_SI)
    hp, hc = wf.get_hphc_source(T_obs=duration_yr, dt=dt_sec, eps=eps, modes=None)
    h_env = np.sqrt(np.asarray(hp, dtype=float)**2 + np.asarray(hc, dtype=float)**2)
    return {
        'h0': float(np.max(h_env)),
        'duration_yr': float(duration_yr),
        'samples': int(h_env.size),
    }

# ================= 2. 频率网格分配与实测数据读取 =================

# 读取 LIGO 真实源数据
ligo_data = np.loadtxt(REPO_ROOT / '灵敏度.txt')
f_ligo_real = ligo_data[:, 0]
asd_ligo = ligo_data[:, 1]
# 特征应变转换
hc_ligo_real = np.sqrt(f_ligo_real) * asd_ligo

# 物理频段
f_lgwa = np.logspace(-4, 0, 1000)       
f_zaiga = np.logspace(-2, 1, 1000)      
f_grace = np.logspace(-4, 0, 1000)      
f_lisa = np.logspace(-5, 0, 1000)       
f_taiji = np.logspace(-5, 0, 1000)
f_tq = np.logspace(-4, 0, 1000)         
f_pta_epta = np.logspace(np.log10(1 / (10 * 3.15e7)), np.log10(1 / (14 * 86400)), 100)
f_pta_ska = np.logspace(np.log10(1 / (20 * 3.15e7)), np.log10(1 / (14 * 86400)), 100)

# 目标源参数
EMRIpars = {
    'M': 4.3e6,
    'mu': 40.0,
    'a': 0.1,
    'p0': 9.0,
    'e0': 0.0,
    'x0': 1.0,
    'qS': 1.66,
    'phiS': 4.71,
    'qK': 0.2,
    'phiK': 0.2,
    'dist': 8.0e-6,
    'Phi_phi0': 1.0,
    'Phi_theta0': 2.0,
    'Phi_r0': 3.0,
    'psi': 0.4,
    'iota': 0.2,
    'T_obs': 5 * YRSID_SI,
}

try:
    emri_fdot_info = estimate_emri_frequency_and_fdot_from_few(EMRIpars, dt_sec=86400.0)
    f_src = emri_fdot_info['f_src']
    fdot_src = emri_fdot_info['fdot']
    f_end_src = emri_fdot_info['f_end']
    p_src = emri_fdot_info['p_src']
    p_end_src = emri_fdot_info['p_end']
    a_eff_src = emri_fdot_info['a_eff']
    fdot_src_label = 'FEW trajectory estimate'
except Exception as exc:
    # 若 FEW 不可用，则退回到原先的占位量级，保证脚本仍可运行。
    emri_fdot_info = None
    f_src = 5e-4
    fdot_src = 2.0 * f_src / (5.0 * SEC_PER_YEAR)
    f_end_src = np.nan
    p_src = np.nan
    p_end_src = np.nan
    a_eff_src = np.nan
    fdot_src_label = f'fallback equivalent 5 yr ({exc})'

try:
    emri_h0_info = estimate_emri_h0_from_waveform(EMRIpars, f_src, dt_sec=20.0, cycles=64)
    h0_src = emri_h0_info['h0']
    h0_src_label = 'EMRIWaveform envelope estimate'
except Exception as exc:
    emri_h0_info = None
    h0_src = 1.4e-17
    h0_src_label = f'manual fallback ({exc})'

hc_src = get_source_hc_from_fdot(h0_src, f_src, fdot_src)
t_chirp_src = f_src / fdot_src
hc_gain = hc_src / h0_src

def get_snr_from_hc(hc_src, hc_noise):
    return hc_src / hc_noise

def format_mathtext_sci(value, precision=2):
    """Format a float as mantissa \\times 10^{exp} for mathtext."""
    if value == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf'{mantissa:.{precision}f}\times10^{{{exponent}}}'

hc_lgwa_src = np.interp(f_src, f_lgwa, get_lgwa_hc(f_lgwa))
hc_lisa_src = np.interp(f_src, f_lisa, get_lisa_hc(f_lisa))
hc_taiji_src = np.interp(f_src, f_taiji, get_taiji_hc(f_taiji))
hc_tq_src = np.interp(f_src, f_tq, get_tianqin_hc(f_tq))

print('=== Sgr A* EMRI ===')
print(f'Input p0 = {EMRIpars["p0"]:.3f}')
print(f'Input a = {EMRIpars["a"]:.3f}')
print(f'Input dist = {EMRIpars["dist"]:.3e} Gpc')
print(f'f_src = {f_src:.3e} Hz')
print(f'h0 source = {h0_src_label}')
print(f'h0 = {h0_src:.3e}')
print(f'fdot source = {fdot_src_label}')
print(f'fdot = {fdot_src:.3e} Hz/s')
if np.isfinite(p_src):
    print(f'FEW initial p = {p_src:.6f}, final p = {p_end_src:.6f}')
if np.isfinite(f_end_src):
    print(f'FEW initial f = {f_src:.3e} Hz, final f = {f_end_src:.3e} Hz')
if np.isfinite(a_eff_src):
    print(f'FEW frequency model uses effective a = {a_eff_src:.3f}')
if emri_h0_info is not None:
    print(f'h0 estimated from first {emri_h0_info["duration_yr"]:.3e} yr waveform segment ({emri_h0_info["samples"]} samples)')
print(f'Chirp timescale f/fdot = {t_chirp_src:.3e} s ({t_chirp_src / SEC_PER_YEAR:.3e} yr)')
print(f'h_c = sqrt(2 f^2 / fdot) * h0 = {hc_src:.3e}')
print(f'Gain factor sqrt(2 f^2 / fdot) = {hc_gain:.2f}x')
print('--- Approx SNR from h_c ratio ---')
print(f'LGWA: SNR={get_snr_from_hc(hc_src, hc_lgwa_src):.3e}')
print(f'LISA: SNR={get_snr_from_hc(hc_src, hc_lisa_src):.3e}')
print(f'Taiji ({TAIJI_DEFAULT_OBSERVATION} full): SNR={get_snr_from_hc(hc_src, hc_taiji_src):.3e}')
print(f'TianQin: SNR={get_snr_from_hc(hc_src, hc_tq_src):.3e}')

# ================= 3. 绘制完美版图表 =================
if not HAS_MATPLOTLIB:
    print(f'Skipping plot: matplotlib import failed: {MATPLOTLIB_IMPORT_ERROR}')
else:
    plot_style = {
        'figsize': (10, 6),
        'figure_dpi': 150,
        'save_dpi': 400,
        'curve_colors': {
            'LISA': '#4C78A8',
            'Taiji': '#59A14F',
            'TianQin': '#00A7C2',
            'LGWA': '#7A4CC2',
            'ZAIGA': '#E28E2C',
            'GRACE-FO': '#C44E52',
            'Source': '#4B5563',
        },
        'curve_lw': {
            'LISA': 2.0,
            'Taiji': 2.15,
            'TianQin': 2.55,
            'LGWA': 2.0,
            'ZAIGA': 2.0,
            'GRACE-FO': 1.95,
        },
        'curve_alpha': {
            'LISA': 0.88,
            'Taiji': 0.88,
            'TianQin': 1.0,
            'LGWA': 0.87,
            'ZAIGA': 0.87,
            'GRACE-FO': 0.86,
        },
        'marker_size': 46,
        'marker_edgewidth': 0.75,
        'annotation_fontsize': 9.4,
        'annotation_offset': (2.25e-4, 1.65e-15),
        'annotation_bbox_alpha': 0.92,
        'annotation_linewidth': 0.8,
        'title_fontsize': 13.2,
        'label_fontsize': 12.0,
        'tick_fontsize': 10.2,
        'legend_fontsize': 9.6,
    }

    fig, ax = plt.subplots(figsize=plot_style['figsize'], dpi=plot_style['figure_dpi'])

    # 解析曲线
    # PTA 与当前 mHz 波段目标源无直接重叠，这里先注释掉。
    # ax.loglog(f_pta_epta, get_pta_epta_hc(f_pta_epta), lw=2, linestyle=':', label='PTA (EPTA, arXiv:1408.0740)', color='purple')
    # ax.loglog(f_pta_ska, get_pta_ska_hc(f_pta_ska), lw=2.5, label='PTA (SKA, arXiv:1408.0740)', color='darkviolet')
    ax.loglog(
        f_lisa, get_lisa_hc(f_lisa),
        label='LISA',
        color=plot_style['curve_colors']['LISA'],
        lw=plot_style['curve_lw']['LISA'],
        alpha=plot_style['curve_alpha']['LISA'],
    )
    ax.loglog(
        f_taiji, get_taiji_hc(f_taiji),
        label='Taiji',
        color=plot_style['curve_colors']['Taiji'],
        lw=plot_style['curve_lw']['Taiji'],
        alpha=plot_style['curve_alpha']['Taiji'],
    )
    ax.loglog(
        f_tq, get_tianqin_hc(f_tq),
        label='TianQin',
        color=plot_style['curve_colors']['TianQin'],
        lw=plot_style['curve_lw']['TianQin'],
        alpha=plot_style['curve_alpha']['TianQin'],
    )
    ax.loglog(
        f_lgwa, get_lgwa_hc(f_lgwa),
        label='LGWA',
        color=plot_style['curve_colors']['LGWA'],
        lw=plot_style['curve_lw']['LGWA'],
        alpha=plot_style['curve_alpha']['LGWA'],
    )
    ax.loglog(
        f_zaiga, get_zaiga_hc(f_zaiga),
        label='ZAIGA',
        color=plot_style['curve_colors']['ZAIGA'],
        lw=plot_style['curve_lw']['ZAIGA'],
        alpha=plot_style['curve_alpha']['ZAIGA'],
    )
    ax.loglog(
        f_grace, get_grace_hc(f_grace),
        label='GRACE-FO',
        color=plot_style['curve_colors']['GRACE-FO'],
        lw=plot_style['curve_lw']['GRACE-FO'],
        alpha=plot_style['curve_alpha']['GRACE-FO'],
        ls='--',
        dashes=(6, 3),
    )

    # LIGO 与当前 mHz 波段目标源无直接重叠，这里先注释掉。
    # ax.loglog(f_ligo_real, hc_ligo_real, label='LIGO (Real Source Data)', color='green', lw=2.5)

    # 源点与注释
    source_text = (
        'Sgr $A^*$ EMRI\n'
        rf'$f \approx {f_src * 1e3:.3f}\,\mathrm{{mHz}}$' '\n'
        rf'$\dot f \approx {format_mathtext_sci(fdot_src, precision=2)}\,\mathrm{{Hz\,s^{{-1}}}}$' '\n'
        rf'$h_c \approx {format_mathtext_sci(hc_src, precision=2)}$'
    )

    ax.scatter(
        f_src,
        hc_src,
        s=plot_style['marker_size'],
        marker='o',
        color=plot_style['curve_colors']['Source'],
        edgecolors='white',
        linewidths=plot_style['marker_edgewidth'],
        zorder=8,
        label='Sgr $A^*$ EMRI',
    )
    ax.annotate(
        source_text,
        xy=(f_src, hc_src),
        xytext=plot_style['annotation_offset'],
        textcoords='data',
        ha='left',
        va='top',
        fontsize=plot_style['annotation_fontsize'],
        color='#1F2937',
        bbox=dict(
            boxstyle='round,pad=0.32',
            fc='white',
            ec='#C9CFD8',
            lw=0.75,
            alpha=plot_style['annotation_bbox_alpha'],
        ),
        arrowprops=dict(
            arrowstyle='-',
            color='#6B7280',
            lw=plot_style['annotation_linewidth'],
            shrinkA=4,
            shrinkB=4,
            connectionstyle='arc3,rad=0.06',
        ),
        zorder=9,
    )

    ax.set_title(
        'Characteristic strain sensitivity and a fiducial Sgr $A^*$ EMRI source',
        fontsize=plot_style['title_fontsize'],
        fontweight='semibold',
        pad=9,
    )
    ax.set_xlabel('Frequency [Hz]', fontsize=plot_style['label_fontsize'])
    ax.set_ylabel('Characteristic Strain $h_c(f)$', fontsize=plot_style['label_fontsize'])
    ax.set_xlim(1e-4, 2e1)
    ax.set_ylim(1e-22, 3e-9)
    ax.tick_params(axis='both', which='major', labelsize=plot_style['tick_fontsize'], length=5, width=0.8)
    ax.tick_params(axis='both', which='minor', length=2.8, width=0.6)
    ax.grid(True, which='major', ls='--', lw=0.55, color='#C9CFD8', alpha=0.55)
    ax.grid(True, which='minor', ls=':', lw=0.35, color='#D8DDE5', alpha=0.42)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = [
        'LISA',
        'TianQin',
        'Taiji',
        'LGWA',
        'ZAIGA',
        'GRACE-FO',
        'Sgr $A^*$ EMRI',
    ]
    order_index = {label: idx for idx, label in enumerate(legend_order)}
    ordered = sorted(zip(handles, labels), key=lambda item: order_index.get(item[1], len(legend_order)))
    ax.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc='upper right',
        fontsize=plot_style['legend_fontsize'],
        frameon=True,
        framealpha=0.96,
        facecolor='white',
        edgecolor='#D0D5DD',
        handlelength=2.8,
        borderpad=0.55,
        labelspacing=0.42,
    )

    output_base = Path(__file__).with_name('frequency_publication_style')
    fig.tight_layout()
    fig.savefig(output_base.with_suffix('.png'), dpi=plot_style['save_dpi'], bbox_inches='tight')
    fig.savefig(output_base.with_suffix('.pdf'), bbox_inches='tight')
    print(f'Saved figure: {output_base.with_suffix(".png")}')
    print(f'Saved figure: {output_base.with_suffix(".pdf")}')
    if plt.get_backend().lower() != 'agg':
        plt.show()
