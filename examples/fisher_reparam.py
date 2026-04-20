import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fisher as base

from gwspace.Noise import TianQinNoise
from gwspace.Waveform import EMRIWaveform
from gwspace.fishertool import fisher_matrix


RAW_PARAMS = base._parse_csv_strings(os.getenv("FISHER_EXP_RAW_PARAMS", "M,mu,p0"))
BASIS_MODE = os.getenv("FISHER_EXP_BASIS", "log_selected").strip().lower()
_LOG_PARAMS_RAW = os.getenv("FISHER_EXP_LOG_PARAMS", "M").strip()
LOG_PARAMS = base._parse_csv_strings(_LOG_PARAMS_RAW) if _LOG_PARAMS_RAW else []
M_P0_ALPHA = float(os.getenv("FISHER_EXP_M_P0_ALPHA", "1.5"))
FREQ_P0_ALPHA = float(os.getenv("FISHER_EXP_FREQ_P0_ALPHA", "1.5"))
CHIRP_M_POWER = float(os.getenv("FISHER_EXP_CHIRP_M_POWER", "3.0"))
CHIRP_P0_POWER = float(os.getenv("FISHER_EXP_CHIRP_P0_POWER", "7.0"))
COMPARE_PHYSICAL = base._env_flag("FISHER_EXP_COMPARE_PHYSICAL", True)
PLOT_CORNER = base._env_flag("FISHER_EXP_PLOT_CORNER", False)
CORNER_DIR = os.getenv("FISHER_EXP_CORNER_DIR", str(Path(base.CORNER_DIR) / "reparam"))
CORNER_TAG = os.getenv("FISHER_EXP_CORNER_TAG", "")
_AXIS_ALIASES_RAW = os.getenv("FISHER_EXP_AXIS_ALIASES", "").strip()
AXIS_ALIASES = base._parse_csv_strings(_AXIS_ALIASES_RAW) if _AXIS_ALIASES_RAW else []
EIGEN_SOURCE_MODE = os.getenv("FISHER_EXP_EIGEN_SOURCE", "emri_freq_chirp").strip().lower()
_EIGEN_REF_STEP_RAW = os.getenv("FISHER_EXP_EIGEN_REF_STEP", "").strip()
EIGEN_REF_STEP = float(_EIGEN_REF_STEP_RAW) if _EIGEN_REF_STEP_RAW else None


@dataclass
class BasisParam:
    name: str
    kind: str
    formula: str
    linear_ref: float | None = None


@dataclass
class ParamTransform:
    mode: str
    raw_params: list[str]
    basis_params: list[BasisParam]
    basis_values: np.ndarray
    jacobian_theta_wrt_basis: np.ndarray
    summary: str
    axis_alias_prefix: str = "B"
    info_lines: list[str] | None = None

    @property
    def names(self):
        return [item.name for item in self.basis_params]


def _safe_power_token(value):
    text = f"{value:.6g}".replace("-", "m").replace(".", "p").replace("+", "")
    return text


def _value_map(params, values):
    return {name: float(val) for name, val in zip(params, values)}


def _require_positive(name, value):
    if float(value) <= 0:
        raise ValueError(f"Parameter '{name}' must be positive to use a log transform, got {value}.")


def build_identity_transform(raw_params, raw_values):
    basis_params = [BasisParam(name=p, kind="raw", formula=p, linear_ref=float(v)) for p, v in zip(raw_params, raw_values)]
    jac = np.eye(len(raw_params), dtype=float)
    return ParamTransform(
        mode="physical",
        raw_params=list(raw_params),
        basis_params=basis_params,
        basis_values=np.asarray(raw_values, dtype=float),
        jacobian_theta_wrt_basis=jac,
        summary="Identity transform: use the original physical parameters directly.",
        axis_alias_prefix="B",
    )


def build_log_transform(raw_params, raw_values, log_params):
    value_map = _value_map(raw_params, raw_values)
    log_set = set(log_params)
    unknown = sorted(log_set.difference(raw_params))
    if unknown:
        raise ValueError(f"FISHER_EXP_LOG_PARAMS contains unknown parameters: {unknown}")

    jac = np.eye(len(raw_params), dtype=float)
    basis_values = []
    basis_params = []
    for i, name in enumerate(raw_params):
        value = value_map[name]
        if name in log_set:
            _require_positive(name, value)
            basis_values.append(np.log(value))
            jac[i, i] = value
            basis_params.append(
                BasisParam(
                    name=f"ln_{name}",
                    kind="log",
                    formula=f"ln({name})",
                    linear_ref=value,
                )
            )
        else:
            basis_values.append(value)
            basis_params.append(BasisParam(name=name, kind="raw", formula=name, linear_ref=value))

    summary = (
        "Mixed basis: selected positive parameters are mapped to ln(theta), "
        "others remain in their original coordinates."
    )
    return ParamTransform(
        mode="log",
        raw_params=list(raw_params),
        basis_params=basis_params,
        basis_values=np.asarray(basis_values, dtype=float),
        jacobian_theta_wrt_basis=jac,
        summary=summary,
        axis_alias_prefix="B",
    )


def build_emri_combo_transform(raw_params, raw_values, alpha):
    value_map = _value_map(raw_params, raw_values)
    for name in ("M", "mu", "p0"):
        if name not in value_map:
            raise ValueError("FISHER_EXP_BASIS=emri_combo requires RAW_PARAMS to include M, mu, and p0.")
        _require_positive(name, value_map[name])

    raw_index = {name: i for i, name in enumerate(raw_params)}
    extra_params = [name for name in raw_params if name not in {"M", "mu", "p0"}]
    n_basis = 3 + len(extra_params)
    jac = np.zeros((len(raw_params), n_basis), dtype=float)
    basis_params = []
    basis_values = []

    combo_name = f"ln_M_p0_pow_{_safe_power_token(alpha)}"
    combo_linear = value_map["M"] * (value_map["p0"] ** alpha)
    basis_params.append(
        BasisParam(
            name=combo_name,
            kind="log_combo",
            formula=f"ln(M * p0^{alpha:.6g})",
            linear_ref=combo_linear,
        )
    )
    basis_values.append(np.log(combo_linear))
    jac[raw_index["M"], 0] = value_map["M"]

    basis_params.append(BasisParam(name="ln_mu", kind="log", formula="ln(mu)", linear_ref=value_map["mu"]))
    basis_values.append(np.log(value_map["mu"]))
    jac[raw_index["mu"], 1] = value_map["mu"]

    basis_params.append(BasisParam(name="ln_p0", kind="log", formula="ln(p0)", linear_ref=value_map["p0"]))
    basis_values.append(np.log(value_map["p0"]))
    jac[raw_index["M"], 2] = -alpha * value_map["M"]
    jac[raw_index["p0"], 2] = value_map["p0"]

    for col, name in enumerate(extra_params, start=3):
        value = value_map[name]
        basis_params.append(BasisParam(name=name, kind="raw", formula=name, linear_ref=value))
        basis_values.append(value)
        jac[raw_index[name], col] = 1.0

    summary = (
        "EMRI test basis: replace (M, mu, p0) with "
        f"(ln(M * p0^{alpha:.6g}), ln(mu), ln(p0)); extra parameters stay unchanged."
    )
    return ParamTransform(
        mode="emri_combo",
        raw_params=list(raw_params),
        basis_params=basis_params,
        basis_values=np.asarray(basis_values, dtype=float),
        jacobian_theta_wrt_basis=jac,
        summary=summary,
        axis_alias_prefix="B",
    )


def build_emri_freq_chirp_transform(raw_params, raw_values, freq_alpha, chirp_m_power, chirp_p0_power):
    value_map = _value_map(raw_params, raw_values)
    for name in ("M", "mu", "p0"):
        if name not in value_map:
            raise ValueError(
                "FISHER_EXP_BASIS=emri_freq_chirp requires RAW_PARAMS to include M, mu, and p0."
            )
        _require_positive(name, value_map[name])

    raw_index = {name: i for i, name in enumerate(raw_params)}
    extra_params = [name for name in raw_params if name not in {"M", "mu", "p0"}]
    n_basis = 3 + len(extra_params)
    jac = np.zeros((len(raw_params), n_basis), dtype=float)
    basis_params = []
    basis_values = []

    freq_name = "ln_freq_like"
    freq_linear = value_map["M"] * (value_map["p0"] ** freq_alpha)
    basis_params.append(
        BasisParam(
            name=freq_name,
            kind="log_combo",
            formula=f"ln(M * p0^{freq_alpha:.6g})",
            linear_ref=freq_linear,
        )
    )
    basis_values.append(np.log(freq_linear))
    jac[raw_index["M"], 0] = value_map["M"]

    chirp_name = "ln_chirp_like"
    chirp_linear = value_map["mu"] / (
        (value_map["M"] ** chirp_m_power) * (value_map["p0"] ** chirp_p0_power)
    )
    basis_params.append(
        BasisParam(
            name=chirp_name,
            kind="log_combo",
            formula=f"ln(mu / (M^{chirp_m_power:.6g} * p0^{chirp_p0_power:.6g}))",
            linear_ref=chirp_linear,
        )
    )
    basis_values.append(np.log(chirp_linear))
    jac[raw_index["mu"], 1] = value_map["mu"]

    residual_name = "ln_p0_residual"
    basis_params.append(
        BasisParam(
            name=residual_name,
            kind="log",
            formula="ln(p0)",
            linear_ref=value_map["p0"],
        )
    )
    basis_values.append(np.log(value_map["p0"]))
    jac[raw_index["M"], 2] = -freq_alpha * value_map["M"]
    jac[raw_index["mu"], 0] = chirp_m_power * value_map["mu"]
    jac[raw_index["mu"], 2] = (chirp_p0_power - freq_alpha * chirp_m_power) * value_map["mu"]
    jac[raw_index["p0"], 2] = value_map["p0"]

    for col, name in enumerate(extra_params, start=3):
        value = value_map[name]
        basis_params.append(BasisParam(name=name, kind="raw", formula=name, linear_ref=value))
        basis_values.append(value)
        jac[raw_index[name], col] = 1.0

    summary = (
        "EMRI frequency/chirp basis: use ln(M * p0^alpha) as a frequency-like direction, "
        "ln(mu / (M^beta * p0^gamma)) as a chirp-like direction, and keep ln(p0) as the residual "
        "orbit-scale coordinate."
    )
    return ParamTransform(
        mode="emri_freq_chirp",
        raw_params=list(raw_params),
        basis_params=basis_params,
        basis_values=np.asarray(basis_values, dtype=float),
        jacobian_theta_wrt_basis=jac,
        summary=summary,
        axis_alias_prefix="B",
    )


def build_transform_from_mode(mode, raw_params, raw_values):
    if mode == "physical":
        return build_identity_transform(raw_params, raw_values)
    if mode == "log_all":
        return build_log_transform(raw_params, raw_values, raw_params)
    if mode == "log_selected":
        return build_log_transform(raw_params, raw_values, LOG_PARAMS)
    if mode == "emri_combo":
        return build_emri_combo_transform(raw_params, raw_values, M_P0_ALPHA)
    if mode in ("emri_freq_chirp", "emri_phys_combo"):
        return build_emri_freq_chirp_transform(
            raw_params,
            raw_values,
            FREQ_P0_ALPHA,
            CHIRP_M_POWER,
            CHIRP_P0_POWER,
        )
    if mode == "eigenbasis":
        raise ValueError("eigenbasis requires a reference Fisher matrix; build it after the step scan.")
    raise ValueError(
        f"Unsupported FISHER_EXP_BASIS='{mode}'. "
        "Use one of: physical, log_all, log_selected, emri_combo, emri_freq_chirp, eigenbasis."
    )


def build_transform(raw_params, raw_values):
    return build_transform_from_mode(BASIS_MODE, raw_params, raw_values)


def build_transformed_fisher(raw_fisher, transform):
    jac = transform.jacobian_theta_wrt_basis
    return jac.T @ np.asarray(raw_fisher, dtype=float) @ jac


def _select_reference_record(records, rel_step_override):
    if not records:
        raise ValueError("No step-scan records are available to define the reference eigenbasis.")
    if rel_step_override is None:
        return records[0]
    tol = max(1e-18, abs(rel_step_override) * 1e-9)
    for rec in records:
        if abs(rec["rel_step"] - rel_step_override) <= tol:
            return rec
    available = ", ".join(f"{rec['rel_step']:.1e}" for rec in records)
    raise ValueError(
        f"FISHER_EXP_EIGEN_REF_STEP={rel_step_override:.6g} does not match any scanned rel_step. "
        f"Available values: {available}"
    )


def _format_linear_combo(weights, labels):
    pieces = []
    for i, (weight, label) in enumerate(zip(weights, labels)):
        if abs(weight) < 5e-3:
            continue
        sign = "-" if weight < 0 else "+"
        coeff = abs(weight)
        if not pieces:
            lead = "-" if weight < 0 else ""
            pieces.append(f"{lead}{coeff:.3f}*{label}")
        else:
            pieces.append(f"{sign} {coeff:.3f}*{label}")
    if not pieces:
        idx = int(np.argmax(np.abs(weights)))
        return f"{weights[idx]:.3f}*{labels[idx]}"
    return " ".join(pieces)


def build_eigenbasis_transform(source_transform, source_record):
    cov_source = np.asarray(source_record["metrics"]["cov_raw"], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov_source)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    basis_values = eigvecs.T @ np.asarray(source_transform.basis_values, dtype=float)
    jac = np.asarray(source_transform.jacobian_theta_wrt_basis, dtype=float) @ eigvecs

    source_aliases = [f"B{i + 1}" for i in range(len(source_transform.names))]
    basis_params = []
    info_lines = ["Source Basis Definitions:"]
    for alias, item in zip(source_aliases, source_transform.basis_params):
        info_lines.append(f"{alias}: {item.formula}")
    info_lines.append("Principal-Axis Basis:")

    for i in range(len(source_transform.names)):
        alias = f"C{i + 1}"
        formula = _format_linear_combo(eigvecs[:, i], source_aliases)
        basis_params.append(
            BasisParam(
                name=alias,
                kind="orthogonal_combo",
                formula=formula,
                linear_ref=None,
            )
        )
        info_lines.append(f"{alias}: {formula}")

    summary = (
        "Eigenbasis from the covariance principal axes of the source transformed basis "
        f"'{source_transform.mode}' at rel_step={source_record['rel_step']:.1e}. "
        "C1 is the loosest direction; later axes are progressively tighter."
    )
    return ParamTransform(
        mode="eigenbasis",
        raw_params=list(source_transform.raw_params),
        basis_params=basis_params,
        basis_values=basis_values,
        jacobian_theta_wrt_basis=jac,
        summary=summary,
        axis_alias_prefix="C",
        info_lines=info_lines,
    ), eigvals


def finalize_records(records, enable_scaled):
    reference = records[0]
    suffix = base.active_metric_suffix(enable_scaled)
    sigma_key = f"sigma_{suffix}"

    for i, rec in enumerate(records):
        rec["delta_fisher_ref"] = base.relative_fisher_change(rec["result"]["fisher"], reference["result"]["fisher"])
        rec[f"delta_sigma_{suffix}_ref"] = base.relative_sigma_change(
            rec["metrics"][sigma_key], reference["metrics"][sigma_key]
        )
        rec[f"delta_sigma_{suffix}_ref_map"] = base.per_param_relative_change(
            rec["metrics"][sigma_key], reference["metrics"][sigma_key]
        )
        if i == 0:
            rec["delta_fisher_prev"] = np.nan
            rec[f"delta_sigma_{suffix}_prev"] = np.nan
            rec[f"delta_sigma_{suffix}_prev_map"] = None
        else:
            prev = records[i - 1]
            rec["delta_fisher_prev"] = base.relative_fisher_change(rec["result"]["fisher"], prev["result"]["fisher"])
            rec[f"delta_sigma_{suffix}_prev"] = base.relative_sigma_change(
                rec["metrics"][sigma_key], prev["metrics"][sigma_key]
            )
            rec[f"delta_sigma_{suffix}_prev_map"] = base.per_param_relative_change(
                rec["metrics"][sigma_key], prev["metrics"][sigma_key]
            )


def print_step_scan_summary(records, enable_scaled, title):
    suffix = base.active_metric_suffix(enable_scaled)
    cond_key = f"cond_{suffix}"
    rank_key = f"rank_{suffix}"
    delta_ref_key = f"delta_sigma_{suffix}_ref"
    delta_prev_key = f"delta_sigma_{suffix}_prev"
    sigma_key = f"sigma_{suffix}"

    print(f"\n=== {title} ===")
    print(f"Report mode: {suffix} | reference_rel_step={records[0]['rel_step']:.1e}")
    for rec in records:
        met = rec["metrics"]
        res = rec["result"]
        print(
            f"rel_step={rec['rel_step']:.1e} | "
            f"SNR={res['snr']:.6e} | "
            f"cond_{suffix}={met[cond_key]:.3e} | "
            f"rank_{suffix}={met[rank_key]}/{len(res['params'])} | "
            f"dF_ref={rec['delta_fisher_ref']:.3e} | "
            f"dSigma_{suffix}_ref={rec[delta_ref_key]:.3e} | "
            f"dF_prev={rec['delta_fisher_prev']:.3e} | "
            f"dSigma_{suffix}_prev={rec[delta_prev_key]:.3e}"
        )
        sigma_text = ", ".join(f"{p}={met[sigma_key][p]:.3e}" for p in res["params"])
        print(f"  sigma_{suffix}: {sigma_text}")


def _format_basis_constraint(meta, value, sigma):
    if meta.kind == "orthogonal_combo":
        return f"  {meta.name}: value={value:.6e} | sigma={sigma:.6e}"

    if meta.kind == "raw":
        rel = np.nan if value == 0 else sigma / abs(value)
        rel_text = "n/a" if not np.isfinite(rel) else f"{rel:.3e}"
        return (
            f"  {meta.name}: value={value:.6e} | sigma={sigma:.6e} | "
            f"rel_sigma={rel_text}"
        )

    frac_exact = np.expm1(sigma) if sigma < 50 else np.inf
    frac_text = "n/a" if not np.isfinite(frac_exact) else f"{frac_exact:.3e}"
    linear_text = ""
    if meta.linear_ref is not None:
        linear_text = f" | linear_ref={meta.linear_ref:.6e}"
    return (
        f"  {meta.name}: value={value:.6e} | sigma={sigma:.6e} | "
        f"frac_sigma~={sigma:.3e} | exp(sigma)-1={frac_text}{linear_text}"
    )


def print_preferred_report(records, enable_scaled, transform, title):
    suffix = base.active_metric_suffix(enable_scaled)
    sigma_key = f"sigma_{suffix}"
    cond_key = f"cond_{suffix}"
    rank_key = f"rank_{suffix}"
    corr_key = f"corr_{suffix}"
    delta_ref_key = f"delta_sigma_{suffix}_ref"
    delta_prev_key = f"delta_sigma_{suffix}_prev"

    preferred = base.choose_preferred_record(records, enable_scaled)
    metrics = preferred["metrics"]
    result = preferred["result"]
    sigma_map = metrics[sigma_key]

    print(f"\n=== {title} ===")
    print(
        f"rel_step={preferred['rel_step']:.1e} | "
        f"SNR={result['snr']:.6e} | "
        f"cond_{suffix}={metrics[cond_key]:.3e} | "
        f"rank_{suffix}={metrics[rank_key]}/{len(result['params'])}"
    )
    print("Parameter constraints:")
    for meta, value in zip(transform.basis_params, transform.basis_values):
        print(_format_basis_constraint(meta, value, sigma_map[meta.name]))

    corr = metrics[corr_key]
    if corr is not None:
        print("Top correlations:")
        for _, rho, p0, p1 in base.top_correlation_pairs(corr, result["params"], top_k=min(5, len(result["params"]) * (len(result["params"]) - 1) // 2)):
            print(f"  rho({p0}, {p1}) = {rho:.3f}")
        if len(result["params"]) <= 6:
            print(f"Correlation matrix ({suffix}):")
            for line in base.format_corr_matrix(corr, result["params"]):
                print(f"  {line}")

    print("Numerical reliability:")
    print(
        f"  cond_input={metrics['cond_input']:.3e} | "
        f"cond_{suffix}={metrics[cond_key]:.3e} | "
        f"rank_{suffix}={metrics[rank_key]}/{len(result['params'])}"
    )
    print(
        f"  dF_ref={preferred['delta_fisher_ref']:.3e} | "
        f"dSigma_{suffix}_ref={preferred[delta_ref_key]:.3e} | "
        f"dF_prev={preferred['delta_fisher_prev']:.3e} | "
        f"dSigma_{suffix}_prev={preferred[delta_prev_key]:.3e}"
    )
    print(base.build_verdict(preferred, enable_scaled, len(result["params"])))
    return preferred


def build_corner_aliases(transform):
    if AXIS_ALIASES:
        if len(AXIS_ALIASES) != len(transform.names):
            raise ValueError("FISHER_EXP_AXIS_ALIASES must have the same length as the transformed basis.")
        return list(AXIS_ALIASES)
    return [f"{transform.axis_alias_prefix}{i + 1}" for i in range(len(transform.names))]


def build_corner_info_lines(transform, axis_labels):
    if transform.info_lines:
        return list(transform.info_lines)
    lines = ["Basis Definitions:"]
    for alias, item in zip(axis_labels, transform.basis_params):
        lines.append(f"{alias}: {item.formula}")
    return lines


def maybe_plot_corner(records, transform):
    if not PLOT_CORNER:
        return

    tag = CORNER_TAG.strip()
    if not tag:
        tag = f"reparam_{transform.mode}_{'_'.join(transform.names)}".replace(".", "p")
    axis_labels = build_corner_aliases(transform)
    info_lines = build_corner_info_lines(transform, axis_labels)
    out_png = base.plot_corner_from_records(
        records=records,
        params=transform.names,
        param_values=transform.basis_values,
        outdir=CORNER_DIR,
        tag=tag,
        cov_mode="raw",
        max_models=min(base.CORNER_MAX_MODELS, len(records)),
        model_labels=base.CORNER_LABELS,
        axis_labels=axis_labels,
        info_lines=info_lines,
        title_text=f"EMRI Reparameterized Fisher Corner (raw) | basis={transform.mode}",
    )
    if out_png is not None:
        print(f"\n[OK] Corner 图已保存: {out_png}")


def main():
    wf = EMRIWaveform(**base.EMRIpars)
    adapter = base.EMRIAETAdapter(
        wf,
        dt=base.DT,
        det="TQ",
        TDIgen=1,
        window_alpha=base.WINDOW_ALPHA,
        window_power_correction=True,
        use_interp_response=base.USE_INTERP_RESPONSE,
    )

    f_series, stride, n_full_fft = base.build_frequency_series(
        T_obs=wf.T_obs,
        dt=base.DT,
        fmin=base.FMIN,
        fmax=base.FMAX,
        max_bins=base.MAX_FREQ_BINS,
    )
    raw_values = np.array([base.get_param_value(wf, p) for p in RAW_PARAMS], dtype=float)
    source_transform = None
    transform = None
    if BASIS_MODE == "eigenbasis":
        if EIGEN_SOURCE_MODE == "eigenbasis":
            raise ValueError("FISHER_EXP_EIGEN_SOURCE cannot itself be 'eigenbasis'.")
        source_transform = build_transform_from_mode(EIGEN_SOURCE_MODE, RAW_PARAMS, raw_values)
    else:
        transform = build_transform(RAW_PARAMS, raw_values)

    print("=== Run Configuration ===")
    print(f"Experiment script: {Path(__file__).name}")
    print(f"Raw physical parameters: {RAW_PARAMS}")
    print(f"Basis mode: {BASIS_MODE}")
    if BASIS_MODE == "log_selected":
        print(f"Log-selected parameters: {LOG_PARAMS}")
    if BASIS_MODE == "emri_combo":
        print(f"EMRI combo alpha: {M_P0_ALPHA:.6g}")
    if BASIS_MODE in ("emri_freq_chirp", "emri_phys_combo"):
        print(
            "EMRI frequency/chirp exponents: "
            f"alpha={FREQ_P0_ALPHA:.6g}, beta={CHIRP_M_POWER:.6g}, gamma={CHIRP_P0_POWER:.6g}"
        )
    if BASIS_MODE == "eigenbasis":
        print(f"Eigenbasis source mode: {EIGEN_SOURCE_MODE}")
        if EIGEN_REF_STEP is None:
            print(f"Eigenbasis reference rel_step: first scanned step ({base.REL_STEPS[0]:.1e})")
        else:
            print(f"Eigenbasis reference rel_step: {EIGEN_REF_STEP:.1e}")
        print("Source basis formulas:")
        for item in source_transform.basis_params:
            print(f"  {item.name} = {item.formula}")
        print(
            "Transform summary: build a physically motivated source basis first, "
            "then rotate it onto the covariance principal axes to obtain C1/C2/C3."
        )
    else:
        print("Transformed basis formulas:")
        for item in transform.basis_params:
            print(f"  {item.name} = {item.formula}")
        print(f"Transform summary: {transform.summary}")
    print(
        f"Frequency setup: T_obs={wf.T_obs / base.YRSID_SI:.4f} yr | DT={base.DT:.3f} s | "
        f"FMIN={base.FMIN:.3e} Hz | FMAX={base.FMAX:.3e} Hz"
    )
    print(
        f"Grid: N_fft={n_full_fft} | N_band_used={f_series.size} | stride={stride} | "
        f"max_bins={base.MAX_FREQ_BINS}"
    )
    print(f"REL_STEPS={base.REL_STEPS}")

    if BASIS_MODE in ("log_all", "log_selected"):
        print(
            "[Info] Pure per-parameter ln transform primarily fixes scale conditioning. "
            "It will not by itself remove true physical correlations."
        )
    if base.PRIOR_REL > 0 and BASIS_MODE != "physical":
        print(
            "[Warning] PRIOR_REL is defined in physical coordinates. "
            "This experiment applies it only to the physical-basis reference report, "
            "not to the transformed-basis report."
        )

    physical_records = []
    source_records = []
    for rel_step in base.REL_STEPS:
        raw_result = fisher_matrix(
            adapter,
            params=RAW_PARAMS,
            det="TQ",
            channel="AET",
            TDIgen=1,
            f_series=f_series,
            noise=TianQinNoise(),
            use_T=base.USE_T,
            rel_step=rel_step,
        )
        raw_result["frequency_size"] = len(f_series)
        physical_metrics = base.analyze_fisher_matrix(
            raw_result["fisher"],
            params=RAW_PARAMS,
            param_values=raw_values,
            rcond=base.PINV_RCOND,
            enable_scaled=True,
            scale_mode=base.SCALE_MODE,
            prior_rel=base.PRIOR_REL,
        )
        physical_records.append(
            {
                "rel_step": rel_step,
                "result": raw_result,
                "metrics": physical_metrics,
            }
        )

        active_transform = source_transform if BASIS_MODE == "eigenbasis" else transform
        basis_fisher = build_transformed_fisher(raw_result["fisher"], active_transform)
        basis_result = {
            "params": active_transform.names,
            "fisher": basis_fisher,
            "snr": raw_result["snr"],
            "frequency": raw_result["frequency"],
            "frequency_size": raw_result["frequency_size"],
        }
        basis_metrics = base.analyze_fisher_matrix(
            basis_fisher,
            params=active_transform.names,
            param_values=active_transform.basis_values,
            rcond=base.PINV_RCOND,
            enable_scaled=False,
            scale_mode="ones",
            prior_rel=0.0,
        )
        source_records.append(
            {
                "rel_step": rel_step,
                "result": basis_result,
                "metrics": basis_metrics,
            }
        )

    finalize_records(physical_records, enable_scaled=True)

    basis_records = source_records
    if BASIS_MODE == "eigenbasis":
        reference_source = _select_reference_record(source_records, EIGEN_REF_STEP)
        transform, eigvals = build_eigenbasis_transform(source_transform, reference_source)
        print("\n=== Derived Eigenbasis ===")
        print(f"Reference rel_step={reference_source['rel_step']:.1e} | source_mode={source_transform.mode}")
        for item in transform.basis_params:
            print(f"  {item.name} = {item.formula}")
        eig_text = ", ".join(f"{item.name} var≈{val:.3e}" for item, val in zip(transform.basis_params, eigvals))
        print(f"Principal variances: {eig_text}")
        print(f"Transform summary: {transform.summary}")

        basis_records = []
        for rec in physical_records:
            raw_result = rec["result"]
            basis_fisher = build_transformed_fisher(raw_result["fisher"], transform)
            basis_result = {
                "params": transform.names,
                "fisher": basis_fisher,
                "snr": raw_result["snr"],
                "frequency": raw_result["frequency"],
                "frequency_size": raw_result["frequency_size"],
            }
            basis_metrics = base.analyze_fisher_matrix(
                basis_fisher,
                params=transform.names,
                param_values=transform.basis_values,
                rcond=base.PINV_RCOND,
                enable_scaled=False,
                scale_mode="ones",
                prior_rel=0.0,
            )
            basis_records.append(
                {
                    "rel_step": rec["rel_step"],
                    "result": basis_result,
                    "metrics": basis_metrics,
                }
            )

    finalize_records(basis_records, enable_scaled=False)

    if COMPARE_PHYSICAL:
        print_step_scan_summary(physical_records, enable_scaled=True, title="Physical Basis Step Scan (scaled)")
    print_step_scan_summary(basis_records, enable_scaled=False, title="Transformed Basis Step Scan (raw)")

    preferred_physical = None
    if COMPARE_PHYSICAL:
        physical_transform = build_identity_transform(RAW_PARAMS, raw_values)
        preferred_physical = print_preferred_report(
            physical_records,
            enable_scaled=True,
            transform=physical_transform,
            title="Preferred Physical Report (scaled)",
        )

    preferred_basis = print_preferred_report(
        basis_records,
        enable_scaled=False,
        transform=transform,
        title="Preferred Transformed Report (raw)",
    )

    if preferred_physical is not None:
        print("\n=== Basis Comparison ===")
        print(
            f"physical/scaled cond={preferred_physical['metrics']['cond_scaled']:.3e} | "
            f"transformed/raw cond={preferred_basis['metrics']['cond_raw']:.3e}"
        )
        phys_corr = preferred_physical["metrics"]["corr_scaled"]
        basis_corr = preferred_basis["metrics"]["corr_raw"]
        if transform.mode == "log" and phys_corr is not None and basis_corr is not None and phys_corr.shape == basis_corr.shape:
            diff = np.max(np.abs(phys_corr - basis_corr))
            print(f"max |corr_physical_scaled - corr_transformed_raw| = {diff:.3e}")

    maybe_plot_corner(basis_records, transform)


if __name__ == "__main__":
    main()
