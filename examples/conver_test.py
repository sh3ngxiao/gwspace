#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run convergence tests for the EMRI Fisher calculation.

This script reuses the setup in examples/fisher.py and scans:
- finite-difference relative step sizes
- frequency-grid caps (max_bins)

For each run it compares the result against a reference configuration:
- smallest rel_step
- largest max_bins
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fisher as fcfg
from gwspace.Noise import TianQinNoise
from gwspace.Waveform import EMRIWaveform
from gwspace.constants import YRSID_SI
from gwspace.fishertool import fisher_matrix


def _parse_csv_floats(text: str) -> list[float]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty float csv string")
    return [float(x) for x in values]


def _parse_csv_ints(text: str) -> list[int]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty int csv string")
    return [int(x) for x in values]


def _parse_csv_strings(text: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("empty string csv")
    return values


def _default_max_bins_grid(base: int) -> list[int]:
    base = int(base)
    candidates = {
        max(2000, int(np.ceil(base / 4))),
        max(4000, int(np.ceil(base / 2))),
        max(8000, base),
    }
    return sorted(candidates)


def _format_sci(value: float, precision: int = 3) -> str:
    if value is None:
        return "None"
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.{precision}e}"


def _format_pct(value: float) -> str:
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{100.0 * value:.3g}%"


def _format_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def pad(text: str, width: int, align: str) -> str:
        return str(text).rjust(width) if align == "right" else str(text).ljust(width)

    header_line = " | ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(pad(cell, widths[i], aligns[i]) for i, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def _quality_label(rank: int, total: int, cond: float, delta_sigma: float) -> str:
    if rank < total:
        return "rank-def"
    if not np.isfinite(cond):
        return "ill-cond"
    if cond > 1e10:
        return "warn-cond"
    if delta_sigma > 5e-2:
        return "drift"
    if delta_sigma > 1e-2:
        return "ok"
    return "stable"


def _record_to_jsonable(rec: dict) -> dict:
    out = dict(rec)
    out["result"] = {
        "params": list(rec["result"]["params"]),
        "snr": float(rec["result"]["snr"]),
        "fisher": np.asarray(rec["result"]["fisher"]).tolist(),
        "cov": np.asarray(rec["result"]["cov"]).tolist(),
        "sigma": dict(rec["result"]["sigma"]),
        "frequency_size": int(np.asarray(rec["result"]["frequency"]).size),
    }
    out["metrics"] = {
        "cond_input": float(rec["metrics"]["cond_input"]),
        "min_eig_input": float(rec["metrics"]["min_eig_input"]),
        "max_eig_input": float(rec["metrics"]["max_eig_input"]),
        "cond_raw": float(rec["metrics"]["cond_raw"]),
        "min_eig_raw": float(rec["metrics"]["min_eig_raw"]),
        "max_eig_raw": float(rec["metrics"]["max_eig_raw"]),
        "rank_raw": int(rec["metrics"]["rank_raw"]),
        "sigma_raw": dict(rec["metrics"]["sigma_raw"]),
        "corr_raw": np.asarray(rec["metrics"]["corr_raw"]).tolist(),
        "cond_scaled": (
            None if not np.isfinite(rec["metrics"]["cond_scaled"])
            else float(rec["metrics"]["cond_scaled"])
        ),
        "rank_scaled": int(rec["metrics"]["rank_scaled"]),
        "sigma_scaled": (
            None if rec["metrics"]["sigma_scaled"] is None
            else dict(rec["metrics"]["sigma_scaled"])
        ),
        "corr_scaled": (
            None if rec["metrics"]["corr_scaled"] is None
            else np.asarray(rec["metrics"]["corr_scaled"]).tolist()
        ),
        "prior_rel": float(rec["metrics"]["prior_rel"]),
        "prior_sigmas": rec["metrics"]["prior_sigmas"],
    }
    return out


def _maybe_plot(records: list[dict], rel_steps: list[float], max_bins_grid: list[int], out_png: Path) -> None:
    try:
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[WARN] Skip plot: cannot import matplotlib ({exc})")
        return

    rel_steps = list(rel_steps)
    max_bins_grid = list(max_bins_grid)
    use_scaled = fcfg.ENABLE_SCALED_FISHER
    sigma_key = "delta_sigma_scaled_ref" if use_scaled else "delta_sigma_raw_ref"
    cond_key = "cond_scaled" if use_scaled else "cond_raw"

    sigma_grid = np.full((len(max_bins_grid), len(rel_steps)), np.nan, dtype=float)
    cond_grid = np.full((len(max_bins_grid), len(rel_steps)), np.nan, dtype=float)
    fisher_grid = np.full((len(max_bins_grid), len(rel_steps)), np.nan, dtype=float)
    rank_grid = np.full((len(max_bins_grid), len(rel_steps)), np.nan, dtype=float)

    max_idx = {int(v): i for i, v in enumerate(max_bins_grid)}
    step_idx = {float(v): i for i, v in enumerate(rel_steps)}
    for rec in records:
        i = max_idx[int(rec["max_bins"])]
        j = step_idx[float(rec["rel_step"])]
        sigma_grid[i, j] = float(rec[sigma_key])
        cond_grid[i, j] = float(rec["metrics"][cond_key])
        fisher_grid[i, j] = float(rec["delta_fisher_ref"])
        rank_grid[i, j] = float(rec["metrics"]["rank_scaled" if use_scaled else "rank_raw"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def draw(ax, data, title, cmap="viridis", log10=False):
        arr = np.array(data, copy=True)
        if log10:
            positive = arr > 0
            arr[positive] = np.log10(arr[positive])
            arr[~positive] = np.nan
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(rel_steps)))
        ax.set_xticklabels([f"{v:.1e}" for v in rel_steps], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(max_bins_grid)))
        ax.set_yticklabels([str(v) for v in max_bins_grid])
        ax.set_xlabel("rel_step")
        ax.set_ylabel("max_bins")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw(axes[0, 0], sigma_grid, f"log10({sigma_key})", log10=True)
    draw(axes[0, 1], cond_grid, f"log10({cond_key})", log10=True)
    draw(axes[1, 0], fisher_grid, "log10(delta_fisher_ref)", log10=True)
    draw(axes[1, 1], rank_grid, "effective rank", cmap="cividis")

    fig.suptitle(
        f"EMRI Fisher Convergence | params={','.join(records[0]['params'])}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[OK] Plot saved: {out_png}")


def _build_records(params: list[str], rel_steps: list[float], max_bins_grid: list[int]) -> tuple[list[dict], dict]:
    wf = EMRIWaveform(**fcfg.EMRIpars)
    adapter = fcfg.EMRIAETAdapter(
        wf,
        dt=fcfg.DT,
        det="TQ",
        TDIgen=1,
        window_alpha=fcfg.WINDOW_ALPHA,
        window_power_correction=True,
        use_interp_response=fcfg.USE_INTERP_RESPONSE,
    )

    param_values = np.array([fcfg.get_param_value(wf, p) for p in params], dtype=float)
    records: list[dict] = []

    for max_bins in max_bins_grid:
        f_series, stride, n_full_fft = fcfg.build_frequency_series(
            T_obs=wf.T_obs,
            dt=fcfg.DT,
            fmin=fcfg.FMIN,
            fmax=fcfg.FMAX,
            max_bins=max_bins,
        )
        for rel_step in rel_steps:
            result = fisher_matrix(
                adapter,
                params=params,
                det="TQ",
                channel="AET",
                TDIgen=1,
                f_series=f_series,
                noise=TianQinNoise(),
                use_T=fcfg.USE_T,
                rel_step=float(rel_step),
            )
            metrics = fcfg.analyze_fisher_matrix(
                result["fisher"],
                params=params,
                param_values=param_values,
                rcond=fcfg.PINV_RCOND,
                enable_scaled=fcfg.ENABLE_SCALED_FISHER,
                scale_mode=fcfg.SCALE_MODE,
                prior_rel=fcfg.PRIOR_REL,
            )
            records.append(
                {
                    "params": list(params),
                    "max_bins": int(max_bins),
                    "stride": int(stride),
                    "n_full_fft": int(n_full_fft),
                    "n_band_used": int(f_series.size),
                    "rel_step": float(rel_step),
                    "result": result,
                    "metrics": metrics,
                }
            )

    reference = min(
        records,
        key=lambda x: (
            -int(x["max_bins"]),
            float(x["rel_step"]),
        ),
    )
    sigma_key = "sigma_scaled" if fcfg.ENABLE_SCALED_FISHER else "sigma_raw"

    for rec in records:
        rec["delta_fisher_ref"] = fcfg.relative_fisher_change(
            rec["result"]["fisher"],
            reference["result"]["fisher"],
        )
        rec["delta_sigma_raw_ref"] = fcfg.relative_sigma_change(
            rec["metrics"]["sigma_raw"],
            reference["metrics"]["sigma_raw"],
        )
        rec["delta_sigma_raw_ref_map"] = fcfg.per_param_relative_change(
            rec["metrics"]["sigma_raw"],
            reference["metrics"]["sigma_raw"],
        )

        if fcfg.ENABLE_SCALED_FISHER and rec["metrics"]["sigma_scaled"] is not None:
            rec["delta_sigma_scaled_ref"] = fcfg.relative_sigma_change(
                rec["metrics"]["sigma_scaled"],
                reference["metrics"]["sigma_scaled"],
            )
            rec["delta_sigma_scaled_ref_map"] = fcfg.per_param_relative_change(
                rec["metrics"]["sigma_scaled"],
                reference["metrics"]["sigma_scaled"],
            )
        else:
            rec["delta_sigma_scaled_ref"] = np.nan
            rec["delta_sigma_scaled_ref_map"] = None

        ref_sigma = reference["metrics"][sigma_key]
        cur_sigma = rec["metrics"][sigma_key]
        rec["max_sigma_ref_rel"] = max(
            abs(cur_sigma[p] - ref_sigma[p]) / (abs(ref_sigma[p]) if ref_sigma[p] != 0 else 1.0)
            for p in params
        )

    meta = {
        "params": list(params),
        "param_values": {p: float(v) for p, v in zip(params, param_values)},
        "T_obs_yr": float(wf.T_obs / YRSID_SI),
        "DT": float(fcfg.DT),
        "FMIN": float(fcfg.FMIN),
        "FMAX": float(fcfg.FMAX),
        "USE_T": bool(fcfg.USE_T),
        "PINV_RCOND": float(fcfg.PINV_RCOND),
        "ENABLE_SCALED_FISHER": bool(fcfg.ENABLE_SCALED_FISHER),
        "SCALE_MODE": str(fcfg.SCALE_MODE),
        "USE_INTERP_RESPONSE": bool(fcfg.USE_INTERP_RESPONSE),
        "rel_steps": [float(v) for v in rel_steps],
        "max_bins_grid": [int(v) for v in max_bins_grid],
        "reference": {
            "max_bins": int(reference["max_bins"]),
            "rel_step": float(reference["rel_step"]),
        },
    }
    return records, meta


def _find_reference_record(records: list[dict], meta: dict) -> dict:
    for rec in records:
        if (
            int(rec["max_bins"]) == int(meta["reference"]["max_bins"])
            and np.isclose(float(rec["rel_step"]), float(meta["reference"]["rel_step"]))
        ):
            return rec
    raise RuntimeError("Reference record not found.")


def _print_summary(records: list[dict], meta: dict) -> None:
    use_scaled = fcfg.ENABLE_SCALED_FISHER
    sigma_key = "sigma_scaled" if use_scaled else "sigma_raw"
    cond_key = "cond_scaled" if use_scaled else "cond_raw"
    rank_key = "rank_scaled" if use_scaled else "rank_raw"
    delta_sigma_key = "delta_sigma_scaled_ref" if use_scaled else "delta_sigma_raw_ref"
    delta_sigma_map_key = "delta_sigma_scaled_ref_map" if use_scaled else "delta_sigma_raw_ref_map"
    reference = _find_reference_record(records, meta)

    print(
        "Reference run: "
        f"max_bins={meta['reference']['max_bins']}, "
        f"rel_step={meta['reference']['rel_step']:.1e}"
    )
    print(
        f"T_obs={meta['T_obs_yr']:.4f} yr | DT={meta['DT']:.3f} s | "
        f"params={meta['params']} | use_scaled={use_scaled}"
    )
    print(f"Reading guide: rank -> independent directions; dSigma_ref -> sigma change vs reference.")
    print("\n=== Convergence Table ===")

    records_sorted = sorted(records, key=lambda x: (int(x["max_bins"]), float(x["rel_step"])))
    total = len(meta["params"])
    summary_rows = []
    for rec in records_sorted:
        key_map = rec[delta_sigma_map_key]
        worst_param = "-"
        if key_map:
            worst_param = max(key_map, key=key_map.get)
        quality = _quality_label(
            int(rec["metrics"][rank_key]),
            total,
            float(rec["metrics"][cond_key]),
            float(rec[delta_sigma_key]),
        )
        summary_rows.append(
            [
                str(rec["max_bins"]),
                str(rec["n_band_used"]),
                str(rec["stride"]),
                f"{rec['rel_step']:.1e}",
                _format_sci(rec["result"]["snr"]),
                f"{rec['metrics'][rank_key]}/{total}",
                _format_sci(rec["metrics"][cond_key]),
                _format_pct(rec["delta_fisher_ref"]),
                _format_pct(rec[delta_sigma_key]),
                worst_param,
                quality,
            ]
        )

    print(
        _format_table(
            headers=[
                "max_bins", "N_used", "stride", "rel_step", "SNR",
                "rank", "cond", "dF_ref", "dSigma_ref", "worst_p", "status",
            ],
            rows=summary_rows,
            aligns=[
                "right", "right", "right", "right", "right",
                "right", "right", "right", "right", "left", "left",
            ],
        )
    )

    print("\n=== Reference Sigma ===")
    sigma_rows = []
    for param in meta["params"]:
        value = float(meta["param_values"][param])
        sigma = float(reference["metrics"][sigma_key][param])
        rel_err = abs(sigma / value) if value != 0 else np.nan
        sigma_rows.append(
            [
                param,
                _format_sci(value, precision=6),
                _format_sci(sigma, precision=6),
                _format_pct(rel_err),
            ]
        )
    print(
        _format_table(
            headers=["param", "value", sigma_key, "rel_err"],
            rows=sigma_rows,
            aligns=["left", "right", "right", "right"],
        )
    )

    if fcfg.REPORT_PER_PARAM:
        print("\n=== Per-Parameter dSigma_ref ===")
        detail_headers = ["max_bins", "rel_step", *meta["params"]]
        detail_rows = []
        for rec in records_sorted:
            key_map = rec[delta_sigma_map_key]
            row = [str(rec["max_bins"]), f"{rec['rel_step']:.1e}"]
            for param in meta["params"]:
                row.append(_format_pct(key_map[param]))
            detail_rows.append(row)
        print(
            _format_table(
                headers=detail_headers,
                rows=detail_rows,
                aligns=["right", "right", *(["right"] * len(meta["params"]))],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run convergence tests for examples/fisher.py.")
    parser.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated parameter list. Default: reuse FISHER_PARAMS / fisher.py PARAMS.",
    )
    parser.add_argument(
        "--rel-steps",
        type=str,
        default="",
        help="Comma-separated rel_step grid. Default: reuse fisher.py REL_STEPS.",
    )
    parser.add_argument(
        "--max-bins",
        type=str,
        default="",
        help="Comma-separated max_bins grid. Default: {MAX/4, MAX/2, MAX}.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="examples/fisher_results",
        help="Output directory.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional output tag.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Do not generate the summary plot.",
    )
    args = parser.parse_args()

    params = _parse_csv_strings(args.params) if args.params.strip() else list(fcfg.PARAMS)
    rel_steps = _parse_csv_floats(args.rel_steps) if args.rel_steps.strip() else list(fcfg.REL_STEPS)
    max_bins_grid = (
        _parse_csv_ints(args.max_bins)
        if args.max_bins.strip()
        else _default_max_bins_grid(int(fcfg.MAX_FREQ_BINS))
    )

    rel_steps = sorted(set(float(v) for v in rel_steps))
    max_bins_grid = sorted(set(int(v) for v in max_bins_grid))

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records, meta = _build_records(params=params, rel_steps=rel_steps, max_bins_grid=max_bins_grid)
    _print_summary(records, meta)

    tag = args.tag.strip()
    if not tag:
        tag = (
            f"T{meta['T_obs_yr']:.3f}yr_"
            f"{'_'.join(meta['params'])}_"
            f"mb{'-'.join(str(v) for v in max_bins_grid)}"
        ).replace(".", "p")

    out_json = outdir / f"conver_test_{tag}.json"
    payload = {"meta": meta, "records": [_record_to_jsonable(rec) for rec in records]}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] JSON saved: {out_json}")

    if not args.skip_plot:
        out_png = outdir / f"conver_test_{tag}.png"
        _maybe_plot(records, rel_steps=rel_steps, max_bins_grid=max_bins_grid, out_png=out_png)


if __name__ == "__main__":
    main()
