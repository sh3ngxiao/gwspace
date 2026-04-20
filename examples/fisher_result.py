#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 examples/fisher.py 的配置运行 Fisher 扫描并生成展示图。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import fisher as fcfg
from gwspace.Noise import TianQinNoise
from gwspace.Waveform import EMRIWaveform
from gwspace.fishertool import fisher_matrix
from gwspace.constants import YRSID_SI


def _build_records():
    """按 fisher.py 当前配置运行步长扫描。"""
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

    f_series, stride, n_full_fft = fcfg.build_frequency_series(
        T_obs=wf.T_obs,
        dt=fcfg.DT,
        fmin=fcfg.FMIN,
        fmax=fcfg.FMAX,
        max_bins=fcfg.MAX_FREQ_BINS,
    )

    params = list(fcfg.PARAMS)
    param_values = np.array([fcfg.get_param_value(wf, p) for p in params], dtype=float)
    records = []
    for rel_step in fcfg.REL_STEPS:
        res = fisher_matrix(
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
            res["fisher"],
            params=params,
            param_values=param_values,
            rcond=fcfg.PINV_RCOND,
            enable_scaled=fcfg.ENABLE_SCALED_FISHER,
            scale_mode=fcfg.SCALE_MODE,
        )
        cov = np.linalg.pinv(res["fisher"], rcond=fcfg.PINV_RCOND)
        records.append(
            {
                "rel_step": float(rel_step),
                "result": res,
                "metrics": metrics,
                "cov": cov,
            }
        )

    meta = {
        "T_obs_yr": float(wf.T_obs / YRSID_SI),
        "N_time": int(np.arange(0, wf.T_obs, fcfg.DT).size),
        "N_fft": int(n_full_fft),
        "N_band_used": int(f_series.size),
        "stride": int(stride),
        "params": params,
        "fisher_pinv_rcond": float(fcfg.PINV_RCOND),
        "use_interp_response": bool(fcfg.USE_INTERP_RESPONSE),
        "enable_scaled_fisher": bool(fcfg.ENABLE_SCALED_FISHER),
        "scale_mode": str(fcfg.SCALE_MODE),
    }
    return records, meta


def _plot_summary(records, meta, out_png: Path):
    """绘制综合展示图（条件数、sigma、相关系数、误差椭圆）。"""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
    except Exception as exc:  # pragma: no cover - 运行时环境依赖
        raise RuntimeError(
            "无法导入 matplotlib（常见原因是 numpy/matplotlib 二进制版本不匹配）。"
        ) from exc

    rel_steps = np.array([r["rel_step"] for r in records], dtype=float)
    cond_raw = np.array([r["metrics"]["cond_raw"] for r in records], dtype=float)
    cond_scaled = np.array([r["metrics"]["cond_scaled"] for r in records], dtype=float)
    params = meta["params"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：条件数随步长变化
    ax = axes[0, 0]
    ax.loglog(rel_steps, cond_raw, "o-", label="cond_raw")
    if np.all(np.isfinite(cond_scaled)):
        ax.loglog(rel_steps, cond_scaled, "s--", label="cond_scaled")
    ax.set_xlabel("rel_step")
    ax.set_ylabel("condition number")
    ax.set_title("Fisher 条件数 vs rel_step")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    # 右上：各参数 sigma 随步长变化（raw）
    ax = axes[0, 1]
    for p in params:
        y = np.array([r["metrics"]["sigma_raw"][p] for r in records], dtype=float)
        ax.loglog(rel_steps, y, "o-", label=f"sigma_raw({p})")
    ax.set_xlabel("rel_step")
    ax.set_ylabel("sigma")
    ax.set_title("参数误差条 vs rel_step (raw)")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    # 左下：参考步长（最小步长）相关系数热图
    ax = axes[1, 0]
    corr = records[0]["metrics"]["corr_raw"]
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_xticklabels(params, rotation=30, ha="right")
    ax.set_yticklabels(params)
    ax.set_title(f"相关系数热图（rel_step={rel_steps[0]:.1e}）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 右下：前两个参数的一阶误差椭圆
    ax = axes[1, 1]
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.5)
    if len(params) >= 2:
        p0, p1 = params[0], params[1]
        color_map = plt.cm.viridis(np.linspace(0.15, 0.9, len(records)))
        for i, rec in enumerate(records):
            cov2 = rec["cov"][:2, :2]
            vals, vecs = np.linalg.eigh(cov2)
            vals = np.clip(vals, 0.0, None)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width = 2.0 * np.sqrt(vals[0])
            height = 2.0 * np.sqrt(vals[1])
            ell = Ellipse(
                xy=(0.0, 0.0),
                width=width,
                height=height,
                angle=angle,
                edgecolor=color_map[i],
                facecolor="none",
                lw=1.6,
                label=f"rel_step={rec['rel_step']:.1e}",
            )
            ax.add_patch(ell)
        ax.set_xlabel(f"Δ{p0}")
        ax.set_ylabel(f"Δ{p1}")
        ax.set_title(f"1σ 误差椭圆（{p0}, {p1}）")
        ax.legend(fontsize=8)
        ax.autoscale_view()
    else:
        ax.text(0.5, 0.5, "参数少于2个，无法绘制误差椭圆", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("误差椭圆")

    fig.suptitle(
        f"EMRI Fisher 展示图 | T_obs={meta['T_obs_yr']:.3f} yr | "
        f"N_band={meta['N_band_used']} | stride={meta['stride']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _save_json(records, meta, out_json: Path):
    """保存机器可读结果，便于后续文章制表。"""
    payload = {"meta": meta, "records": []}
    for rec in records:
        payload["records"].append(
            {
                "rel_step": rec["rel_step"],
                "snr": float(rec["result"]["snr"]),
                "fisher": np.asarray(rec["result"]["fisher"]).tolist(),
                "cov": np.asarray(rec["cov"]).tolist(),
                "sigma_raw": rec["metrics"]["sigma_raw"],
                "sigma_scaled": rec["metrics"]["sigma_scaled"],
                "cond_raw": rec["metrics"]["cond_raw"],
                "cond_scaled": rec["metrics"]["cond_scaled"],
                "rank_raw": rec["metrics"]["rank_raw"],
                "rank_scaled": rec["metrics"]["rank_scaled"],
                "corr_raw": np.asarray(rec["metrics"]["corr_raw"]).tolist(),
                "corr_scaled": (
                    None if rec["metrics"]["corr_scaled"] is None
                    else np.asarray(rec["metrics"]["corr_scaled"]).tolist()
                ),
            }
        )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行 Fisher 扫描并绘图。")
    parser.add_argument(
        "--outdir",
        type=str,
        default="examples/fisher_results",
        help="输出目录（默认: examples/fisher_results）",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="输出文件名标签（可选）",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records, meta = _build_records()

    tag = args.tag.strip()
    if not tag:
        tag = f"T{meta['T_obs_yr']:.3f}yr_{'_'.join(meta['params'])}"
        tag = tag.replace(".", "p")

    out_png = outdir / f"fisher_summary_{tag}.png"
    out_json = outdir / f"fisher_summary_{tag}.json"

    _save_json(records, meta, out_json)
    print(f"[OK] 数据已保存: {out_json}")

    try:
        _plot_summary(records, meta, out_png)
        print(f"[OK] 图已保存: {out_png}")
    except RuntimeError as exc:
        print(f"[WARN] 绘图失败: {exc}")
        print("[WARN] 已输出 JSON，可在修复 matplotlib 后重绘。")


if __name__ == "__main__":
    main()
