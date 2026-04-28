from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gwspace.constants import MTSUN_SI, PI, YRSID_SI
from tianqin_dc.bbh_catalog import redshift_to_luminosity_distance_mpc
from tianqin_dc.config import ObservationConfig
from tianqin_dc.sources.compact_binary import SMBHBSourceFactory


DEFAULT_CSV_PATH = REPO_ROOT / "tianqin_dc" / "codes" / "mbhb_catalogs" / "tianqin_mbhb_catalog_v1.csv"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "tianqin_dc" / "only_smbhb_catalog_explicit.json"

YEAR_S = float(YRSID_SI)
TWO_PI = float(2.0 * math.pi)
LOG10_Q_MIN = math.log10(0.12)
LOG10_MTOT_MEAN = 5.85
LOG10_MTOT_STD = 0.42
LOG10_MTOT_MIN = 5.0
LOG10_MTOT_MAX = 7.0
MIN_Z = 0.15
MAX_Z = 6.0
MIN_F_GW_HZ = 5.0e-5
MAX_F_GW_HZ = 2.0e-3
MIN_TC_YR = 2.0 / 365.25
MAX_TC_YR = 2.0
MIN_F_ISCO_HZ = 3.0e-4
DEFAULT_POOL_FACTOR = 8


@dataclass
class MBHBCandidate:
    source_id: str
    z: float
    m1_Msun: float
    m2_Msun: float
    total_mass_Msun: float
    chirp_mass_Msun: float
    q: float
    DL_Mpc: float
    t_c_yr: float
    f_gw_start_hz: float
    f_gw_isco_hz: float
    Lambda: float
    Beta: float
    iota: float
    psi_rad: float
    var_phi: float
    phi_c: float
    eccentricity: float
    engine: str
    score_proxy: float
    signal_rss: float | None = None
    signal_max: float | None = None

    def waveform_parameters(self) -> dict[str, Any]:
        return {
            "mass1": self.m1_Msun,
            "mass2": self.m2_Msun,
            "DL": self.DL_Mpc,
            "Lambda": self.Lambda,
            "Beta": self.Beta,
            "iota": self.iota,
            "psi": self.psi_rad,
            "var_phi": self.var_phi,
            "phi_c": self.phi_c,
            "tc": self.t_c_yr * YEAR_S,
            "eccentricity": self.eccentricity,
            "engine": self.engine,
        }

    def csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["catalog_kind"] = "tianqin_validated_mbhb"
        return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a TianQin-friendly MBHB catalog and an explicit SMBHB config."
    )
    parser.add_argument("--count", type=int, default=12, help="Number of MBHB sources to keep.")
    parser.add_argument("--seed", type=int, default=20260424, help="Random seed.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Catalog CSV output path. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Explicit SMBHB JSON config output path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--observation-duration-s",
        type=float,
        default=32768.0,
        help="Observation duration used for waveform validation and the generated JSON config.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=0.25,
        help="Sample rate used for waveform validation and the generated JSON config.",
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=DEFAULT_POOL_FACTOR,
        help="Oversampling factor before keeping the loudest validated candidates.",
    )
    return parser


def chirp_mass_msun(m1_msun: float, m2_msun: float) -> float:
    total_mass = m1_msun + m2_msun
    return float((m1_msun * m2_msun) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0))


def gw_frequency_from_time_to_coalescence_hz(mchirp_redshifted_msun: float, tau_s: float) -> float:
    mc_sec = mchirp_redshifted_msun * MTSUN_SI
    return float((5.0 / 256.0 / tau_s / mc_sec ** (5.0 / 3.0)) ** (3.0 / 8.0) / PI)


def time_to_coalescence_years(mchirp_redshifted_msun: float, f_gw_hz: float) -> float:
    mc_sec = mchirp_redshifted_msun * MTSUN_SI
    tau_s = 5.0 / 256.0 * (PI * f_gw_hz) ** (-8.0 / 3.0) * mc_sec ** (-5.0 / 3.0)
    return float(tau_s / YEAR_S)


def f_isco_hz(total_mass_redshifted_msun: float) -> float:
    return float(1.0 / (PI * (6.0 ** 1.5) * total_mass_redshifted_msun * MTSUN_SI))


def sample_redshift(rng: np.random.Generator) -> float:
    while True:
        value = float(rng.gamma(shape=3.0, scale=0.75))
        if MIN_Z <= value <= MAX_Z:
            return value


def sample_angles(rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    lam = float(rng.uniform(0.0, TWO_PI))
    beta = float(math.asin(rng.uniform(-1.0, 1.0)))
    iota = float(math.acos(rng.uniform(-1.0, 1.0)))
    psi = float(rng.uniform(0.0, math.pi))
    var_phi = float(rng.uniform(0.0, TWO_PI))
    phi_c = float(rng.uniform(0.0, TWO_PI))
    return lam, beta, iota, psi, var_phi, phi_c


def sample_candidate(rng: np.random.Generator, index: int) -> MBHBCandidate | None:
    z = sample_redshift(rng)

    log10_mtot = float(np.clip(rng.normal(LOG10_MTOT_MEAN, LOG10_MTOT_STD), LOG10_MTOT_MIN, LOG10_MTOT_MAX))
    total_mass_msun = float(10.0 ** log10_mtot)
    q = float(10.0 ** rng.uniform(LOG10_Q_MIN, 0.0))
    m1_msun = float(total_mass_msun / (1.0 + q))
    m2_msun = float(total_mass_msun - m1_msun)
    if m2_msun > m1_msun:
        m1_msun, m2_msun = m2_msun, m1_msun
        q = m2_msun / m1_msun

    total_mass_redshifted_msun = total_mass_msun * (1.0 + z)
    f_gw_isco_hz = f_isco_hz(total_mass_redshifted_msun)
    if f_gw_isco_hz < MIN_F_ISCO_HZ:
        return None

    f_low = max(MIN_F_GW_HZ, 0.03 * f_gw_isco_hz)
    f_high = min(MAX_F_GW_HZ, 0.15 * f_gw_isco_hz)
    if f_low >= f_high:
        return None

    f_gw_start_hz = float(math.exp(rng.uniform(math.log(f_low), math.log(f_high))))
    mchirp_msun = chirp_mass_msun(m1_msun, m2_msun)
    t_c_yr = time_to_coalescence_years(mchirp_msun * (1.0 + z), f_gw_start_hz)
    if not (MIN_TC_YR <= t_c_yr <= MAX_TC_YR):
        return None

    distance_mpc = float(redshift_to_luminosity_distance_mpc(z))
    lam, beta, iota, psi, var_phi, phi_c = sample_angles(rng)
    score_proxy = float(((mchirp_msun * (1.0 + z)) ** (5.0 / 3.0)) * (f_gw_start_hz ** (2.0 / 3.0)) / distance_mpc)

    return MBHBCandidate(
        source_id=f"mbhb_{index:04d}",
        z=z,
        m1_Msun=m1_msun,
        m2_Msun=m2_msun,
        total_mass_Msun=total_mass_msun,
        chirp_mass_Msun=mchirp_msun,
        q=q,
        DL_Mpc=distance_mpc,
        t_c_yr=t_c_yr,
        f_gw_start_hz=f_gw_start_hz,
        f_gw_isco_hz=f_gw_isco_hz,
        Lambda=lam,
        Beta=beta,
        iota=iota,
        psi_rad=psi,
        var_phi=var_phi,
        phi_c=phi_c,
        eccentricity=0.0,
        engine="bhb_EccFD",
        score_proxy=score_proxy,
    )


def select_candidates(count: int, rng: np.random.Generator, pool_factor: int) -> list[MBHBCandidate]:
    target_pool = max(count, count * pool_factor)
    pool: list[MBHBCandidate] = []
    attempts = 0
    max_attempts = target_pool * 200

    while len(pool) < target_pool and attempts < max_attempts:
        candidate = sample_candidate(rng, attempts + 1)
        attempts += 1
        if candidate is not None:
            pool.append(candidate)

    if len(pool) < count:
        raise RuntimeError(
            f"Only generated {len(pool)} valid MBHB candidates after {attempts} attempts; need at least {count}."
        )

    pool.sort(key=lambda item: item.score_proxy, reverse=True)
    selected = pool[:count]
    renumbered: list[MBHBCandidate] = []
    for index, item in enumerate(selected, start=1):
        renumbered.append(MBHBCandidate(**{**asdict(item), "source_id": f"mbhb_{index:04d}"}))
    return renumbered


def validate_candidates(candidates: list[MBHBCandidate], observation: ObservationConfig) -> list[MBHBCandidate]:
    factory = SMBHBSourceFactory()
    validated: list[MBHBCandidate] = []
    for candidate in candidates:
        result = factory.generate(candidate.waveform_parameters(), observation)
        concatenated = np.concatenate([np.asarray(result.channels[name], dtype=np.float64) for name in observation.channels])
        signal_rss = float(np.sqrt(np.mean(concatenated ** 2)))
        signal_max = float(np.max(np.abs(concatenated)))
        validated.append(
            MBHBCandidate(**{**asdict(candidate), "signal_rss": signal_rss, "signal_max": signal_max})
        )
    validated.sort(key=lambda item: ((item.signal_rss or 0.0), item.score_proxy), reverse=True)
    renumbered: list[MBHBCandidate] = []
    for index, item in enumerate(validated, start=1):
        renumbered.append(MBHBCandidate(**{**asdict(item), "source_id": f"mbhb_{index:04d}"}))
    return renumbered


def write_catalog_csv(path: Path, candidates: list[MBHBCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(candidates[0].csv_row().keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.csv_row())


def build_catalog_config(candidates: list[MBHBCandidate], observation: ObservationConfig, seed: int) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "tianqin_dc_only_smbhb_catalog",
            "description": "SMBHB-only TianQin XYZ dataset generated from the validated MBHB catalog.",
        },
        "seed": seed,
        "catalog": {
            "path": str(DEFAULT_CSV_PATH.relative_to(REPO_ROOT)),
            "rows_per_file": len(candidates),
            "selection": "first",
        },
        "observation": {
            "duration_s": observation.duration_s,
            "sample_rate_hz": observation.sample_rate_hz,
            "detector": observation.detector,
            "tdi_generation": observation.tdi_generation,
            "use_gpu": observation.use_gpu,
        },
        "output": {
            "path": "outputs/tianqin_dc_only_smbhb_catalog.h5",
            "overwrite": True,
            "compression": "gzip",
            "compression_level": 4,
        },
        "smbhb": {
            "seconds_per_year": YEAR_S,
        },
        "meta": {
            "catalog_csv": str(DEFAULT_CSV_PATH.relative_to(REPO_ROOT)),
            "selection_notes": [
                "Masses and redshifts are sampled from a TianQin-friendly MBHB prior with source-frame total masses around 1e5-1e7 Msun.",
                "Entries are filtered to keep merger frequencies and observation-start frequencies inside the TianQin/LISA mHz band.",
                "Each retained source is validated through the local SMBHB waveform generator.",
            ],
        },
    }


def write_config_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    observation = ObservationConfig.from_config(
        {
            "duration_s": args.observation_duration_s,
            "sample_rate_hz": args.sample_rate_hz,
            "detector": "TQ",
            "tdi_generation": 1,
            "channels": ["A", "E", "T"],
            "use_gpu": False,
        }
    )

    selected = select_candidates(args.count, rng, args.pool_factor)
    validated = validate_candidates(selected, observation)
    write_catalog_csv(args.csv_output, validated)
    config_payload = build_catalog_config(validated, observation, args.seed)
    config_payload["catalog"]["path"] = str(args.csv_output.relative_to(REPO_ROOT))
    config_payload["meta"]["catalog_csv"] = str(args.csv_output.relative_to(REPO_ROOT))
    write_config_json(args.config_output, config_payload)

    print(f"Wrote catalog: {args.csv_output}")
    print(f"Wrote config: {args.config_output}")
    print(f"Validated sources: {len(validated)}")
    print(
        "Mass range [Msun]: "
        f"{min(item.total_mass_Msun for item in validated):.3e} .. {max(item.total_mass_Msun for item in validated):.3e}"
    )
    print(
        "Redshift range: "
        f"{min(item.z for item in validated):.3f} .. {max(item.z for item in validated):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
