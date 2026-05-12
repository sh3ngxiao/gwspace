from __future__ import annotations

import argparse
import csv
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


DEFAULT_SOURCE_ROOT = REPO_ROOT / "tianqin_dc" / "smbhb_sources"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tianqin_dc" / "codes" / "mbhb_catalogs"
DEFAULT_COUNT = 500
DEFAULT_SEED = 20260512
YEAR_S = float(YRSID_SI)
TWO_PI = float(2.0 * math.pi)

MIN_F_GW_HZ = 5.0e-5
MAX_F_GW_HZ = 2.0e-3
MIN_TC_YR = 2.0 / 365.25
MAX_TC_YR = 2.0
MIN_F_ISCO_HZ = 3.0e-4

OMEGA_LAMBDA = 0.689
OMEGA_MATTER = 0.311
OMEGA_RADIATION_AND_NU = 2.47e-5 * pow(0.677, -2) + 0.1 * pow(93.12, -1) * pow(0.677, -2)

K16_CATALOGS = {
    "popIII": DEFAULT_SOURCE_ROOT / "popIII_K16.dat" / "popIII_K16.dat",
    "Q3d": DEFAULT_SOURCE_ROOT / "Q3d_K16.dat" / "Q3d_K16.dat",
    "Q3nod": DEFAULT_SOURCE_ROOT / "Q3nod_K16.dat" / "Q3nod_K16.dat",
}

FIELDNAMES = (
    "source_id",
    "z",
    "m1_Msun",
    "m2_Msun",
    "total_mass_Msun",
    "chirp_mass_Msun",
    "q",
    "DL_Mpc",
    "t_c_yr",
    "f_gw_start_hz",
    "f_gw_isco_hz",
    "Lambda",
    "Beta",
    "iota",
    "psi_rad",
    "var_phi",
    "phi_c",
    "eccentricity",
    "engine",
    "score_proxy",
    "signal_rss",
    "signal_max",
    "catalog_kind",
    "population_model",
    "population_weight",
    "observer_rate_weight_proxy",
    "original_row_number",
    "original_m1_Msun",
    "original_m2_Msun",
    "spin1",
    "spin2",
    "spin_angle_alpha",
    "spin_angle_beta",
    "spin_angle_gamma",
    "spin_plane_psi",
    "final_spin",
    "final_mass_Msun",
    "kick_velocity_km_s",
    "escape_velocity_halo_km_s",
    "escape_velocity_baryon_km_s",
    "cluster_mass_Msun",
    "disk_stars_mass_Msun",
    "disk_gas_mass_Msun",
    "bulge_stars_mass_Msun",
    "bulge_gas_mass_Msun",
    "nuclear_star_cluster_mass_Msun",
    "reservoir_mass_Msun",
    "final_halo_mass_Msun",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Barausse/Klein K16 SMBHB population catalogs into the TianQin-friendly "
            "MBHB CSV format used by minimal_catalog_aet."
        )
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Rows to sample from each K16 catalog.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Directory containing the extracted K16 .dat folders. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated CSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--engine", default="bhb_EccFD", help="Waveform engine value written to the CSV.")
    parser.add_argument(
        "--allow-replacement",
        action="store_true",
        help="Allow replacement if fewer valid rows than --count pass the mHz/window filter.",
    )
    return parser


def chirp_mass_msun(m1_msun: np.ndarray, m2_msun: np.ndarray) -> np.ndarray:
    total_mass = m1_msun + m2_msun
    return (m1_msun * m2_msun) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)


def f_isco_hz(total_mass_redshifted_msun: np.ndarray) -> np.ndarray:
    return 1.0 / (PI * (6.0**1.5) * total_mass_redshifted_msun * MTSUN_SI)


def gw_frequency_from_time_to_coalescence_hz(
    mchirp_redshifted_msun: np.ndarray,
    tau_s: float,
) -> np.ndarray:
    mc_sec = mchirp_redshifted_msun * MTSUN_SI
    return (5.0 / 256.0 / tau_s / mc_sec ** (5.0 / 3.0)) ** (3.0 / 8.0) / PI


def time_to_coalescence_years(mchirp_redshifted_msun: float, f_gw_hz: float) -> float:
    mc_sec = mchirp_redshifted_msun * MTSUN_SI
    tau_s = 5.0 / 256.0 * (PI * f_gw_hz) ** (-8.0 / 3.0) * mc_sec ** (-5.0 / 3.0)
    return float(tau_s / YEAR_S)


def sample_angles(rng: np.random.Generator) -> tuple[float, float, float, float, float, float]:
    lambda_rad = float(rng.uniform(0.0, TWO_PI))
    beta_rad = float(math.asin(rng.uniform(-1.0, 1.0)))
    iota_rad = float(math.acos(rng.uniform(-1.0, 1.0)))
    psi_rad = float(rng.uniform(0.0, math.pi))
    var_phi = float(rng.uniform(0.0, TWO_PI))
    phi_c = float(rng.uniform(0.0, TWO_PI))
    return lambda_rad, beta_rad, iota_rad, psi_rad, var_phi, phi_c


def expansion_e(redshift: np.ndarray) -> np.ndarray:
    return np.sqrt(
        OMEGA_LAMBDA
        + OMEGA_MATTER * (1.0 + redshift) ** 3
        + OMEGA_RADIATION_AND_NU * (1.0 + redshift) ** 4
    )


def observer_rate_weight_proxy(redshift: np.ndarray, luminosity_distance_mpc: np.ndarray, wps: np.ndarray) -> np.ndarray:
    comoving_distance_mpc = luminosity_distance_mpc / (1.0 + redshift)
    weights = wps * comoving_distance_mpc**2 / expansion_e(redshift) / (1.0 + redshift)
    return np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)


def source_path(source_root: Path, population: str) -> Path:
    if population == "popIII":
        return source_root / "popIII_K16.dat" / "popIII_K16.dat"
    if population == "Q3d":
        return source_root / "Q3d_K16.dat" / "Q3d_K16.dat"
    if population == "Q3nod":
        return source_root / "Q3nod_K16.dat" / "Q3nod_K16.dat"
    raise ValueError(f"Unsupported K16 population {population!r}.")


def select_indices(
    table: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    allow_replacement: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    z = table[:, 0]
    original_m1 = table[:, 1]
    original_m2 = table[:, 2]
    m1 = np.maximum(original_m1, original_m2)
    m2 = np.minimum(original_m1, original_m2)
    total_mass = m1 + m2
    chirp_mass = chirp_mass_msun(m1, m2)
    luminosity_distance = redshift_to_luminosity_distance_mpc(z)
    fisco = f_isco_hz(total_mass * (1.0 + z))
    mchirp_redshifted = chirp_mass * (1.0 + z)

    f_low = np.maximum(MIN_F_GW_HZ, 0.03 * fisco)
    f_high = np.minimum(MAX_F_GW_HZ, 0.15 * fisco)
    f_low = np.maximum(f_low, gw_frequency_from_time_to_coalescence_hz(mchirp_redshifted, MAX_TC_YR * YEAR_S))
    f_high = np.minimum(f_high, gw_frequency_from_time_to_coalescence_hz(mchirp_redshifted, MIN_TC_YR * YEAR_S))

    valid = (
        np.isfinite(z)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & np.isfinite(luminosity_distance)
        & (z >= 0.0)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (luminosity_distance > 0.0)
        & (fisco >= MIN_F_ISCO_HZ)
        & (f_low < f_high)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError("No K16 rows passed the mHz/window selection.")

    replace = False
    if valid_indices.size < count:
        if not allow_replacement:
            raise ValueError(
                f"Only {valid_indices.size} valid rows are available, fewer than requested count={count}. "
                "Use --allow-replacement to sample with replacement."
            )
        replace = True

    weights = observer_rate_weight_proxy(z, np.asarray(luminosity_distance), table[:, 22])[valid_indices]
    if float(np.sum(weights)) <= 0.0:
        probabilities = None
    else:
        probabilities = weights / np.sum(weights)

    chosen = rng.choice(valid_indices, size=count, replace=replace, p=probabilities)
    return np.sort(chosen), {
        "input_rows": int(table.shape[0]),
        "valid_rows": int(valid_indices.size),
        "sampled_rows": int(count),
    }


def build_rows(
    table: np.ndarray,
    selected_indices: np.ndarray,
    *,
    population: str,
    rng: np.random.Generator,
    engine: str,
) -> list[dict[str, Any]]:
    selected = table[selected_indices]
    z_all = selected[:, 0]
    original_m1_all = selected[:, 1]
    original_m2_all = selected[:, 2]
    m1_all = np.maximum(original_m1_all, original_m2_all)
    m2_all = np.minimum(original_m1_all, original_m2_all)
    total_mass_all = m1_all + m2_all
    chirp_mass_all = chirp_mass_msun(m1_all, m2_all)
    luminosity_distance_all = np.asarray(redshift_to_luminosity_distance_mpc(z_all), dtype=np.float64)
    fisco_all = f_isco_hz(total_mass_all * (1.0 + z_all))
    mchirp_redshifted_all = chirp_mass_all * (1.0 + z_all)
    rate_weights = observer_rate_weight_proxy(z_all, luminosity_distance_all, selected[:, 22])

    rows: list[dict[str, Any]] = []
    for output_index, row_index in enumerate(selected_indices, start=1):
        row = table[int(row_index)]
        array_index = output_index - 1
        z = float(z_all[array_index])
        m1 = float(m1_all[array_index])
        m2 = float(m2_all[array_index])
        total_mass = float(total_mass_all[array_index])
        chirp_mass = float(chirp_mass_all[array_index])
        fisco = float(fisco_all[array_index])
        mchirp_redshifted = float(mchirp_redshifted_all[array_index])

        f_low = max(MIN_F_GW_HZ, 0.03 * fisco)
        f_high = min(MAX_F_GW_HZ, 0.15 * fisco)
        f_low = max(f_low, float(gw_frequency_from_time_to_coalescence_hz(np.asarray(mchirp_redshifted), MAX_TC_YR * YEAR_S)))
        f_high = min(f_high, float(gw_frequency_from_time_to_coalescence_hz(np.asarray(mchirp_redshifted), MIN_TC_YR * YEAR_S)))
        if f_low >= f_high:
            raise RuntimeError(f"Internal selection error: row {row_index + 1} has empty frequency range.")

        f_start = float(math.exp(rng.uniform(math.log(f_low), math.log(f_high))))
        t_c_yr = time_to_coalescence_years(mchirp_redshifted, f_start)
        lambda_rad, beta_rad, iota_rad, psi_rad, var_phi, phi_c = sample_angles(rng)
        distance_mpc = float(luminosity_distance_all[array_index])
        score_proxy = float((mchirp_redshifted ** (5.0 / 3.0)) * (f_start ** (2.0 / 3.0)) / distance_mpc)

        rows.append(
            {
                "source_id": f"{population}_K16_{output_index:04d}",
                "z": z,
                "m1_Msun": m1,
                "m2_Msun": m2,
                "total_mass_Msun": total_mass,
                "chirp_mass_Msun": chirp_mass,
                "q": m2 / m1,
                "DL_Mpc": distance_mpc,
                "t_c_yr": t_c_yr,
                "f_gw_start_hz": f_start,
                "f_gw_isco_hz": fisco,
                "Lambda": lambda_rad,
                "Beta": beta_rad,
                "iota": iota_rad,
                "psi_rad": psi_rad,
                "var_phi": var_phi,
                "phi_c": phi_c,
                "eccentricity": 0.0,
                "engine": engine,
                "score_proxy": score_proxy,
                "signal_rss": "",
                "signal_max": "",
                "catalog_kind": f"K16_{population}_weighted_sample",
                "population_model": population,
                "population_weight": float(row[22]),
                "observer_rate_weight_proxy": float(rate_weights[array_index]),
                "original_row_number": int(row_index + 1),
                "original_m1_Msun": float(row[1]),
                "original_m2_Msun": float(row[2]),
                "spin1": float(row[3]),
                "spin2": float(row[4]),
                "spin_angle_alpha": float(row[5]),
                "spin_angle_beta": float(row[6]),
                "spin_angle_gamma": float(row[7]),
                "spin_plane_psi": float(row[8]),
                "final_spin": float(row[9]),
                "final_mass_Msun": float(row[10]),
                "kick_velocity_km_s": float(row[11]),
                "escape_velocity_halo_km_s": float(row[12]),
                "escape_velocity_baryon_km_s": float(row[13]),
                "cluster_mass_Msun": float(row[14]),
                "disk_stars_mass_Msun": float(row[15]),
                "disk_gas_mass_Msun": float(row[16]),
                "bulge_stars_mass_Msun": float(row[17]),
                "bulge_gas_mass_Msun": float(row[18]),
                "nuclear_star_cluster_mass_Msun": float(row[19]),
                "reservoir_mass_Msun": float(row[20]),
                "final_halo_mass_Msun": float(row[21]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def convert_catalog(
    *,
    population: str,
    input_path: Path,
    output_path: Path,
    count: int,
    rng: np.random.Generator,
    engine: str,
    allow_replacement: bool,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"K16 input catalog does not exist: {input_path}")
    table = np.loadtxt(input_path)
    if table.ndim != 2 or table.shape[1] != 23:
        raise ValueError(f"K16 catalog {input_path} must have 23 columns, got shape {table.shape}.")

    selected_indices, stats = select_indices(
        table,
        count=count,
        rng=rng,
        allow_replacement=allow_replacement,
    )
    rows = build_rows(table, selected_indices, population=population, rng=rng, engine=engine)
    write_csv(output_path, rows)
    stats.update(
        {
            "population": population,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
    )
    return stats


def main() -> int:
    args = build_parser().parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")

    seed_sequence = np.random.SeedSequence(args.seed)
    population_sequences = seed_sequence.spawn(len(K16_CATALOGS))

    for (population, _default_path), sequence in zip(K16_CATALOGS.items(), population_sequences, strict=True):
        input_path = source_path(args.source_root, population)
        output_path = args.output_dir / f"tianqin_mbhb_{population}_K16_{args.count}.csv"
        rng = np.random.default_rng(int(sequence.generate_state(1, dtype=np.uint64)[0]))
        stats = convert_catalog(
            population=population,
            input_path=input_path,
            output_path=output_path,
            count=args.count,
            rng=rng,
            engine=str(args.engine),
            allow_replacement=bool(args.allow_replacement),
        )
        print(
            f"Wrote {stats['sampled_rows']} {population} rows to {stats['output_path']} "
            f"from {stats['valid_rows']}/{stats['input_rows']} valid K16 rows.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
