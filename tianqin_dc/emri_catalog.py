from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


_EMRI_CATALOG_FILE_RE = re.compile(
    r"EMRICAT(?P<catalog_id>\d+)_MBH(?P<compact_object_mass>\d+)_SIGMA(?P<sigma>\d+)_"
    r"NPL(?P<npl>\d+)_CUSP(?P<cusp>\d+)_JON(?P<jon>\d+)_SPIN(?P<spin>\d+)\.OUT"
)


def _catalog_year(catalog_id: int) -> int:
    return catalog_id - 100 if catalog_id >= 100 else catalog_id


@dataclass(frozen=True)
class EMRICatalogEntry:
    file_path: str
    file_name: str
    catalog_id: int
    catalog_year: int
    row_number: int
    compact_object_mass_msun: float
    sigma_model: int
    npl_code: int
    plunges_per_emri: int
    cusp_model: int
    jon_model: int
    spin_model: int
    log10_mbh_mass: float
    redshift: float
    mbh_spin: float
    inclination: float
    distance_gpc: float

    @property
    def mbh_mass_msun(self) -> float:
        return float(10.0 ** self.log10_mbh_mass)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "catalog_id": self.catalog_id,
            "catalog_year": self.catalog_year,
            "row_number": self.row_number,
            "compact_object_mass_msun": self.compact_object_mass_msun,
            "sigma_model": self.sigma_model,
            "npl_code": self.npl_code,
            "plunges_per_emri": self.plunges_per_emri,
            "cusp_model": self.cusp_model,
            "jon_model": self.jon_model,
            "spin_model": self.spin_model,
            "log10_mbh_mass": self.log10_mbh_mass,
            "mbh_mass_msun": self.mbh_mass_msun,
            "redshift": self.redshift,
            "mbh_spin": self.mbh_spin,
            "inclination": self.inclination,
            "distance_gpc": self.distance_gpc,
        }


def parse_emri_catalog_filename(path: str | Path) -> dict[str, Any]:
    catalog_path = Path(path)
    match = _EMRI_CATALOG_FILE_RE.fullmatch(catalog_path.name)
    if match is None:
        raise ValueError(
            f"EMRI catalog file '{catalog_path.name}' does not match the expected EMRICAT*.OUT pattern."
        )

    catalog_id = int(match.group("catalog_id"))
    npl_code = int(match.group("npl"))
    return {
        "file_path": str(catalog_path.resolve()),
        "file_name": catalog_path.name,
        "catalog_id": catalog_id,
        "catalog_year": _catalog_year(catalog_id),
        "compact_object_mass_msun": float(match.group("compact_object_mass")),
        "sigma_model": int(match.group("sigma")),
        "npl_code": npl_code,
        "plunges_per_emri": int(str(npl_code)[-3:]),
        "cusp_model": int(match.group("cusp")),
        "jon_model": int(match.group("jon")),
        "spin_model": int(match.group("spin")),
    }


def load_emri_catalog(path: str | Path) -> list[EMRICatalogEntry]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"EMRI catalog file '{catalog_path}' does not exist.")

    metadata = parse_emri_catalog_filename(catalog_path)
    rows = np.loadtxt(catalog_path, dtype=np.float64, ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError(
            f"EMRI catalog file '{catalog_path}' must contain exactly 5 numeric columns, got shape {rows.shape}."
        )

    entries: list[EMRICatalogEntry] = []
    for row_number, row in enumerate(rows, start=1):
        entries.append(
            EMRICatalogEntry(
                file_path=metadata["file_path"],
                file_name=metadata["file_name"],
                catalog_id=metadata["catalog_id"],
                catalog_year=metadata["catalog_year"],
                row_number=row_number,
                compact_object_mass_msun=metadata["compact_object_mass_msun"],
                sigma_model=metadata["sigma_model"],
                npl_code=metadata["npl_code"],
                plunges_per_emri=metadata["plunges_per_emri"],
                cusp_model=metadata["cusp_model"],
                jon_model=metadata["jon_model"],
                spin_model=metadata["spin_model"],
                log10_mbh_mass=float(row[0]),
                redshift=float(row[1]),
                mbh_spin=float(row[2]),
                inclination=float(row[3]),
                distance_gpc=float(row[4]),
            )
        )
    return entries
