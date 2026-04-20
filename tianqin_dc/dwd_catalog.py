from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


_TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class DWDCatalogEntry:
    file_path: str
    file_name: str
    row_number: int
    f0: float
    dfdt_0: float
    b_ecl: float
    l_ecl: float
    amp: float
    iota: float
    psi: float
    phi_0: float

    @property
    def Lambda(self) -> float:
        return float(np.mod(self.l_ecl, _TWO_PI))

    @property
    def Beta(self) -> float:
        return self.b_ecl

    @property
    def phi0(self) -> float:
        return float(np.mod(self.phi_0, _TWO_PI))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "row_number": self.row_number,
            "f0": self.f0,
            "dfdt_0": self.dfdt_0,
            "b_ecl": self.b_ecl,
            "l_ecl": self.l_ecl,
            "Amp": self.amp,
            "iota": self.iota,
            "psi": self.psi,
            "phi_0": self.phi_0,
            "Lambda": self.Lambda,
            "Beta": self.Beta,
            "phi0": self.phi0,
        }

    def to_source_parameters(self) -> dict[str, float]:
        return {
            "f0": self.f0,
            "dfdt_0": self.dfdt_0,
            "b_ecl": self.b_ecl,
            "l_ecl": self.l_ecl,
            "Amp": self.amp,
            "iota": self.iota,
            "psi": self.psi,
            "phi_0": self.phi_0,
        }


def _parse_catalog_line(path: Path, row_number: int, line: str) -> DWDCatalogEntry:
    values = np.fromstring(line, sep=" ", dtype=np.float64)
    if values.shape != (8,):
        raise ValueError(
            f"DWD catalog file '{path}' row {row_number} must contain exactly 8 numeric columns, got {values.shape}."
        )

    return DWDCatalogEntry(
        file_path=str(path.resolve()),
        file_name=path.name,
        row_number=row_number,
        f0=float(values[0]),
        dfdt_0=float(values[1]),
        b_ecl=float(values[2]),
        l_ecl=float(values[3]),
        amp=float(values[4]),
        iota=float(values[5]),
        psi=float(values[6]),
        phi_0=float(values[7]),
    )


def iter_dwd_catalog(path: str | Path) -> Iterator[DWDCatalogEntry]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"DWD catalog file '{catalog_path}' does not exist.")

    with catalog_path.open("r", encoding="utf-8") as handle:
        row_number = 0
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            row_number += 1
            yield _parse_catalog_line(catalog_path, row_number, line)


def load_dwd_catalog(path: str | Path, *, limit: int | None = None) -> list[DWDCatalogEntry]:
    entries: list[DWDCatalogEntry] = []
    for entry in iter_dwd_catalog(path):
        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break
    return entries
