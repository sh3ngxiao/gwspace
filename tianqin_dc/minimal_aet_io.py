from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


MINIMAL_AET_DTYPE = np.dtype(
    [
        ("time", np.float64),
        ("a", np.float64),
        ("e", np.float64),
        ("t", np.float64),
    ]
)


@dataclass(frozen=True)
class MinimalOutputConfig:
    path: str
    overwrite: bool = False
    compression: str | None = "gzip"
    compression_level: int = 4
    chunk_rows: int = 65536

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "MinimalOutputConfig":
        return cls(
            path=str(value["path"]),
            overwrite=bool(value.get("overwrite", False)),
            compression=value.get("compression", "gzip"),
            compression_level=int(value.get("compression_level", 4)),
            chunk_rows=int(value.get("chunk_rows", 65536)),
        )


def _dataset_kwargs(config: MinimalOutputConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.compression:
        kwargs["compression"] = config.compression
        kwargs["compression_opts"] = config.compression_level
    return kwargs


def save_minimal_aet_hdf5(
    output: MinimalOutputConfig,
    *,
    time_s: np.ndarray,
    a: np.ndarray,
    e: np.ndarray,
    t: np.ndarray,
) -> Path:
    output_path = Path(output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not output.overwrite:
        raise FileExistsError(f"Output file '{output_path}' already exists. Set output.overwrite=true to replace it.")

    arrays = {
        "time": np.asarray(time_s, dtype=np.float64),
        "a": np.asarray(a, dtype=np.float64),
        "e": np.asarray(e, dtype=np.float64),
        "t": np.asarray(t, dtype=np.float64),
    }
    n_rows = arrays["time"].shape[0]
    for name, values in arrays.items():
        if values.shape != (n_rows,):
            raise ValueError(f"Column '{name}' has shape {values.shape}, expected {(n_rows,)}.")

    chunk_rows = max(1, min(int(output.chunk_rows), n_rows))
    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=(n_rows,),
            dtype=MINIMAL_AET_DTYPE,
            chunks=(chunk_rows,),
            **_dataset_kwargs(output),
        )
        buffer = np.empty(chunk_rows, dtype=MINIMAL_AET_DTYPE)
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            span = stop - start
            buffer_view = buffer[:span]
            buffer_view["time"] = arrays["time"][start:stop]
            buffer_view["a"] = arrays["a"][start:stop]
            buffer_view["e"] = arrays["e"][start:stop]
            buffer_view["t"] = arrays["t"][start:stop]
            dataset[start:stop] = buffer_view

    return output_path


def read_minimal_aet_hdf5(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if tuple(handle.keys()) != ("data",):
            raise ValueError(f"Minimal AET file '{path}' must contain only the root dataset '/data'.")
        dataset = handle["data"]
        if dataset.dtype.fields is None:
            values = np.asarray(dataset)
            if values.ndim != 2 or values.shape[1] != 4:
                raise ValueError(f"Minimal AET dataset '{path}:/data' must have 4 columns.")
            return (
                values[:, 0].astype(np.float64, copy=True),
                {
                    "A": values[:, 1].astype(np.float64, copy=True),
                    "E": values[:, 2].astype(np.float64, copy=True),
                    "T": values[:, 3].astype(np.float64, copy=True),
                },
            )

        missing = [name for name in MINIMAL_AET_DTYPE.names or () if name not in dataset.dtype.fields]
        if missing:
            raise ValueError(f"Minimal AET dataset '{path}:/data' is missing fields: {missing}.")
        return (
            dataset["time"][:].astype(np.float64, copy=False),
            {
                "A": dataset["a"][:].astype(np.float64, copy=False),
                "E": dataset["e"][:].astype(np.float64, copy=False),
                "T": dataset["t"][:].astype(np.float64, copy=False),
            },
        )
