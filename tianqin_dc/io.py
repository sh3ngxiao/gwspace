from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tianqin_dc.config import RunConfig
from tianqin_dc.models import DatasetBundle
from tianqin_dc.plotting import save_time_domain_preview


def _json_dataset(group: h5py.Group, name: str, payload: Any) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=json.dumps(payload, indent=2, sort_keys=True), dtype=dtype)


def _array_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compression: str | None,
    compression_level: int,
) -> None:
    kwargs: dict[str, Any] = {}
    if compression:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = compression_level
    group.create_dataset(name, data=data, **kwargs)


def save_dataset_hdf5(bundle: DatasetBundle, config: RunConfig) -> Path:
    output_path = Path(config.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Set output.overwrite=true to replace it."
        )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_name"] = config.dataset.name
        handle.attrs["dataset_description"] = config.dataset.description
        handle.attrs["format_name"] = config.dataset.format_name
        handle.attrs["format_version"] = config.dataset.format_version
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        observation_group = handle.create_group("observation")
        observation_group.attrs["detector"] = config.observation.detector
        observation_group.attrs["tdi_generation"] = config.observation.tdi_generation
        observation_group.attrs["sample_rate_hz"] = config.observation.sample_rate_hz
        observation_group.attrs["sample_spacing_s"] = config.observation.sample_spacing_s
        observation_group.attrs["requested_duration_s"] = config.observation.duration_s
        observation_group.attrs["effective_duration_s"] = config.observation.effective_duration_s
        observation_group.attrs["num_samples"] = config.observation.num_samples
        observation_group.attrs["channels"] = json.dumps(list(config.observation.channels))
        _array_dataset(
            observation_group,
            "time_s",
            bundle.time_s,
            compression=config.output.compression,
            compression_level=config.output.compression_level,
        )

        channels_group = handle.create_group("channels")
        for section_name, payload in (
            ("signal", bundle.signal),
            ("noise", bundle.noise),
            ("observed", bundle.observed),
        ):
            section_group = channels_group.create_group(section_name)
            for channel, series in payload.items():
                _array_dataset(
                    section_group,
                    channel,
                    series,
                    compression=config.output.compression,
                    compression_level=config.output.compression_level,
                )

        if bundle.per_source:
            per_source_group = handle.create_group("per_source")
            for source_id, channel_map in bundle.per_source.items():
                source_group = per_source_group.create_group(source_id)
                for channel, series in channel_map.items():
                    _array_dataset(
                        source_group,
                        channel,
                        series,
                        compression=config.output.compression,
                        compression_level=config.output.compression_level,
                    )

        noise_group = handle.create_group("noise_model")
        _array_dataset(
            noise_group,
            "frequency_hz",
            bundle.frequency_hz,
            compression=config.output.compression,
            compression_level=config.output.compression_level,
        )
        psd_group = noise_group.create_group("psd")
        for channel, psd in bundle.noise_psd.items():
            _array_dataset(
                psd_group,
                channel,
                psd,
                compression=config.output.compression,
                compression_level=config.output.compression_level,
            )

        injections_group = handle.create_group("injections")
        _json_dataset(injections_group, "index_json", [record.to_mapping() for record in bundle.injections])
        for record in bundle.injections:
            source_group = injections_group.create_group(record.source_id)
            source_group.attrs["kind"] = record.kind
            source_group.attrs["family"] = record.family
            source_group.attrs["population_name"] = record.population_name
            source_group.attrs["population_role"] = record.population_role
            source_group.attrs["engine"] = record.engine
            source_group.attrs["implementation"] = record.implementation
            source_group.attrs["domain"] = record.domain
            source_group.attrs["seed"] = record.seed
            _json_dataset(source_group, "parameters_json", record.parameters)
            _json_dataset(source_group, "notes_json", record.notes)
            _json_dataset(source_group, "metadata_json", record.metadata)

        labels_group = handle.create_group("labels")
        _json_dataset(labels_group, "dataset_labels_json", bundle.labels)

        config_group = handle.create_group("config")
        _json_dataset(config_group, "run_config_json", bundle.run_config)

        provenance_group = handle.create_group("provenance")
        _json_dataset(provenance_group, "seed_book_json", bundle.seed_book)
        _json_dataset(provenance_group, "metadata_json", bundle.metadata)

    preview_channels = bundle.observed or bundle.signal or bundle.noise
    save_time_domain_preview(
        output_path,
        bundle.time_s,
        preview_channels,
        title=f"{config.dataset.name} observed channels",
    )
    return output_path
