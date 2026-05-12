from __future__ import annotations

from typing import Any

import numpy as np

from gwspace.Waveform import GCBWaveform

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.sources.gcb import GCBSourceFactory


_TWO_PI = float(2.0 * np.pi)
_FASTGB_ENGINE = "gwspace:fastgb"
_FASTGB_AET_IMPLEMENTATION = "fastgb_fd_xyz_irfft_aet"


class CatalogDWDWaveform(GCBWaveform):
    """GCB waveform adapter for amplitude-driven DWD catalogs."""

    __slots__ = ("catalog_amp",)

    def __init__(
        self,
        *,
        T_obs: float,
        f0: float,
        fdot: float,
        Beta: float,
        Lambda: float,
        amp: float,
        iota: float,
        psi: float,
        phi0: float,
        fddot: float | None = None,
    ) -> None:
        super().__init__(
            mass1=1.0,
            mass2=1.0,
            T_obs=T_obs,
            phi0=phi0,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            DL=1.0,
            Lambda=Lambda,
            Beta=Beta,
            iota=iota,
            psi=psi,
        )
        self.catalog_amp = float(amp)

    @property
    def amplitude(self) -> float:
        return self.catalog_amp


def _fastgb_detector_name(detector: str) -> str:
    normalized = GCBWaveform._fastgb_detector_name(detector)
    if normalized not in ("TianQin", "LISA", "TaiJi"):
        raise ValueError(f"Unsupported FastGB detector '{detector}'.")
    return normalized


def empty_fastgb_xyz_frequency_buffers(observation: ObservationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = observation.num_samples // 2 + 1
    return (
        np.zeros(length, dtype=np.complex128),
        np.zeros(length, dtype=np.complex128),
        np.zeros(length, dtype=np.complex128),
    )


def fastgb_xyz_frequency_buffers_to_xyz_td(
    buffers: tuple[np.ndarray, np.ndarray, np.ndarray],
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    expected = observation.num_samples // 2 + 1
    n_samples = observation.num_samples
    channels: dict[str, np.ndarray] = {}
    for name, spectrum in zip(("X", "Y", "Z"), buffers, strict=True):
        array = np.asarray(spectrum, dtype=np.complex128)
        if array.shape != (expected,):
            raise ValueError(f"FastGB {name} spectrum has shape {array.shape}, expected {(expected,)}.")
        # FastGB buffers already use the rFFT normalization expected by numpy.irfft.
        channels[name] = np.fft.irfft(array, n=n_samples).astype(np.float64, copy=False)
    return channels


def fastgb_xyz_to_aet_channels(
    xyz: dict[str, np.ndarray],
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    x = np.asarray(xyz["X"], dtype=np.float64)
    y = np.asarray(xyz["Y"], dtype=np.float64)
    z = np.asarray(xyz["Z"], dtype=np.float64)
    full = {
        "A": (z - x) / np.sqrt(2.0),
        "E": (x - 2.0 * y + z) / np.sqrt(6.0),
        "T": (x + y + z) / np.sqrt(3.0),
    }
    return {channel: full[channel].astype(np.float64, copy=False) for channel in observation.channels}


def fastgb_xyz_frequency_buffers_to_aet_channels(
    buffers: tuple[np.ndarray, np.ndarray, np.ndarray],
    observation: ObservationConfig,
) -> dict[str, np.ndarray]:
    return fastgb_xyz_to_aet_channels(fastgb_xyz_frequency_buffers_to_xyz_td(buffers, observation), observation)


def _fastgb_metadata(observation: ObservationConfig, *, oversample: int) -> dict[str, Any]:
    return {
        "detector_requested": observation.detector,
        "detector_fastgb": _fastgb_detector_name(observation.detector),
        "tdi_generation_requested": observation.tdi_generation,
        "fastgb_oversample": int(oversample),
        "frequency_bins": observation.num_samples // 2 + 1,
    }


def _fastgb_notes(
    observation: ObservationConfig,
    *,
    catalog_parameterization: bool,
) -> list[str]:
    notes = [
        "DWD generated with FastGB frequency-domain XYZ response and converted to time-domain channels by inverse rFFT.",
    ]
    if catalog_parameterization:
        notes.append(
            "Input parameters came from an amplitude-driven DWD source table instead of the mass-distance GCB parameterization."
        )
    if observation.tdi_generation != 1:
        notes.append(
            "FastGB XYZ generation does not expose a TDI-generation selector; the requested tdi_generation is recorded in metadata."
        )
    return notes


def _build_fastgb_dwd_waveform(
    factory: "DWDSourceFactory",
    parameters: dict[str, Any],
    observation: ObservationConfig,
) -> tuple[GCBWaveform, dict[str, Any], bool]:
    catalog_parameterization = (
        factory._is_catalog_parameterization(parameters)
        or factory._is_normalized_catalog_parameterization(parameters)
    )
    prepared = factory.prepare_parameters(parameters, observation)
    if catalog_parameterization:
        waveform = CatalogDWDWaveform(**prepared)
    else:
        waveform = GCBWaveform(**prepared)
    return waveform, prepared, catalog_parameterization


def add_dwd_to_fastgb_frequency_buffers(
    factory: "DWDSourceFactory",
    parameters: dict[str, Any],
    observation: ObservationConfig,
    buffers: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    oversample: int = 1,
) -> tuple[dict[str, Any], bool]:
    waveform, prepared, catalog_parameterization = _build_fastgb_dwd_waveform(factory, parameters, observation)
    waveform.get_fastgb_fd_single(
        observation.sample_spacing_s,
        oversample=oversample,
        detector=_fastgb_detector_name(observation.detector),
        buffer=buffers,
    )
    return prepared, catalog_parameterization


def generate_fastgb_dwd_aet_channels(
    factory: "DWDSourceFactory",
    parameters: dict[str, Any],
    observation: ObservationConfig,
    *,
    oversample: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    buffers = empty_fastgb_xyz_frequency_buffers(observation)
    prepared, catalog_parameterization = add_dwd_to_fastgb_frequency_buffers(
        factory,
        parameters,
        observation,
        buffers,
        oversample=oversample,
    )
    return fastgb_xyz_frequency_buffers_to_aet_channels(buffers, observation), prepared, catalog_parameterization


class DWDSourceFactory(GCBSourceFactory):
    kind = "dwd"
    family = "galactic_binary"
    default_engine = _FASTGB_ENGINE
    default_implementation = _FASTGB_AET_IMPLEMENTATION
    domain = "frequency_to_time"
    catalog_required_parameters = (
        "f0",
        "dfdt_0",
        "b_ecl",
        "l_ecl",
        "Amp",
        "iota",
        "psi",
        "phi_0",
    )
    normalized_catalog_required_parameters = (
        "f0",
        "fdot",
        "Beta",
        "Lambda",
        "amp",
        "iota",
        "psi",
        "phi0",
    )
    notes = (
        "DWD uses GWspace FastGB through the GCBWaveform FastGB adapter.",
        "The distinction is semantic: 'dwd' is the challenge-level detached white-dwarf-binary label, while 'gcb' remains a broader GWspace/foreground label.",
        "DWD catalogs in the 8-column LISA source-table format are also supported through a local amplitude-driven waveform adapter.",
    )

    def _is_catalog_parameterization(self, parameters: dict[str, Any]) -> bool:
        return all(name in parameters for name in self.catalog_required_parameters)

    def _is_normalized_catalog_parameterization(self, parameters: dict[str, Any]) -> bool:
        return all(name in parameters for name in self.normalized_catalog_required_parameters)

    def _normalize_catalog_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, float]:
        missing = [name for name in self.catalog_required_parameters if name not in parameters]
        if missing:
            raise ValueError(f"Source '{self.kind}' is missing required catalog parameters: {missing}.")

        prepared: dict[str, float] = {
            "T_obs": observation.effective_duration_s,
            "f0": float(parameters["f0"]),
            "fdot": float(parameters["dfdt_0"]),
            "Beta": float(parameters["b_ecl"]),
            "Lambda": float(np.mod(float(parameters["l_ecl"]), _TWO_PI)),
            "amp": float(parameters["Amp"]),
            "iota": float(parameters["iota"]),
            "psi": float(parameters["psi"]),
            "phi0": float(np.mod(float(parameters["phi_0"]), _TWO_PI)),
        }
        if "fddot" in parameters:
            prepared["fddot"] = float(parameters["fddot"])
        if "T_obs" in parameters:
            prepared["T_obs"] = float(parameters["T_obs"])
        return prepared

    def _prepare_normalized_catalog_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, float]:
        missing = [name for name in self.normalized_catalog_required_parameters if name not in parameters]
        if missing:
            raise ValueError(f"Source '{self.kind}' is missing required normalized catalog parameters: {missing}.")

        prepared: dict[str, float] = {
            "T_obs": observation.effective_duration_s,
            "f0": float(parameters["f0"]),
            "fdot": float(parameters["fdot"]),
            "Beta": float(parameters["Beta"]),
            "Lambda": float(np.mod(float(parameters["Lambda"]), _TWO_PI)),
            "amp": float(parameters["amp"]),
            "iota": float(parameters["iota"]),
            "psi": float(parameters["psi"]),
            "phi0": float(np.mod(float(parameters["phi0"]), _TWO_PI)),
        }
        if "fddot" in parameters:
            prepared["fddot"] = float(parameters["fddot"])
        if "T_obs" in parameters:
            prepared["T_obs"] = float(parameters["T_obs"])
        return prepared

    def prepare_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, Any]:
        if self._is_normalized_catalog_parameterization(parameters):
            return self._prepare_normalized_catalog_parameters(parameters, observation)
        if self._is_catalog_parameterization(parameters):
            return self._normalize_catalog_parameters(parameters, observation)
        return super().prepare_parameters(parameters, observation)

    def generate(self, parameters: dict[str, Any], observation: ObservationConfig) -> SourceGenerationResult:
        channels, prepared, catalog_parameterization = generate_fastgb_dwd_aet_channels(self, parameters, observation)
        return self.make_fastgb_result(
            channels,
            prepared,
            observation,
            catalog_parameterization=catalog_parameterization,
        )

    def make_fastgb_result(
        self,
        channels: dict[str, Any],
        prepared: dict[str, Any],
        observation: ObservationConfig,
        *,
        catalog_parameterization: bool,
        implementation: str = _FASTGB_AET_IMPLEMENTATION,
        oversample: int = 1,
    ) -> SourceGenerationResult:
        return self.make_result(
            channels,
            prepared,
            engine=_FASTGB_ENGINE,
            implementation=implementation,
            domain="frequency_to_time",
            notes=_fastgb_notes(observation, catalog_parameterization=catalog_parameterization),
            metadata={
                **_fastgb_metadata(observation, oversample=oversample),
                "catalog_parameterization": bool(catalog_parameterization),
            },
        )
