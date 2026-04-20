from __future__ import annotations

from typing import Any

import numpy as np

from gwspace.Waveform import GCBWaveform

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_td
from tianqin_dc.sources.gcb import GCBSourceFactory


_TWO_PI = float(2.0 * np.pi)


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


class DWDSourceFactory(GCBSourceFactory):
    kind = "dwd"
    family = "galactic_binary"
    default_engine = "gwspace:gcb"
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
        "In this prototype, 'dwd' is implemented with the same GWspace GCBWaveform backend used for 'gcb'.",
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
        if self._is_catalog_parameterization(parameters) or self._is_normalized_catalog_parameterization(parameters):
            prepared = self.prepare_parameters(parameters, observation)
            waveform = CatalogDWDWaveform(**prepared)
            channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
            return self.make_result(
                channels,
                prepared,
                implementation="catalog_td_response",
                notes=[
                    "Input parameters came from an amplitude-driven DWD source table instead of the mass-distance GCB parameterization.",
                ],
                metadata={"catalog_parameterization": True},
            )
        return super().generate(parameters, observation)
