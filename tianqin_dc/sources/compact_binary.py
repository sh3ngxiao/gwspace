from __future__ import annotations

from copy import deepcopy
from typing import Any

from gwspace.Waveform import waveforms

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_fd
from tianqin_dc.sources.base import SourceFactory


class GWspaceCompactBinaryFactory(SourceFactory):
    family = "compact_binary"
    default_engine = "gwspace:bhb_EccFD"
    default_implementation = "gwspace_fd_response_adapter"
    domain = "frequency"
    required_parameters = (
        "mass1",
        "mass2",
        "DL",
        "Lambda",
        "Beta",
        "iota",
    )
    notes = (
        "Compact-binary sources are currently injected through a frequency-domain GWspace adapter and converted back to A/E/T time series on the observation rFFT grid.",
    )

    def prepare_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, Any]:
        prepared = super().prepare_parameters(parameters, observation)
        prepared.setdefault("T_obs", observation.effective_duration_s)
        prepared.setdefault("psi", 0.0)
        prepared.setdefault("phi_c", 0.0)
        prepared.setdefault("var_phi", 0.0)
        prepared.setdefault("tc", 0.0)
        prepared.setdefault("eccentricity", 0.0)
        prepared.setdefault("engine", "auto")
        return prepared

    def _resolve_engine_candidates(self, engine_request: str) -> tuple[str, ...]:
        normalized = engine_request.strip()
        if normalized in ("auto", ""):
            return ("bhb_PhenomD", "bhb_EccFD")
        aliases = {
            "bhb_phenomd": "bhb_PhenomD",
            "phenomd": "bhb_PhenomD",
            "bhb_eccfd": "bhb_EccFD",
            "eccfd": "bhb_EccFD",
        }
        engine = aliases.get(normalized.lower(), normalized)
        if engine not in ("bhb_PhenomD", "bhb_EccFD"):
            raise ValueError(f"Unsupported compact-binary engine request '{engine_request}'.")
        return (engine,)

    def _build_waveform(self, engine: str, parameters: dict[str, Any]):
        wave_parameters = deepcopy(parameters)
        wave_parameters.pop("engine", None)
        if engine == "bhb_PhenomD":
            wave_parameters.pop("eccentricity", None)
        return waveforms[engine](**wave_parameters)

    def _engine_specific_notes(self, engine: str, requested_kind: str) -> list[str]:
        notes: list[str] = []
        if engine == "bhb_EccFD":
            notes.append("Resolved compact-binary engine: GWspace bhb_EccFD.")
            if requested_kind == "smbhb":
                notes.append(
                    "SMBHB currently uses the same GWspace compact-binary FD engine as SBBH and is distinguished by source-type label plus population priors."
                )
        elif engine == "bhb_PhenomD":
            notes.append("Resolved compact-binary engine: GWspace bhb_PhenomD.")
        return notes

    def generate(self, parameters: dict[str, Any], observation: ObservationConfig) -> SourceGenerationResult:
        prepared = self.prepare_parameters(parameters, observation)
        engine_request = str(prepared.get("engine", "auto"))
        last_error: Exception | None = None

        for engine in self._resolve_engine_candidates(engine_request):
            try:
                waveform = self._build_waveform(engine, prepared)
                channels = generate_tdi_channels_fd(waveform, observation)
                metadata = {
                    "engine_request": engine_request,
                    "engine_resolved": engine,
                }
                return self.make_result(
                    channels,
                    prepared,
                    engine=f"gwspace:{engine}",
                    notes=self._engine_specific_notes(engine, self.kind),
                    metadata=metadata,
                )
            except (ImportError, ModuleNotFoundError) as exc:
                last_error = exc
                continue

        if last_error is None:
            raise RuntimeError(f"Failed to resolve any compact-binary engine for source kind '{self.kind}'.")
        raise RuntimeError(
            f"Compact-binary source '{self.kind}' could not be generated with engine request '{engine_request}'."
        ) from last_error


class SBBHSourceFactory(GWspaceCompactBinaryFactory):
    kind = "sbbh"
    notes = GWspaceCompactBinaryFactory.notes + (
        "SBBH is modeled as a compact-binary population. By default the engine resolver falls back to bhb_EccFD, which is available in this repository.",
    )


class SMBHBSourceFactory(GWspaceCompactBinaryFactory):
    kind = "smbhb"
    notes = GWspaceCompactBinaryFactory.notes + (
        "SMBHB is currently represented with the same GWspace compact-binary engine family as SBBH, using different population priors and labels.",
        "If PyIMRPhenomD becomes available later, you can request engine='bhb_PhenomD' in the source config.",
    )
