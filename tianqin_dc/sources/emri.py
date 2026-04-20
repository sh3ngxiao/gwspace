from __future__ import annotations

from typing import Any

from gwspace.Waveform import waveforms

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_td
from tianqin_dc.sources.base import SourceFactory


class EMRISourceFactory(SourceFactory):
    kind = "emri"
    family = "emri"
    default_engine = "gwspace:emri"
    default_implementation = "gwspace_td_response"
    required_parameters = (
        "M",
        "mu",
        "a",
        "p0",
        "e0",
        "x0",
        "dist",
        "qS",
        "phiS",
        "qK",
        "phiK",
    )
    notes = (
        "EMRI generation relies on GWspace.EMRIWaveform and therefore requires FEW to be installed.",
    )

    def prepare_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, Any]:
        prepared = super().prepare_parameters(parameters, observation)
        prepared.setdefault("T_obs", observation.effective_duration_s)
        prepared.setdefault("backend", "cpu")
        return prepared

    def generate(self, parameters: dict[str, Any], observation: ObservationConfig) -> SourceGenerationResult:
        prepared = self.prepare_parameters(parameters, observation)
        waveform = waveforms["emri"](**prepared)
        channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
        return self.make_result(
            channels,
            prepared,
            metadata={"few_backend_request": prepared.get("backend", "cpu")},
        )
