from __future__ import annotations

from gwspace.Waveform import waveforms

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_td
from tianqin_dc.sources.base import SourceFactory


class GCBSourceFactory(SourceFactory):
    kind = "gcb"
    family = "galactic_binary"
    default_engine = "gwspace:gcb"
    default_implementation = "gwspace_td_response"
    required_parameters = (
        "mass1",
        "mass2",
        "DL",
        "phi0",
        "f0",
        "psi",
        "iota",
        "Lambda",
        "Beta",
    )

    def prepare_parameters(
        self,
        parameters: dict[str, object],
        observation: ObservationConfig,
    ) -> dict[str, object]:
        prepared = super().prepare_parameters(parameters, observation)
        prepared.setdefault("T_obs", observation.effective_duration_s)
        return prepared

    def build_waveform(self, parameters: dict[str, object], observation: ObservationConfig):
        prepared = self.prepare_parameters(parameters, observation)
        return waveforms["gcb"](**prepared)

    def generate(self, parameters: dict[str, object], observation: ObservationConfig) -> SourceGenerationResult:
        prepared = self.prepare_parameters(parameters, observation)
        waveform = self.build_waveform(prepared, observation)
        channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
        return self.make_result(channels, prepared)
