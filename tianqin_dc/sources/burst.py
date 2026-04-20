from __future__ import annotations

from typing import Any

import numpy as np

from gwspace.Waveform import BurstWaveform, p0_plus_cross

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_td
from tianqin_dc.sources.base import SourceFactory


class DirectionalBurstWaveform(BurstWaveform):
    """Adapter that adds sky location and polarization to GWspace.BurstWaveform.

    GWspace's native BurstWaveform provides h_plus/h_cross only. The TianQin
    response helpers also need source direction and polarization tensors, so this
    adapter supplies the same geometry interface used by GWspace.BasicWaveform.
    """

    def __init__(
        self,
        amp: float,
        tau: float,
        fc: float,
        tc: float,
        Lambda: float,
        Beta: float,
        psi: float = 0.0,
    ) -> None:
        super().__init__(amp=amp, tau=tau, fc=fc, tc=tc)
        self.Lambda = Lambda
        self.Beta = Beta
        self.psi = psi

    @property
    def vec_k(self) -> np.ndarray:
        return np.array(
            [
                -np.cos(self.Beta) * np.cos(self.Lambda),
                -np.cos(self.Beta) * np.sin(self.Lambda),
                -np.sin(self.Beta),
            ],
            dtype=np.float64,
        )

    def polarization(self) -> tuple[np.ndarray, np.ndarray]:
        p0_plus, p0_cross = p0_plus_cross(self.Lambda, self.Beta)
        p_plus = p0_plus * np.cos(2 * self.psi) + p0_cross * np.sin(2 * self.psi)
        p_cross = -p0_plus * np.sin(2 * self.psi) + p0_cross * np.cos(2 * self.psi)
        return p_plus, p_cross


class BurstSourceFactory(SourceFactory):
    kind = "burst"
    family = "burst"
    default_engine = "gwspace:burst"
    default_implementation = "gwspace_td_response"
    required_parameters = (
        "amp",
        "tau",
        "fc",
        "Lambda",
        "Beta",
    )
    notes = (
        "This source uses GWspace.BurstWaveform for h_plus/h_cross and a local adapter for sky geometry.",
    )

    def prepare_parameters(
        self,
        parameters: dict[str, Any],
        observation: ObservationConfig,
    ) -> dict[str, Any]:
        prepared = super().prepare_parameters(parameters, observation)
        prepared.setdefault("tc", 0.5 * observation.effective_duration_s)
        prepared.setdefault("psi", 0.0)
        return prepared

    def build_waveform(self, parameters: dict[str, Any], observation: ObservationConfig) -> Any:
        prepared = self.prepare_parameters(parameters, observation)
        return DirectionalBurstWaveform(
            amp=float(prepared["amp"]),
            tau=float(prepared["tau"]),
            fc=float(prepared["fc"]),
            tc=float(prepared["tc"]),
            Lambda=float(prepared["Lambda"]),
            Beta=float(prepared["Beta"]),
            psi=float(prepared["psi"]),
        )

    def generate(self, parameters: dict[str, Any], observation: ObservationConfig) -> SourceGenerationResult:
        prepared = self.prepare_parameters(parameters, observation)
        waveform = self.build_waveform(prepared, observation)
        channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
        return self.make_result(
            channels,
            prepared,
            metadata={"geometry_adapter": "DirectionalBurstWaveform"},
        )
