from __future__ import annotations

from typing import Any

import numpy as np
from gwspace.constants import YRSID_SI
from gwspace.Waveform import waveforms

from tianqin_dc.config import ObservationConfig
from tianqin_dc.models import SourceGenerationResult
from tianqin_dc.response import generate_tdi_channels_td
from tianqin_dc.sources.base import SourceFactory


class InterpolatedEMRIWaveform:
    """EMRI waveform adapter that supports arbitrary delayed time samples.

    GWspace's EMRI waveform generator produces a uniformly sampled source
    waveform. The TDI time-domain response evaluates h(t) at Doppler- and
    light-travel-time-delayed samples, so this adapter caches the uniform
    waveform once and interpolates those delayed queries.
    """

    def __init__(
        self,
        waveform: Any,
        *,
        sample_spacing_s: float,
        duration_s: float,
        time_margin_s: float = 4096.0,
    ) -> None:
        if sample_spacing_s <= 0.0:
            raise ValueError("sample_spacing_s must be positive for EMRI interpolation.")
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive for EMRI interpolation.")

        self._waveform = waveform
        self._sample_spacing_s = float(sample_spacing_s)
        self._cache_duration_s = float(duration_s) + max(float(time_margin_s), 0.0)
        self._time_s: np.ndarray | None = None
        self._hp: np.ndarray | None = None
        self._hc: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._waveform, name)

    def _ensure_cache(self) -> None:
        if self._time_s is not None:
            return

        hp, hc = self._waveform.get_hphc_source(
            self._cache_duration_s / YRSID_SI,
            self._sample_spacing_s,
        )
        hp = np.asarray(hp, dtype=np.float64)
        hc = np.asarray(hc, dtype=np.float64)
        if hp.shape != hc.shape:
            raise ValueError(f"EMRI hplus/hcross shapes differ: {hp.shape} != {hc.shape}.")
        if hp.ndim != 1:
            hp = np.ravel(hp)
            hc = np.ravel(hc)
        if hp.size == 0:
            raise ValueError("EMRI waveform generator returned an empty source waveform.")

        self._time_s = np.arange(hp.size, dtype=np.float64) * self._sample_spacing_s
        self._hp = hp
        self._hc = hc

    def get_hphc(self, tf: np.ndarray, eps: float = 1e-5, modes: Any = None) -> tuple[np.ndarray, np.ndarray]:
        if eps != 1e-5 or modes is not None:
            raise ValueError("InterpolatedEMRIWaveform only supports the cached default FEW mode selection.")
        self._ensure_cache()
        if self._time_s is None or self._hp is None or self._hc is None:
            raise RuntimeError("Internal error: EMRI interpolation cache was not initialized.")

        query = np.asarray(tf, dtype=np.float64)
        flat_query = query.reshape(-1)
        hp_out = np.zeros(flat_query.shape, dtype=np.float64)
        hc_out = np.zeros(flat_query.shape, dtype=np.float64)

        valid = np.isfinite(flat_query) & (flat_query >= self._time_s[0]) & (flat_query <= self._time_s[-1])
        if np.any(valid):
            hp_out[valid] = np.interp(flat_query[valid], self._time_s, self._hp)
            hc_out[valid] = np.interp(flat_query[valid], self._time_s, self._hc)

        return hp_out.reshape(query.shape), hc_out.reshape(query.shape)


def build_interpolated_emri_waveform(
    prepared_parameters: dict[str, Any],
    observation: ObservationConfig,
) -> InterpolatedEMRIWaveform:
    waveform = waveforms["emri"](**prepared_parameters)
    return InterpolatedEMRIWaveform(
        waveform,
        sample_spacing_s=observation.sample_spacing_s,
        duration_s=observation.effective_duration_s,
    )


class EMRISourceFactory(SourceFactory):
    kind = "emri"
    family = "emri"
    default_engine = "gwspace:emri"
    default_implementation = "gwspace_td_response_interpolated"
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
        waveform = build_interpolated_emri_waveform(prepared, observation)
        channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
        return self.make_result(
            channels,
            prepared,
            notes=[
                "EMRI source waveform is cached on the observation grid and interpolated at delayed TDI sample times.",
            ],
            metadata={
                "few_backend_request": prepared.get("backend", "cpu"),
                "time_interpolation": "linear",
            },
        )
