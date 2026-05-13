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
        time_shift_s: float = 0.0,
        taper_duration_s: float = 86400.0,
    ) -> None:
        if sample_spacing_s <= 0.0:
            raise ValueError("sample_spacing_s must be positive for EMRI interpolation.")
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive for EMRI interpolation.")

        self._waveform = waveform
        self._sample_spacing_s = float(sample_spacing_s)
        self._source_duration_s = float(duration_s)
        self._cache_duration_s = self._source_duration_s + max(float(time_margin_s), 0.0)
        self._time_shift_s = float(time_shift_s)
        self._taper_duration_s = max(float(taper_duration_s), 0.0)
        self._time_s: np.ndarray | None = None
        self._hp: np.ndarray | None = None
        self._hc: np.ndarray | None = None
        self._trajectory_end: dict[str, Any] | None = None

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

        source_stop = min(hp.size, int(np.floor(self._source_duration_s / self._sample_spacing_s)) + 1)
        self._apply_edge_taper(hp, hc, source_stop=source_stop)
        hp[source_stop:] = 0.0
        hc[source_stop:] = 0.0
        self._time_s = np.arange(hp.size, dtype=np.float64) * self._sample_spacing_s
        self._hp = hp
        self._hc = hc

    @property
    def time_shift_s(self) -> float:
        return self._time_shift_s

    def set_time_shift_s(self, time_shift_s: float) -> None:
        self._time_shift_s = float(time_shift_s)

    @property
    def taper_duration_s(self) -> float:
        return self._taper_duration_s

    def _apply_edge_taper(self, hp: np.ndarray, hc: np.ndarray, *, source_stop: int) -> None:
        if source_stop <= 0:
            return
        taper_samples = int(round(self._taper_duration_s / self._sample_spacing_s))
        taper_samples = min(taper_samples, source_stop // 2)
        if taper_samples <= 1:
            return

        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_samples)))
        hp[:taper_samples] *= ramp
        hc[:taper_samples] *= ramp
        hp[source_stop - taper_samples : source_stop] *= ramp[::-1]
        hc[source_stop - taper_samples : source_stop] *= ramp[::-1]

    def source_activity(
        self,
        *,
        threshold_fraction: float,
        threshold_abs: float = 0.0,
    ) -> dict[str, Any]:
        self._ensure_cache()
        if self._time_s is None or self._hp is None or self._hc is None:
            raise RuntimeError("Internal error: EMRI interpolation cache was not initialized.")

        amplitude = np.hypot(self._hp, self._hc)
        peak_index = int(np.argmax(amplitude))
        peak_value = float(amplitude[peak_index])
        threshold = max(float(threshold_abs), peak_value * float(threshold_fraction))
        active = np.flatnonzero(amplitude > threshold)
        if active.size:
            active_start_index = int(active[0])
            active_end_index = int(active[-1])
        else:
            active_start_index = peak_index
            active_end_index = peak_index

        return {
            "threshold": threshold,
            "threshold_fraction": float(threshold_fraction),
            "threshold_abs": float(threshold_abs),
            "original_peak_time_s": float(self._time_s[peak_index]),
            "original_peak_index": peak_index,
            "original_peak_value": peak_value,
            "original_active_start_time_s": float(self._time_s[active_start_index]),
            "original_active_start_index": active_start_index,
            "original_active_end_time_s": float(self._time_s[active_end_index]),
            "original_active_end_index": active_end_index,
            "original_active_samples": int(active_end_index - active_start_index + 1),
        }

    def trajectory_end(self) -> dict[str, Any]:
        """Return the FEW trajectory end time used as the physical end anchor."""

        if self._trajectory_end is not None:
            return dict(self._trajectory_end)

        wave_func = getattr(self._waveform, "wave_func", None)
        waveform_generator = getattr(wave_func, "waveform_generator", None)
        inspiral_generator = getattr(waveform_generator, "inspiral_generator", None)
        if inspiral_generator is None:
            self._trajectory_end = {
                "trajectory_end_time_s": self._source_duration_s,
                "trajectory_end_index": int(round(self._source_duration_s / self._sample_spacing_s)),
                "trajectory_end_reason": "unavailable",
                "trajectory_background": getattr(self._waveform, "_few_background", None),
            }
            return dict(self._trajectory_end)

        background = getattr(self._waveform, "_few_background", None)
        if background is None:
            background = getattr(waveform_generator, "background", None)
        if background == "Schwarzschild":
            a_eff = 0.0
            x_eff = 1.0
        else:
            a_eff = float(self._waveform.a)
            x_eff = float(self._waveform.x0)

        out = inspiral_generator(
            float(self._waveform.M),
            float(self._waveform.mu),
            a_eff,
            float(self._waveform.p0),
            float(self._waveform.e0),
            x_eff,
            T=self._source_duration_s / YRSID_SI,
            dt=self._sample_spacing_s,
            DENSE_STEPPING=False,
            buffer_length=1000,
        )
        trajectory_time = np.asarray(out[0], dtype=np.float64)
        if trajectory_time.size == 0:
            end_time_s = self._source_duration_s
        else:
            end_time_s = float(trajectory_time[-1])
        end_time_s = float(np.clip(end_time_s, 0.0, self._source_duration_s))
        tolerance_s = max(10.0 * self._sample_spacing_s, 1.0e-6 * self._source_duration_s)
        if end_time_s < self._source_duration_s - tolerance_s:
            reason = "trajectory_stopped_before_requested_duration"
        else:
            reason = "requested_duration_reached"

        self._trajectory_end = {
            "trajectory_end_time_s": end_time_s,
            "trajectory_end_index": int(round(end_time_s / self._sample_spacing_s)),
            "trajectory_end_reason": reason,
            "trajectory_background": background,
            "trajectory_points": int(trajectory_time.size),
        }
        if len(out) >= 4 and trajectory_time.size:
            self._trajectory_end.update(
                {
                    "trajectory_p_end": float(np.asarray(out[1], dtype=np.float64)[-1]),
                    "trajectory_e_end": float(np.asarray(out[2], dtype=np.float64)[-1]),
                    "trajectory_x_end": float(np.asarray(out[3], dtype=np.float64)[-1]),
                }
            )
        return dict(self._trajectory_end)

    def get_hphc(self, tf: np.ndarray, eps: float = 1e-5, modes: Any = None) -> tuple[np.ndarray, np.ndarray]:
        if eps != 1e-5 or modes is not None:
            raise ValueError("InterpolatedEMRIWaveform only supports the cached default FEW mode selection.")
        self._ensure_cache()
        if self._time_s is None or self._hp is None or self._hc is None:
            raise RuntimeError("Internal error: EMRI interpolation cache was not initialized.")

        query = np.asarray(tf, dtype=np.float64) - self._time_shift_s
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
                "time_taper_duration_s": waveform.taper_duration_s,
            },
        )
