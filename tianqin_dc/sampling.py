from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from tianqin_dc.config import SamplerConfig, SourcePopulationConfig


def sample_value(spec: SamplerConfig, rng: np.random.Generator) -> Any:
    distribution = spec.distribution.lower()

    if distribution == "fixed":
        return deepcopy(spec.value)

    if distribution == "uniform":
        if spec.low is None or spec.high is None:
            raise ValueError("Uniform sampler requires 'low' and 'high'.")
        return float(rng.uniform(spec.low, spec.high))

    if distribution == "isotropic_polar":
        low = 0.0 if spec.low is None else float(spec.low)
        high = float(np.pi) if spec.high is None else float(spec.high)
        if not (0.0 <= low <= np.pi and 0.0 <= high <= np.pi):
            raise ValueError("Isotropic polar sampler bounds must lie within [0, pi].")
        if high < low:
            raise ValueError("Isotropic polar sampler requires high >= low.")
        cos_low = float(np.cos(low))
        cos_high = float(np.cos(high))
        return float(np.arccos(rng.uniform(cos_high, cos_low)))

    if distribution == "isotropic_latitude":
        low = -float(np.pi) / 2.0 if spec.low is None else float(spec.low)
        high = float(np.pi) / 2.0 if spec.high is None else float(spec.high)
        if not (-float(np.pi) / 2.0 <= low <= float(np.pi) / 2.0 and -float(np.pi) / 2.0 <= high <= float(np.pi) / 2.0):
            raise ValueError("Isotropic latitude sampler bounds must lie within [-pi/2, pi/2].")
        if high < low:
            raise ValueError("Isotropic latitude sampler requires high >= low.")
        sin_low = float(np.sin(low))
        sin_high = float(np.sin(high))
        return float(np.arcsin(rng.uniform(sin_low, sin_high)))

    if distribution == "loguniform":
        if spec.low is None or spec.high is None:
            raise ValueError("Loguniform sampler requires 'low' and 'high'.")
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError("Loguniform sampler bounds must be positive.")
        return float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))

    if distribution == "normal":
        if spec.mean is None or spec.std is None:
            raise ValueError("Normal sampler requires 'mean' and 'std'.")
        return float(rng.normal(spec.mean, spec.std))

    if distribution == "choice":
        if not spec.choices:
            raise ValueError("Choice sampler requires a non-empty 'choices' list.")
        value = rng.choice(spec.choices)
        return value.item() if hasattr(value, "item") else value

    raise ValueError(f"Unsupported sampler distribution '{spec.distribution}'.")


def sample_population_parameters(
    population: SourcePopulationConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if population.parameters is not None:
        return [deepcopy(item) for item in population.parameters]

    sampled: list[dict[str, Any]] = []
    for _ in range(population.count):
        parameters = deepcopy(population.fixed)
        for name, spec in population.sampler.items():
            parameters[name] = sample_value(spec, rng)
        sampled.append(parameters)
    return sampled
