#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package init for gwspace.

Provides optional libFastGB binding and exports GPU-aware response helpers.
"""

# Optional C extension: libFastGB
try:
    from . import libFastGB as _libfastgb  # type: ignore
    libFastGB = _libfastgb
except Exception:  # pragma: no cover
    try:
        import libFastGB as _libfastgb  # type: ignore
        libFastGB = _libfastgb
    except Exception:
        libFastGB = None  # placeholder when extension is not built

# GPU/CPU selectable response helpers
from gwspace.response_gpu import (
    get_AET_td_backend,
    get_XYZ_td_backend,
    get_y_slr_td_backend,
)

__all__ = [
    "libFastGB",
    "get_AET_td_backend",
    "get_XYZ_td_backend",
    "get_y_slr_td_backend",
]
