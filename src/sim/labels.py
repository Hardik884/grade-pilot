"""Episode labels, exactly as the episode schema defines them.

All five are computed from measured ``bw`` against the active ``bw_sp`` -- never
from ``bw_true``. A label the downstream modules could not have derived from what
the mill actually sees would be a leak dressed up as a target.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

__all__ = ["SPEC_BAND", "STABILISATION_WINDOW_SEC", "Labels", "compute_labels"]

#: Off-spec threshold: abs(bw - bw_sp) / bw_sp > 0.025.
SPEC_BAND: float = 0.025

#: A stabilisation instant must be followed by at least this much continuous
#: in-band running to count.
STABILISATION_WINDOW_SEC: float = 120.0


class Labels(TypedDict):
    off_spec: bool
    max_dev_pct: float
    breach_t_sec: float | None
    stabilisation_t_sec: float | None
    broke_tonnes: float


def compute_labels(df: pd.DataFrame, *, trim_m: float, dt_sec: float = 1.0) -> Labels:
    """Compute the five schema labels over the post-trigger part of the episode."""
    post = df.loc[df.index > 0.0]
    t = post.index.to_numpy(dtype=float)
    dev = np.abs(post["bw"].to_numpy(dtype=float) - post["bw_sp"].to_numpy(dtype=float)) / post[
        "bw_sp"
    ].to_numpy(dtype=float)
    out_of_band = dev > SPEC_BAND

    off_spec = bool(out_of_band.any())
    max_dev_pct = float(dev.max() * 100.0) if dev.size else 0.0
    breach_t_sec = float(t[out_of_band][0]) if off_spec else None

    stabilisation_t_sec = _stabilisation(t, out_of_band)

    # broke_tonnes: bw [g/m2] * speed [m/min] * trim [m] = g/min, times dt in
    # minutes, over off-spec samples only, converted to tonnes.
    bw = post["bw"].to_numpy(dtype=float)
    speed = post["speed"].to_numpy(dtype=float)
    dt_min = dt_sec / 60.0
    broke_g = float(np.sum(bw[out_of_band] * speed[out_of_band] * trim_m * dt_min))
    broke_tonnes = broke_g / 1e6

    return Labels(
        off_spec=off_spec,
        max_dev_pct=round(max_dev_pct, 3),
        breach_t_sec=breach_t_sec,
        stabilisation_t_sec=stabilisation_t_sec,
        broke_tonnes=round(broke_tonnes, 4),
    )


def _stabilisation(t: np.ndarray, out_of_band: np.ndarray) -> float | None:
    """First instant after which ``bw`` *remains* in band, requiring a continuous
    window of at least :data:`STABILISATION_WINDOW_SEC`.

    "Remains inside the band" is taken literally: the qualifying instant opens the
    final in-band run of the episode. The 120 s requirement is what stops a brief
    pass through the band on the way to the next breach from being reported as
    stabilisation.
    """
    if t.size == 0:
        return None
    breaches = np.flatnonzero(out_of_band)
    start_idx = 0 if breaches.size == 0 else int(breaches[-1]) + 1
    if start_idx >= t.size:
        return None
    t0 = float(t[start_idx])
    if float(t[-1]) - t0 < STABILISATION_WINDOW_SEC:
        return None
    return t0
