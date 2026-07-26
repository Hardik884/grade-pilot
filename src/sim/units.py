"""Canonical units, physical ranges and unit conversions.

Single source of truth for the range table in the ``papermaking-process`` skill.
Every module boundary in the simulator validates against :data:`RANGES`; a value
outside its range is a bug, not a disturbance.

Unit discipline: any name that could be ambiguous carries its unit as a suffix
(``theta_sec``, ``speed_m_min``, ``stock_flow_m3_h``). Conversions happen here and
at module boundaries, never mid-calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RANGES",
    "VarRange",
    "BW_RATE_LIMIT_G_M2_MIN",
    "SPEED_RATE_LIMIT_M_MIN_MIN",
    "m3_h_to_L_min",
    "L_min_to_m3_h",
    "transport_delay_sec",
    "range_violations",
    "validate_ranges",
    "RangeViolation",
]


@dataclass(frozen=True)
class VarRange:
    """Inclusive physical range for one process variable."""

    lo: float
    hi: float
    unit: str

    def violations(self, values: ArrayLike) -> tuple[float, float, int]:
        """Return ``(min, max, n_outside)`` for ``values`` against this range."""
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (float("nan"), float("nan"), int(arr.size))
        outside = int(np.count_nonzero((finite < self.lo) | (finite > self.hi)))
        outside += int(arr.size - finite.size)
        return (float(finite.min()), float(finite.max()), outside)


#: Canonical variable ranges, verbatim from the ``papermaking-process`` skill.
RANGES: dict[str, VarRange] = {
    "bw": VarRange(40.0, 300.0, "g/m2"),
    "moist": VarRange(4.0, 9.0, "%"),
    "ash": VarRange(5.0, 30.0, "%"),
    "caliper": VarRange(60.0, 400.0, "um"),
    "stock_flow": VarRange(500.0, 4000.0, "m3/h"),
    "stock_cons": VarRange(2.5, 4.5, "%"),
    "filler_flow": VarRange(0.0, 800.0, "L/min"),
    "steam_p": VarRange(1.0, 6.0, "bar"),
    "speed": VarRange(400.0, 1800.0, "m/min"),
    "trim": VarRange(3.0, 10.0, "m"),
}


#: Maximum physically plausible rates of change, from the ``papermaking-process``
#: plausibility gate. Canonical here rather than in the gate itself: the simulator
#: has to *plan* transitions that respect them, and the gate then checks the result
#: against the same numbers.
BW_RATE_LIMIT_G_M2_MIN: float = 15.0
SPEED_RATE_LIMIT_M_MIN_MIN: float = 100.0


class RangeViolation(ValueError):
    """Raised when a physical quantity leaves its canonical range."""


def m3_h_to_L_min(flow_m3_h: ArrayLike) -> NDArray[np.float64] | float:
    """Convert a volumetric flow from m3/h to L/min.

    The thick stock flow is tabled in m3/h and **must** pass through this before
    entering the mass balance. Omitting it under-predicts basis weight ~17x.
    """
    return np.asarray(flow_m3_h, dtype=float) * 1000.0 / 60.0


def L_min_to_m3_h(flow_L_min: ArrayLike) -> NDArray[np.float64] | float:
    """Convert a volumetric flow from L/min to m3/h."""
    return np.asarray(flow_L_min, dtype=float) * 60.0 / 1000.0


def transport_delay_sec(distance_m: float, speed_m_min: ArrayLike) -> NDArray[np.float64] | float:
    """Pure transport delay over ``distance_m`` at ``speed_m_min``.

    This is the *steady-speed* closed form. During a transition the speed changes
    while the parcel is in transit, so the simulator integrates cumulative travel
    instead; see :class:`src.sim.machine.VariableTransportDelay`. Use this only for
    nominal sizing and sanity checks.
    """
    speed = np.asarray(speed_m_min, dtype=float)
    return 60.0 * distance_m / speed


def range_violations(values: Mapping[str, ArrayLike], *, strict: bool = True) -> list[str]:
    """Return a human-readable list of range violations.

    Unknown keys are ignored when ``strict`` is False, otherwise they are reported.
    """
    problems: list[str] = []
    for name, value in values.items():
        rng = RANGES.get(name)
        if rng is None:
            if strict:
                problems.append(f"{name}: no canonical range defined")
            continue
        lo_seen, hi_seen, n_outside = rng.violations(value)
        if n_outside:
            problems.append(
                f"{name}: {n_outside} sample(s) outside [{rng.lo}, {rng.hi}] {rng.unit} "
                f"(observed min {lo_seen:.4g}, max {hi_seen:.4g})"
            )
    return problems


def validate_ranges(values: Mapping[str, ArrayLike], *, context: str = "") -> None:
    """Raise :class:`RangeViolation` if any value leaves its canonical range.

    Called at module boundaries: the mass balance validates its inputs, the episode
    builder validates the finished frame.
    """
    problems = range_violations(values, strict=False)
    if problems:
        where = f" [{context}]" if context else ""
        raise RangeViolation(f"physical range violation{where}: " + "; ".join(problems))
