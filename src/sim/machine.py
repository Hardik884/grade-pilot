"""Machine configuration and the four physical stages of the sheet path.

The chain, kept deliberately distinct (never conflate two of these):

1. **Mass balance** -> ``bw_ideal``, from the *instantaneous* manipulated variables.
2. **Wet-end first-order lag** (tau 20-60 s) on the thick-stock delivery -> ``bw_true``.
   This is the schema's headbox ground truth: post-lag, pre-delay, pre-noise.
3. **Variable transport delay**, integrated as cumulative travel so that it is exact
   when speed changes mid-transit.
4. **Scanner**: traverse average, zero-order hold at 1 Hz, plus gaussian noise.

Why the lag sits on the stock line rather than on ``bw_ideal`` as a whole: the wet
end is a mixing volume on the *thick stock* path, while machine speed changes the
grammage the instant the wire sees them. Lagging the composite ``bw_ideal`` would
give speed and stock flow the same total lag and destroy the planted causal
structure (speed -> bw at theta, stock_flow -> bw at theta + tau_wet). Both
quantities are computed and exposed, so stage 2 remains a genuine first-order lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.sim.units import m3_h_to_L_min, validate_ranges

__all__ = [
    "MachineConfig",
    "DryingModel",
    "CaliperModel",
    "dry_mass_rate_g_min",
    "basis_weight_g_m2",
    "ash_pct_from_flows",
    "required_flows",
    "first_order_step",
    "VariableTransportDelay",
    "Scanner",
]


@dataclass(frozen=True)
class MachineConfig:
    """Fixed machine geometry. Serialised verbatim into ``meta.json['machine']``."""

    trim_m: float = 6.4
    scanner_distance_m: float = 140.0
    retention: float = 0.78
    filler_cons_pct: float = 30.0

    def as_meta(self) -> dict[str, float]:
        """The exact four keys the episode schema defines for ``machine``."""
        return {
            "trim_m": float(self.trim_m),
            "scanner_distance_m": float(self.scanner_distance_m),
            "retention": float(self.retention),
            "filler_cons_pct": float(self.filler_cons_pct),
        }


@dataclass(frozen=True)
class DryingModel:
    """Steady-state moisture as a function of steam pressure and drying load.

    ``moist_ss = moist_ref - beta * (steam_p - steam_ref)
                 + alpha * (load - load_ref) / load_ref``

    with ``load = bw * speed / 1e5`` (a stand-in for the water mass rate the dryer
    section must evaporate). ``beta > 0`` is what makes moisture move *opposite* to
    steam pressure at steady state, which the plausibility gate checks.
    """

    moist_ref_pct: float = 6.0
    steam_ref_bar: float = 3.0
    beta_pct_per_bar: float = 1.6
    alpha_pct: float = 2.6
    load_ref: float = 0.75

    def load(self, bw_g_m2: ArrayLike, speed_m_min: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(bw_g_m2, dtype=float) * np.asarray(speed_m_min, dtype=float) / 1e5

    def moisture_pct(
        self, steam_p_bar: ArrayLike, bw_g_m2: ArrayLike, speed_m_min: ArrayLike
    ) -> NDArray[np.float64]:
        """Steady-state moisture (%) at the given steam pressure and load."""
        load_term = (self.load(bw_g_m2, speed_m_min) - self.load_ref) / self.load_ref
        return (
            self.moist_ref_pct
            - self.beta_pct_per_bar * (np.asarray(steam_p_bar, dtype=float) - self.steam_ref_bar)
            + self.alpha_pct * load_term
        )

    def steam_p_bar(
        self, moist_pct: ArrayLike, bw_g_m2: ArrayLike, speed_m_min: ArrayLike
    ) -> NDArray[np.float64]:
        """Steam pressure needed to hold ``moist_pct`` at the given load (inverse)."""
        load_term = (self.load(bw_g_m2, speed_m_min) - self.load_ref) / self.load_ref
        return self.steam_ref_bar + (
            self.moist_ref_pct - np.asarray(moist_pct, dtype=float) + self.alpha_pct * load_term
        ) / self.beta_pct_per_bar


@dataclass(frozen=True)
class CaliperModel:
    """Caliper from basis weight and ash: ``caliper = bw * (c0 - c1 * ash)``.

    Bulk falls as filler replaces fibre. Constants are fixed by the two worked
    grades in the episode schema: (82 g/m2, 18% ash) -> 105 um and
    (64 g/m2, 12% ash) -> 88 um.
    """

    c0_um_m2_per_g: float = 1.564
    c1_per_ash_pct: float = 0.01575

    def caliper_um(self, bw_g_m2: ArrayLike, ash_pct: ArrayLike) -> NDArray[np.float64]:
        bulk = self.c0_um_m2_per_g - self.c1_per_ash_pct * np.asarray(ash_pct, dtype=float)
        return np.asarray(bw_g_m2, dtype=float) * bulk


# --------------------------------------------------------------------------------------
# Mass balance
# --------------------------------------------------------------------------------------


def dry_mass_rate_g_min(
    stock_flow_m3_h: ArrayLike,
    stock_cons_pct: ArrayLike,
    filler_flow_L_min: ArrayLike,
    filler_cons_pct: float,
) -> NDArray[np.float64]:
    """Dry solids delivered to the wire, g/min, before retention.

    ``stock_flow_m3_h`` is converted to L/min here and nowhere else. The factor 10
    converts L/min x % to g/min at ~1 kg/L slurry density.
    """
    stock_L_min = m3_h_to_L_min(stock_flow_m3_h)
    fibre = np.asarray(stock_L_min, dtype=float) * np.asarray(stock_cons_pct, dtype=float) * 10.0
    filler = np.asarray(filler_flow_L_min, dtype=float) * float(filler_cons_pct) * 10.0
    return fibre + filler


def basis_weight_g_m2(
    stock_flow_m3_h: ArrayLike,
    stock_cons_pct: ArrayLike,
    filler_flow_L_min: ArrayLike,
    speed_m_min: ArrayLike,
    *,
    trim_m: float,
    retention: float,
    filler_cons_pct: float,
    validate: bool = False,
) -> NDArray[np.float64]:
    """Basis weight from the mass balance, g/m2.

    Arithmetic check (the one the whole project hangs on): trim 6.4 m, 1000 m/min,
    946 m3/h of 3.5% stock, 404 L/min of 30% filler, retention 0.78 -> 82.0 g/m2.
    """
    if validate:
        validate_ranges(
            {
                "stock_flow": stock_flow_m3_h,
                "stock_cons": stock_cons_pct,
                "filler_flow": filler_flow_L_min,
                "speed": speed_m_min,
                "trim": trim_m,
            },
            context="basis_weight_g_m2 inputs",
        )
    delivered = dry_mass_rate_g_min(
        stock_flow_m3_h, stock_cons_pct, filler_flow_L_min, filler_cons_pct
    )
    area_rate = np.asarray(speed_m_min, dtype=float) * float(trim_m)
    return delivered / area_rate * float(retention)


def ash_pct_from_flows(
    stock_flow_m3_h: ArrayLike,
    stock_cons_pct: ArrayLike,
    filler_flow_L_min: ArrayLike,
    filler_cons_pct: float,
) -> NDArray[np.float64]:
    """Sheet ash content, %. Retention applies equally to fibre and filler, so the
    retained ash fraction equals the delivered filler mass fraction."""
    stock_L_min = m3_h_to_L_min(stock_flow_m3_h)
    fibre = np.asarray(stock_L_min, dtype=float) * np.asarray(stock_cons_pct, dtype=float) * 10.0
    filler = np.asarray(filler_flow_L_min, dtype=float) * float(filler_cons_pct) * 10.0
    total = fibre + filler
    return 100.0 * filler / total


def required_flows(
    bw_g_m2: float,
    ash_pct: float,
    speed_m_min: float,
    stock_cons_pct: float,
    *,
    trim_m: float,
    retention: float,
    filler_cons_pct: float,
) -> tuple[float, float]:
    """Inverse mass balance -> ``(stock_flow_m3_h, filler_flow_L_min)``.

    Used by the grade catalogue's operating-point solver and by the controller's
    feedforward. Exact inverse of :func:`basis_weight_g_m2` / :func:`ash_pct_from_flows`.
    """
    retained_g_min = bw_g_m2 * speed_m_min * trim_m
    delivered_g_min = retained_g_min / retention
    ash_frac = ash_pct / 100.0
    filler_dry_g_min = ash_frac * delivered_g_min
    fibre_dry_g_min = (1.0 - ash_frac) * delivered_g_min
    filler_flow_L_min = filler_dry_g_min / (filler_cons_pct * 10.0)
    stock_flow_L_min = fibre_dry_g_min / (stock_cons_pct * 10.0)
    stock_flow_m3_h = stock_flow_L_min * 60.0 / 1000.0
    return float(stock_flow_m3_h), float(filler_flow_L_min)


# --------------------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------------------


def first_order_step(y_prev: float, u: float, tau_sec: float, dt_sec: float) -> float:
    """One exact step of ``tau * dy/dt + y = u`` under a zero-order-hold input."""
    if tau_sec <= 0.0:
        return float(u)
    a = float(np.exp(-dt_sec / tau_sec))
    return float(y_prev * a + u * (1.0 - a))


class VariableTransportDelay:
    """Speed-dependent dead time by cumulative-travel inversion.

    Integrates ``D(t) = int speed/60 dt`` (metres of sheet made). A parcel formed at
    ``t_f`` reaches the scanner when ``D(t) - D(t_f) = distance_m``. Inverting by
    interpolation on ``D`` makes the delay exact even when speed changes while the
    parcel is in transit -- which is precisely what happens during a grade change.

    Never a fixed lag and never a fixed-length deque: a deque of N samples is a
    constant *time* delay, which is wrong the moment speed moves.
    """

    def __init__(
        self,
        distance_m: float,
        *,
        capacity: int,
        n_signals: int,
        t0_sec: float,
        dt_sec: float,
        initial_speed_m_min: float,
        initial_values: ArrayLike,
        prefill: int = 512,
    ) -> None:
        self.distance_m = float(distance_m)
        self._n = int(n_signals)
        total = int(capacity) + int(prefill) + 8
        self._t = np.empty(total, dtype=float)
        self._d = np.empty(total, dtype=float)
        self._v = np.empty((total, self._n), dtype=float)
        # Pre-fill with steady operation before the episode window so that the very
        # first sample already has a full sheet length of history behind it.
        init = np.asarray(initial_values, dtype=float).reshape(self._n)
        back = np.arange(prefill, 0, -1) * dt_sec
        self._t[:prefill] = t0_sec - back
        self._d[:prefill] = -back * initial_speed_m_min / 60.0
        self._v[:prefill] = init
        self._k = prefill
        self._d_last = 0.0
        self._read = 1
        self.theta_sec = 60.0 * self.distance_m / initial_speed_m_min

    def step(
        self, t_sec: float, speed_m_min: float, dt_sec: float, values: ArrayLike
    ) -> NDArray[np.float64]:
        """Push the freshly formed sheet and read what arrives at the sensor now.

        Interpolation happens in the travel coordinate ``D``, which is the natural
        one: the sensor sees the sheet at a fixed distance behind the headbox, not
        at a fixed time behind it.
        """
        self._d_last += speed_m_min / 60.0 * dt_sec
        k = self._k
        self._t[k] = t_sec
        self._d[k] = self._d_last
        self._v[k] = np.asarray(values, dtype=float).reshape(self._n)
        self._k = k + 1

        target_d = self._d_last - self.distance_m
        # D is strictly increasing and target_d advances monotonically, so the read
        # pointer only ever moves forward: O(1) amortised, no repeated binary search.
        i = self._read
        while i < self._k - 1 and self._d[i] < target_d:
            i += 1
        self._read = i
        d0, d1 = self._d[i - 1], self._d[i]
        w = 0.0 if d1 <= d0 else (target_d - d0) / (d1 - d0)
        t_formed = self._t[i - 1] + w * (self._t[i] - self._t[i - 1])
        self.theta_sec = float(t_sec - t_formed)
        return self._v[i - 1] + w * (self._v[i] - self._v[i - 1])


class Scanner:
    """QCS traversing sensor: scan-average, zero-order hold, noise.

    A reading is the mean of the arriving sheet over one traverse (20-45 s), held
    until the next traverse completes, so noise enters once per scan rather than
    once per second -- a held reading is one measurement repeated, not a fresh one.

    Two noise terms, because they average differently. ``sensor_noise_std`` is the
    per-point sensor noise from the process reference (0.3-0.8 g/m2 on basis
    weight); the traverse averages hundreds of points, so it reaches the reported
    value divided by sqrt(N). ``scan_noise_std`` is the residual scan-to-scan
    variation that does *not* average out. Drawing one gaussian per scan with
    ``sqrt(sensor^2/N + scan^2)`` is distributionally identical to averaging N
    per-point draws and far cheaper.

    Treating the reference's per-point figure as if it were the noise on the scan
    average would put a lightweight grade off-spec on sensor noise alone: at
    45 g/m2 the 2.5% band is +/-1.1 g/m2, which is barely two sigma.
    """

    def __init__(
        self,
        scan_period_sec: float,
        sensor_noise_std: ArrayLike,
        scan_noise_std: ArrayLike,
        rng: np.random.Generator,
        *,
        dt_sec: float,
        initial_values: ArrayLike,
    ) -> None:
        self.scan_period_sec = float(scan_period_sec)
        sensor = np.asarray(sensor_noise_std, dtype=float)
        scan = np.asarray(scan_noise_std, dtype=float)
        n_points = max(self.scan_period_sec / dt_sec, 1.0)
        self._noise = np.sqrt(sensor**2 / n_points + scan**2)
        self._rng = rng
        self._n = self._noise.size
        self._held = np.asarray(initial_values, dtype=float).reshape(self._n).copy()
        self._acc = np.zeros(self._n, dtype=float)
        self._acc_t = 0.0

    def step(self, dt_sec: float, values: ArrayLike) -> NDArray[np.float64]:
        self._acc += np.asarray(values, dtype=float).reshape(self._n) * dt_sec
        self._acc_t += dt_sec
        if self._acc_t >= self.scan_period_sec:
            mean = self._acc / self._acc_t
            self._held = mean + self._rng.normal(0.0, self._noise)
            self._acc[:] = 0.0
            self._acc_t = 0.0
        return self._held.copy()
