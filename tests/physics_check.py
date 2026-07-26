"""Physical plausibility gate.

``python -m tests.physics_check data/episodes``

Implements the six rejection rules from the ``papermaking-process`` skill. Exits
non-zero naming the rule and the offending episodes.

Two rules are not per-sample tests and are implemented accordingly:

* **Rate limits** use a rolling 60 s window, never sample-to-sample differencing.
  The scanner is a zero-order hold, so a fresh scan landing after 30 s of silence
  is a legitimate step; differencing per sample would score it as an instantaneous
  jump and spuriously breach the 15 g/m2/min limit on perfectly good data.
* **Moisture vs steam pressure** is a steady-state *gain sign*, which no single
  episode can identify: in closed loop steam is whatever holds moisture at its
  target, so within one steady window the two are related by the controller, not
  by the process. Pooling the steady phase across the dataset and regressing
  moisture on steam while controlling for drying load recovers the process gain,
  and its sign is the thing worth checking.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.sim.machine import basis_weight_g_m2
from src.sim.units import (
    BW_RATE_LIMIT_G_M2_MIN,
    RANGES,
    SPEED_RATE_LIMIT_M_MIN_MIN,
)

__all__ = [
    "RULES",
    "GateResult",
    "check_dataset",
    "main",
    "mass_balance_lag_sec",
    "response_travel",
]

RULES: tuple[str, ...] = (
    "range",
    "bw_rate",
    "speed_rate",
    "bw_zero_lag",
    "ash_mass",
    "moisture_steam_sign",
)

RATE_WINDOW_SEC = 60.0
#: Smallest lag, in seconds, at which basis weight may respond to a manipulated
#: variable. Anything below this is a zero-lag response.
MIN_RESPONSE_LAG_SEC = 3.0
MAX_RESPONSE_LAG_SEC = 240.0
#: An instantaneous mass-balance fit tighter than this cannot happen on measured
#: data - scanner noise alone floors the residual well above it. Below it, basis
#: weight was computed algebraically from the manipulated variables.
ALGEBRAIC_FIT_RMS_G_M2 = 0.05
#: Fractional RMS improvement of the best lag over zero lag required before the
#: lag is considered identifiable at all.
LAG_IDENTIFIABLE_IMPROVEMENT = 0.20

#: Columns whose ranges are checked. ``bw_true`` is included: it is simulator
#: ground truth, but an out-of-range ground truth is a simulator bug and this is
#: the simulator's own gate.
_RANGE_COLUMNS = (
    "bw",
    "moist",
    "ash",
    "caliper",
    "stock_flow",
    "stock_cons",
    "filler_flow",
    "steam_p",
    "speed",
)


@dataclass
class GateResult:
    """Per-rule failures, keyed by rule name -> list of human-readable failures."""

    failures: dict[str, list[str]] = field(default_factory=lambda: {r: [] for r in RULES})
    n_episodes: int = 0

    @property
    def passed(self) -> bool:
        return not any(self.failures[r] for r in RULES)

    def add(self, rule: str, message: str) -> None:
        self.failures[rule].append(message)


# --------------------------------------------------------------------------------------


def rolling_rate_per_min(series: pd.Series, window_sec: float = RATE_WINDOW_SEC) -> np.ndarray:
    """Change over a trailing ``window_sec`` window, expressed per minute.

    The index is ``t_sec`` at 1 Hz, so the window is a fixed number of samples.
    Returns an array the length of the series with NaN before the window fills.
    """
    values = series.to_numpy(dtype=float)
    t = series.index.to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    k = max(int(round(window_sec / dt)), 1)
    if values.size <= k:
        return np.full(values.size, np.nan)
    out = np.full(values.size, np.nan)
    span_min = (k * dt) / 60.0
    out[k:] = (values[k:] - values[:-k]) / span_min
    return out


def response_travel(series: pd.Series) -> float:
    """Peak-to-peak travel of a signal after the transition trigger."""
    post = series.loc[series.index > 0.0].to_numpy(dtype=float)
    return float(post.max() - post.min()) if post.size else 0.0


def mass_balance_lag_sec(
    df: pd.DataFrame, machine: dict[str, float], max_lag_sec: float = MAX_RESPONSE_LAG_SEC
) -> tuple[float, float, float]:
    """Lag that best explains measured ``bw`` as a delayed mass balance.

    Rebuilds basis weight from the *contemporaneous* manipulated variables and finds
    the shift that minimises RMS error against the measurement. Returns
    ``(lag_sec, rms_at_that_lag, rms_at_zero_lag)``.

    This is the discriminator the zero-lag rule needs. A simulator that emitted
    basis weight as an algebraic function of the manipulated variables scores an
    exact fit at lag 0; a correct one, carrying the wet-end lag, the transport
    delay and the scanner traverse, has its optimum tens of seconds out and fits
    markedly worse at zero.
    """
    mb = np.asarray(
        basis_weight_g_m2(
            df["stock_flow"],
            df["stock_cons"],
            df["filler_flow"],
            df["speed"],
            trim_m=machine["trim_m"],
            retention=machine["retention"],
            filler_cons_pct=machine["filler_cons_pct"],
        ),
        dtype=float,
    )
    bw = df["bw"].to_numpy(dtype=float)
    t = df.index.to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    best_lag, best_rms, rms_zero = 0.0, float("inf"), float("nan")
    for k in range(0, int(max_lag_sec / dt) + 1):
        resid = bw[k:] - mb[: bw.size - k] if k else bw - mb
        if resid.size == 0:
            break
        rms = float(np.sqrt(np.mean(resid**2)))
        if k == 0:
            rms_zero = rms
        if rms < best_rms:
            best_lag, best_rms = k * dt, rms
    return best_lag, best_rms, rms_zero


# --------------------------------------------------------------------------------------


def check_dataset(root: str | Path, *, verbose: bool = False) -> GateResult:
    """Run all six rules over an episode directory."""
    root_p = Path(root)
    episode_dirs = sorted(d for d in root_p.glob("*") if (d / "series.parquet").is_file())
    result = GateResult(n_episodes=len(episode_dirs))
    if not episode_dirs:
        result.add("range", f"no episodes found under {root_p}")
        return result

    steady_pool: list[pd.DataFrame] = []

    for d in episode_dirs:
        name = d.name
        df = pd.read_parquet(d / "series.parquet")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

        _check_ranges(df, name, meta, result)
        _check_rates(df, name, result)
        _check_zero_lag(df, name, meta["machine"], result)
        _check_ash_mass(df, name, result)

        steady = df.loc[df["phase"] == "steady", ["moist", "steam_p", "bw", "speed"]]
        if len(steady) > 30:
            steady_pool.append(steady.mean().to_frame().T)

        if verbose:
            print(f"checked {name}", file=sys.stderr)

    _check_moisture_steam_sign(steady_pool, result)
    return result


def _check_ranges(df: pd.DataFrame, name: str, meta: dict, result: GateResult) -> None:
    for column in _RANGE_COLUMNS:
        rng = RANGES[column]
        values = df[column].to_numpy(dtype=float)
        bad = (values < rng.lo) | (values > rng.hi) | ~np.isfinite(values)
        if bad.any():
            result.add(
                "range",
                f"{name}: {column} has {int(bad.sum())} sample(s) outside "
                f"[{rng.lo}, {rng.hi}] {rng.unit} "
                f"(min {np.nanmin(values):.4g}, max {np.nanmax(values):.4g})",
            )
    trim = float(meta["machine"]["trim_m"])
    if not RANGES["trim"].lo <= trim <= RANGES["trim"].hi:
        result.add("range", f"{name}: trim_m {trim} outside {RANGES['trim']}")
    # bw_true is simulator ground truth; an out-of-range value is still a bug.
    truth = df["bw_true"].to_numpy(dtype=float)
    if ((truth < RANGES["bw"].lo) | (truth > RANGES["bw"].hi)).any():
        result.add("range", f"{name}: bw_true leaves the basis weight range")


def _check_rates(df: pd.DataFrame, name: str, result: GateResult) -> None:
    bw_rate = rolling_rate_per_min(df["bw"])
    worst_bw = np.nanmax(np.abs(bw_rate)) if np.isfinite(bw_rate).any() else 0.0
    if worst_bw > BW_RATE_LIMIT_G_M2_MIN:
        result.add(
            "bw_rate",
            f"{name}: basis weight changed {worst_bw:.1f} g/m2 over a 60 s window "
            f"(limit {BW_RATE_LIMIT_G_M2_MIN} g/m2/min)",
        )
    speed_rate = rolling_rate_per_min(df["speed"])
    worst_speed = np.nanmax(np.abs(speed_rate)) if np.isfinite(speed_rate).any() else 0.0
    if worst_speed > SPEED_RATE_LIMIT_M_MIN_MIN:
        result.add(
            "speed_rate",
            f"{name}: speed changed {worst_speed:.1f} m/min over a 60 s window "
            f"(limit {SPEED_RATE_LIMIT_M_MIN_MIN} m/min/min)",
        )


def _check_zero_lag(
    df: pd.DataFrame, name: str, machine: dict[str, float], result: GateResult
) -> None:
    """Measured basis weight must not be an instantaneous function of the
    manipulated variables.

    Fit residual against lag, not cross-correlation. Correlation cannot identify
    this lag for two independent reasons: the loop is closed, so stock flow and
    basis weight are both driven by the same setpoint ramp and correlate at lag 0
    by construction; and the scanner is a zero-order hold refreshing every 20-45 s,
    so a transport delay of order 10 s sits below the resolution of the measured
    signal whatever estimator is used.

    Known limit, deliberately accepted: on a gentle ramp the lag-induced offset is
    the ramp rate times the lag, which falls under the scanner noise, so this
    statistic cannot separate "algebraic plus realistic noise" from correct data.
    An earlier attempt to force it to - by demanding an identifiable optimum on any
    large transition - failed four sound episodes whose measurement provably lagged
    the headbox by 30-40 s. The delay mechanism is pinned directly by unit test
    instead (``test_transport_delay_tracks_speed``,
    ``test_bw_true_leads_the_measurement``), and this rule keeps to the gross case
    it can actually decide.
    """
    pre = df.loc[df.index <= 0.0, "bw"]
    if response_travel(df["bw"]) < max(4.0 * float(pre.std()), 0.5):
        return  # transition too small to identify any lag against scanner noise

    lag, rms, rms_zero = mass_balance_lag_sec(df, machine)

    # An instantaneous fit this good is not physically reachable: even with no lag
    # at all, scanner noise alone would leave a residual. It means basis weight was
    # computed algebraically from the manipulated variables.
    if rms_zero < ALGEBRAIC_FIT_RMS_G_M2:
        result.add(
            "bw_zero_lag",
            f"{name}: mass balance fits bw at zero lag to {rms_zero:.4f} g/m2, below "
            f"the {ALGEBRAIC_FIT_RMS_G_M2} g/m2 scanner noise floor. Basis weight is an "
            f"algebraic function of the manipulated variables - no lag, delay or scanner.",
        )
        return

    # Otherwise the lag must be identifiable before it can be judged. On a small
    # transition the RMS curve rises monotonically from zero with no real optimum,
    # and its argmin carries no information; only enforce where a genuine optimum
    # exists to be found.
    improvement = (rms_zero - rms) / rms_zero if rms_zero > 0 else 0.0
    if improvement >= LAG_IDENTIFIABLE_IMPROVEMENT and lag < MIN_RESPONSE_LAG_SEC:
        result.add(
            "bw_zero_lag",
            f"{name}: bw is best explained by the mass balance at {lag:.0f} s lag "
            f"(rms {rms:.3f} vs {rms_zero:.3f} at zero); must be >= "
            f"{MIN_RESPONSE_LAG_SEC:.0f} s.",
        )


def _check_ash_mass(df: pd.DataFrame, name: str, result: GateResult) -> None:
    ash_mass = df["ash"].to_numpy(dtype=float) / 100.0 * df["bw"].to_numpy(dtype=float)
    bw = df["bw"].to_numpy(dtype=float)
    if (ash_mass > bw).any():
        result.add("ash_mass", f"{name}: ash mass exceeds total basis weight")


def _check_moisture_steam_sign(steady_pool: list[pd.DataFrame], result: GateResult) -> None:
    if len(steady_pool) < 10:
        return
    pooled = pd.concat(steady_pool, ignore_index=True)
    load = pooled["bw"].to_numpy(dtype=float) * pooled["speed"].to_numpy(dtype=float) / 1e5
    design = np.column_stack(
        [np.ones(len(pooled)), pooled["steam_p"].to_numpy(dtype=float), load]
    )
    target = pooled["moist"].to_numpy(dtype=float)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    if coeffs[1] >= 0.0:
        result.add(
            "moisture_steam_sign",
            "pooled steady phase: moisture rises with steam pressure "
            f"(coefficient {coeffs[1]:+.3f} %/bar, controlling for drying load)",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Physical plausibility gate for episode data.")
    parser.add_argument("root", nargs="?", default="data/episodes")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-report", type=int, default=8, help="failures listed per rule")
    args = parser.parse_args(argv)

    result = check_dataset(args.root, verbose=args.verbose)
    print(f"physics gate: {result.n_episodes} episode(s) under {args.root}")

    if result.passed:
        for rule in RULES:
            print(f"  PASS  {rule}")
        return 0

    for rule in RULES:
        problems = result.failures[rule]
        if not problems:
            print(f"  PASS  {rule}")
            continue
        print(f"  FAIL  {rule}: {len(problems)} episode(s)", file=sys.stderr)
        for line in problems[: args.max_report]:
            print(f"          {line}", file=sys.stderr)
        if len(problems) > args.max_report:
            print(f"          ... and {len(problems) - args.max_report} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
