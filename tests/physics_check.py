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

from src.sim.units import RANGES

__all__ = ["RULES", "GateResult", "check_dataset", "main"]

RULES: tuple[str, ...] = (
    "range",
    "bw_rate",
    "speed_rate",
    "bw_zero_lag",
    "ash_mass",
    "moisture_steam_sign",
)

BW_RATE_LIMIT_G_M2_MIN = 15.0
SPEED_RATE_LIMIT_M_MIN_MIN = 100.0
RATE_WINDOW_SEC = 60.0
#: Smallest lag, in seconds, at which basis weight may respond to a manipulated
#: variable. Anything below this is a zero-lag response.
MIN_RESPONSE_LAG_SEC = 3.0
MAX_RESPONSE_LAG_SEC = 240.0

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


def best_response_lag_sec(
    bw: pd.Series, driver: pd.Series, max_lag_sec: float = MAX_RESPONSE_LAG_SEC
) -> float | None:
    """Lag at which ``bw`` responds most strongly to ``driver``.

    Both signals are smoothed and differenced over a 30 s span before correlating,
    which strips the scanner's zero-order-hold staircase and the slow ramp trend and
    leaves the response. Returns None when the driver does not move enough to
    identify anything.
    """
    t = bw.index.to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    span = max(int(round(30.0 / dt)), 1)

    def prep(s: pd.Series) -> np.ndarray:
        smooth = s.rolling(span, center=True, min_periods=1).mean().to_numpy(dtype=float)
        d = np.full(smooth.size, np.nan)
        d[span:] = smooth[span:] - smooth[:-span]
        return d

    x, y = prep(driver), prep(bw)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 120 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    max_lag = int(round(max_lag_sec / dt))
    lags = np.arange(0, min(max_lag, x.size // 2))
    corr = np.array([np.dot(x[: x.size - lag], y[lag:]) / (x.size - lag) for lag in lags])
    if np.max(np.abs(corr)) < 0.15:
        return None
    return float(lags[int(np.argmax(np.abs(corr)))] * dt)


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
        _check_zero_lag(df, name, result)
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


def _check_zero_lag(df: pd.DataFrame, name: str, result: GateResult) -> None:
    for driver in ("stock_flow", "speed", "filler_flow"):
        lag = best_response_lag_sec(df["bw"], df[driver])
        if lag is not None and lag < MIN_RESPONSE_LAG_SEC:
            result.add(
                "bw_zero_lag",
                f"{name}: bw responds to {driver} at {lag:.0f} s "
                f"(must be >= {MIN_RESPONSE_LAG_SEC:.0f} s)",
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
