"""Episode loading, derived lag fields, and the 90-second feature table.

Ground-truth discipline is enforced by **allowlist, not denylist**. The loader selects
the columns and metadata keys the product is allowed to see; simulator-only fields are
never named here, so they cannot leak through a typo or a future schema widening.
Tests assert the forbidden names are absent from this package's source.

Three derived quantities live here, all computed from measured data only:

``detected_scan_period_sec``
    The QCS reading is zero-order held between traverses, so the measured basis-weight
    trace is a staircase. The median interval between value changes is the hold period.

``effective_measurement_lag_sec``
    ``theta_sec + detected_scan_period_sec``. Transport delay alone (5-17 s) badly
    understates how stale the operator's screen is; the traverse average plus hold adds
    roughly a full scan period on top. Both components stay separately addressable
    because the dashboard shows the decomposition.

``stab_from_ramp_end_sec``
    Stabilisation measured from the end of the setpoint ramp rather than from the
    trigger, fixing the degenerate label in the episode store. See
    :func:`stab_from_ramp_end_sec` for why this is still not a usable regression target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "SERIES_COLUMNS",
    "META_KEYS",
    "SPEC_BAND",
    "STABILISATION_WINDOW_SEC",
    "FEATURE_HORIZON_SEC",
    "MANIPULATED",
    "EpisodeData",
    "episode_dirs",
    "load_index",
    "load_series",
    "load_meta",
    "load_episode",
    "load_grades",
    "detect_scan_period_sec",
    "ramp_end_sec",
    "effective_measurement_lag",
    "stab_from_ramp_end_sec",
    "episode_features",
    "build_feature_table",
]

#: Columns the product is permitted to read from ``series.parquet``. Everything not
#: listed is either derived or simulator-only.
SERIES_COLUMNS: tuple[str, ...] = (
    "bw",
    "moist",
    "ash",
    "caliper",
    "bw_sp",
    "moist_sp",
    "ash_sp",
    "stock_flow",
    "filler_flow",
    "steam_p",
    "speed",
    "stock_cons",
    "theta_sec",
    "phase",
    "op_action",
    "alarm",
)

#: Metadata keys the product is permitted to read from ``meta.json``.
META_KEYS: tuple[str, ...] = (
    "episode_id",
    "grade_from",
    "grade_to",
    "grade_from_props",
    "grade_to_props",
    "machine",
    "recipe_limits",
    "actuator_rates",
    "labels",
)

#: Off-spec threshold from the process reference: 2.5% of the active setpoint.
SPEC_BAND: float = 0.025

#: Continuous in-band running required to call an episode stabilised.
STABILISATION_WINDOW_SEC: float = 120.0

#: The predictor sees only this much of the episode. Early data, not hindsight.
FEATURE_HORIZON_SEC: float = 90.0

#: Manipulated variables plus the slow disturbance, in canonical units.
MANIPULATED: tuple[str, ...] = ("stock_flow", "filler_flow", "steam_p", "speed", "stock_cons")

_INDEX_NAME = "index.parquet"


class EpisodeData:
    """One episode, safe columns only, with the derived lag fields attached."""

    __slots__ = ("series", "meta", "episode_id")

    def __init__(self, series: pd.DataFrame, meta: dict[str, Any]) -> None:
        self.series = series
        self.meta = meta
        self.episode_id = str(meta["episode_id"])

    @property
    def trim_m(self) -> float:
        return float(self.meta["machine"]["trim_m"])

    @property
    def retention(self) -> float:
        return float(self.meta["machine"]["retention"])

    @property
    def filler_cons_pct(self) -> float:
        return float(self.meta["machine"]["filler_cons_pct"])

    def post_trigger(self) -> pd.DataFrame:
        return self.series.loc[self.series.index >= 0.0]

    def window(self, horizon_sec: float = FEATURE_HORIZON_SEC) -> pd.DataFrame:
        """Post-trigger slice: the transition so far, up to ``horizon_sec``."""
        idx = self.series.index
        return self.series.loc[(idx >= 0.0) & (idx <= horizon_sec)]

    def history(self, horizon_sec: float = FEATURE_HORIZON_SEC) -> pd.DataFrame:
        """Everything the operator has at ``horizon_sec``, pre-roll included.

        This is the slice the predictor is allowed to see at inference time. It stops at
        the horizon, so there is still no lookahead; it simply does not pretend the
        machine had no history before the grade change was triggered.
        """
        return self.series.loc[self.series.index <= horizon_sec]

    def deviation_pct(self) -> pd.Series:
        """Signed deviation of measured basis weight from the active setpoint, %."""
        sp = self.series["bw_sp"]
        return (self.series["bw"] - sp) / sp * 100.0


def episode_dirs(root: str | Path = "data/episodes") -> list[Path]:
    return sorted(p for p in Path(root).glob("EP-*") if p.is_dir())


def load_index(root: str | Path = "data/episodes") -> pd.DataFrame:
    """Read ``index.parquet``, the fast query surface.

    Preferred over walking ``meta.json`` files: it is one read, and it is built to
    exclude simulator ground truth entirely.
    """
    path = Path(root) / _INDEX_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - regenerate it with src.sim.writer.rebuild_index")
    return pd.read_parquet(path)


def load_series(episode_dir: str | Path) -> pd.DataFrame:
    """Read one ``series.parquet``, projected onto :data:`SERIES_COLUMNS`."""
    df = pd.read_parquet(Path(episode_dir) / "series.parquet")
    missing = [c for c in SERIES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{episode_dir}: series.parquet missing columns {missing}")
    return df.loc[:, list(SERIES_COLUMNS)]


def load_meta(episode_dir: str | Path) -> dict[str, Any]:
    """Read one ``meta.json``, projected onto :data:`META_KEYS`."""
    raw = json.loads((Path(episode_dir) / "meta.json").read_text(encoding="utf-8"))
    return {k: raw[k] for k in META_KEYS if k in raw}


def load_episode(episode_dir: str | Path) -> EpisodeData:
    return EpisodeData(load_series(episode_dir), load_meta(episode_dir))


def load_grades(path: str | Path = "data/grades.json") -> dict[str, dict[str, float]]:
    """Grade catalogue. ``nominal_speed_m_min`` is an operating parameter, not part of
    the grade-space embedding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detect_scan_period_sec(bw: np.ndarray, dt_sec: float = 1.0) -> float:
    """Recover the scanner hold interval from the measured trace alone.

    The QCS holds its reading until the next traverse completes, so ``bw`` is a
    staircase and the spacing between steps is the scan period. Median rather than
    mean: a couple of coincidentally equal consecutive readings would otherwise drag
    the estimate down.
    """
    changed = np.flatnonzero(np.diff(np.asarray(bw, dtype=float)) != 0.0)
    if changed.size < 3:
        return float("nan")
    return float(np.median(np.diff(changed)) * dt_sec)


def ramp_end_sec(series: pd.DataFrame) -> float | None:
    """Last instant of the ``ramp`` phase, from the ``phase`` column."""
    phase = series["phase"].astype("string").to_numpy()
    idx = np.flatnonzero(phase == "ramp")
    if idx.size == 0:
        return None
    return float(series.index.to_numpy(dtype=float)[int(idx[-1])])


def effective_measurement_lag(series: pd.DataFrame) -> dict[str, Any]:
    """Decompose the lag between sheet formation and what the operator sees.

    Returns the three components separately. The composed figure is the one the evidence
    card must quote: transport delay on its own reads ~6 s, and claiming 40 s of staleness
    next to a 6 s number on screen is indefensible.

    The scan period is detected over **everything passed in**, including any pre-trigger
    baseline, while transport delay is taken over the post-trigger part. That asymmetry is
    deliberate: a 90 s window holds only two or three scanner steps, which is not enough
    to measure a hold interval, but the operator has the pre-roll history too, so using it
    costs nothing and is not lookahead. Pass the series from the start of history up to
    "now"; the transition itself is what the transport median should describe.
    """
    scan = detect_scan_period_sec(series["bw"].to_numpy(dtype=float))
    post = series.loc[series.index >= 0.0]
    if post.empty:
        post = series
    theta = post["theta_sec"].to_numpy(dtype=float)
    transport = float(np.median(theta))
    per_sample = pd.Series(theta + scan, index=post.index, name="effective_measurement_lag_sec")
    return {
        "transport_sec": transport,
        "scanner_sec": float(scan),
        "composed_sec": float(transport + scan),
        "per_sample": per_sample,
    }


def stab_from_ramp_end_sec(series: pd.DataFrame, band: float = SPEC_BAND) -> float | None:
    """Stabilisation time measured from the end of the setpoint ramp.

    The episode store's ``stabilisation_t_sec`` is degenerate: an episode that never
    breaches "stabilises" at the first post-trigger sample, which is 187 of 300
    episodes. Measuring from ramp end removes that artefact but does not make the
    field informative -- 80% of episodes are already in-band when the ramp completes,
    so the distribution piles up at 0.0. Kept because the dashboard reports it and
    because it is the honest version of the quantity; the impact ranking uses the
    off-spec subset instead. See ``impact.py``.

    Returns ``None`` if the episode never holds the band for
    :data:`STABILISATION_WINDOW_SEC` before the episode ends.
    """
    t_end = ramp_end_sec(series)
    if t_end is None:
        return None
    post = series.loc[series.index >= t_end]
    t = post.index.to_numpy(dtype=float)
    sp = post["bw_sp"].to_numpy(dtype=float)
    dev = np.abs(post["bw"].to_numpy(dtype=float) - sp) / sp
    breaches = np.flatnonzero(dev > band)
    start = 0 if breaches.size == 0 else int(breaches[-1]) + 1
    if start >= t.size or float(t[-1]) - float(t[start]) < STABILISATION_WINDOW_SEC:
        return None
    return float(t[start]) - t_end


def _rate_per_min(t_sec: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope in units per minute. Robust to the staircase in measured
    signals, unlike a first/last difference."""
    if t_sec.size < 2:
        return 0.0
    slope = float(np.polyfit(t_sec, y, 1)[0])
    return slope * 60.0


def episode_features(
    ep: EpisodeData,
    *,
    horizon_sec: float = FEATURE_HORIZON_SEC,
    grades: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """One feature row from the first ``horizon_sec`` seconds after the trigger.

    Everything here is available to the operator 90 s into a transition. No label, no
    ground truth, no lookahead.
    """
    w = ep.window(horizon_sec)
    t = w.index.to_numpy(dtype=float)
    lag = effective_measurement_lag(ep.history(horizon_sec))

    from_props = ep.meta["grade_from_props"]
    to_props = ep.meta["grade_to_props"]
    grades = grades if grades is not None else {}
    nom_from = float(grades.get(ep.meta["grade_from"], {}).get("nominal_speed_m_min", np.nan))
    nom_to = float(grades.get(ep.meta["grade_to"], {}).get("nominal_speed_m_min", np.nan))

    row: dict[str, Any] = {
        "episode_id": ep.episode_id,
        "grade_from": ep.meta["grade_from"],
        "grade_to": ep.meta["grade_to"],
    }

    # --- ramp rates of the manipulated variables over the window
    for name in MANIPULATED:
        row[f"rate_{name}"] = _rate_per_min(t, w[name].to_numpy(dtype=float))
        row[f"start_{name}"] = float(w[name].to_numpy(dtype=float)[0])

    # --- normalised rates and the desync signature.
    # Physics: bw is proportional to stock flow and inversely proportional to speed, so
    # a coordinated transition holds the two normalised rates equal. Their ratio is the
    # ramp-desynchronisation signature, and it is dimensionless, so it is comparable
    # across a 45 g/m2 and a 160 g/m2 grade.
    speed_start = max(row["start_speed"], 1e-6)
    stock_start = max(row["start_stock_flow"], 1e-6)
    speed_rate_frac = row["rate_speed"] / speed_start
    stock_rate_frac = row["rate_stock_flow"] / stock_start
    row["speed_rate_frac_per_min"] = speed_rate_frac
    row["stock_rate_frac_per_min"] = stock_rate_frac
    row["speed_stock_rate_ratio"] = float(
        np.clip(speed_rate_frac / (stock_rate_frac + np.sign(stock_rate_frac or 1.0) * 1e-4), -20.0, 20.0)
    )
    # Signed mismatch: positive means speed is running ahead of stock, which drives the
    # sheet light. This is the term the advisor ends up acting on.
    row["desync_frac_per_min"] = speed_rate_frac - stock_rate_frac

    # --- deviation so far
    dev = (w["bw"].to_numpy(dtype=float) - w["bw_sp"].to_numpy(dtype=float)) / w[
        "bw_sp"
    ].to_numpy(dtype=float) * 100.0
    row["dev_pct_now"] = float(dev[-1])
    row["dev_pct_abs_now"] = float(abs(dev[-1]))
    row["dev_pct_max_abs"] = float(np.max(np.abs(dev)))
    row["dev_pct_rate_per_min"] = _rate_per_min(t, dev)
    row["bw_sp_rate_per_min"] = _rate_per_min(t, w["bw_sp"].to_numpy(dtype=float))

    # --- grade property deltas
    for prop in ("bw", "ash", "moist", "caliper"):
        row[f"d_{prop}"] = float(to_props[prop]) - float(from_props[prop])
    row["d_bw_frac"] = row["d_bw"] / max(float(from_props["bw"]), 1e-6)
    row["nominal_speed_from"] = nom_from
    row["nominal_speed_to"] = nom_to
    row["d_nominal_speed"] = nom_to - nom_from

    # --- lag decomposition
    row["transport_lag_sec"] = lag["transport_sec"]
    row["scanner_lag_sec"] = lag["scanner_sec"]
    row["effective_measurement_lag_sec"] = lag["composed_sec"]

    # --- machine constants that enter the mass balance
    row["trim_m"] = ep.trim_m
    row["retention"] = ep.retention
    row["filler_cons_pct"] = ep.filler_cons_pct

    # --- targets, for training and evaluation only
    labels = ep.meta["labels"]
    row["off_spec"] = bool(labels["off_spec"])
    row["max_dev_pct"] = float(labels["max_dev_pct"])
    row["breach_t_sec"] = labels["breach_t_sec"]
    row["stabilisation_t_sec"] = labels["stabilisation_t_sec"]
    row["stab_from_ramp_end_sec"] = stab_from_ramp_end_sec(ep.series)
    row["broke_tonnes"] = float(labels["broke_tonnes"])
    row["ramp_end_sec"] = ramp_end_sec(ep.series)
    return row


def build_feature_table(
    root: str | Path = "data/episodes",
    *,
    horizon_sec: float = FEATURE_HORIZON_SEC,
    grades_path: str | Path = "data/grades.json",
    cache: str | Path | None = None,
) -> pd.DataFrame:
    """One row per episode, features from the first ``horizon_sec`` seconds only."""
    if cache is not None and Path(cache).exists():
        return pd.read_parquet(cache)
    grades = load_grades(grades_path) if Path(grades_path).exists() else {}
    rows = [
        episode_features(load_episode(d), horizon_sec=horizon_sec, grades=grades)
        for d in episode_dirs(root)
    ]
    table = pd.DataFrame(rows)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
    return table
