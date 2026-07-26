"""Physics-first breach predictor. Gray-box: mass balance leads, ML trims the residual.

Why this is not a classifier
----------------------------
The mass balance is not an approximation of the mill, it is the relation the mill obeys:
``bw = (stock_L_min * cons * 10 + filler * filler_cons * 10) / (speed * trim) * retention``.
Given the manipulated variables, basis weight at the headbox is *determined*. So the
prediction is arithmetic, and the learned model only has to explain what the arithmetic
cannot see from 90 seconds of data: the wet-end time constant, moisture coupling into the
QCS total-weight reading, and controller behaviour through the settle.

The core claim, stated plainly
------------------------------
What the operator's screen shows now is sheet that was formed
``effective_measurement_lag_sec`` ago -- transport delay plus a full scanner traverse and
hold, roughly 40 s in this dataset, not the 6 s that ``theta_sec`` alone suggests. The
headbox state is therefore already ahead of the display. We compute basis weight at the
headbox from the manipulated variables *now*, so we know what the sheet already is before
the scanner reports it. That is why a breach can be forecast that the operator cannot yet
see: it is not extrapolation of a trend, it is reading the sheet that is already made.

Everything here reads measured columns only. The physics path never touches ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.analysis.loader import (
    FEATURE_HORIZON_SEC,
    SPEC_BAND,
    build_feature_table,
    effective_measurement_lag,
    load_episode,
    load_grades,
)
from src.sim.machine import basis_weight_g_m2, first_order_step, required_flows

__all__ = [
    "TAU_WET_NOMINAL_SEC",
    "SETTLE_ALLOWANCE_SEC",
    "PhysicsProjection",
    "Predictor",
    "project_physics",
    "evaluate",
    "main",
]

#: Wet-end mixing and retention lag, seconds. The process reference gives 20-60 s and the
#: simulator draws per episode; 90 s of data is not enough to identify it, so the
#: projection uses the midpoint as a physics constant. The residual model absorbs the
#: per-episode difference -- this is exactly the kind of small, bounded error the learned
#: term is for.
TAU_WET_NOMINAL_SEC: float = 40.0

#: Display margin past the physics validity horizon, seconds. The trajectory is drawn a
#: little further than the physics is trusted so the chart shows where the loop takes
#: over, but no breach claim is made from that region.
DISPLAY_MARGIN_SEC: float = 60.0

#: Filler ramp rate, L/min per minute. Not present in ``actuator_rates``, so a nominal is
#: needed for the case where the filler ramp has not visibly started inside the window.
FILLER_RATE_L_MIN_PER_MIN: float = 160.0

_MAX_HORIZON_SEC: float = 1200.0
_TRAJ_STEP_SEC: float = 5.0

#: Feature columns handed to the residual model. Numeric, all available at 90 s.
RESIDUAL_FEATURES: tuple[str, ...] = (
    "rate_stock_flow",
    "rate_filler_flow",
    "rate_steam_p",
    "rate_speed",
    "rate_stock_cons",
    "start_stock_flow",
    "start_speed",
    "speed_rate_frac_per_min",
    "stock_rate_frac_per_min",
    "speed_stock_rate_ratio",
    "desync_frac_per_min",
    "dev_pct_now",
    "dev_pct_abs_now",
    "dev_pct_max_abs",
    "dev_pct_rate_per_min",
    "bw_sp_rate_per_min",
    "d_bw",
    "d_ash",
    "d_moist",
    "d_caliper",
    "d_bw_frac",
    "d_nominal_speed",
    "transport_lag_sec",
    "scanner_lag_sec",
    "effective_measurement_lag_sec",
    "physics_max_dev_pct",
    "physics_committed_dev_pct",
    "physics_signed_dev_pct",
)


@dataclass
class PhysicsProjection:
    """Result of the pure-physics forward projection. No learned term involved."""

    t_now_sec: float
    trajectory: list[tuple[float, float]]
    setpoint_trajectory: list[tuple[float, float]]
    max_dev_pct: float
    signed_dev_at_max_pct: float
    time_to_breach_sec: float | None
    lag_components: dict[str, float]
    headbox_bw_now: float
    measured_bw_now: float
    committed_dev_pct: float = 0.0
    calibration_offset: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def _ramp_rate_per_sec(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def _target_operating_point(
    meta: dict[str, Any],
    cons_pct: float,
    target_speed_m_min: float,
) -> tuple[float, float]:
    """Stock and filler flow the target grade needs, from the inverse mass balance.

    This is what bounds the projection. Extrapolating the manipulated variables at their
    current ramp rates without a destination is what makes a naive forward projection
    diverge; the mill is not ramping into the void, it is ramping to the operating point
    the target grade requires, and that point is computable.
    """
    machine = meta["machine"]
    to_props = meta["grade_to_props"]
    return required_flows(
        float(to_props["bw"]),
        float(to_props["ash"]),
        target_speed_m_min,
        cons_pct,
        trim_m=float(machine["trim_m"]),
        retention=float(machine["retention"]),
        filler_cons_pct=float(machine["filler_cons_pct"]),
    )


def project_physics(
    series_so_far: pd.DataFrame,
    meta: dict[str, Any],
    *,
    tau_wet_sec: float = TAU_WET_NOMINAL_SEC,
    horizon_sec: float | None = None,
    target_speed_m_min: float | None = None,
) -> PhysicsProjection:
    """Project basis weight forward from the mass balance alone.

    The projection is built in the **headbox domain** and then shifted into the
    measurement domain by the composed lag. The first ``lag`` seconds of the predicted
    measurement are therefore already committed: they are sheet that exists but has not
    reached the scanner, so no extrapolation is involved in that part at all. Beyond it,
    each manipulated variable continues at its observed ramp rate but is clamped at the
    operating point the target grade requires, so the projection converges instead of
    diverging. The breach signal is the *arrival mismatch*: speed reaching its target
    ahead of stock flow leaves the sheet light for the interval between them.

    ``series_so_far`` should run from the start of available history up to "now", pre-roll
    included. Ramp rates are measured on the post-trigger part only; the pre-roll is there
    because the scanner hold interval cannot be measured from 90 s alone.
    """
    window = series_so_far.loc[series_so_far.index >= 0.0]
    if window.empty:
        window = series_so_far
    t = window.index.to_numpy(dtype=float)
    t_now = float(t[-1])
    lag_info = effective_measurement_lag(series_so_far)
    lag = float(lag_info["composed_sec"])
    if not np.isfinite(lag) or lag <= 0.0:
        lag = float(np.median(window["theta_sec"].to_numpy(dtype=float)))

    machine = meta["machine"]
    limits = meta["recipe_limits"]
    rates_limit = meta["actuator_rates"]
    trim_m = float(machine["trim_m"])
    retention = float(machine["retention"])
    filler_cons_pct = float(machine["filler_cons_pct"])

    mv_names = ("stock_flow", "filler_flow", "steam_p", "speed")
    now = {n: float(window[n].to_numpy(dtype=float)[-1]) for n in mv_names}
    rate = {n: _ramp_rate_per_sec(t, window[n].to_numpy(dtype=float)) for n in mv_names}
    cons_now = float(window["stock_cons"].to_numpy(dtype=float)[-1])

    sp_now = float(window["bw_sp"].to_numpy(dtype=float)[-1])
    sp_rate = _ramp_rate_per_sec(t, window["bw_sp"].to_numpy(dtype=float))
    sp_target = float(meta["grade_to_props"]["bw"])

    # --- destination for each manipulated variable
    speed_target = float(target_speed_m_min) if target_speed_m_min is not None else now["speed"]
    stock_target, filler_target = _target_operating_point(meta, cons_now, speed_target)
    target = {
        "stock_flow": float(np.clip(stock_target, *limits["stock_flow"])),
        "filler_flow": float(np.clip(filler_target, *limits["filler_flow"])),
        "speed": float(np.clip(speed_target, *limits["speed"])),
        "steam_p": now["steam_p"],
    }
    # Where the observed rate is too small to be a ramp, fall back to the actuator limit:
    # the variable has not started moving yet but it still has to get there.
    rate_floor = {
        "stock_flow": float(rates_limit.get("stock_flow", 120.0)) / 60.0,
        "speed": float(rates_limit.get("speed", 60.0)) / 60.0,
        "steam_p": float(rates_limit.get("steam_p", 0.4)) / 60.0,
        "filler_flow": FILLER_RATE_L_MIN_PER_MIN / 60.0,
    }
    eff_rate: dict[str, float] = {}
    arrival: dict[str, float] = {}
    for name in mv_names:
        gap = target[name] - now[name]
        r = rate[name]
        if abs(r) < 0.05 * rate_floor[name] or r * gap <= 0.0:
            r = float(np.sign(gap)) * rate_floor[name]
        eff_rate[name] = r
        arrival[name] = abs(gap) / abs(r) if abs(r) > 1e-9 else 0.0

    # --- physics validity horizon.
    # The mill is closed-loop. Projecting the mass balance open-loop for ten minutes is
    # meaningless, because the controller acts long before then; an early version of this
    # function did exactly that and produced 26% mean predicted deviation against a 2.8%
    # actual. The interval over which the loop *cannot* act is the measurement lag: until
    # the sheet formed now reaches the scanner, the controller has no information about
    # it and cannot respond. That window is where the mass balance is not merely a model
    # but a statement about sheet that already exists. Everything past it belongs to the
    # residual model and to the causal ranking, not to open-loop physics.
    validity_sec = lag
    if horizon_sec is None:
        horizon_sec = float(min(validity_sec + DISPLAY_MARGIN_SEC, _MAX_HORIZON_SEC))
    t_sp_done = t_now + float(max(arrival.values()))

    # --- headbox trajectory: window history, then bounded projection
    dt = 1.0
    t_hb = np.arange(t[0], t_now + horizon_sec + dt, dt)
    mv_traj: dict[str, np.ndarray] = {}
    for name in mv_names:
        past = np.interp(np.clip(t_hb, t[0], t_now), t, window[name].to_numpy(dtype=float))
        ahead = t_hb - t_now
        projected = now[name] + eff_rate[name] * np.clip(ahead, 0.0, None)
        lo_t, hi_t = min(now[name], target[name]), max(now[name], target[name])
        projected = np.clip(projected, lo_t, hi_t)
        lo, hi = float(limits[name][0]), float(limits[name][1])
        mv_traj[name] = np.where(ahead > 0.0, np.clip(projected, lo, hi), past)

    # Wet-end first-order lag on the thick stock delivery. Filler and speed act on the
    # sheet without this lag, which is exactly why a desynchronised ramp bites.
    stock_hb = np.empty_like(t_hb)
    stock_hb[0] = mv_traj["stock_flow"][0]
    for k in range(1, t_hb.size):
        stock_hb[k] = first_order_step(
            float(stock_hb[k - 1]), float(mv_traj["stock_flow"][k]), tau_wet_sec, dt
        )

    bw_hb = np.asarray(
        basis_weight_g_m2(
            stock_hb,
            cons_now,
            mv_traj["filler_flow"],
            mv_traj["speed"],
            trim_m=trim_m,
            retention=retention,
            filler_cons_pct=filler_cons_pct,
        ),
        dtype=float,
    )

    # --- anchor the mass balance to the measurement.
    # The QCS reports total basis weight, so it sits a little above the dry mass balance
    # whenever moisture is off target, and the sensor carries its own calibration. Both
    # are slow relative to a transition, so a single offset fitted over the window
    # removes them. Fitted from measured bw against the lag-shifted physics only -- no
    # ground truth, no label.
    bw_meas = window["bw"].to_numpy(dtype=float)
    phys_at_meas_time = np.interp(t - lag, t_hb, bw_hb)
    usable = t - lag >= t[0]
    offset = float(np.mean(bw_meas[usable] - phys_at_meas_time[usable])) if usable.any() else 0.0

    # --- into the measurement domain
    t_pred = np.arange(t_now, t_now + horizon_sec + _TRAJ_STEP_SEC, _TRAJ_STEP_SEC)
    bw_pred = np.interp(t_pred - lag, t_hb, bw_hb) + offset

    sp_pred = np.clip(
        sp_now + sp_rate * np.clip(np.minimum(t_pred, t_sp_done) - t_now, 0.0, None),
        min(sp_now, sp_target),
        max(sp_now, sp_target),
    )

    dev_pct = (bw_pred - sp_pred) / sp_pred * 100.0

    # Claims are made only inside the validity horizon.
    committed_window = t_pred <= t_now + validity_sec
    dev_committed = dev_pct[committed_window]
    idx_max = int(np.argmax(np.abs(dev_committed)))
    breach = np.flatnonzero(np.abs(dev_committed) > SPEC_BAND * 100.0)
    ttb = float(t_pred[committed_window][breach[0]] - t_now) if breach.size else None

    # Committed deviation: the sheet at the headbox right now, against the setpoint that
    # will be in force when that sheet reaches the scanner. This number is already
    # decided -- it is in the sheet -- and the operator cannot see it yet.
    bw_headbox_now = float(np.interp(t_now, t_hb, bw_hb) + offset)
    sp_when_seen = float(
        np.clip(sp_now + sp_rate * lag, min(sp_now, sp_target), max(sp_now, sp_target))
    )
    committed_dev_pct = (bw_headbox_now - sp_when_seen) / sp_when_seen * 100.0

    return PhysicsProjection(
        t_now_sec=t_now,
        trajectory=[(round(float(a), 1), round(float(b), 3)) for a, b in zip(t_pred, bw_pred)],
        setpoint_trajectory=[
            (round(float(a), 1), round(float(b), 3)) for a, b in zip(t_pred, sp_pred)
        ],
        max_dev_pct=float(np.max(np.abs(dev_committed))),
        signed_dev_at_max_pct=float(dev_committed[idx_max]),
        time_to_breach_sec=ttb,
        lag_components={
            "transport": round(float(lag_info["transport_sec"]), 2),
            "scanner": round(float(lag_info["scanner_sec"]), 2),
            "composed": round(lag, 2),
        },
        headbox_bw_now=bw_headbox_now,
        measured_bw_now=float(bw_meas[-1]),
        committed_dev_pct=float(committed_dev_pct),
        calibration_offset=round(offset, 3),
        detail={
            "t_mv_arrival_sec": round(t_sp_done, 1),
            "horizon_sec": round(float(horizon_sec), 1),
            "validity_sec": round(float(validity_sec), 1),
            "sp_now": round(sp_now, 2),
            "sp_target": round(sp_target, 2),
            "sp_when_seen": round(sp_when_seen, 2),
            "tau_wet_sec": tau_wet_sec,
            "mv_rates_per_min": {n: round(rate[n] * 60.0, 3) for n in mv_names},
            "mv_targets": {n: round(target[n], 2) for n in mv_names},
            "mv_arrival_sec": {n: round(arrival[n], 1) for n in mv_names},
            "headbox_lead_g_m2": round(bw_headbox_now - float(bw_meas[-1]), 3),
        },
    )


class Predictor:
    """Physics projection plus a residual correction.

    ``fit`` trains only on the gap between the physics projection and what actually
    happened. If that gap ever stops being small, the physics path is broken and the
    honest move is to say so rather than let the model paper over it -- see
    :meth:`residual_report`.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: GradientBoostingRegressor | None = None
        self.features: list[str] = []
        self._fit_report: dict[str, Any] = {}

    # -- training ---------------------------------------------------------------
    @staticmethod
    def physics_table(
        features: pd.DataFrame,
        root: str | Path = "data/episodes",
        *,
        horizon_sec: float = FEATURE_HORIZON_SEC,
        grades: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Add the physics projection columns to the feature table."""
        out = features.copy()
        grades = grades if grades is not None else load_grades()
        phys: list[float] = []
        ttb: list[float | None] = []
        committed: list[float] = []
        signed: list[float] = []
        for episode_id in out["episode_id"]:
            ep = load_episode(Path(root) / str(episode_id))
            proj = project_physics(
                ep.history(horizon_sec),
                ep.meta,
                target_speed_m_min=grades.get(ep.meta["grade_to"], {}).get("nominal_speed_m_min"),
            )
            phys.append(proj.max_dev_pct)
            ttb.append(proj.time_to_breach_sec)
            committed.append(proj.committed_dev_pct)
            signed.append(proj.signed_dev_at_max_pct)
        out["physics_max_dev_pct"] = phys
        out["physics_time_to_breach_sec"] = ttb
        out["physics_committed_dev_pct"] = committed
        out["physics_signed_dev_pct"] = signed
        return out

    def fit(self, table: pd.DataFrame) -> dict[str, Any]:
        """Train the residual model. ``table`` must already carry the physics column."""
        self.features = [c for c in RESIDUAL_FEATURES if c in table.columns]
        x = table[self.features].astype(float).fillna(0.0)
        residual = table["max_dev_pct"].astype(float) - table["physics_max_dev_pct"].astype(float)
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=self.random_state
        )
        self.model.fit(x, residual)
        self._fit_report = {
            "n_train": int(len(table)),
            "residual_mean_abs_pct": round(float(residual.abs().mean()), 3),
            "residual_std_pct": round(float(residual.std()), 3),
        }
        return self._fit_report

    def correct(self, table: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Predictor.fit must be called before correct")
        x = table[self.features].astype(float).fillna(0.0)
        return np.asarray(self.model.predict(x), dtype=float)

    def residual_report(self, table: pd.DataFrame) -> dict[str, Any]:
        """How much of the answer physics is carrying. Demo asset and sanity check."""
        phys = table["physics_max_dev_pct"].astype(float).abs()
        corr = np.abs(self.correct(table))
        share = float((phys / (phys + corr)).mean())
        return {
            "physics_contribution_mean": round(share, 3),
            "model_correction_mean": round(1.0 - share, 3),
            "physics_mean_abs_pct": round(float(phys.mean()), 3),
            "correction_mean_abs_pct": round(float(corr.mean()), 3),
            "verdict": (
                "physics leads" if share >= 0.6 else "MODEL IS DOING TOO MUCH - check physics path"
            ),
        }

    # -- inference --------------------------------------------------------------
    def predict(
        self,
        series_so_far: pd.DataFrame,
        meta: dict[str, Any],
        *,
        feature_row: pd.DataFrame | None = None,
        target_speed_m_min: float | None = None,
    ) -> dict[str, Any]:
        """Predict from a partial episode. ``series_so_far`` must be measured columns only."""
        proj = project_physics(
            series_so_far, meta, target_speed_m_min=target_speed_m_min
        )
        physics_dev = proj.max_dev_pct
        correction = 0.0
        if self.model is not None and feature_row is not None:
            row = feature_row.copy()
            row["physics_max_dev_pct"] = physics_dev
            row["physics_committed_dev_pct"] = proj.committed_dev_pct
            row["physics_signed_dev_pct"] = proj.signed_dev_at_max_pct
            correction = float(self.correct(row)[0])

        corrected = physics_dev + correction
        spread = max(float(self._fit_report.get("residual_std_pct", 1.0)), 0.25)
        margin = (abs(corrected) - SPEC_BAND * 100.0) / spread
        confidence = float(1.0 / (1.0 + np.exp(-abs(margin))))

        denom = abs(physics_dev) + abs(correction)
        return {
            "will_breach": bool(abs(corrected) > SPEC_BAND * 100.0),
            "confidence": round(confidence, 3),
            "predicted_max_dev_pct": round(corrected, 3),
            "time_to_breach_sec": proj.time_to_breach_sec,
            "predicted_trajectory": proj.trajectory,
            "setpoint_trajectory": proj.setpoint_trajectory,
            "physics_contribution": round(abs(physics_dev) / denom, 3) if denom else 1.0,
            "model_correction": round(abs(correction) / denom, 3) if denom else 0.0,
            "physics_max_dev_pct": round(physics_dev, 3),
            "model_correction_pct": round(correction, 3),
            "lag_components": proj.lag_components,
            "headbox_bw_now": round(proj.headbox_bw_now, 2),
            "measured_bw_now": round(proj.measured_bw_now, 2),
            "headbox_lead_g_m2": proj.detail["headbox_lead_g_m2"],
            "physics_detail": proj.detail,
        }


# -- evaluation -----------------------------------------------------------------
def _classification(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    total = tp + fp + fn + tn
    return {
        "accuracy": round((tp + tn) / total, 3) if total else float("nan"),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _best_threshold(truth: np.ndarray, score: np.ndarray) -> float:
    """Accuracy-maximising cut point, chosen on training data only."""
    candidates = np.arange(0.5, 5.01, 0.05)
    accuracies = [float(np.mean((score > c) == truth)) for c in candidates]
    return float(candidates[int(np.argmax(accuracies))])


def evaluate(
    root: str | Path = "data/episodes",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    naive_threshold_pct: float = 1.5,
    table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """80/20 split, three predictors, one honest comparison.

    Baselines are non-negotiable: a breach predictor that cannot beat "the deviation is
    already large" is not a predictor, it is a threshold with extra steps.
    """
    feats = table if table is not None else Predictor.physics_table(build_feature_table(root), root)
    train, test = train_test_split(
        feats, test_size=test_size, random_state=random_state, stratify=feats["off_spec"]
    )

    predictor = Predictor(random_state=random_state)
    fit_report = predictor.fit(train)

    truth = test["off_spec"].to_numpy(dtype=bool)
    physics_dev = test["physics_max_dev_pct"].to_numpy(dtype=float)
    corrected = physics_dev + predictor.correct(test)

    # Decision thresholds are calibrated on the training split only. The predicted
    # quantity is a deviation the physics can only partly see at 90 s, so the spec band
    # itself is not the optimal cut point for it; picking the cut on test data would be
    # leakage, so it is picked on train and applied blind.
    train_truth = train["off_spec"].to_numpy(dtype=bool)
    train_physics = train["physics_max_dev_pct"].to_numpy(dtype=float)
    train_corrected = train_physics + predictor.correct(train)
    thr_gray = _best_threshold(train_truth, np.abs(train_corrected))
    thr_physics = _best_threshold(train_truth, np.abs(train_physics))

    results = {
        "gray_box": _classification(truth, np.abs(corrected) > thr_gray),
        "physics_only": _classification(truth, np.abs(physics_dev) > thr_physics),
        "naive_current_deviation": _classification(
            truth, test["dev_pct_abs_now"].to_numpy(dtype=float) > naive_threshold_pct
        ),
    }
    actual = test["max_dev_pct"].to_numpy(dtype=float)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_breach_rate": round(float(truth.mean()), 3),
        "naive_threshold_pct": naive_threshold_pct,
        "thresholds_from_train": {
            "gray_box_pct": round(thr_gray, 2),
            "physics_only_pct": round(thr_physics, 2),
        },
        "results": results,
        "max_dev_mae_pct": {
            "physics_only": round(float(np.mean(np.abs(physics_dev - actual))), 3),
            "gray_box": round(float(np.mean(np.abs(corrected - actual))), 3),
        },
        "fit_report": fit_report,
        "residual_report": predictor.residual_report(test),
        "predictor": predictor,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the gray-box breach predictor.")
    parser.add_argument("--root", default="data/episodes")
    parser.add_argument("--cache", default="data/features_90s.parquet")
    args = parser.parse_args(argv)

    feats = build_feature_table(args.root, cache=args.cache)
    table = Predictor.physics_table(feats, args.root)
    report = evaluate(args.root, table=table)

    print(f"train {report['n_train']}   test {report['n_test']}   "
          f"test breach rate {report['test_breach_rate']:.3f}")
    print(f"\nBREACH PREDICTION FROM 90 s OF DATA")
    print(f"{'model':<28}{'accuracy':>10}{'precision':>11}{'recall':>9}")
    labels = {
        "gray_box": "gray-box (physics + residual)",
        "physics_only": "physics only",
        "naive_current_deviation": f"naive (dev > {report['naive_threshold_pct']}%)",
    }
    for key, label in labels.items():
        m = report["results"][key]
        print(f"{label:<28}{m['accuracy']:>10.3f}{m['precision']:>11.3f}{m['recall']:>9.3f}")

    thr = report["thresholds_from_train"]
    print(
        f"thresholds calibrated on train: gray-box {thr['gray_box_pct']}%  "
        f"physics {thr['physics_only_pct']}%"
    )
    mae = report["max_dev_mae_pct"]
    print(f"\nmax-deviation MAE: physics {mae['physics_only']}%  gray-box {mae['gray_box']}%")
    rr = report["residual_report"]
    print(
        f"contribution split: physics {rr['physics_contribution_mean']:.1%} / "
        f"model {rr['model_correction_mean']:.1%}  -> {rr['verdict']}"
    )
    print(
        f"residual: mean abs {report['fit_report']['residual_mean_abs_pct']}%  "
        f"std {report['fit_report']['residual_std_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
