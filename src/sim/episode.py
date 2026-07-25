"""One grade-change episode: closed-loop simulation to ``(DataFrame, meta)``.

Planted causal structure, and the lags that separate it. A plain correlation matrix
cannot recover this, because the coordinated ramp makes every variable correlate at
lag 0; only a lagged method separates the edges:

===========================  ==================================
edge                          lag
===========================  ==================================
``speed`` -> ``bw``           theta (transport only)
``stock_flow`` -> ``bw``      theta + tau_wet (20-60 s)
``filler_flow`` -> ``ash``    theta
``steam_p`` -> ``moist``      tau_dry (60-180 s), sign negative
``moist`` -> ``bw``           small, via the total-weight measurement
===========================  ==================================

``bw_true`` and the fault list are simulator ground truth. ``bw_true`` never enters
the control loop; the controller only ever sees the delayed, scan-averaged, noisy
measurement, which is the whole reason overshoot happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.sim.controller import ControllerTuning, MPCSurrogate, MVCommand, Ramp, TransitionPlan
from src.sim.faults import FaultEffects, TransitionContext, sample_faults
from src.sim.grades import (
    ACTUATOR_RATES,
    RECIPE_LIMITS,
    GradeProps,
    operating_point,
)
from src.sim.labels import compute_labels
from src.sim.machine import (
    CaliperModel,
    DryingModel,
    MachineConfig,
    Scanner,
    VariableTransportDelay,
    ash_pct_from_flows,
    basis_weight_g_m2,
    first_order_step,
)
from src.sim.units import validate_ranges

__all__ = ["SimConfig", "PHASES", "SERIES_COLUMNS", "simulate_episode", "episode_id"]

PHASES: tuple[str, ...] = ("pre", "ramp", "settle", "steady")

#: Exact column order of ``series.parquet``.
SERIES_COLUMNS: tuple[str, ...] = (
    "bw",
    "moist",
    "ash",
    "caliper",
    "bw_true",
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

_FILLER_RATE_L_MIN_PER_MIN: float = 160.0
_MAX_MOIST_UNDER_FAULT_PCT: float = 8.7


@dataclass(frozen=True)
class SimConfig:
    """Knobs on the generator. ``fault_prob`` is the calibration knob for the
    25-45% off-spec rate; nothing else should be used to force that number."""

    dt_sec: float = 1.0
    pre_roll_sec: float = 180.0
    settle_sec: float = 420.0
    steady_sec: float = 420.0
    #: Share of the 2.5% spec band the schedule is willing to spend on scanner
    #: resolution. A scan-averaged reading held for one traverse sits anywhere from
    #: half to one and a half scan periods behind the sheet, so against a ramping
    #: target it swings by ``ramp_rate * scan_period / 2`` no matter how good the
    #: control is. Planning the ramp against that is what a mill does when it says a
    #: grade change "has to be taken slowly"; it is also why the widest transitions
    #: in the catalogue cannot be taken inside 15 minutes without making broke.
    ramp_dev_budget: float = 0.009
    speed_rate_use_frac: float = 0.85
    min_ramp_sec: float = 180.0
    max_ramp_sec: float = 900.0
    #: Calibrated so the dataset lands mid-band on the required 25-45% off-spec
    #: rate. This is the only knob that should ever be used to move that number.
    fault_prob: float = 0.55
    second_fault_prob: float = 0.16
    operator_action_prob: float = 0.28
    #: How strongly a moisture excursion shows up in the total basis weight the QCS
    #: reports. 1.0 means +1% moisture reads as +1% basis weight.
    moist_bw_coupling: float = 1.0
    #: Fraction of a caliper excursion that survives the calender. Caliper is a
    #: separately controlled quality with its own loop, so it does not follow basis
    #: weight one for one; the bw -> caliper and ash -> caliper edges stay, damped.
    calender_pass_through: float = 0.55


def episode_id(grade_from: str, grade_to: str, seq: int) -> str:
    return f"EP-{grade_from}-{grade_to}-{seq:04d}"


def simulate_episode(
    grade_from: GradeProps,
    grade_to: GradeProps,
    *,
    seed: int,
    seq: int = 1,
    machine: MachineConfig | None = None,
    config: SimConfig | None = None,
    drying: DryingModel | None = None,
    caliper_model: CaliperModel | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one grade change end to end.

    ``seed`` is an explicit argument and the only source of randomness; the same
    seed reproduces the episode byte for byte.
    """
    cfg = config or SimConfig()
    mac = machine or MachineConfig()
    dry = drying or DryingModel()
    cal = caliper_model or CaliperModel()
    rng = np.random.default_rng(seed)

    # ---- per-episode process character ------------------------------------------
    tau_wet_sec = float(rng.uniform(20.0, 60.0))
    tau_dry_sec = float(rng.uniform(60.0, 180.0))
    scan_period_sec = float(rng.uniform(20.0, 45.0))
    # Per-point sensor noise, straight from the process reference.
    sensor_noise_std = np.array(
        [
            rng.uniform(0.30, 0.80),  # bw, g/m2
            rng.uniform(0.04, 0.10),  # moist, %
            rng.uniform(0.15, 0.35),  # ash, %
            rng.uniform(0.6, 1.6),  # caliper, um
        ],
        dtype=float,
    )
    # Residual scan-to-scan variation that the traverse average does not remove.
    scan_noise_std = np.array(
        [
            rng.uniform(0.06, 0.16),
            rng.uniform(0.02, 0.05),
            rng.uniform(0.06, 0.14),
            rng.uniform(0.30, 0.80),
        ],
        dtype=float,
    )
    cons_nominal_pct = float(np.clip(rng.normal(3.5, 0.10), 3.25, 3.75))
    moist_distance_m = mac.scanner_distance_m * float(rng.uniform(0.30, 0.42))

    op_a = operating_point(grade_from, mac, stock_cons_pct=cons_nominal_pct, drying=dry)
    op_b = operating_point(grade_to, mac, stock_cons_pct=cons_nominal_pct, drying=dry)

    # ---- transition schedule --------------------------------------------------------
    # Scheduled before the faults are drawn, because fault severity is scaled
    # against the ramp it perturbs.
    bw_min = min(grade_from.bw, grade_to.bw)
    d_bw = (
        abs(grade_to.bw - grade_from.bw)
        * (scan_period_sec / 2.0)
        / (cfg.ramp_dev_budget * bw_min)
    )
    d_speed = (
        abs(grade_to.nominal_speed_m_min - grade_from.nominal_speed_m_min)
        / (cfg.speed_rate_use_frac * ACTUATOR_RATES["speed"])
        * 60.0
    )
    ramp_dur_sec = float(
        np.clip(
            max(d_bw, d_speed, cfg.min_ramp_sec) * rng.uniform(0.95, 1.15),
            cfg.min_ramp_sec,
            cfg.max_ramp_sec,
        )
    )

    # ---- faults -------------------------------------------------------------------
    delta_speed = grade_to.nominal_speed_m_min - grade_from.nominal_speed_m_min
    faults, fx = sample_faults(
        rng,
        TransitionContext(
            bw_min_g_m2=bw_min,
            bw_ramp_rate_g_m2_sec=abs(grade_to.bw - grade_from.bw) / ramp_dur_sec,
            delta_speed_m_min=delta_speed,
            ramp_dur_sec=ramp_dur_sec,
        ),
        fault_prob=cfg.fault_prob,
        second_fault_prob=cfg.second_fault_prob,
        raises_basis_weight=grade_to.bw > grade_from.bw,
        changes_speed=abs(delta_speed) > 40.0,
    )

    speed_mean = 0.5 * (grade_from.nominal_speed_m_min + grade_to.nominal_speed_m_min)
    theta_nom_sec = 60.0 * mac.scanner_distance_m / speed_mean
    # Effective measurement delay is theta plus a full scan period: half from the
    # traverse average being centred, half from the reading being held until the
    # next traverse completes.
    meas_lead_nom_sec = theta_nom_sec + scan_period_sec
    lag_nom_sec = meas_lead_nom_sec + tau_wet_sec

    ash_start = max(lag_nom_sec - fx.ash_lead_sec, 5.0)
    plan = TransitionPlan(
        bw_sp=Ramp(lag_nom_sec, ramp_dur_sec, grade_from.bw, grade_to.bw),
        ash_sp=Ramp(ash_start, ramp_dur_sec * fx.ash_dur_scale, grade_from.ash, grade_to.ash),
        moist_sp=Ramp(lag_nom_sec, ramp_dur_sec, grade_from.moist, grade_to.moist),
        speed_plan=Ramp(
            tau_wet_sec,
            ramp_dur_sec,
            grade_from.nominal_speed_m_min,
            grade_to.nominal_speed_m_min,
        ),
    )

    # Partial speed feedforward. The deficit acts on the *absolute* speed change, so
    # a healthy machine can only afford a percent or so: on the widest transition in
    # the catalogue (895 m/min) even 1.5% is a 2.7% basis-weight error, right at the
    # spec band. The large deficits belong to the speed-first fault, not to baseline.
    ff_gain = (
        fx.ff_speed_gain
        if fx.ff_speed_gain is not None
        else float(np.clip(rng.normal(0.992, 0.006), 0.980, 1.0))
    )
    tuning = ControllerTuning(
        kp_bw_frac=float(rng.uniform(0.38, 0.55)) * fx.kp_bw_scale,
        ti_bw_sec=float(rng.uniform(105.0, 155.0)) * fx.ti_bw_scale,
        kp_ash_frac=float(rng.uniform(0.20, 0.32)),
        ti_ash_sec=float(rng.uniform(190.0, 260.0)),
        kp_moist_frac=float(rng.uniform(0.24, 0.38)),
        ti_moist_sec=float(rng.uniform(220.0, 320.0)),
        ff_speed_gain=ff_gain,
        bw_lead_sec=lag_nom_sec * float(rng.normal(1.0, 0.12)) * fx.bw_lead_scale,
        speed_lead_sec=tau_wet_sec * float(rng.normal(1.0, 0.15)),
        moist_lead_sec=tau_dry_sec * float(rng.normal(1.0, 0.15)),
        stock_delay_sec=fx.stock_delay_sec,
        cons_hat_pct=cons_nominal_pct,
        meas_filter_tau_sec=scan_period_sec / 2.0,
    )

    steam_max_bar = _steam_ceiling(fx, grade_to, op_a, op_b, dry)

    # ---- time grid ------------------------------------------------------------------
    dt = cfg.dt_sec
    ramp_end_sec = plan.ramp_end_sec
    t_end = ramp_end_sec + cfg.settle_sec + cfg.steady_sec
    n = int(round((t_end + cfg.pre_roll_sec) / dt)) + 1
    t_grid = -cfg.pre_roll_sec + np.arange(n, dtype=float) * dt

    # ---- initial steady state --------------------------------------------------------
    bw0 = grade_from.bw
    ash0 = grade_from.ash
    moist0 = grade_from.moist
    cal0 = float(cal.caliper_um(bw0, ash0))

    controller = MPCSurrogate(
        plan=plan,
        tuning=tuning,
        machine=mac,
        drying=dry,
        recipe_limits=RECIPE_LIMITS,
        actuator_rates=ACTUATOR_RATES,
        filler_rate_L_min_per_min=_FILLER_RATE_L_MIN_PER_MIN,
        initial=MVCommand(
            op_a.stock_flow_m3_h, op_a.filler_flow_L_min, op_a.steam_p_bar, op_a.speed_m_min
        ),
        t0_sec=-cfg.pre_roll_sec,
    )
    sheet_delay = VariableTransportDelay(
        mac.scanner_distance_m,
        capacity=n,
        n_signals=3,
        t0_sec=t_grid[0],
        dt_sec=dt,
        initial_speed_m_min=op_a.speed_m_min,
        initial_values=[bw0, ash0, cal0],
    )
    dryer_delay = VariableTransportDelay(
        moist_distance_m,
        capacity=n,
        n_signals=1,
        t0_sec=t_grid[0],
        dt_sec=dt,
        initial_speed_m_min=op_a.speed_m_min,
        initial_values=[moist0],
    )
    scanner = Scanner(
        scan_period_sec,
        sensor_noise_std,
        scan_noise_std,
        rng,
        dt_sec=dt,
        initial_values=[bw0, moist0, ash0, cal0],
    )

    stock_headbox = op_a.stock_flow_m3_h
    moist_true = moist0
    meas = np.array([bw0, moist0, ash0, cal0], dtype=float)

    # Consistency disturbance: a slow OU wander always present, plus the drift fault.
    cons_ou = 0.0
    ou_tau, ou_sigma = 240.0, 0.016
    ou_a = float(np.exp(-dt / ou_tau))
    ou_kick = ou_sigma * np.sqrt(1.0 - ou_a**2)
    cons_noise = rng.normal(0.0, 1.0, size=n)

    # Operator intervention: armed for the settle window, fires only if the deviation
    # is actually visible to the operator.
    op_armed = rng.random() < cfg.operator_action_prob
    op_watch_from = ramp_end_sec + float(rng.uniform(30.0, cfg.settle_sec * 0.6))
    op_bias_dur = float(rng.uniform(120.0, 240.0))
    op_bias_frac = float(rng.uniform(0.02, 0.05))
    op_fired_at: float | None = None
    op_bias = 0.0

    out = {name: np.empty(n, dtype=float) for name in SERIES_COLUMNS[:14]}
    op_action = np.full(n, None, dtype=object)

    for k in range(n):
        t = float(t_grid[k])

        bw_sp = plan.bw_sp(t)
        ash_sp = plan.ash_sp(t)
        moist_sp = plan.moist_sp(t)

        # --- thick stock consistency (slow-varying disturbance)
        cons_ou = cons_ou * ou_a + ou_kick * float(cons_noise[k])
        drift = 0.0
        if fx.cons_drift_pct > 0.0 and t > 0.0:
            gate = min(t / 120.0, 1.0)
            drift = (
                fx.cons_drift_pct
                * gate
                * float(np.sin(2.0 * np.pi * t / fx.cons_drift_period_sec + fx.cons_drift_phase))
            )
        stock_cons = cons_nominal_pct + cons_ou + drift

        # --- operator intervention
        if op_armed and op_fired_at is None and t >= op_watch_from:
            dev = (bw_sp - meas[0]) / bw_sp
            if abs(dev) > 0.012:
                op_fired_at = t
                op_bias = float(np.sign(dev)) * op_bias_frac * op_a.stock_flow_m3_h
                op_action[k] = "manual_stock_bias_up" if dev > 0 else "manual_stock_bias_down"
        bias = op_bias if (op_fired_at is not None and t - op_fired_at < op_bias_dur) else 0.0
        if k > 0 and t_grid[k - 1] < 0.0 <= t:
            op_action[k] = "grade_change_start"

        # --- control (acts on the stale measurement: the delay is inside the loop)
        mv = controller.step(
            t,
            dt,
            bw_meas=float(meas[0]),
            ash_meas=float(meas[2]),
            moist_meas=float(meas[1]),
            stock_bias_m3_h=bias,
            steam_max_bar=steam_max_bar,
        )

        # --- stage 1: mass balance from the instantaneous manipulated variables.
        # bw_ideal is not stored because it is a pure function of columns that are:
        # basis_weight_g_m2(stock_flow, stock_cons, filler_flow, speed). Recompute it
        # from the frame whenever the un-lagged value is wanted.
        # --- stage 2: wet-end first-order lag on the thick stock delivery
        stock_headbox = first_order_step(stock_headbox, mv.stock_flow_m3_h, tau_wet_sec, dt)
        bw_dry = float(
            basis_weight_g_m2(
                stock_headbox,
                stock_cons,
                mv.filler_flow_L_min,
                mv.speed_m_min,
                trim_m=mac.trim_m,
                retention=mac.retention,
                filler_cons_pct=mac.filler_cons_pct,
            )
        )
        ash_true = float(
            ash_pct_from_flows(
                stock_headbox, stock_cons, mv.filler_flow_L_min, mac.filler_cons_pct
            )
        )

        # dryer response, slower than the wet end: this is why moisture recovers last
        moist_ss = float(dry.moisture_pct(mv.steam_p_bar, bw_dry, mv.speed_m_min))
        moist_true = first_order_step(moist_true, moist_ss, tau_dry_sec, dt)

        # The QCS reports total basis weight, so a moisture excursion reads as a
        # basis-weight excursion. At target moisture this term is exactly zero and
        # bw_true is the plain mass balance.
        bw_true = bw_dry * (1.0 + cfg.moist_bw_coupling * (moist_true - moist_sp) / 100.0)
        # Caliper: the sheet's own bulk, pulled back toward the grade target by the
        # calender loop. The target is read at the measurement lead so that it lines
        # up with the sheet being formed now, not with what is at the scanner.
        cal_open = float(cal.caliper_um(bw_true, ash_true))
        cal_ref = float(
            cal.caliper_um(plan.bw_sp(t + meas_lead_nom_sec), plan.ash_sp(t + meas_lead_nom_sec))
        )
        cal_true = cal_ref + cfg.calender_pass_through * (cal_open - cal_ref)

        # --- stage 3: variable transport delay (speed-dependent, exact in transit)
        delayed = sheet_delay.step(t, mv.speed_m_min, dt, (bw_true, ash_true, cal_true))
        delayed_moist = dryer_delay.step(t, mv.speed_m_min, dt, (moist_true,))

        # --- stage 4: scanner traverse average + zero-order hold + noise
        meas = scanner.step(
            dt, (delayed[0], delayed_moist[0], delayed[1], delayed[2])
        )

        out["bw"][k] = meas[0]
        out["moist"][k] = meas[1]
        out["ash"][k] = meas[2]
        out["caliper"][k] = meas[3]
        out["bw_true"][k] = bw_true
        out["bw_sp"][k] = bw_sp
        out["moist_sp"][k] = moist_sp
        out["ash_sp"][k] = ash_sp
        out["stock_flow"][k] = mv.stock_flow_m3_h
        out["filler_flow"][k] = mv.filler_flow_L_min
        out["steam_p"][k] = mv.steam_p_bar
        out["speed"][k] = mv.speed_m_min
        out["stock_cons"][k] = stock_cons
        out["theta_sec"][k] = sheet_delay.theta_sec

    df = _assemble(out, t_grid, plan, ramp_end_sec, cfg, op_action)
    validate_ranges(
        {name: df[name].to_numpy() for name in
         ("bw", "moist", "ash", "caliper", "stock_flow", "stock_cons",
          "filler_flow", "steam_p", "speed")},
        context="episode series",
    )

    labels = compute_labels(df, trim_m=mac.trim_m, dt_sec=dt)
    meta: dict[str, Any] = {
        "episode_id": episode_id(grade_from.code, grade_to.code, seq),
        "grade_from": grade_from.code,
        "grade_to": grade_to.code,
        "grade_from_props": grade_from.props(),
        "grade_to_props": grade_to.props(),
        "machine": mac.as_meta(),
        "recipe_limits": {k: list(v) for k, v in RECIPE_LIMITS.items()},
        "actuator_rates": dict(ACTUATOR_RATES),
        "injected_faults": faults,
        "seed": int(seed),
        "labels": dict(labels),
    }
    return df, meta


def _steam_ceiling(
    fx: FaultEffects,
    grade_to: GradeProps,
    op_a: Any,
    op_b: Any,
    dry: DryingModel,
) -> float | None:
    """Steam ceiling for the steam-limited-drying fault, capped so the resulting
    moisture stays inside its canonical range. A fault that pushes a variable out of
    range is a bug, not a harder fault."""
    if fx.steam_shortfall_bar is None:
        return None
    headroom_pct = _MAX_MOIST_UNDER_FAULT_PCT - grade_to.moist
    max_shortfall = headroom_pct / dry.beta_pct_per_bar
    shortfall = min(fx.steam_shortfall_bar, max_shortfall)
    if shortfall < 0.2:
        return None
    ceiling = op_b.steam_p_bar - shortfall
    lo = RECIPE_LIMITS["steam_p"][0]
    return float(max(ceiling, lo + 0.05))


def _assemble(
    out: dict[str, np.ndarray],
    t_grid: np.ndarray,
    plan: TransitionPlan,
    ramp_end_sec: float,
    cfg: SimConfig,
    op_action: np.ndarray,
) -> pd.DataFrame:
    index = pd.Index(t_grid, name="t_sec", dtype="float64")
    df = pd.DataFrame({name: out[name] for name in SERIES_COLUMNS[:14]}, index=index)

    phase = np.full(t_grid.size, "steady", dtype=object)
    phase[t_grid < 0.0] = "pre"
    phase[(t_grid >= 0.0) & (t_grid < ramp_end_sec)] = "ramp"
    phase[(t_grid >= ramp_end_sec) & (t_grid < ramp_end_sec + cfg.settle_sec)] = "settle"
    df["phase"] = pd.Categorical(phase, categories=list(PHASES))
    df["op_action"] = pd.array(op_action, dtype="string")
    df["alarm"] = pd.array(_alarms(df), dtype="string")
    return df[list(SERIES_COLUMNS)]


def _alarms(df: pd.DataFrame) -> np.ndarray:
    """DCS alarm tags, derived only from measured values and active setpoints."""
    bw_dev = np.abs(df["bw"].to_numpy() - df["bw_sp"].to_numpy()) / df["bw_sp"].to_numpy()
    moist_dev = np.abs(df["moist"].to_numpy() - df["moist_sp"].to_numpy())
    ash_dev = np.abs(df["ash"].to_numpy() - df["ash_sp"].to_numpy())
    tags = np.full(len(df), None, dtype=object)
    tags[ash_dev > 1.5] = "ASH_DEV"
    tags[moist_dev > 0.6] = "MOIST_DEV"
    tags[bw_dev > 0.015] = "BW_DEV_WARN"
    tags[bw_dev > 0.025] = "BW_DEV_HIGH"
    return tags
