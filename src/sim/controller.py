"""MPC surrogate: constrained PI loops with model-based feedforward.

This stands in for the mill's MD multivariable MPC. It is deliberately *not* a real
MPC -- it is the smallest thing that reproduces the dynamics that matter:

* feedforward from the scheduled setpoint and speed trajectories, inverted through
  the same mass balance the process uses;
* the delay sits **inside** the feedback loop -- the PI acts on a measurement that
  is one transport delay plus half a scan period stale, which is what makes
  overshoot *emerge* rather than being painted on;
* **partial** speed feedforward, so a speed ramp leaves a residual the feedback has
  to clean up late.

Why the feedforward leads the setpoint. At the scanner,
``bw_meas(t) ~ mdot(t - theta - s - tau_wet) / (speed(t - theta - s) * trim)``
with ``s`` half a scan period. Setting ``u = t - theta - s - tau_wet`` gives
``mdot(u) = bw_sp(u + theta + s + tau_wet) * speed(u + tau_wet) * trim``:
the flow feedforward leads the basis-weight target by ``theta + s + tau_wet`` and
the speed it compensates for by ``tau_wet``. Those two leads are different, and
getting them right is what lets a real machine track a ramp at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sim.machine import DryingModel, MachineConfig, required_flows

__all__ = ["Ramp", "TransitionPlan", "ControllerTuning", "MVCommand", "MPCSurrogate"]


@dataclass(frozen=True)
class Ramp:
    """Linear setpoint ramp, clamped outside its window."""

    t_start_sec: float
    duration_sec: float
    v_from: float
    v_to: float

    def __call__(self, t_sec: float) -> float:
        if t_sec <= self.t_start_sec:
            return float(self.v_from)
        if self.duration_sec <= 0.0 or t_sec >= self.t_start_sec + self.duration_sec:
            return float(self.v_to)
        frac = (t_sec - self.t_start_sec) / self.duration_sec
        return float(self.v_from + frac * (self.v_to - self.v_from))

    @property
    def t_end_sec(self) -> float:
        return self.t_start_sec + self.duration_sec


@dataclass(frozen=True)
class TransitionPlan:
    """The scheduled grade change: quality targets at the reel, speed at the wire.

    ``bw_sp`` is the target for the *measured* sheet, so it starts ramping one
    nominal process lag after the trigger at ``t = 0``; the flows start moving at
    ``t = 0`` to make that happen. ``ash_sp`` normally shares the basis-weight
    window and leads it only under the filler-desync fault.
    """

    bw_sp: Ramp
    ash_sp: Ramp
    moist_sp: Ramp
    speed_plan: Ramp

    @property
    def ramp_end_sec(self) -> float:
        return max(self.bw_sp.t_end_sec, self.ash_sp.t_end_sec, self.speed_plan.t_end_sec)


@dataclass(frozen=True)
class ControllerTuning:
    """Everything the controller believes, correctly or otherwise."""

    kp_bw_frac: float = 0.35
    ti_bw_sec: float = 150.0
    kp_ash_frac: float = 0.25
    ti_ash_sec: float = 220.0
    kp_moist_frac: float = 0.30
    ti_moist_sec: float = 260.0
    #: Fraction of the speed change the feedforward compensates. < 1 by design.
    ff_speed_gain: float = 0.96
    #: Estimated theta + scan/2 + tau_wet. Errors here become ramp tracking error.
    bw_lead_sec: float = 60.0
    #: Estimated tau_wet: how far the flows must lead the speed they compensate.
    speed_lead_sec: float = 35.0
    moist_lead_sec: float = 120.0
    #: Extra sequencing lag on the stock ramp; the speed-first failure mode.
    stock_delay_sec: float = 0.0
    #: Consistency the controller assumes. Drift away from it biases the balance.
    cons_hat_pct: float = 3.5
    #: First-order filter on the QCS reading before it reaches the PI. A scan-held
    #: measurement steps once per traverse; feeding those steps straight into a
    #: proportional term would kick the actuator every scan. Real QCS loops filter.
    meas_filter_tau_sec: float = 15.0
    #: Integral gain multiplier while the transition is ramping. During a ramp the
    #: feedback error is dominated by how well the controller's lead estimate lines
    #: the measurement up with a moving target, not by a real bias; integrating it
    #: charges the loop with an offset that discharges as an excursion the moment
    #: the ramp ends. Mills hold or detune the integral through a grade change for
    #: exactly this reason. Proportional action is untouched, so the delay is still
    #: fully inside the loop.
    ramp_integral_scale: float = 0.25


@dataclass
class MVCommand:
    """Applied manipulated variables after rate limiting and constraint clipping."""

    stock_flow_m3_h: float
    filler_flow_L_min: float
    steam_p_bar: float
    speed_m_min: float
    stock_ff_m3_h: float = 0.0


@dataclass
class _PIState:
    integral: float = 0.0


class MPCSurrogate:
    """Three PI loops (bw<-stock, ash<-filler, moist<-steam) plus speed sequencing.

    Constraint handling is structural: every command is rate-limited to the
    ``actuator_rates`` and clipped to the ``recipe_limits`` before it reaches the
    process, and the integrator is back-calculated against what was actually
    applied so a saturated actuator cannot wind up.
    """

    def __init__(
        self,
        *,
        plan: TransitionPlan,
        tuning: ControllerTuning,
        machine: MachineConfig,
        drying: DryingModel,
        recipe_limits: dict[str, list[float]],
        actuator_rates: dict[str, float],
        filler_rate_L_min_per_min: float,
        initial: MVCommand,
        t0_sec: float = 0.0,
    ) -> None:
        self.plan = plan
        self.tuning = tuning
        self.machine = machine
        self.drying = drying
        self.limits = recipe_limits
        self.rates = actuator_rates
        self.filler_rate = float(filler_rate_L_min_per_min)
        self.mv = MVCommand(
            stock_flow_m3_h=initial.stock_flow_m3_h,
            filler_flow_L_min=initial.filler_flow_L_min,
            steam_p_bar=initial.steam_p_bar,
            speed_m_min=initial.speed_m_min,
        )
        self._bw = _PIState()
        self._ash = _PIState()
        self._moist = _PIState()
        self._speed_ref = initial.speed_m_min
        self._filtered: dict[str, float] | None = None
        # Pre-charge the integrators so the pre-roll starts genuinely at steady state
        # rather than spending two minutes removing a startup offset.
        ff = self.feedforward(t0_sec)
        self._bw.integral = initial.stock_flow_m3_h - ff[0]
        self._ash.integral = initial.filler_flow_L_min - ff[1]
        self._moist.integral = initial.steam_p_bar - ff[2]

    # -- feedforward -------------------------------------------------------------

    def feedforward(self, t_sec: float) -> tuple[float, float, float]:
        """Model-inverted feedforward ``(stock_m3_h, filler_L_min, steam_bar)``.

        Stock and filler get **different** leads, because they reach the sheet by
        different routes. Fibre set now arrives at the headbox one wet-end time
        constant later; filler is injected close to the headbox and arrives
        essentially at once. Requiring both ``bw_true(v) = bw_sp(v + theta + s)``
        and ``ash_true(v) = ash_sp(v + theta + s)`` gives

        * stock: targets led by ``tau_wet + theta + s``, against the speed it will
          meet, which is the plan led by ``tau_wet``;
        * filler: targets led by ``theta + s`` only, against the speed now.

        Leading the filler by the stock's lead instead -- the obvious shortcut --
        lands the ash change roughly one wet-end constant early. The ash loop then
        pulls filler flow away from its feedforward, the stock feedforward's
        assumption about how much mass the filler contributes stops holding, and
        basis weight drifts off in proportion to the size of the ash change.
        """
        tn = self.tuning
        # --- thick stock: the long route
        t_flow = t_sec - tn.stock_delay_sec
        speed_stock = self._partial_speed(self.plan.speed_plan(t_flow + tn.speed_lead_sec))
        stock_ff, _ = required_flows(
            self.plan.bw_sp(t_flow + tn.bw_lead_sec),
            self.plan.ash_sp(t_flow + tn.bw_lead_sec),
            speed_stock,
            tn.cons_hat_pct,
            trim_m=self.machine.trim_m,
            retention=self.machine.retention,
            filler_cons_pct=self.machine.filler_cons_pct,
        )
        # --- filler: the short route, led only by the measurement delay
        meas_lead = max(tn.bw_lead_sec - tn.speed_lead_sec, 0.0)
        speed_filler = self._partial_speed(self.plan.speed_plan(t_sec))
        _, filler_ff = required_flows(
            self.plan.bw_sp(t_sec + meas_lead),
            self.plan.ash_sp(t_sec + meas_lead),
            speed_filler,
            tn.cons_hat_pct,
            trim_m=self.machine.trim_m,
            retention=self.machine.retention,
            filler_cons_pct=self.machine.filler_cons_pct,
        )
        # --- steam: the dryer is slower still
        moist_target = self.plan.moist_sp(t_sec + tn.moist_lead_sec)
        steam_ff = float(
            self.drying.steam_p_bar(
                moist_target, self.plan.bw_sp(t_sec + tn.moist_lead_sec), speed_stock
            )
        )
        return stock_ff, filler_ff, steam_ff

    def _partial_speed(self, speed_planned: float) -> float:
        """Partial speed feedforward: only ``ff_speed_gain`` of the speed move is
        credited, so a speed ramp leaves a residual for the feedback to find late."""
        return self._speed_ref + self.tuning.ff_speed_gain * (speed_planned - self._speed_ref)

    # -- one control step --------------------------------------------------------

    def step(
        self,
        t_sec: float,
        dt_sec: float,
        *,
        bw_meas: float,
        ash_meas: float,
        moist_meas: float,
        stock_bias_m3_h: float = 0.0,
        steam_max_bar: float | None = None,
    ) -> MVCommand:
        tn = self.tuning
        stock_ff, filler_ff, steam_ff = self.feedforward(t_sec)

        bw_sp = self.plan.bw_sp(t_sec)
        ash_sp = self.plan.ash_sp(t_sec)
        moist_sp = self.plan.moist_sp(t_sec)

        bw_f, ash_f, moist_f = self._filter(bw_meas, ash_meas, moist_meas, dt_sec)
        i_scale = (
            tn.ramp_integral_scale if 0.0 <= t_sec < self.plan.ramp_end_sec else 1.0
        )

        # --- basis weight -> thick stock flow
        # Gain scheduling on the local sensitivity d(stock)/d(bw) = stock/bw.
        kp_bw = tn.kp_bw_frac * stock_ff / max(bw_sp, 1e-6)
        e_bw = bw_sp - bw_f
        self._bw.integral += kp_bw / tn.ti_bw_sec * e_bw * dt_sec * i_scale
        stock_cmd = stock_ff + kp_bw * e_bw + self._bw.integral + stock_bias_m3_h
        stock_app, stock_excess = self._apply(
            self.mv.stock_flow_m3_h, stock_cmd, self.rates["stock_flow"], "stock_flow", dt_sec
        )
        self._bw.integral += stock_excess

        # --- ash -> filler flow
        kp_ash = tn.kp_ash_frac * filler_ff / max(ash_sp, 1e-6)
        e_ash = ash_sp - ash_f
        self._ash.integral += kp_ash / tn.ti_ash_sec * e_ash * dt_sec * i_scale
        filler_cmd = filler_ff + kp_ash * e_ash + self._ash.integral
        filler_app, filler_excess = self._apply(
            self.mv.filler_flow_L_min, filler_cmd, self.filler_rate, "filler_flow", dt_sec
        )
        self._ash.integral += filler_excess

        # --- moisture -> steam pressure.
        # Moisture falls as steam rises, so the error is taken as (measured - target):
        # a wet sheet calls for more steam.
        kp_moist = tn.kp_moist_frac / self.drying.beta_pct_per_bar
        e_moist = moist_f - moist_sp
        self._moist.integral += kp_moist / tn.ti_moist_sec * e_moist * dt_sec
        steam_cmd = steam_ff + kp_moist * e_moist + self._moist.integral
        steam_app, steam_excess = self._apply(
            self.mv.steam_p_bar,
            steam_cmd,
            self.rates["steam_p"],
            "steam_p",
            dt_sec,
            hard_max=steam_max_bar,
        )
        self._moist.integral += steam_excess

        # --- speed follows its schedule directly (fast drive loop, not a PI here)
        speed_app, _ = self._apply(
            self.mv.speed_m_min, self.plan.speed_plan(t_sec), self.rates["speed"], "speed", dt_sec
        )

        self.mv = MVCommand(stock_app, filler_app, steam_app, speed_app, stock_ff)
        return self.mv

    def _filter(
        self, bw: float, ash: float, moist: float, dt_sec: float
    ) -> tuple[float, float, float]:
        tau = self.tuning.meas_filter_tau_sec
        if self._filtered is None:
            self._filtered = {"bw": bw, "ash": ash, "moist": moist}
            return bw, ash, moist
        a = float(np.exp(-dt_sec / tau)) if tau > 0.0 else 0.0
        f = self._filtered
        f["bw"] = f["bw"] * a + bw * (1.0 - a)
        f["ash"] = f["ash"] * a + ash * (1.0 - a)
        f["moist"] = f["moist"] * a + moist * (1.0 - a)
        return f["bw"], f["ash"], f["moist"]

    def _apply(
        self,
        current: float,
        command: float,
        rate_per_min: float,
        name: str,
        dt_sec: float,
        *,
        hard_max: float | None = None,
    ) -> tuple[float, float]:
        """Rate limit, then clip to the recipe envelope. Constraints bind before the
        value reaches the process, never after.

        Returns ``(applied, windup_excess)``. The excess is measured against the
        *position* envelope only. The slew limit must not feed the integrator: a
        scan-held measurement steps once per traverse, and charging that transient
        slew to the integral turns the rate limiter into a slow, one-way drift.
        """
        max_step = rate_per_min * dt_sec / 60.0
        stepped = float(np.clip(command, current - max_step, current + max_step))
        lo, hi = self.limits[name]
        if hard_max is not None:
            hi = min(hi, hard_max)
        applied = float(np.clip(stepped, lo, hi))
        excess = float(np.clip(command, lo, hi)) - command
        return applied, excess
