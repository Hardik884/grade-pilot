"""The five failure modes, expressed as perturbations to the plan and the tuning.

None of these paints a waveform onto the output. Each changes something a mill
engineer would recognise as a cause -- a mistuned loop, a mis-sequenced ramp, a
drifting consistency, a dryer without headroom -- and the excursion falls out of
the closed loop. In particular ``overshoot_from_delay`` only makes the controller
aggressive and its delay estimate optimistic; the overshoot itself is produced by
the transport delay sitting inside the feedback path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FAULT_NAMES", "FaultEffects", "TransitionContext", "sample_faults"]

FAULT_NAMES: tuple[str, ...] = (
    "overshoot_from_delay",
    "ramp_desync_filler_lead",
    "speed_first",
    "consistency_drift",
    "steam_limited_drying",
)

#: Relative weights when a fault is drawn. Roughly even, with the two most
#: characteristic grade-change modes slightly favoured.
_WEIGHTS: dict[str, float] = {
    "overshoot_from_delay": 1.25,
    "ramp_desync_filler_lead": 1.15,
    "speed_first": 1.0,
    "consistency_drift": 0.95,
    "steam_limited_drying": 0.9,
}


@dataclass(frozen=True)
class TransitionContext:
    """What the fault magnitudes have to be scaled against.

    A fixed perturbation is the wrong model. Sequencing the stock ramp 3 minutes
    late costs ``ramp_rate * 180`` g/m2, which is a nuisance on a slow 18 g/m2
    transition and drives the sheet clean out of its physical range on a fast
    115 g/m2 one. Every fault below is therefore sized to produce a *deviation of
    a chosen severity* rather than a fixed input offset.
    """

    bw_min_g_m2: float
    bw_ramp_rate_g_m2_sec: float
    delta_speed_m_min: float
    ramp_dur_sec: float


@dataclass
class FaultEffects:
    """Accumulated effect of the drawn faults. Defaults are the healthy machine."""

    kp_bw_scale: float = 1.0
    ti_bw_scale: float = 1.0
    #: Multiplies the controller's lead estimate. < 1 means it under-estimates the
    #: delay it is compensating for.
    bw_lead_scale: float = 1.0
    ff_speed_gain: float | None = None
    #: Filler/ash ramp starts this many seconds before the basis-weight ramp.
    ash_lead_sec: float = 0.0
    ash_dur_scale: float = 1.0
    #: Sequencing lag applied to the flow ramp only, so speed moves first.
    stock_delay_sec: float = 0.0
    #: Amplitude (% absolute) and period of thick-stock consistency wander.
    cons_drift_pct: float = 0.0
    cons_drift_period_sec: float = 600.0
    cons_drift_phase: float = 0.0
    #: Steam ceiling shortfall relative to what the target grade needs, in bar.
    steam_shortfall_bar: float | None = None


def sample_faults(
    rng: np.random.Generator,
    context: TransitionContext,
    *,
    fault_prob: float,
    second_fault_prob: float,
    raises_basis_weight: bool,
    changes_speed: bool,
) -> tuple[list[str], FaultEffects]:
    """Draw the faults for one episode and fold them into a :class:`FaultEffects`.

    Applicability is respected rather than forced: ``steam_limited_drying`` is the
    failure mode of *a large basis-weight increase without adequate steam headroom*,
    so it only applies when basis weight is going up, and ``speed_first`` only when
    the speed actually moves. Ineligible faults are excluded from the draw rather
    than drawn and discarded, so the fault probability stays the honest knob on the
    off-spec rate.
    """
    effects = FaultEffects()
    if rng.random() >= fault_prob:
        return [], effects

    eligible = [
        name
        for name in FAULT_NAMES
        if not (name == "steam_limited_drying" and not raises_basis_weight)
        and not (name == "speed_first" and not changes_speed)
    ]
    chosen = [_draw(rng, eligible)]
    if rng.random() < second_fault_prob:
        rest = [n for n in eligible if n != chosen[0]]
        if rest:
            chosen.append(_draw(rng, rest))

    for name in chosen:
        _apply(name, rng, effects, context)
    return sorted(chosen), effects


def _draw(rng: np.random.Generator, names: list[str]) -> str:
    w = np.array([_WEIGHTS[n] for n in names], dtype=float)
    return str(rng.choice(names, p=w / w.sum()))


def _apply(
    name: str, rng: np.random.Generator, fx: FaultEffects, ctx: TransitionContext
) -> None:
    if name == "overshoot_from_delay":
        # Aggressive loop plus an optimistic delay estimate. The overshoot is not
        # written anywhere: it emerges because the PI is acting on stale sheet.
        #
        # These bounds were originally set on the assumption that the loop's own
        # stability margin caps the result. It does not: a PI controller around a
        # dead time has no bounded overshoot, it simply goes unstable past the
        # margin. The old range drove basis weight 16-24 g/m2 beyond target -- more
        # than the entire transition on the narrower grade pairs -- and breached the
        # 15 g/m2/min plausibility limit. Sized now to overshoot by a few percent:
        # comfortably outside the 2.5% spec band, which is what makes the episode
        # off-spec and worth advising on, while staying a physical excursion rather
        # than a divergence.
        fx.kp_bw_scale *= float(rng.uniform(1.5, 2.2))
        fx.ti_bw_scale *= float(rng.uniform(0.55, 0.78))
        fx.bw_lead_scale *= float(rng.uniform(0.65, 0.85))
    elif name == "ramp_desync_filler_lead":
        # Filler ramps ahead of stock: ash arrives early, the extra filler mass
        # drags basis weight, and the stock loop chases it a delay too late. The
        # size of the drag follows the ash change, so the lead is set as a share of
        # the ramp rather than as a fixed number of seconds.
        fx.ash_lead_sec += float(rng.uniform(0.35, 0.85)) * ctx.ramp_dur_sec
        fx.ash_dur_scale *= float(rng.uniform(0.40, 0.65))
    elif name == "speed_first":
        # Speed goes on schedule, the flow ramp is sequenced late and only partly
        # feeds forward the speed change. Basis weight moves with the speed ratio.
        # The sequencing lag is sized to a target deviation: late by
        # dev_frac * bw / ramp_rate seconds costs dev_frac of basis weight.
        dev_frac = float(rng.uniform(0.030, 0.075))
        rate = max(ctx.bw_ramp_rate_g_m2_sec, 1e-4)
        fx.stock_delay_sec += float(np.clip(dev_frac * ctx.bw_min_g_m2 / rate, 30.0, 220.0))
        # The un-credited part of the speed move is bounded in m/min, not as a
        # fraction: a 3% deficit is trivial on a 60 m/min ramp and catastrophic on
        # a 900 m/min one.
        deficit = float(rng.uniform(15.0, 45.0))
        fx.ff_speed_gain = float(
            np.clip(1.0 - deficit / max(abs(ctx.delta_speed_m_min), 1.0), 0.30, 0.94)
        )
    elif name == "consistency_drift":
        fx.cons_drift_pct = float(rng.uniform(0.07, 0.20))
        fx.cons_drift_period_sec = float(rng.uniform(500.0, 1400.0))
        fx.cons_drift_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    elif name == "steam_limited_drying":
        fx.steam_shortfall_bar = float(rng.uniform(0.35, 0.95))
    else:  # pragma: no cover - guarded by FAULT_NAMES
        raise ValueError(f"unknown fault {name!r}")
