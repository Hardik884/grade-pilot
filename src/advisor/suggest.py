"""Constrained counterfactual advisory: what to move, how far, and never past a limit.

Shape of the answer
-------------------
1. Fire only when the predictor says the sheet is going off spec. No breach, no advice.
2. Retrieve ``k`` similar transitions in grade-property space and keep the ones that
   stayed in spec.
3. Compare their manipulated-variable *progress* against this episode's at the same
   ``t_sec``, pick the one variable with the largest consistent difference, tie-broken
   toward the shorter causal lag - a variable that acts in 40 s is worth more at 90 s
   into a transition than one that acts in 80 s. Only differences that point the way the
   mass balance needs are eligible: see :func:`corrective_direction`. History says which
   knob and how far; physics says which way. Where the two disagree, there is no advice.
4. Build candidate moves toward what the successful neighbours did.
5. **Filter every candidate against this episode's own recipe limits and actuator rates
   before anything is scored.** See :func:`admissible`.
6. Score only the survivors, and report the expected outcome from the neighbours.

On step 5: the constraint filter is structural, not a ranking penalty. Candidates enter
scoring as a separate type (:class:`ScoredCandidate`) that can only be constructed from a
candidate that already passed :func:`admissible`, so "unsafe but ranked last" is not a
state this module can represent.

Progress, not absolute value
----------------------------
Neighbour MV trajectories cannot be compared in absolute units: the catalogue spans
505-1400 m/min and stock flows differ by more than 2x across grades, so G10 -> G11's stock
flow at 90 s says nothing about G01 -> G03's. What transfers is *fractional progress from
the episode's own pre-transition operating point*, ``(v(t) - v(0)) / v(0)``. "The ones that
worked had stock flow 6.2% up by now, you are 1.8% up" is a comparison that holds across
grades and is also how an operator would put it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.advisor.retrieval import (
    DEFAULT_K,
    Neighbour,
    failed,
    find_similar,
    succeeded,
)
from src.analysis import loader
from src.analysis.loader import SPEC_BAND
from src.analysis.predictor import FILLER_RATE_L_MIN_PER_MIN, project_physics
from src.sim.units import m3_h_to_L_min

__all__ = [
    "ADVISABLE",
    "UNITS",
    "MOVE_FRACTIONS",
    "MIN_CONSISTENCY",
    "MIN_EFFECT_PCT",
    "bw_effect_pct",
    "Candidate",
    "ScoredCandidate",
    "DiscardedCandidate",
    "VariableDiff",
    "Suggestion",
    "actuator_rate_per_min",
    "admissible",
    "signed_deviation_pct",
    "corrective_direction",
    "variable_diffs",
    "select_variable",
    "build_candidates",
    "suggest",
]

#: Variables the advisor may propose moving. ``stock_cons`` is excluded: it is a
#: slow-varying disturbance with no recipe limit and no actuator, so it is not advice.
ADVISABLE: tuple[str, ...] = ("stock_flow", "filler_flow", "steam_p", "speed")

#: Sign of d(bw)/d(variable) from the mass balance. Both flows sit in the numerator, speed
#: sits in the denominator. ``steam_p`` is 0 because it acts on moisture, not on delivered
#: mass - it correlates with basis weight during a transition only because everything ramps
#: together, and the claim being corrected here is always a basis-weight breach. Treating a
#: correlation as a lever is exactly the mistake the project's causal work exists to avoid.
BW_SENSITIVITY_SIGN: dict[str, int] = {
    "stock_flow": 1,
    "filler_flow": 1,
    "steam_p": 0,
    "speed": -1,
}

#: Canonical units, from the process reference, plus the resolution a setpoint is actually
#: entered at. ``display`` avoids ASCII digits so the narration numeral validator never
#: trips over a unit string, and ``decimals`` keeps proposals at operator resolution: a
#: recommendation of 428.5 L/min is not a number anyone types into a DCS.
UNITS: dict[str, dict[str, Any]] = {
    "stock_flow": {"canonical": "m3/h", "display": "m³/h", "decimals": 0},
    "filler_flow": {"canonical": "L/min", "display": "L/min", "decimals": 0},
    "steam_p": {"canonical": "bar", "display": "bar", "decimals": 2},
    "speed": {"canonical": "m/min", "display": "m/min", "decimals": 0},
}

#: Candidates are generated as fractions of the full neighbour-median move. A large move
#: that breaks a limit is discarded and a smaller one can still survive, which is what an
#: operator would do rather than abandoning the correction entirely.
MOVE_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

#: A difference counts only if this share of successful neighbours agree on its sign.
MIN_CONSISTENCY: float = 0.6

#: Smallest basis-weight effect worth an operator's attention, % - a quarter of the 2.5%
#: spec band. Imitating the neighbours can produce arithmetically real but useless moves
#: ("filler back 2 L/min" against a 3.2% excursion); those are not advice, they are noise
#: with a confidence interval attached, and issuing them is how an advisory system gets
#: switched off.
MIN_EFFECT_PCT: float = SPEC_BAND * 100.0 / 4.0

#: Window the move is executed over, seconds. Sized to the composed measurement lag: a
#: correction that takes longer than the operator's blind window is not a correction.
DEFAULT_MOVE_WINDOW_SEC: float = 60.0

#: ``meta.json`` carries no filler-flow rate; the projection assumes this one, and the
#: advisor imports it rather than re-declaring a second truth.
_IMPLIED_RATES: dict[str, float] = {"filler_flow": FILLER_RATE_L_MIN_PER_MIN}

_RANKING_PATH = Path("data/impact_ranking.json")


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------


def actuator_rate_per_min(meta: Mapping[str, Any], variable: str) -> float:
    """Maximum rate of change per minute for ``variable`` on **this** episode.

    Read from the episode's own ``meta.json``, never a module default: the limits vary
    between episodes and a hardcoded bound would eventually authorise a move the machine
    in question cannot make.
    """
    rates = meta.get("actuator_rates", {})
    if variable in rates:
        return float(rates[variable])
    if variable in _IMPLIED_RATES:
        return float(_IMPLIED_RATES[variable])
    raise KeyError(f"no actuator rate available for {variable!r}")


def recipe_limits(meta: Mapping[str, Any], variable: str) -> tuple[float, float]:
    lo, hi = meta["recipe_limits"][variable]
    return float(lo), float(hi)


@dataclass(frozen=True)
class Candidate:
    """A proposed move. Carries no score - scoring happens elsewhere, and only after
    :func:`admissible` has passed."""

    variable: str
    v_from: float
    v_to: float
    window_sec: float
    move_fraction: float

    @property
    def delta(self) -> float:
        return self.v_to - self.v_from

    @property
    def ramp_rate_per_min(self) -> float:
        return self.delta / (self.window_sec / 60.0)

    @property
    def unit(self) -> str:
        return UNITS[self.variable]["canonical"]


@dataclass(frozen=True)
class DiscardedCandidate:
    """A candidate that failed the constraint filter. Has no score field, by design."""

    candidate: Candidate
    reason: str
    detail: str


@dataclass(frozen=True)
class ScoredCandidate:
    """An admissible candidate with a rank score.

    Constructible only through :meth:`of`, which re-checks admissibility. This is the
    single door between the constraint filter and the ranking.
    """

    candidate: Candidate
    score: float
    constraint_report: dict[str, Any]

    @classmethod
    def of(cls, candidate: Candidate, meta: Mapping[str, Any], score: float) -> "ScoredCandidate":
        ok, report = admissible(candidate, meta)
        if not ok:
            raise ValueError(
                f"refusing to score an inadmissible candidate: {report['reason']} - "
                f"{report['detail']}"
            )
        return cls(candidate=candidate, score=float(score), constraint_report=report)


def admissible(candidate: Candidate, meta: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Does the candidate respect this episode's recipe limits and actuator rates?

    Both checks are hard. Returns the report either way so a rejection is explainable.
    """
    lo, hi = recipe_limits(meta, candidate.variable)
    rate_limit = actuator_rate_per_min(meta, candidate.variable)
    rate = abs(candidate.ramp_rate_per_min)

    if not (lo <= candidate.v_to <= hi):
        return False, {
            "reason": "recipe_limit",
            "detail": (
                f"{candidate.variable} target {candidate.v_to:.1f} outside "
                f"[{lo:.1f}, {hi:.1f}] {candidate.unit}"
            ),
            "recipe_limits": "fail",
            "actuator_rate": "not_reached",
            "limit_lo": lo,
            "limit_hi": hi,
            "rate_limit_per_min": rate_limit,
        }
    if rate > rate_limit + 1e-9:
        return False, {
            "reason": "actuator_rate",
            "detail": (
                f"{candidate.variable} needs {rate:.1f} {candidate.unit}/min, "
                f"actuator allows {rate_limit:.1f}"
            ),
            "recipe_limits": "pass",
            "actuator_rate": "fail",
            "limit_lo": lo,
            "limit_hi": hi,
            "rate_limit_per_min": rate_limit,
        }

    span = hi - lo
    headroom_used = abs(candidate.delta) / span if span > 0 else 0.0
    bound = "upper" if candidate.delta >= 0 else "lower"
    bound_value = hi if candidate.delta >= 0 else lo
    return True, {
        "reason": "ok",
        "detail": (
            f"{candidate.variable} {bound} limit {bound_value:.1f} {candidate.unit}; "
            f"proposal uses {headroom_used * 100:.0f}% of the range"
        ),
        "recipe_limits": "pass",
        "actuator_rate": "pass",
        "limit_lo": lo,
        "limit_hi": hi,
        "bound": bound,
        "bound_value": bound_value,
        "headroom_used_pct": round(headroom_used * 100.0, 1),
        "rate_limit_per_min": rate_limit,
        "rate_used_per_min": round(rate, 2),
        "rate_used_pct": round(rate / rate_limit * 100.0, 1) if rate_limit else 0.0,
    }


# --------------------------------------------------------------------------------------
# Which variable
# --------------------------------------------------------------------------------------


def _causal_edges(path: str | Path = _RANKING_PATH) -> dict[str, dict[str, Any]]:
    """Deviation-affecting edges from the discovered lagged graph, by variable.

    Only ``kind == "deviation"`` records are used. The stabilisation records exist but
    none reach significance at 0.05, and the ranking file says so; leaning on them would
    be dressing noise up as evidence.
    """
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(rec["variable"]): rec
        for rec in payload.get("rankings", [])
        if rec.get("kind") == "deviation"
    }


def _progress(series: pd.DataFrame, variable: str, t_now: float) -> float | None:
    """Fractional change in ``variable`` from the pre-transition point up to ``t_now``."""
    if variable not in series.columns:
        return None
    idx = series.index.to_numpy(dtype=float)
    post = series.loc[(idx >= 0.0) & (idx <= t_now), variable].to_numpy(dtype=float)
    if post.size < 2:
        return None
    base = float(post[0])
    if abs(base) < 1e-9:
        return None
    return float(post[-1] - base) / base


@dataclass(frozen=True)
class VariableDiff:
    """How this episode's handling of one variable differs from the ones that worked."""

    variable: str
    current_progress: float
    neighbour_progress: float
    diff: float
    consistency: float
    n_agree: int
    n_compared: int
    strength: float
    best_lag_sec: float | None
    corrective: bool = True
    effect_pct: float = 0.0

    @property
    def material(self) -> bool:
        """Would closing this gap actually move basis weight enough to matter?"""
        return abs(self.effect_pct) >= MIN_EFFECT_PCT

    @property
    def score(self) -> float:
        """Magnitude of the consistent difference, weighted by causal strength.

        Consistency gates rather than scales: a difference half the neighbours disagree
        about is not evidence, and :func:`select_variable` drops it.
        """
        return abs(self.diff) * self.consistency * max(self.strength, 0.05)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "current_progress_pct": round(self.current_progress * 100.0, 2),
            "successful_progress_pct": round(self.neighbour_progress * 100.0, 2),
            "diff_pct": round(self.diff * 100.0, 2),
            "consistency": round(self.consistency, 2),
            "n_agree": self.n_agree,
            "n_compared": self.n_compared,
            "strength": self.strength,
            "best_lag_sec": self.best_lag_sec,
            "corrective": self.corrective,
            "effect_pct": round(self.effect_pct, 3),
            "material": self.material,
            "score": round(self.score, 5),
        }


def signed_deviation_pct(
    prediction: Mapping[str, Any],
    series_so_far: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    target_speed_m_min: float | None = None,
) -> float:
    """Predicted deviation with its sign restored, %.

    ``Predictor.predict`` reports ``predicted_max_dev_pct`` as the *magnitude* of the worst
    projected excursion - it is built from ``max(abs(dev))`` plus a residual correction. The
    advisor cannot use a magnitude: whether the sheet is light or heavy decides which way
    every actuator must move, and getting it backwards turns a correction into the fault.

    So the magnitude comes from the gray-box and the sign comes from the physics
    projection's ``signed_dev_at_max_pct``, which is the same excursion with its direction
    intact. If a future predictor reports the signed value itself, that is preferred.
    """
    magnitude = abs(float(prediction.get("predicted_max_dev_pct", 0.0)))
    signed = prediction.get("physics_signed_dev_pct")
    if signed is None:
        signed = project_physics(
            series_so_far, meta, target_speed_m_min=target_speed_m_min
        ).signed_dev_at_max_pct
    sign = -1.0 if float(signed) < 0.0 else 1.0
    return sign * magnitude


def bw_effect_pct(
    variable: str,
    delta: float,
    series_so_far: pd.DataFrame,
    meta: Mapping[str, Any],
) -> float:
    """Estimated effect of moving ``variable`` by ``delta`` on basis weight, %.

    Straight off the mass balance, linearised at the current operating point: the flows sit
    in the numerator so their effect is their share of delivered dry mass, and speed sits in
    the denominator. This is what makes a proposal's size meaningful rather than merely
    non-zero.
    """
    window = series_so_far.loc[series_so_far.index >= 0.0]
    row = window.iloc[-1]
    cons = float(row["stock_cons"])
    filler_cons = float(meta["machine"]["filler_cons_pct"])
    mass_now = (
        m3_h_to_L_min(float(row["stock_flow"])) * cons * 10.0
        + float(row["filler_flow"]) * filler_cons * 10.0
    )
    if variable == "stock_flow":
        return (m3_h_to_L_min(delta) * cons * 10.0) / mass_now * 100.0 if mass_now else 0.0
    if variable == "filler_flow":
        return (delta * filler_cons * 10.0) / mass_now * 100.0 if mass_now else 0.0
    if variable == "speed":
        speed_now = float(row["speed"])
        return -delta / speed_now * 100.0 if speed_now else 0.0
    return 0.0


def corrective_direction(variable: str, dev_pct: float) -> int:
    """Which way ``variable`` must move to pull basis weight back toward setpoint.

    ``+1`` up, ``-1`` down, ``0`` if the variable is not a basis-weight lever. This is the
    mass balance talking, not history: a sheet running heavy needs less delivered mass or
    more speed, whatever the neighbours happened to do.
    """
    sensitivity = BW_SENSITIVITY_SIGN.get(variable, 0)
    if sensitivity == 0 or dev_pct == 0.0:
        return 0
    required_bw_sign = -1 if dev_pct > 0 else 1
    return required_bw_sign * sensitivity


def variable_diffs(
    series_so_far: pd.DataFrame,
    winners: Sequence[Neighbour],
    meta: Mapping[str, Any],
    *,
    t_now: float,
    dev_pct: float,
    root: str | Path = "data/episodes",
    edges: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[VariableDiff]:
    """Per-variable comparison against the neighbours that stayed in spec.

    Each neighbour's series is read only up to ``t_now``, matching what is knowable now.
    ``dev_pct`` is the predicted signed deviation and decides which direction counts as
    corrective; a difference that points the wrong way is recorded but marked
    ``corrective=False`` so the reasoning stays visible on the card.
    """
    edges = edges if edges is not None else _causal_edges()
    winner_series = [loader.load_series(Path(root) / n.episode_id) for n in winners]
    idx = series_so_far.index.to_numpy(dtype=float)

    out: list[VariableDiff] = []
    for var in ADVISABLE:
        mine = _progress(series_so_far, var, t_now)
        if mine is None:
            continue
        theirs = [p for s in winner_series if (p := _progress(s, var, t_now)) is not None]
        if not theirs:
            continue
        diffs = np.array([p - mine for p in theirs], dtype=float)
        median_diff = float(np.median(diffs))
        if abs(median_diff) < 1e-9:
            continue
        n_agree = int(np.sum(np.sign(diffs) == np.sign(median_diff)))
        edge = edges.get(var, {})
        needed = corrective_direction(var, dev_pct)
        v_start = float(series_so_far.loc[(idx >= 0.0) & (idx <= t_now), var].to_numpy()[0])
        effect = bw_effect_pct(var, median_diff * v_start, series_so_far, meta)
        out.append(
            VariableDiff(
                variable=var,
                current_progress=mine,
                neighbour_progress=float(np.median(theirs)),
                diff=median_diff,
                consistency=n_agree / len(diffs),
                n_agree=n_agree,
                n_compared=len(diffs),
                strength=float(edge.get("strength", 0.0)),
                best_lag_sec=(
                    float(edge["best_lag_sec"]) if edge.get("best_lag_sec") is not None else None
                ),
                corrective=bool(needed != 0 and np.sign(median_diff) == needed),
                effect_pct=effect,
            )
        )
    return sorted(out, key=lambda d: d.score, reverse=True)


def select_variable(diffs: Sequence[VariableDiff], *, tie_tolerance: float = 0.15) -> VariableDiff | None:
    """The single variable to advise on.

    Largest consistent difference wins, among variables whose difference points the
    corrective way *and* is large enough to move basis weight by a material amount. Where
    two are within ``tie_tolerance`` of each other, the shorter ``best_lag_sec`` breaks the
    tie: it is actionable sooner, which at 90 s into a transition is the whole game.
    """
    usable = [
        d for d in diffs if d.corrective and d.material and d.consistency >= MIN_CONSISTENCY
    ]
    if not usable:
        return None
    best = max(d.score for d in usable)
    if best <= 0.0:
        return None
    contenders = [d for d in usable if d.score >= best * (1.0 - tie_tolerance)]
    return min(contenders, key=lambda d: (d.best_lag_sec if d.best_lag_sec is not None else 1e9))


# --------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------


def build_candidates(
    diff: VariableDiff,
    series_so_far: pd.DataFrame,
    *,
    t_now: float,
    window_sec: float = DEFAULT_MOVE_WINDOW_SEC,
    fractions: Sequence[float] = MOVE_FRACTIONS,
) -> list[Candidate]:
    """Moves toward what the successful neighbours had done by now, at several sizes.

    Targets are quantised to the variable's setpoint resolution *before* the constraint
    check, so the number that gets validated is the number that gets recommended. Rounding
    after the check could nudge a proposal back over a limit by a fraction of a unit.
    Quantised no-ops are dropped: "move it by nothing" is not advice.
    """
    idx = series_so_far.index.to_numpy(dtype=float)
    post = series_so_far.loc[(idx >= 0.0) & (idx <= t_now), diff.variable].to_numpy(dtype=float)
    v_start, v_now = float(post[0]), float(post[-1])
    full_delta = diff.diff * v_start  # close the progress gap, in canonical units
    decimals = int(UNITS[diff.variable]["decimals"])

    out: list[Candidate] = []
    for f in fractions:
        v_to = round(v_now + full_delta * float(f), decimals)
        if v_to == round(v_now, decimals):
            continue
        out.append(
            Candidate(
                variable=diff.variable,
                v_from=round(v_now, decimals),
                v_to=v_to,
                window_sec=float(window_sec),
                move_fraction=float(f),
            )
        )
    return out


def _score(candidate: Candidate, diff: VariableDiff) -> float:
    """Prefer the largest admissible step toward the neighbour target.

    Deliberately simple and monotone in move size: the direction and the variable are
    already decided by the evidence, so the only remaining question is how much of the
    known-good move the constraints allow.
    """
    return candidate.move_fraction * diff.consistency


# --------------------------------------------------------------------------------------
# The suggestion
# --------------------------------------------------------------------------------------


@dataclass
class Suggestion:
    """One recommendation plus everything needed to justify or audit it."""

    episode_id: str
    issued_t_sec: float
    variable: str
    v_from: float
    v_to: float
    unit: str
    ramp_rate_per_min: float
    window_sec: float
    signed_dev_pct: float
    expected_max_dev_pct: float
    expected_stab_from_ramp_end_sec: float | None
    no_action_max_dev_pct: float
    no_action_stab_from_ramp_end_sec: float | None
    constraint_report: dict[str, Any]
    diff: VariableDiff
    considered: list[VariableDiff]
    neighbours: list[Neighbour]
    winners: list[Neighbour]
    losers: list[Neighbour]
    scored: list[ScoredCandidate] = field(default_factory=list)
    discarded: list[DiscardedCandidate] = field(default_factory=list)
    immaterial: list[DiscardedCandidate] = field(default_factory=list)
    effect_pct: float = 0.0

    @property
    def k(self) -> int:
        return len(self.neighbours)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "issued_t_sec": self.issued_t_sec,
            "variable": self.variable,
            "from": round(self.v_from, 2),
            "to": round(self.v_to, 2),
            "unit": self.unit,
            "ramp_rate_per_min": round(self.ramp_rate_per_min, 2),
            "expected_max_dev_pct": round(self.expected_max_dev_pct, 3),
            "expected_stab_from_ramp_end_sec": self.expected_stab_from_ramp_end_sec,
            "constraints": self.constraint_report,
            "considered": [d.as_dict() for d in self.considered],
            "discarded": [
                {"to": round(d.candidate.v_to, 2), "reason": d.reason, "detail": d.detail}
                for d in self.discarded
            ],
        }


def _median_or_none(values: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else None


def suggest(
    series_so_far: pd.DataFrame,
    meta: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    k: int = DEFAULT_K,
    root: str | Path = "data/episodes",
    catalogue: pd.DataFrame | None = None,
    window_sec: float = DEFAULT_MOVE_WINDOW_SEC,
    target_speed_m_min: float | None = None,
) -> Suggestion | None:
    """Advise on this episode, or return ``None`` when there is nothing safe to say.

    ``None`` is returned when the predictor sees no breach, when no successful neighbour
    exists to imitate, when no variable shows a consistent difference in the corrective
    direction, or when every candidate move violates this episode's constraints. That last
    case is a real answer:
    the machine is boxed in, and inventing an illegal setpoint would be worse than silence.
    """
    if not prediction.get("will_breach", False):
        return None

    t_now = float(series_so_far.index.to_numpy(dtype=float)[-1])
    neighbours = find_similar(meta, k, catalogue=catalogue, root=root)
    winners, losers = succeeded(neighbours), failed(neighbours)
    if not winners:
        return None

    dev_pct = signed_deviation_pct(
        prediction, series_so_far, meta, target_speed_m_min=target_speed_m_min
    )
    diffs = variable_diffs(
        series_so_far, winners, meta, t_now=t_now, dev_pct=dev_pct, root=root
    )
    chosen = select_variable(diffs)
    if chosen is None:
        return None

    candidates = build_candidates(chosen, series_so_far, t_now=t_now, window_sec=window_sec)

    # ---- constraint filter, ahead of scoring -----------------------------------------
    survivors: list[Candidate] = []
    discarded: list[DiscardedCandidate] = []
    immaterial: list[DiscardedCandidate] = []
    for cand in candidates:
        ok, report = admissible(cand, meta)
        if not ok:
            discarded.append(
                DiscardedCandidate(candidate=cand, reason=report["reason"], detail=report["detail"])
            )
            continue
        effect = bw_effect_pct(cand.variable, cand.delta, series_so_far, meta)
        if abs(effect) < MIN_EFFECT_PCT:
            immaterial.append(
                DiscardedCandidate(
                    candidate=cand,
                    reason="immaterial",
                    detail=(
                        f"{cand.variable} {cand.delta:+.1f} {cand.unit} moves basis weight "
                        f"{effect:+.2f}%, under the {MIN_EFFECT_PCT:.2f}% worth advising on"
                    ),
                )
            )
            continue
        survivors.append(cand)
    if not survivors:
        return None

    # ---- scoring, survivors only ------------------------------------------------------
    scored = sorted(
        (ScoredCandidate.of(c, meta, _score(c, chosen)) for c in survivors),
        key=lambda s: s.score,
        reverse=True,
    )
    best = scored[0]

    exp_dev = float(np.median([n.max_dev_pct for n in winners]))
    exp_stab = _median_or_none([n.stab_from_ramp_end_sec for n in winners])
    no_action_dev = (
        float(np.median([n.max_dev_pct for n in losers])) if losers else abs(dev_pct)
    )
    no_action_stab = _median_or_none([n.stab_from_ramp_end_sec for n in losers])

    return Suggestion(
        episode_id=str(meta["episode_id"]),
        issued_t_sec=t_now,
        variable=best.candidate.variable,
        v_from=best.candidate.v_from,
        v_to=best.candidate.v_to,
        unit=best.candidate.unit,
        ramp_rate_per_min=best.candidate.ramp_rate_per_min,
        window_sec=best.candidate.window_sec,
        signed_dev_pct=dev_pct,
        expected_max_dev_pct=exp_dev,
        expected_stab_from_ramp_end_sec=exp_stab,
        no_action_max_dev_pct=no_action_dev,
        no_action_stab_from_ramp_end_sec=no_action_stab,
        constraint_report=best.constraint_report,
        diff=chosen,
        considered=list(diffs),
        neighbours=list(neighbours),
        winners=winners,
        losers=losers,
        scored=scored,
        discarded=discarded,
        immaterial=immaterial,
        effect_pct=bw_effect_pct(
            best.candidate.variable, best.candidate.delta, series_so_far, meta
        ),
    )
