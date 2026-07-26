"""Evidence cards: the "why" behind every suggestion, and the narration of it.

Five source types, exactly as the evidence-card contract specifies, each with a concrete
checkable detail, and weights that sum to 1.0. Nothing reaches the operator that is not on
the card, and the narrator is handed the card and nothing else.

Two things in here are load-bearing for honesty:

* ``physics_detail.validity_sec`` is carried onto the card verbatim. The open-loop
  projection is only valid for that window - it is the composed measurement lag, the span
  of sheet already made but not yet measured. Beyond it the closed loop is reacting to
  data we do not have. Quoting a 300 s forecast off a 41 s valid window would be the
  single easiest way to make this system untrustworthy, so the caveat travels with the
  claim rather than sitting in a footnote.
* ``effective_measurement_lag_sec`` is the composed figure: transport delay plus scanner
  hold. Transport on its own reads about 7 s and the scanner adds about 34 s. The composed
  median across 300 episodes is 41.44 s. The card carries the real number and its two
  components, never a hand-waved range.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.advisor.suggest import UNITS, Suggestion
from src.analysis.loader import SPEC_BAND
from src.sim.units import m3_h_to_L_min

__all__ = [
    "SOURCE_TYPES",
    "WEIGHT_BUDGET",
    "NUMERAL_RE",
    "mass_balance_attribution",
    "build_card",
    "narrate",
    "card_numerals",
    "validate_narration",
    "narrate_validated",
    "attach_narration",
    "NarrationError",
]

#: The only permitted source types.
SOURCE_TYPES: tuple[str, ...] = ("physics", "causal", "historical", "recipe", "model")

#: Weight budget before normalisation. ``physics_model`` is split between the two by the
#: predictor's own reported split, so a card where the learned term did the work says so.
WEIGHT_BUDGET: dict[str, float] = {
    "physics_model": 0.55,
    "causal": 0.18,
    "historical": 0.18,
    "recipe": 0.09,
}

#: A numeral as the validator sees it. Signs and decimals included, thousands separators
#: not - nothing in the card is formatted with them.
NUMERAL_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: Identifier patterns stripped before numeral extraction. Episode and card IDs contain
#: digits that are not quantities; leaving them in would let any two-digit claim pass.
_ID_PATTERNS = (re.compile(r"EP-G\d+-G\d+-\d+"), re.compile(r"EC-[\w\d-]+"))

_LAG_MEDIAN_NOTE = "composed = transport + scanner hold"


# --------------------------------------------------------------------------------------
# Physics attribution
# --------------------------------------------------------------------------------------


def mass_balance_attribution(
    series_so_far: pd.DataFrame,
    meta: Mapping[str, Any],
    physics_detail: Mapping[str, Any],
) -> dict[str, Any]:
    """Which mass-balance term is driving the forecast, with the actual numbers.

    ``bw = delivered_mass_rate / (speed * trim) * retention``, so the change still to come
    decomposes into numerator terms (fibre and filler, separately) and a denominator term
    (speed). Whichever is largest in magnitude is the term that explains the excursion, and
    it is the one the card names.

    Terms are evaluated over the **validity window**, not over the whole remaining ramp. On
    a wide grade change the ramp still to run is enormous - 1357 to 505 m/min is a 63%
    denominator move - and quoting that as the reason for a 7.5% excursion is not an
    explanation, it is a category error, because the setpoint is travelling with it. Each
    variable is advanced at its observed rate for as long as the projection is valid and
    clamped at its target, which keeps the attribution on the same footing as the claim.
    """
    window = series_so_far.loc[series_so_far.index >= 0.0]
    row = window.iloc[-1]
    machine = meta["machine"]
    filler_cons = float(machine["filler_cons_pct"])
    cons = float(row["stock_cons"])
    targets = dict(physics_detail.get("mv_targets", {}))
    rates = dict(physics_detail.get("mv_rates_per_min", {}))
    horizon_sec = float(physics_detail.get("validity_sec") or 0.0)

    def advance(name: str) -> tuple[float, float]:
        now = float(row[name])
        target = float(targets.get(name, now))
        moved = now + float(rates.get(name, 0.0)) / 60.0 * horizon_sec
        return now, float(np.clip(moved, min(now, target), max(now, target)))

    speed_now, speed_ahead = advance("speed")
    stock_now, stock_ahead = advance("stock_flow")
    filler_now, filler_ahead = advance("filler_flow")
    speed_target, stock_target, filler_target = speed_ahead, stock_ahead, filler_ahead

    def mass(stock_m3_h: float, filler_L_min: float) -> float:
        return m3_h_to_L_min(stock_m3_h) * cons * 10.0 + filler_L_min * filler_cons * 10.0

    fibre_now = m3_h_to_L_min(stock_now) * cons * 10.0
    fibre_target = m3_h_to_L_min(stock_target) * cons * 10.0
    filler_dry_now = filler_now * filler_cons * 10.0
    filler_dry_target = filler_target * filler_cons * 10.0
    mass_now = fibre_now + filler_dry_now
    mass_target = fibre_target + filler_dry_target

    # The numerator is split into its two physical streams. Naming "mass" as the driver
    # while quoting stock-flow numbers would be wrong whenever filler is the stream that
    # is actually moving, which happens on every high-ash to low-ash change.
    fibre_term_pct = (fibre_target - fibre_now) / mass_now * 100.0 if mass_now else 0.0
    filler_term_pct = (filler_dry_target - filler_dry_now) / mass_now * 100.0 if mass_now else 0.0
    mass_term_pct = fibre_term_pct + filler_term_pct
    speed_term_pct = -(speed_target - speed_now) / speed_now * 100.0 if speed_now else 0.0

    terms = {
        "machine_speed": speed_term_pct,
        "thick_stock": fibre_term_pct,
        "filler": filler_term_pct,
    }
    dominant = max(terms, key=lambda k: abs(terms[k]))
    return {
        "dominant_term": dominant,
        "window_sec": round(horizon_sec, 1),
        "mass_term_pct": round(mass_term_pct, 2),
        "fibre_term_pct": round(fibre_term_pct, 2),
        "filler_term_pct": round(filler_term_pct, 2),
        "speed_term_pct": round(speed_term_pct, 2),
        "net_term_pct": round(mass_term_pct + speed_term_pct, 2),
        "speed_now_m_min": round(speed_now, 1),
        "speed_ahead_m_min": round(speed_ahead, 1),
        "stock_flow_now_m3_h": round(stock_now, 1),
        "stock_flow_ahead_m3_h": round(stock_ahead, 1),
        "filler_flow_now_L_min": round(filler_now, 1),
        "filler_flow_ahead_L_min": round(filler_ahead, 1),
        "dry_mass_now_g_min": round(mass_now, 0),
        "dry_mass_ahead_g_min": round(mass_target, 0),
    }


def _worst_setpoint(prediction: Mapping[str, Any]) -> float:
    """Setpoint in force at the projected worst moment."""
    traj = np.asarray(prediction["predicted_trajectory"], dtype=float)
    sp = np.asarray(prediction["setpoint_trajectory"], dtype=float)
    if traj.size == 0 or sp.size == 0:
        return float("nan")
    dev = np.abs(traj[:, 1] - sp[:, 1]) / sp[:, 1]
    return float(sp[int(np.argmax(dev)), 1])


# --------------------------------------------------------------------------------------
# Card
# --------------------------------------------------------------------------------------


def build_card(
    suggestion: Suggestion,
    prediction: Mapping[str, Any],
    series_so_far: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    card_id: str | None = None,
    spread_pct: float | None = None,
    ranking_path: str | Path = "data/impact_ranking.json",
) -> dict[str, Any]:
    """Assemble the evidence card for one suggestion. Weights sum to 1.0."""
    detail = dict(prediction.get("physics_detail", {}))
    lag = dict(prediction.get("lag_components", {}))
    phys = mass_balance_attribution(series_so_far, meta, detail)

    # Signed, from the suggestion: the predictor reports a magnitude and the advisor has
    # already recovered the direction from the physics projection. The card must not
    # re-derive it, or the claim and the action could disagree about which way the sheet
    # is off.
    dev_pct = float(suggestion.signed_dev_pct)
    sp_worst = _worst_setpoint(prediction)
    predicted_value = sp_worst * (1.0 + dev_pct / 100.0)
    spread = float(spread_pct) if spread_pct else max(abs(float(prediction["model_correction_pct"])), 0.5)
    interval = [
        round(predicted_value * (1.0 - spread / 100.0), 2),
        round(predicted_value * (1.0 + spread / 100.0), 2),
    ]

    ttb = prediction.get("time_to_breach_sec")
    side = "low" if dev_pct < 0 else "high"
    # Quote the rounded figure the card stores, so the statement, the fields and the
    # narration cannot disagree in the first decimal.
    dev_shown = abs(round(dev_pct, 2))
    if ttb is None:
        statement = f"Basis weight is on track to run {dev_shown:.1f}% off target."
    elif float(ttb) <= 0.0:
        statement = (
            f"Basis weight is already past the {side} limit in sheet not yet measured, "
            f"heading for {dev_shown:.1f}% off target."
        )
    else:
        statement = f"Basis weight is on track to breach the {side} limit in {ttb:.0f} s."

    n_win, k = len(suggestion.winners), suggestion.k
    edge = suggestion.diff
    ranking = _lag_decomposition(ranking_path)
    composed = float(lag.get("composed", float("nan")))

    raw: dict[str, float] = {
        "physics": WEIGHT_BUDGET["physics_model"] * float(prediction.get("physics_contribution", 1.0)),
        "model": WEIGHT_BUDGET["physics_model"] * float(prediction.get("model_correction", 0.0)),
        "causal": WEIGHT_BUDGET["causal"] * min(max(edge.strength, 0.0), 1.0),
        "historical": WEIGHT_BUDGET["historical"] * (n_win / k if k else 0.0),
        "recipe": WEIGHT_BUDGET["recipe"],
    }
    weights = _normalise_weights(raw)

    unit_display = UNITS[suggestion.variable]["display"]
    decimals = int(UNITS[suggestion.variable]["decimals"])
    sources: list[dict[str, Any]] = [
        {
            "type": "physics",
            "detail": (
                f"Mass balance over the valid window, {_term_phrase(phys)}: net "
                f"{phys['net_term_pct']}% on basis weight (fibre {phys['fibre_term_pct']}%, "
                f"filler {phys['filler_term_pct']}%, speed {phys['speed_term_pct']}%)."
            ),
            "physics_detail": {
                # The whole attribution, not a subset: the narrator reads only the card, so
                # anything trimmed here becomes a gap it has to talk around.
                **phys,
                "validity_sec": detail.get("validity_sec"),
                "validity_note": (
                    "Open-loop projection is valid for this window only - it is sheet "
                    "already formed but not yet measured. Beyond it the loop is reacting "
                    "to data not yet available."
                ),
                "effective_measurement_lag_sec": round(composed, 2) if np.isfinite(composed) else None,
                "transport_lag_sec": lag.get("transport"),
                "scanner_lag_sec": lag.get("scanner"),
                "lag_note": _LAG_MEDIAN_NOTE,
                "dataset_median_effective_lag_sec": ranking.get("composed_sec_median"),
                "headbox_bw_now_g_m2": prediction.get("headbox_bw_now"),
                "measured_bw_now_g_m2": prediction.get("measured_bw_now"),
                "headbox_lead_g_m2": prediction.get("headbox_lead_g_m2"),
            },
            "weight": weights["physics"],
        },
        {
            "type": "causal",
            "detail": (
                f"{edge.variable} -> bw, lag {_fmt(edge.best_lag_sec)} s, strength "
                f"{edge.strength} (lagged discovery, {_edge_n(ranking_path, edge.variable)} "
                f"episodes)."
            ),
            "weight": weights["causal"],
        },
        {
            "type": "historical",
            "detail": (
                f"{k} nearest transitions in grade-property space; {n_win} stayed in spec, "
                f"{k - n_win} went off. By this point the ones that held had "
                f"{edge.variable} {_signed_progress(edge.neighbour_progress)} against your "
                f"{_signed_progress(edge.current_progress)}."
            ),
            "k": k,
            "n_in_spec": n_win,
            "n_off_spec": k - n_win,
            "episode_ids": [n.episode_id for n in suggestion.winners],
            "off_spec_episode_ids": [n.episode_id for n in suggestion.losers],
            "weight": weights["historical"],
        },
        {
            "type": "recipe",
            "detail": suggestion.constraint_report["detail"],
            "recipe_detail": {
                "limit_lo": suggestion.constraint_report["limit_lo"],
                "limit_hi": suggestion.constraint_report["limit_hi"],
                "binding_bound": suggestion.constraint_report.get("bound"),
                "binding_value": suggestion.constraint_report.get("bound_value"),
                "headroom_used_pct": suggestion.constraint_report.get("headroom_used_pct"),
                "room_to_bound": _room_to_bound(suggestion, decimals),
                "rate_limit_per_min": suggestion.constraint_report["rate_limit_per_min"],
                "rate_used_per_min": suggestion.constraint_report.get("rate_used_per_min"),
                "rate_used_pct": suggestion.constraint_report.get("rate_used_pct"),
                "candidates_discarded": len(suggestion.discarded),
                "candidates_immaterial": len(suggestion.immaterial),
            },
            "weight": weights["recipe"],
        },
        {
            "type": "model",
            "detail": (
                f"Residual model moved the forecast by "
                f"{_fmt(prediction.get('model_correction_pct'))}% on top of the physics "
                f"{_fmt(prediction.get('physics_max_dev_pct'))}%; physics carries "
                f"{_fmt(float(prediction.get('physics_contribution', 1.0)) * 100, 0)}% of "
                f"the claim."
            ),
            "weight": weights["model"],
        },
    ]

    return {
        "card_id": card_id or f"EC-{suggestion.episode_id}-{int(suggestion.issued_t_sec)}",
        "episode_id": suggestion.episode_id,
        "issued_t_sec": round(suggestion.issued_t_sec, 1),
        "kind": "recommendation",
        "claim": {
            "statement": statement,
            "predicted_value": round(predicted_value, 2),
            "unit": "g/m2",
            "dev_pct": round(dev_pct, 2),
            "spec_band_pct": round(SPEC_BAND * 100.0, 1),
            "horizon_sec": (round(float(ttb), 1) if ttb is not None else None),
            "confidence": prediction.get("confidence"),
            "interval": interval,
        },
        "action": {
            "variable": suggestion.variable,
            "from": round(suggestion.v_from, decimals),
            "to": round(suggestion.v_to, decimals),
            "unit": UNITS[suggestion.variable]["canonical"],
            "unit_display": unit_display,
            "decimals": decimals,
            "delta": round(suggestion.v_to - suggestion.v_from, decimals),
            "window_sec": round(suggestion.window_sec, 0),
            "ramp_rate_per_min": round(suggestion.ramp_rate_per_min, decimals),
            "estimated_bw_effect_pct": round(suggestion.effect_pct, 2),
            "expected_max_dev_pct": round(suggestion.expected_max_dev_pct, 2),
            "expected_stab_from_ramp_end_sec": _round_or_none(
                suggestion.expected_stab_from_ramp_end_sec, 0
            ),
            "expected_stabilisation_gain_sec": _round_or_none(
                None
                if suggestion.expected_stab_from_ramp_end_sec is None
                or suggestion.no_action_stab_from_ramp_end_sec is None
                else suggestion.no_action_stab_from_ramp_end_sec
                - suggestion.expected_stab_from_ramp_end_sec,
                0,
            ),
        },
        "sources": sources,
        "constraints_checked": {
            "recipe_limits": suggestion.constraint_report["recipe_limits"],
            "actuator_rate": suggestion.constraint_report["actuator_rate"],
            "filtered_before_scoring": True,
            "candidates_generated": (
                len(suggestion.scored) + len(suggestion.discarded) + len(suggestion.immaterial)
            ),
            "candidates_scored": len(suggestion.scored),
            "candidates_discarded": len(suggestion.discarded),
            "candidates_immaterial": len(suggestion.immaterial),
        },
        "counterfactual": {
            "no_action_max_dev_pct": round(suggestion.no_action_max_dev_pct, 2),
            "no_action_stab_from_ramp_end_sec": _round_or_none(
                suggestion.no_action_stab_from_ramp_end_sec, 0
            ),
            "with_action_max_dev_pct": round(suggestion.expected_max_dev_pct, 2),
        },
        "narration": None,
    }


def _room_to_bound(suggestion: Suggestion, decimals: int) -> float | None:
    """Distance from the proposed setpoint to the limit it is heading toward.

    More useful to an operator than the limit on its own: "leaves 465 L/min to the floor"
    answers the question, where "clear of the 0 L/min floor" states a triviality.
    """
    bound = suggestion.constraint_report.get("bound_value")
    if bound is None:
        return None
    return round(abs(suggestion.v_to - float(bound)), decimals)


def _moving(now: Any, ahead: Any) -> str:
    """"coming up" / "coming down" / "holding", read off the two values."""
    if now is None or ahead is None:
        return "moving"
    delta = float(ahead) - float(now)
    if abs(delta) < 1e-9:
        return "holding"
    return "coming up" if delta > 0 else "coming down"


def _puts(term_pct: Any) -> str:
    """Direction and size of a term's effect on basis weight, in operator phrasing."""
    if term_pct is None:
        return "shifts the weight"
    value = float(term_pct)
    if value >= 0:
        return f"puts {abs(value):.1f}% on"
    return f"takes {abs(value):.1f}% off"


def _signed_progress(fraction: float) -> str:
    """Fractional progress as an operator would read it: "6.2% up", "43.9% down"."""
    pct = fraction * 100.0
    return f"{abs(pct):.1f}% {'up' if pct >= 0 else 'down'}"


def _term_phrase(phys: Mapping[str, Any]) -> str:
    """The driving term named with the values it was computed from."""
    dominant = phys["dominant_term"]
    window = phys["window_sec"]
    if dominant == "machine_speed":
        return (
            f"speed going {phys['speed_now_m_min']} to {phys['speed_ahead_m_min']} m/min "
            f"over {window} s"
        )
    if dominant == "filler":
        return (
            f"filler going {phys['filler_flow_now_L_min']} to "
            f"{phys['filler_flow_ahead_L_min']} L/min over {window} s"
        )
    return (
        f"thick stock going {phys['stock_flow_now_m3_h']} to "
        f"{phys['stock_flow_ahead_m3_h']} m3/h over {window} s"
    )


def _lag_decomposition(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("lag_decomposition", {})


def _edge_n(path: str | Path, variable: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    for rec in json.loads(p.read_text(encoding="utf-8")).get("rankings", []):
        if rec.get("kind") == "deviation" and rec.get("variable") == variable:
            return int(rec.get("n_episodes", 0))
    return 0


def _normalise_weights(raw: Mapping[str, float]) -> dict[str, float]:
    """Scale to sum 1.0 and absorb rounding error into the largest source."""
    total = sum(raw.values())
    if total <= 0.0:
        even = round(1.0 / len(raw), 3)
        out = {k: even for k in raw}
    else:
        out = {k: round(v / total, 3) for k, v in raw.items()}
    biggest = max(out, key=lambda k: out[k])
    out[biggest] = round(out[biggest] + (1.0 - sum(out.values())), 3)
    return out


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits) if digits else float(round(float(value)))


def _fmt(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "n/a"
    v = float(value)
    if digits is not None:
        return f"{v:.{digits}f}"
    return f"{v:g}"


# --------------------------------------------------------------------------------------
# Narration
# --------------------------------------------------------------------------------------


class NarrationError(RuntimeError):
    """Raised when a narration cites a number the card does not contain."""


_GAP_NOTICE = (
    "Evidence card incomplete - no narration available. Fields missing: {fields}. "
    "Read the card directly."
)

_REQUIRED = (
    ("claim", "dev_pct"),
    ("action", "variable"),
    ("action", "from"),
    ("action", "to"),
)


def narrate(card: Mapping[str, Any]) -> str:
    """Two to four plain sentences, operator register, numbers from the card only.

    The narrator receives the card and nothing else. It rephrases; it does not reason. If
    a required field is missing it emits a gap notice rather than filling the gap.
    """
    missing = [f"{a}.{b}" for a, b in _REQUIRED if card.get(a, {}).get(b) is None]
    if missing:
        return _GAP_NOTICE.format(fields=", ".join(missing))

    claim, action = card["claim"], card["action"]
    phys = _source(card, "physics").get("physics_detail", {})
    hist = _source(card, "historical")
    recipe = _source(card, "recipe").get("recipe_detail", {})

    dev = float(claim["dev_pct"])
    light = dev < 0
    lag = phys.get("effective_measurement_lag_sec")
    var = _spoken(action["variable"])
    unit = action.get("unit_display", "")
    places = int(action.get("decimals", 1))

    # 1. What is happening, and the staleness caveat that makes it worth acting on now.
    first = (
        f"The sheet's running {'light' if light else 'heavy'} - heading "
        f"{_fmt(abs(dev), 1)}% {'under' if light else 'over'} target, and there's "
        f"{_fmt(lag, 0)} s of sheet already made that the scanner hasn't seen yet."
    )

    # 2. Why, in the terms the physics and causal sources actually support.
    dominant = phys.get("dominant_term")
    ahead = _fmt(phys.get("window_sec"), 0)
    if dominant == "machine_speed":
        second = (
            f"Machine speed is {_moving(phys.get('speed_now_m_min'), phys.get('speed_ahead_m_min'))}, "
            f"{_fmt(phys.get('speed_now_m_min'), 0)} to "
            f"{_fmt(phys.get('speed_ahead_m_min'), 0)} m/min over the next {ahead} s, and "
            f"that on its own {_puts(phys.get('speed_term_pct'))} the weight."
        )
    elif dominant == "filler":
        second = (
            f"Filler is the stream doing the work, "
            f"{_moving(phys.get('filler_flow_now_L_min'), phys.get('filler_flow_ahead_L_min'))} "
            f"from {_fmt(phys.get('filler_flow_now_L_min'), 0)} to "
            f"{_fmt(phys.get('filler_flow_ahead_L_min'), 0)} L/min in that time, which "
            f"{_puts(phys.get('filler_term_pct'))} the weight."
        )
    else:
        second = (
            f"Thick stock is "
            f"{_moving(phys.get('stock_flow_now_m3_h'), phys.get('stock_flow_ahead_m3_h'))}, "
            f"{_fmt(phys.get('stock_flow_now_m3_h'), 0)} to "
            f"{_fmt(phys.get('stock_flow_ahead_m3_h'), 0)} m³/h over the next {ahead} s, "
            f"which {_puts(phys.get('fibre_term_pct'))} the weight."
        )

    # 3. What to do, with the limit that bounds it.
    direction = "up" if float(action["to"]) >= float(action["from"]) else "back"
    edge_word = "ceiling" if recipe.get("binding_bound") == "upper" else "floor"
    third = (
        f"Put {var} {direction} from {_fmt(action['from'], places)} to "
        f"{_fmt(action['to'], places)} {unit} over the next "
        f"{_fmt(action['window_sec'], 0)} s - "
        f"{_fmt(abs(float(action['ramp_rate_per_min'])), places)} {unit} in the minute against "
        f"the {_fmt(recipe.get('rate_limit_per_min'), 0)} the actuator allows, and it still "
        f"leaves {_fmt(recipe.get('room_to_bound'), places)} {unit} to the recipe {edge_word}."
    )

    # 4. What the history says either way.
    fourth = (
        f"Of the {_fmt(hist.get('k'), 0)} closest past transitions, "
        f"{_fmt(hist.get('n_in_spec'), 0)} held spec doing that and topped out at "
        f"{_fmt(action.get('expected_max_dev_pct'), 1)}%; leave it and this one runs to about "
        f"{_fmt(card['counterfactual']['no_action_max_dev_pct'], 1)}% off."
    )

    return " ".join([first, second, third, fourth])


_SPOKEN = {
    "stock_flow": "thick stock",
    "filler_flow": "filler",
    "steam_p": "dryer steam",
    "speed": "machine speed",
}


def _spoken(variable: str) -> str:
    return _SPOKEN.get(variable, variable)


def _source(card: Mapping[str, Any], kind: str) -> dict[str, Any]:
    for src in card.get("sources", []):
        if src.get("type") == kind:
            return dict(src)
    return {}


# --------------------------------------------------------------------------------------
# Validator - fail closed
# --------------------------------------------------------------------------------------


def _numeric_leaves(node: Any) -> Iterable[float]:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield float(node)
    elif isinstance(node, Mapping):
        for v in node.values():
            yield from _numeric_leaves(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _numeric_leaves(v)


def card_numerals(card: Mapping[str, Any]) -> set[str]:
    """Every numeral the card can legitimately be quoted as containing.

    Two sources: numeral tokens in the serialised card (which covers numbers written into
    the ``detail`` strings, since the operator can read those too), and 0/1/2-decimal
    roundings of every numeric leaf, so "41" is allowed for a stored 41.44 but "42" is not.
    Episode and card identifiers are stripped first - their digits are labels, not
    quantities, and leaving them in would quietly weaken the check.
    """
    payload = {k: v for k, v in card.items() if k != "narration"}
    text = json.dumps(payload)
    for pattern in _ID_PATTERNS:
        text = pattern.sub("", text)
    allowed = set(NUMERAL_RE.findall(text))
    for value in _numeric_leaves(payload):
        for digits in (0, 1, 2):
            allowed.add(f"{value:.{digits}f}".rstrip(".") if digits == 0 else f"{value:.{digits}f}")
            allowed.add(f"{abs(value):.{digits}f}" if digits else f"{abs(value):.0f}")
        allowed.add(f"{value:g}")
        allowed.add(f"{abs(value):g}")
    return allowed


def validate_narration(card: Mapping[str, Any], narration: str) -> tuple[bool, list[str]]:
    """Every numeral in ``narration`` must appear in the card. Returns the offenders."""
    allowed = card_numerals(card)
    offenders = [tok for tok in NUMERAL_RE.findall(narration) if tok not in allowed]
    return (not offenders), offenders


def narrate_validated(card: Mapping[str, Any]) -> str:
    """Narrate and validate in one step. Raises rather than returning bad text.

    Fail closed: an unvalidated narration is not display material, so the caller gets an
    exception and the UI shows the card without prose.
    """
    text = narrate(card)
    ok, offenders = validate_narration(card, text)
    if not ok:
        raise NarrationError(
            f"narration cites numbers absent from card {card.get('card_id')}: {offenders}"
        )
    return text


def attach_narration(card: dict[str, Any]) -> dict[str, Any]:
    """Validate, then write the narration onto the card. Leaves it ``None`` on failure."""
    try:
        card["narration"] = narrate_validated(card)
    except NarrationError as exc:
        card["narration"] = None
        card["narration_error"] = str(exc)
    return card
