"""Broke tonnes avoided, estimated from the card's own counterfactual.

Kept out of ``server.py`` deliberately: the HTTP layer assembles, it does not model. The
one piece of arithmetic the dashboard needs that no other module provides is the money
translation, so it lives here with its assumption stated in the return value.

The estimate is severity-proportional. ``labels.broke_tonnes`` is what this transition
actually cost with no advice given, computed by :mod:`src.sim.labels` as off-spec
production mass. The advisor's counterfactual gives peak deviation with and without the
move. Scaling the realised tonnage by the reduction in peak deviation assumes off-spec
mass falls in proportion to how far outside the band the sheet goes, which is
conservative: it credits nothing for the shorter time spent off-spec, only the shallower
excursion.

No claim is returned without the numbers it was derived from - the tile has to be able to
show its own working.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["broke_avoided"]


def broke_avoided(
    labels: Mapping[str, Any], counterfactual: Mapping[str, Any]
) -> dict[str, Any]:
    """Estimate broke tonnes avoided if the suggested move is accepted.

    ``labels`` is ``meta["labels"]`` for the episode being replayed; ``counterfactual`` is
    ``card["counterfactual"]``. Returns the estimate plus every input to it and a plain
    sentence naming the assumption.
    """
    no_action_t = float(labels.get("broke_tonnes") or 0.0)
    no_action_dev = float(counterfactual.get("no_action_max_dev_pct") or 0.0)
    with_action_dev = float(counterfactual.get("with_action_max_dev_pct") or 0.0)

    if no_action_dev <= 0.0:
        reduction = 0.0
    else:
        reduction = 1.0 - with_action_dev / no_action_dev
    reduction = min(max(reduction, 0.0), 1.0)

    avoided_t = no_action_t * reduction
    return {
        "no_action_broke_tonnes": round(no_action_t, 3),
        "no_action_max_dev_pct": round(no_action_dev, 2),
        "with_action_max_dev_pct": round(with_action_dev, 2),
        "severity_reduction_pct": round(reduction * 100.0, 1),
        "avoided_broke_tonnes": round(avoided_t, 3),
        "residual_broke_tonnes": round(no_action_t - avoided_t, 3),
        "basis": (
            f"{no_action_t:.2f} t of off-spec production was realised on this transition "
            f"with no advice. The move takes peak deviation from {no_action_dev:.2f}% to "
            f"{with_action_dev:.2f}%, a {reduction * 100.0:.0f}% shallower excursion; "
            f"tonnage is scaled by that reduction."
        ),
    }
