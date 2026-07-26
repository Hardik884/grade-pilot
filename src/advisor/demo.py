"""Run predictor + advisor on one real off-spec episode at the 90 s mark.

    python -m src.advisor.demo                      # first off-spec episode found
    python -m src.advisor.demo EP-G12-G11-0002      # a named episode

Prints the evidence card as JSON and the narration underneath it. The narration is the
validated one; if validation fails the card prints and the prose does not, which is the
intended behaviour rather than a bug in the demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.advisor import evidence, suggest
from src.advisor.retrieval import load_catalogue
from src.analysis import loader
from src.analysis.predictor import Predictor

ROOT = Path("data/episodes")
FEATURES = Path("data/features_90s.parquet")


def _fit_predictor(features: pd.DataFrame) -> tuple[Predictor, dict[str, Any]]:
    table = Predictor.physics_table(features, ROOT)
    pred = Predictor()
    report = pred.fit(table)
    return pred, {"fit": report, "table": table}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode_id", nargs="?", default=None)
    ap.add_argument("--t", type=float, default=loader.FEATURE_HORIZON_SEC)
    ap.add_argument("--k", type=int, default=9)
    args = ap.parse_args(argv)

    features = (
        pd.read_parquet(FEATURES)
        if FEATURES.exists()
        else loader.build_feature_table(ROOT, cache=FEATURES)
    )
    catalogue = load_catalogue(ROOT)

    if args.episode_id:
        episode_id = args.episode_id
    else:
        off = features.loc[features["off_spec"].astype(bool)].sort_values(
            "max_dev_pct", ascending=False
        )
        episode_id = str(off["episode_id"].iloc[0])

    predictor, extra = _fit_predictor(features)
    grades = loader.load_grades()
    ep = loader.load_episode(ROOT / episode_id)
    history = ep.history(args.t)
    row = features.loc[features["episode_id"] == episode_id].reset_index(drop=True)

    prediction = predictor.predict(
        history,
        ep.meta,
        feature_row=row,
        target_speed_m_min=grades[ep.meta["grade_to"]]["nominal_speed_m_min"],
    )

    print(f"episode        {episode_id}   {ep.meta['grade_from']} -> {ep.meta['grade_to']}")
    print(f"truth          off_spec={ep.meta['labels']['off_spec']} "
          f"max_dev_pct={ep.meta['labels']['max_dev_pct']}")
    print(f"at t={args.t:.0f}s    will_breach={prediction['will_breach']} "
          f"predicted_max_dev_pct={prediction['predicted_max_dev_pct']} "
          f"ttb={prediction['time_to_breach_sec']}")
    print(f"residual fit   {extra['fit']}")
    print()

    sug = suggest.suggest(
        history,
        ep.meta,
        prediction,
        k=args.k,
        catalogue=catalogue,
        target_speed_m_min=grades[ep.meta["grade_to"]]["nominal_speed_m_min"],
    )
    if sug is None:
        print("no suggestion: predictor sees no breach, or no admissible candidate exists.")
        return 0

    card = evidence.build_card(sug, prediction, history, ep.meta)
    ok, offenders = True, []
    try:
        card["narration"] = evidence.narrate_validated(card)
    except evidence.NarrationError as exc:
        ok, offenders = False, [str(exc)]
        card["narration"] = None

    print(json.dumps(card, indent=2, default=str))
    print()
    print("weights sum   ", round(sum(s["weight"] for s in card["sources"]), 4))
    print("narration ok  ", ok, offenders)
    print()
    print("NARRATION")
    print(card["narration"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
