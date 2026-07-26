"""Advisor tests: constraint safety, sparse-pair retrieval, and narration grounding."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.advisor import evidence, retrieval, suggest
from src.analysis import loader
from src.analysis.predictor import Predictor

ROOT = Path("data/episodes")
FEATURES = Path("data/features_90s.parquet")
T_NOW = loader.FEATURE_HORIZON_SEC


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grades() -> dict[str, dict[str, float]]:
    if not Path("data/grades.json").exists():
        pytest.skip("grade catalogue missing")
    return loader.load_grades()


@pytest.fixture(scope="module")
def catalogue(grades) -> pd.DataFrame:
    if not (ROOT / "index.parquet").exists():
        pytest.skip("episode index missing - generate the dataset first")
    return retrieval.load_catalogue(ROOT, grades=grades)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not FEATURES.exists():
        pytest.skip("cached feature table missing")
    return pd.read_parquet(FEATURES)


@pytest.fixture(scope="module")
def predictor(features) -> Predictor:
    table = Predictor.physics_table(features, ROOT)
    pred = Predictor()
    pred.fit(table)
    return pred


@pytest.fixture(scope="module")
def breaching_case(features, predictor, catalogue, grades) -> dict[str, Any]:
    """First off-spec episode the predictor also flags, plus its suggestion and card."""
    off = features.loc[features["off_spec"].astype(bool)].sort_values(
        "max_dev_pct", ascending=False
    )
    for episode_id in off["episode_id"].astype(str):
        ep = loader.load_episode(ROOT / episode_id)
        history = ep.history(T_NOW)
        row = features.loc[features["episode_id"] == episode_id].reset_index(drop=True)
        target_speed = grades[ep.meta["grade_to"]]["nominal_speed_m_min"]
        prediction = predictor.predict(
            history, ep.meta, feature_row=row, target_speed_m_min=target_speed
        )
        if not prediction["will_breach"]:
            continue
        sug = suggest.suggest(
            history,
            ep.meta,
            prediction,
            catalogue=catalogue,
            target_speed_m_min=target_speed,
        )
        if sug is None:
            continue
        card = evidence.build_card(sug, prediction, history, ep.meta)
        return {
            "ep": ep,
            "history": history,
            "prediction": prediction,
            "suggestion": sug,
            "card": card,
            "target_speed": target_speed,
        }
    pytest.skip("no off-spec episode produced an advisable suggestion")


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------


def test_grade_space_embedding_is_normalised(grades) -> None:
    norm = retrieval.build_normaliser(grades)
    points = np.vstack([retrieval.embed_grade(p, norm) for p in grades.values()])
    assert points.min() == pytest.approx(0.0)
    assert points.max() == pytest.approx(1.0)
    assert points.shape[1] == len(retrieval.GRADE_DIMS)


def test_transition_vector_is_directional(grades) -> None:
    norm = retrieval.build_normaliser(grades)
    up = retrieval.transition_vector(grades["G07"], grades["G12"], norm)
    down = retrieval.transition_vector(grades["G12"], grades["G07"], norm)
    assert np.allclose(up, -down)


def test_dataset_has_singleton_grade_pairs(catalogue) -> None:
    """The premise of property-space retrieval: code matching would return nothing."""
    counts = retrieval.pair_counts(catalogue)
    assert int((counts == 1).sum()) > 0


def test_retrieval_on_a_pair_seen_exactly_once(catalogue, grades) -> None:
    """A transition whose grade pair occurs once still gets a full neighbourhood.

    This is the selling point: matching on codes yields zero comparable episodes for these
    transitions, matching on properties yields nine.
    """
    counts = retrieval.pair_counts(catalogue)
    singletons = counts[counts == 1]
    if singletons.empty:
        pytest.skip("no singleton grade pair in this dataset")
    grade_from, grade_to = singletons.index[0]
    row = catalogue.loc[
        (catalogue["grade_from"] == grade_from) & (catalogue["grade_to"] == grade_to)
    ].iloc[0]
    meta = loader.load_meta(ROOT / str(row["episode_id"]))

    same_pair = catalogue.loc[
        (catalogue["grade_from"] == grade_from)
        & (catalogue["grade_to"] == grade_to)
        & (catalogue["episode_id"] != meta["episode_id"])
    ]
    assert same_pair.empty, "fixture assumption broken: pair is not a singleton"

    neighbours = retrieval.find_similar(meta, k=9, catalogue=catalogue, grades=grades)
    assert len(neighbours) == 9
    assert meta["episode_id"] not in {n.episode_id for n in neighbours}
    # Sorted by distance, and none of them is the same code pair.
    assert neighbours == sorted(neighbours, key=lambda n: n.distance)
    assert all(f"{n.grade_from}->{n.grade_to}" != f"{grade_from}->{grade_to}" for n in neighbours)
    # Neighbours carry the outcome fields the advisor needs.
    assert all(isinstance(n.off_spec, bool) and np.isfinite(n.max_dev_pct) for n in neighbours)


def test_retrieval_excludes_the_query_episode(catalogue, grades) -> None:
    meta = loader.load_meta(ROOT / str(catalogue["episode_id"].iloc[0]))
    ids = {n.episode_id for n in retrieval.find_similar(meta, k=12, catalogue=catalogue, grades=grades)}
    assert meta["episode_id"] not in ids


# --------------------------------------------------------------------------------------
# Constraints - the structural guarantee
# --------------------------------------------------------------------------------------


def _tighten(meta: dict[str, Any], variable: str, value: float) -> dict[str, Any]:
    """Clamp a variable's recipe band and actuator rate around its current value."""
    tight = copy.deepcopy(meta)
    tight["recipe_limits"][variable] = [value - 0.5, value + 0.5]
    tight["actuator_rates"][variable] = 0.01
    return tight


def test_out_of_bounds_candidate_is_never_returned(breaching_case) -> None:
    """Adversarial case: limits pinched shut around the current operating point.

    The assertion is about *absence from the output*, not about ranking. A candidate that
    breaks a limit must not appear as a scored option at all.
    """
    case = breaching_case
    sug = case["suggestion"]
    variable = sug.variable
    tight_meta = _tighten(case["ep"].meta, variable, sug.v_from)

    tight = suggest.suggest(
        case["history"],
        tight_meta,
        case["prediction"],
        catalogue=None,
        target_speed_m_min=case["target_speed"],
    )
    if tight is not None:
        # If it still advises, it must have switched variables and still be legal.
        assert tight.variable != variable or abs(tight.v_to - tight.v_from) <= 0.5
        lo, hi = tight_meta["recipe_limits"][tight.variable]
        assert lo <= tight.v_to <= hi
        for scored in tight.scored:
            assert suggest.admissible(scored.candidate, tight_meta)[0]

    # And directly on the filter: every candidate on the pinched variable is discarded.
    diff = sug.diff
    candidates = suggest.build_candidates(diff, case["history"], t_now=sug.issued_t_sec)
    admissible = [c for c in candidates if suggest.admissible(c, tight_meta)[0]]
    assert admissible == [], "constraint filter let a candidate through on pinched limits"


def test_scoring_refuses_an_inadmissible_candidate(breaching_case) -> None:
    """The only door into the ranking re-checks admissibility and slams shut."""
    case = breaching_case
    sug = case["suggestion"]
    lo, hi = case["ep"].meta["recipe_limits"][sug.variable]
    illegal = suggest.Candidate(
        variable=sug.variable,
        v_from=sug.v_from,
        v_to=hi + 1000.0,
        window_sec=sug.window_sec,
        move_fraction=1.0,
    )
    assert not suggest.admissible(illegal, case["ep"].meta)[0]
    with pytest.raises(ValueError):
        suggest.ScoredCandidate.of(illegal, case["ep"].meta, score=99.0)


def test_actuator_rate_is_enforced(breaching_case) -> None:
    case = breaching_case
    meta = case["ep"].meta
    sug = case["suggestion"]
    rate_limit = suggest.actuator_rate_per_min(meta, sug.variable)
    lo, hi = meta["recipe_limits"][sug.variable]
    # A move well inside the recipe band but far too fast for the actuator.
    fast = suggest.Candidate(
        variable=sug.variable,
        v_from=sug.v_from,
        v_to=min(hi, max(lo, sug.v_from + rate_limit * 5.0)),
        window_sec=1.0,
        move_fraction=1.0,
    )
    ok, report = suggest.admissible(fast, meta)
    assert not ok and report["reason"] == "actuator_rate"


def test_limits_come_from_the_episode_not_a_default() -> None:
    """Per-episode limits, never hardcoded. Confirmed by disagreeing with the default."""
    from src.sim.grades import RECIPE_LIMITS

    meta = copy.deepcopy(loader.load_meta(ROOT / str(loader.episode_dirs(ROOT)[0].name)))
    meta["recipe_limits"]["stock_flow"] = [900.0, 901.0]
    meta["actuator_rates"]["stock_flow"] = 1.0
    assert suggest.recipe_limits(meta, "stock_flow") == (900.0, 901.0)
    assert suggest.recipe_limits(meta, "stock_flow") != tuple(RECIPE_LIMITS["stock_flow"])
    assert suggest.actuator_rate_per_min(meta, "stock_flow") == 1.0


def test_no_suggestion_without_a_breach(breaching_case) -> None:
    quiet = dict(breaching_case["prediction"])
    quiet["will_breach"] = False
    assert (
        suggest.suggest(breaching_case["history"], breaching_case["ep"].meta, quiet) is None
    )


def test_suggested_move_is_physically_corrective(breaching_case) -> None:
    """A light sheet gets more mass or more speed, never the opposite."""
    sug = breaching_case["suggestion"]
    needed = suggest.corrective_direction(sug.variable, sug.signed_dev_pct)
    assert needed != 0
    assert np.sign(sug.v_to - sug.v_from) == needed


# --------------------------------------------------------------------------------------
# Evidence card and narration
# --------------------------------------------------------------------------------------


def test_card_weights_sum_to_one(breaching_case) -> None:
    card = breaching_case["card"]
    total = sum(float(s["weight"]) for s in card["sources"])
    assert total == pytest.approx(1.0, abs=0.01)


def test_card_has_all_five_source_types_with_details(breaching_case) -> None:
    card = breaching_case["card"]
    types = [s["type"] for s in card["sources"]]
    assert set(types) == set(evidence.SOURCE_TYPES)
    assert len(types) == len(set(types))
    for src in card["sources"]:
        assert src["detail"] and any(ch.isdigit() for ch in src["detail"])


def test_card_carries_validity_window_and_composed_lag(breaching_case) -> None:
    """The two honesty fields: how long the projection is valid, and the real lag."""
    phys = next(s for s in breaching_case["card"]["sources"] if s["type"] == "physics")
    detail = phys["physics_detail"]
    assert detail["validity_sec"] > 0.0
    composed = detail["effective_measurement_lag_sec"]
    assert composed == pytest.approx(
        detail["transport_lag_sec"] + detail["scanner_lag_sec"], abs=0.05
    )
    assert 20.0 < composed < 90.0
    assert detail["dataset_median_effective_lag_sec"] == pytest.approx(41.44, abs=0.5)


def test_historical_source_names_its_episodes(breaching_case) -> None:
    hist = next(s for s in breaching_case["card"]["sources"] if s["type"] == "historical")
    assert hist["k"] == hist["n_in_spec"] + hist["n_off_spec"]
    assert len(hist["episode_ids"]) == hist["n_in_spec"]
    for episode_id in hist["episode_ids"] + hist["off_spec_episode_ids"]:
        assert (ROOT / episode_id).is_dir()


def test_narration_passes_the_numeral_validator(breaching_case) -> None:
    card = breaching_case["card"]
    text = evidence.narrate_validated(card)
    ok, offenders = evidence.validate_narration(card, text)
    assert ok, offenders
    assert 2 <= text.count(". ") + text.endswith(".") <= 4


def test_broken_card_narration_is_rejected(breaching_case) -> None:
    """A deliberately corrupted narration must fail closed, not be displayed."""
    card = breaching_case["card"]
    tampered = evidence.narrate(card) + " Expect a 137.91% excursion."
    ok, offenders = evidence.validate_narration(card, tampered)
    assert not ok
    assert "137.91" in offenders


def test_narration_rejects_a_card_whose_numbers_were_edited(breaching_case) -> None:
    """Same check from the other side: shift the card, keep the prose, still fail.

    Built on a synthetic card rather than a real one so the edited figure is unique. On a
    real card several fields legitimately carry the same number - the proposal's starting
    point *is* the current stock flow - and moving one of them proves nothing.
    """
    card = copy.deepcopy(breaching_case["card"])
    card["counterfactual"]["no_action_max_dev_pct"] = 8.31
    good = evidence.narrate(card)
    assert "8.3" in good
    assert evidence.validate_narration(card, good)[0]

    card["counterfactual"]["no_action_max_dev_pct"] = 4.77
    ok, offenders = evidence.validate_narration(card, good)
    assert not ok
    assert "8.3" in offenders


def test_attach_narration_leaves_none_on_failure(breaching_case, monkeypatch) -> None:
    card = copy.deepcopy(breaching_case["card"])
    monkeypatch.setattr(evidence, "narrate", lambda _c: "Push it to 99999.9 straight away.")
    out = evidence.attach_narration(card)
    assert out["narration"] is None
    assert "narration_error" in out


def test_narration_emits_a_gap_notice_for_a_missing_field(breaching_case) -> None:
    card = copy.deepcopy(breaching_case["card"])
    card["action"]["to"] = None
    text = evidence.narrate(card)
    assert "incomplete" in text
    assert evidence.validate_narration(card, text)[0]


def test_narration_uses_operator_register(breaching_case) -> None:
    text = evidence.narrate(breaching_case["card"]).lower()
    for jargon in ("residual", "regressor", "feature", "p-value", "correlation", "gradient"):
        assert jargon not in text


def test_advisor_never_reads_simulator_ground_truth() -> None:
    """No path from the advisor or the feedback log to planted faults."""
    banned = ("injected_faults", "bw_true")
    for path in list(Path("src/advisor").rglob("*.py")) + list(Path("src/feedback").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path} references {token}"
