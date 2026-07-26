"""Feedback store tests: round-trip, reason-code discipline, and the trust breakdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.feedback import store


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "feedback.db"


def _card(card_id: str = "EC-1", dominant: str = "physics") -> dict:
    weights = {"physics": 0.2, "causal": 0.2, "historical": 0.2, "recipe": 0.2, "model": 0.2}
    weights[dominant] = 0.9
    return {
        "card_id": card_id,
        "episode_id": "EP-G12-G11-0002",
        "sources": [{"type": t, "detail": "x", "weight": w} for t, w in weights.items()],
    }


def test_round_trip(db: Path) -> None:
    row = store.record("EC-1", "EP-G12-G11-0002", "accepted", db_path=db)
    assert row["decision"] == "accepted"

    rows = store.decisions(db)
    assert len(rows) == 1
    assert rows[0]["card_id"] == "EC-1"
    assert rows[0]["episode_id"] == "EP-G12-G11-0002"
    assert rows[0]["reason_code"] is None
    assert rows[0]["timestamp"]

    stats = store.stats(db)
    assert stats["n"] == 1
    assert stats["accepted"] == 1
    assert stats["acceptance_rate"] == 1.0


def test_stats_reflect_every_write(db: Path) -> None:
    store.record("EC-1", "EP-A", "accepted", dominant_source="physics", db_path=db)
    store.record("EC-2", "EP-B", "accepted", dominant_source="physics", db_path=db)
    store.record("EC-3", "EP-C", "rejected", reason_code="too_late", dominant_source="model", db_path=db)
    store.record(
        "EC-4", "EP-D", "rejected", reason_code="wrong_variable", dominant_source="model", db_path=db
    )
    store.record("EC-5", "EP-E", "rejected", reason_code="too_late", dominant_source="physics", db_path=db)

    stats = store.stats(db)
    assert stats["n"] == 5
    assert stats["accepted"] == 2
    assert stats["rejected"] == 3
    assert stats["acceptance_rate"] == pytest.approx(0.4)

    by_source = stats["by_dominant_source"]
    assert by_source["physics"]["n"] == 3
    assert by_source["physics"]["acceptance_rate"] == pytest.approx(0.667, abs=0.001)
    assert by_source["model"]["n"] == 2
    assert by_source["model"]["acceptance_rate"] == 0.0

    assert stats["reason_counts"]["too_late"] == 2
    assert stats["reason_counts"]["wrong_variable"] == 1
    assert stats["reason_counts"]["unsafe"] == 0
    assert set(stats["reason_counts"]) == set(store.REASON_CODES)


def test_rejection_requires_a_reason_code(db: Path) -> None:
    with pytest.raises(ValueError, match="reason_code"):
        store.record("EC-9", "EP-Z", "rejected", db_path=db)
    assert store.stats(db)["n"] == 0


def test_unknown_reason_code_is_refused(db: Path) -> None:
    with pytest.raises(ValueError):
        store.record("EC-9", "EP-Z", "rejected", reason_code="because", db_path=db)


def test_unknown_decision_is_refused(db: Path) -> None:
    with pytest.raises(ValueError):
        store.record("EC-9", "EP-Z", "maybe", db_path=db)


def test_accept_needs_no_reason_but_may_carry_one(db: Path) -> None:
    store.record("EC-1", "EP-A", "accepted", db_path=db)
    store.record("EC-2", "EP-B", "accepted", reason_code="other", db_path=db)
    assert store.stats(db)["accepted"] == 2


def test_dominant_source_is_taken_from_the_card(db: Path) -> None:
    store.record_card_decision(_card("EC-1", "model"), "accepted", db_path=db)
    store.record_card_decision(
        _card("EC-2", "historical"), "rejected", reason_code="disagree_with_cause", db_path=db
    )
    by_source = store.stats(db)["by_dominant_source"]
    assert by_source["model"]["accepted"] == 1
    assert by_source["historical"]["accepted"] == 0


def test_empty_store_has_no_acceptance_rate(db: Path) -> None:
    stats = store.stats(db)
    assert stats["n"] == 0
    assert stats["acceptance_rate"] is None
    assert stats["by_dominant_source"] == {}


def test_rows_without_a_source_are_bucketed_not_dropped(db: Path) -> None:
    store.record("EC-1", "EP-A", "accepted", db_path=db)
    assert store.stats(db)["by_dominant_source"]["unattributed"]["n"] == 1
