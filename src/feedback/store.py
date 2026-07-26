"""Accept/reject capture. A rejection is data, not a failure to be hidden.

One SQLite table at ``data/feedback.db``. Accept and reject are recorded through the same
path with the same effort, because the acceptance rate is only meaningful if the UI does
not bias it - the dashboard conventions require the two buttons to be equally weighted and
this module has to be equally willing to store either.

``dominant_source`` is denormalised onto the row on purpose. The interesting question the
feedback log has to answer is *which kind of evidence operators trust*: cards where the
physics term carried the claim, or cards where the learned residual did. Recovering that
later would mean keeping every card forever, so the one derived field travels with the
decision. Everything else on the row is exactly the contract: card, episode, decision,
reason code, timestamp.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = [
    "DEFAULT_DB",
    "DECISIONS",
    "REASON_CODES",
    "connect",
    "init_db",
    "dominant_source_of",
    "record",
    "record_card_decision",
    "decisions",
    "stats",
]

DEFAULT_DB = Path("data/feedback.db")

#: The two decisions. Nothing else is storable.
DECISIONS: tuple[str, ...] = ("accepted", "rejected")

#: Reason codes from the evidence-card contract. A reject must carry one of these.
REASON_CODES: tuple[str, ...] = (
    "unsafe",
    "already_handling",
    "wrong_variable",
    "too_aggressive",
    "too_late",
    "disagree_with_cause",
    "other",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    card_id         TEXT NOT NULL,
    episode_id      TEXT NOT NULL,
    decision        TEXT NOT NULL CHECK (decision IN ('accepted', 'rejected')),
    reason_code     TEXT,
    timestamp       TEXT NOT NULL,
    dominant_source TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_card ON feedback (card_id);
"""


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open (and create) the feedback database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB) -> Path:
    with _session(db_path):
        pass
    return Path(db_path)


def dominant_source_of(card: Mapping[str, Any]) -> str | None:
    """The source type carrying the most weight on a card."""
    sources = card.get("sources") or []
    if not sources:
        return None
    return str(max(sources, key=lambda s: float(s.get("weight", 0.0))).get("type"))


def record(
    card_id: str,
    episode_id: str,
    decision: str,
    *,
    reason_code: str | None = None,
    dominant_source: str | None = None,
    timestamp: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> dict[str, Any]:
    """Write one decision. Returns the row as stored, for the UI's confirmation.

    A reject without a reason code is refused: the reason is the training signal, and a
    log full of unexplained rejections tells you nothing about suggestion quality.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if decision == "rejected":
        if reason_code is None:
            raise ValueError("a rejection requires a reason_code")
        if reason_code not in REASON_CODES:
            raise ValueError(f"reason_code must be one of {REASON_CODES}, got {reason_code!r}")
    elif reason_code is not None and reason_code not in REASON_CODES:
        raise ValueError(f"reason_code must be one of {REASON_CODES}, got {reason_code!r}")

    row = {
        "card_id": str(card_id),
        "episode_id": str(episode_id),
        "decision": decision,
        "reason_code": reason_code,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dominant_source": dominant_source,
    }
    with _session(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback (card_id, episode_id, decision, reason_code, timestamp, "
            "dominant_source) VALUES (:card_id, :episode_id, :decision, :reason_code, "
            ":timestamp, :dominant_source)",
            row,
        )
    return row


def record_card_decision(
    card: Mapping[str, Any],
    decision: str,
    *,
    reason_code: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> dict[str, Any]:
    """Convenience path for the UI: pull ids and dominant source straight off the card."""
    return record(
        card_id=str(card["card_id"]),
        episode_id=str(card["episode_id"]),
        decision=decision,
        reason_code=reason_code,
        dominant_source=dominant_source_of(card),
        db_path=db_path,
    )


def decisions(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    """Every row, oldest first."""
    with closing(connect(db_path)) as conn:
        conn.executescript(_SCHEMA)
        rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp, rowid").fetchall()
    return [dict(r) for r in rows]


def stats(db_path: str | Path = DEFAULT_DB) -> dict[str, Any]:
    """Acceptance overall, by dominant source type, and rejection reasons by count.

    The by-source breakdown is the trust-calibration view: it says whether operators are
    accepting the physics-led cards and rejecting the model-led ones, which is the signal
    that tells you where the system is actually credible.
    """
    rows = decisions(db_path)
    total = len(rows)
    accepted = sum(1 for r in rows if r["decision"] == "accepted")

    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["dominant_source"] or "unattributed"
        bucket = by_source.setdefault(key, {"n": 0, "accepted": 0, "acceptance_rate": 0.0})
        bucket["n"] += 1
        bucket["accepted"] += int(row["decision"] == "accepted")
    for bucket in by_source.values():
        bucket["acceptance_rate"] = round(bucket["accepted"] / bucket["n"], 3) if bucket["n"] else 0.0

    reasons = Counter(r["reason_code"] for r in rows if r["reason_code"])
    return {
        "n": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": round(accepted / total, 3) if total else None,
        "by_dominant_source": dict(sorted(by_source.items())),
        "reason_counts": {code: int(reasons.get(code, 0)) for code in REASON_CODES},
    }
