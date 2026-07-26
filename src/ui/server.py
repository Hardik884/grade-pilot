"""HTTP surface for the operator dashboard. Thin wrapper, no new logic.

Run it from the repo root, because every module below resolves its data paths relative to
the working directory::

    python -m src.ui.server

Then open http://127.0.0.1:8000/.

Every endpoint is an assembly of calls that already exist elsewhere: loader for episodes,
:class:`~src.analysis.predictor.Predictor` for the forecast, :mod:`src.advisor.suggest` for
the constrained search, :mod:`src.advisor.evidence` for the card and narration,
:mod:`src.feedback.store` for the decision log. The only arithmetic performed here is the
uncertainty cone for the chart, which is documented where it happens.

Security note: this binds to loopback and has no authentication. It is a local demo
surface for a single operator on their own machine, and it writes to ``data/feedback.db``.
Do not expose it on a routable interface.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from flask import Flask, Response, request

from src.advisor import evidence, suggest as advisor
from src.advisor.retrieval import DEFAULT_K, load_catalogue
from src.analysis import loader
from src.analysis.predictor import Predictor
from src.feedback import store
from src.ui.economics import broke_avoided

ROOT = Path("data/episodes")
FEATURES = Path("data/features_90s.parquet")
IMPACT_RANKING = Path("data/impact_ranking.json")
GRADES = Path("data/grades.json")

#: Episode ids are used to build filesystem paths, so they are matched, not trusted.
_ID_RE = re.compile(r"^EP-[A-Za-z0-9]+-[A-Za-z0-9]+-\d+$")

#: Replay advances on this grid, matching the predictor's own trajectory step so the
#: chart never interpolates between two different forecasts.
REPLAY_STEP_SEC = 5.0

app = Flask(__name__, static_folder="static", static_url_path="")


# ----------------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------------
def _plain(obj: Any) -> Any:
    """Coerce numpy/pandas scalars and non-finite floats into JSON-legal values.

    ``NaN`` and ``inf`` are emitted as ``null``: ``json.dumps`` would otherwise write bare
    ``NaN``, which every browser's ``JSON.parse`` rejects.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NaT or (isinstance(obj, float) and pd.isna(obj)):
        return None
    return str(obj)


def _json(payload: Any, status: int = 200) -> Response:
    body = json.dumps(_plain(payload), allow_nan=False)
    return Response(body, status=status, mimetype="application/json")


def _fail(message: str, status: int = 400) -> Response:
    return _json({"error": message}, status)


# ----------------------------------------------------------------------------------
# Warm state. Built once, at startup.
# ----------------------------------------------------------------------------------
class State:
    """Everything expensive, fitted once and shared read-only across requests."""

    def __init__(self) -> None:
        self.features: pd.DataFrame
        self.index: pd.DataFrame
        self.catalogue: pd.DataFrame
        self.grades: dict[str, dict[str, float]]
        self.predictor: Predictor
        self.fit_report: dict[str, Any]
        self.residual_report: dict[str, Any]
        self.default_episode_id: str | None = None
        self._episodes: dict[str, loader.EpisodeData] = {}
        self._predictions: dict[tuple[str, float], dict[str, Any]] = {}
        self._suggestions: dict[tuple[str, float], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def build(self) -> None:
        t0 = time.perf_counter()
        self.features = (
            pd.read_parquet(FEATURES)
            if FEATURES.exists()
            else loader.build_feature_table(ROOT, cache=FEATURES)
        )
        self.index = loader.load_index(ROOT)
        self.catalogue = load_catalogue(ROOT)
        self.grades = loader.load_grades(GRADES)
        table = Predictor.physics_table(self.features, ROOT, grades=self.grades)
        self.predictor = Predictor()
        self.fit_report = self.predictor.fit(table)
        self.residual_report = self.predictor.residual_report(table)
        store.init_db()
        self.default_episode_id = self._pick_default()
        print(
            f"ready in {time.perf_counter() - t0:.1f}s  "
            f"episodes={len(self.index)}  fit={self.fit_report}  "
            f"physics_share={self.residual_report['physics_contribution_mean']}  "
            f"opens_on={self.default_episode_id}"
        )

    def _pick_default(self, candidates: int = 8) -> str | None:
        """Costliest transition that the advisor can actually say something about.

        Ranked by realised broke tonnage, then walked from the top until one produces a
        card at the feature horizon. Opening on an episode where the advisor correctly has
        nothing to propose is an honest state but a useless first screen, and checking a
        handful of candidates costs well under a second.
        """
        ranked = self.index.sort_values("labels.broke_tonnes", ascending=False)
        ids = [str(v) for v in ranked["episode_id"].head(candidates)]
        for episode_id in ids:
            try:
                if _suggestion(episode_id, loader.FEATURE_HORIZON_SEC)["card"] is not None:
                    return episode_id
            except Exception as exc:  # a bad episode must not stop the server booting
                print(f"default pick skipped {episode_id}: {type(exc).__name__}: {exc}")
        return ids[0] if ids else None

    # -- episode access ------------------------------------------------------------
    def episode(self, episode_id: str) -> loader.EpisodeData:
        with self._lock:
            hit = self._episodes.get(episode_id)
        if hit is not None:
            return hit
        ep = loader.load_episode(ROOT / episode_id)
        with self._lock:
            if len(self._episodes) > 8:
                self._episodes.clear()
            self._episodes[episode_id] = ep
        return ep

    def target_speed(self, meta: Mapping[str, Any]) -> float | None:
        grade = self.grades.get(str(meta["grade_to"]))
        return None if grade is None else float(grade["nominal_speed_m_min"])

    def feature_row(self, episode_id: str, t_sec: float) -> pd.DataFrame | None:
        """The residual model's features, or ``None`` when they would be lookahead.

        The feature table is cut at :data:`loader.FEATURE_HORIZON_SEC`. Handing it to the
        predictor at an earlier replay position would let the residual see data the
        operator does not have yet, so before the horizon the forecast is physics only and
        says so in ``residual_applied``.
        """
        if t_sec < loader.FEATURE_HORIZON_SEC:
            return None
        rows = self.features.loc[self.features["episode_id"] == episode_id]
        return None if rows.empty else rows.reset_index(drop=True)


STATE = State()


# ----------------------------------------------------------------------------------
# Prediction and advice, memoised on the replay grid
# ----------------------------------------------------------------------------------
def _snap(t_sec: float) -> float:
    return round(round(float(t_sec) / REPLAY_STEP_SEC) * REPLAY_STEP_SEC, 1)


def _uncertainty_band(
    trajectory: list[tuple[float, float]], t_now: float, spread_pct: float, validity_sec: float
) -> dict[str, list[float]]:
    """Cone half-width per trajectory point, in g/m2.

    ``spread_pct`` is the same figure the evidence card puts in ``claim.interval``, so the
    cone and the card cannot disagree. It is ramped rather than constant: at ``t_now`` the
    sheet is already formed and only sensor calibration is in question, while at the far
    end of the validity window the whole residual is in play. Past the validity horizon it
    keeps widening on the same slope, which is the honest picture - out there the control
    loop is reacting to data that does not exist yet.
    """
    span = max(float(validity_sec), 1.0)
    lo: list[float] = []
    hi: list[float] = []
    for t_sec, value in trajectory:
        frac = max((float(t_sec) - t_now) / span, 0.0)
        half = float(value) * (spread_pct / 100.0) * (0.2 + 0.8 * frac)
        lo.append(round(float(value) - half, 3))
        hi.append(round(float(value) + half, 3))
    return {"lo": lo, "hi": hi}


#: Memo cap. A replay revisits the same grid points constantly, so caching is what keeps
#: a step at ~70 ms, but an unbounded dict across 300 episodes is a slow leak.
_MEMO_MAX = 4000


def _prediction(episode_id: str, t_sec: float) -> dict[str, Any]:
    key = (episode_id, t_sec)
    hit = STATE._predictions.get(key)
    if hit is not None:
        return hit
    if len(STATE._predictions) > _MEMO_MAX:
        STATE._predictions.clear()

    ep = STATE.episode(episode_id)
    history = ep.history(t_sec)
    target_speed = STATE.target_speed(ep.meta)
    row = STATE.feature_row(episode_id, t_sec)
    prediction = STATE.predictor.predict(
        history, ep.meta, feature_row=row, target_speed_m_min=target_speed
    )

    # Same rule as evidence.build_card, so claim.interval and the chart cone agree.
    spread_pct = max(abs(float(prediction["model_correction_pct"])), 0.5)
    validity_sec = float(prediction["physics_detail"]["validity_sec"])
    payload = {
        **prediction,
        "episode_id": episode_id,
        "t_sec": t_sec,
        "residual_applied": row is not None,
        "spread_pct": round(spread_pct, 3),
        "validity_sec": validity_sec,
        "spec_band_pct": round(loader.SPEC_BAND * 100.0, 1),
        "uncertainty": _uncertainty_band(
            prediction["predicted_trajectory"], t_sec, spread_pct, validity_sec
        ),
        "measurement_lag": loader.effective_measurement_lag(history),
    }
    payload["measurement_lag"].pop("per_sample", None)
    STATE._predictions[key] = payload
    return payload


def _suggestion(episode_id: str, t_sec: float) -> dict[str, Any]:
    """Advice plus evidence card at one replay position.

    ``suggestion: None`` is a real answer, not an error: no breach forecast, no in-spec
    neighbour to imitate, or every candidate move blocked by the constraint filter. The
    reason is passed through so the panel can say which.
    """
    key = (episode_id, t_sec)
    hit = STATE._suggestions.get(key)
    if hit is not None:
        return hit
    if len(STATE._suggestions) > _MEMO_MAX:
        STATE._suggestions.clear()

    prediction = _prediction(episode_id, t_sec)
    ep = STATE.episode(episode_id)
    history = ep.history(t_sec)
    payload: dict[str, Any] = {
        "episode_id": episode_id,
        "t_sec": t_sec,
        "will_breach": bool(prediction["will_breach"]),
        "suggestion": None,
        "card": None,
        "economics": None,
        "reason": None,
    }

    if not prediction["will_breach"]:
        payload["reason"] = "No breach forecast at this point in the transition."
        STATE._suggestions[key] = payload
        return payload

    found = advisor.suggest(
        history,
        ep.meta,
        prediction,
        k=DEFAULT_K,
        catalogue=STATE.catalogue,
        target_speed_m_min=STATE.target_speed(ep.meta),
    )
    if found is None:
        payload["reason"] = (
            "Breach forecast, but nothing safe to propose: no similar transition held "
            "spec, or every candidate move failed the constraint filter."
        )
        STATE._suggestions[key] = payload
        return payload

    card = evidence.attach_narration(
        evidence.build_card(found, prediction, history, ep.meta, ranking_path=IMPACT_RANKING)
    )
    payload["suggestion"] = found.as_dict()
    payload["card"] = card
    payload["economics"] = broke_avoided(ep.meta["labels"], card["counterfactual"])
    payload["dominant_source"] = store.dominant_source_of(card)
    STATE._suggestions[key] = payload
    return payload


# ----------------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------------
@app.get("/")
def index() -> Response:
    return app.send_static_file("index.html")


@app.get("/api/episodes")
def api_episodes() -> Response:
    """Episode ids for the selector, with enough label context to pick a demo case."""
    idx = STATE.index
    records = [
        {
            "episode_id": str(row["episode_id"]),
            "grade_from": str(row["grade_from"]),
            "grade_to": str(row["grade_to"]),
            "off_spec": bool(row["labels.off_spec"]),
            "max_dev_pct": float(row["labels.max_dev_pct"]),
            "broke_tonnes": float(row["labels.broke_tonnes"]),
        }
        for _, row in idx.iterrows()
    ]
    records.sort(key=lambda r: r["episode_id"])
    return _json(
        {
            "episodes": records,
            "default_episode_id": STATE.default_episode_id,
            "n": len(records),
        }
    )


@app.get("/api/episode/<episode_id>")
def api_episode(episode_id: str) -> Response:
    if not _ID_RE.match(episode_id) or not (ROOT / episode_id).is_dir():
        return _fail(f"unknown episode {episode_id!r}", 404)

    ep = STATE.episode(episode_id)
    series = ep.series
    t = series.index.to_numpy(dtype=float)

    def col(name: str, decimals: int) -> list[float | None]:
        return [
            None if not math.isfinite(v) else round(float(v), decimals)
            for v in series[name].to_numpy(dtype=float)
        ]

    lag = loader.effective_measurement_lag(series)
    lag.pop("per_sample", None)
    post = t[t >= 0.0]
    return _json(
        {
            "episode_id": episode_id,
            "meta": ep.meta,
            "series": {
                "t_sec": [round(float(v), 1) for v in t],
                "bw": col("bw", 3),
                "bw_sp": col("bw_sp", 3),
                "moist": col("moist", 3),
                "ash": col("ash", 3),
                "stock_flow": col("stock_flow", 2),
                "filler_flow": col("filler_flow", 2),
                "steam_p": col("steam_p", 3),
                "speed": col("speed", 2),
                "stock_cons": col("stock_cons", 3),
                "phase": [str(v) for v in series["phase"].astype("string").to_numpy()],
            },
            "replay": {
                "t_start": float(post[0]) if post.size else 0.0,
                "t_end": float(post[-1]) if post.size else 0.0,
                "step_sec": REPLAY_STEP_SEC,
                "feature_horizon_sec": loader.FEATURE_HORIZON_SEC,
            },
            "spec_band_pct": round(loader.SPEC_BAND * 100.0, 1),
            "measurement_lag": lag,
            "ramp_end_sec": loader.ramp_end_sec(series),
        }
    )


@app.get("/api/predict/<episode_id>")
def api_predict(episode_id: str) -> Response:
    if not _ID_RE.match(episode_id) or not (ROOT / episode_id).is_dir():
        return _fail(f"unknown episode {episode_id!r}", 404)
    try:
        t_sec = _snap(request.args.get("t", loader.FEATURE_HORIZON_SEC))
    except (TypeError, ValueError):
        return _fail("t must be a number")
    if t_sec < 0.0:
        return _fail("t must be at or after the transition trigger (t=0)")
    return _json(_prediction(episode_id, t_sec))


@app.get("/api/suggest/<episode_id>")
def api_suggest(episode_id: str) -> Response:
    if not _ID_RE.match(episode_id) or not (ROOT / episode_id).is_dir():
        return _fail(f"unknown episode {episode_id!r}", 404)
    try:
        t_sec = _snap(request.args.get("t", loader.FEATURE_HORIZON_SEC))
    except (TypeError, ValueError):
        return _fail("t must be a number")
    if t_sec < 0.0:
        return _fail("t must be at or after the transition trigger (t=0)")
    return _json(_suggestion(episode_id, t_sec))


@app.get("/api/impact-ranking")
def api_impact_ranking() -> Response:
    if not IMPACT_RANKING.exists():
        return _fail(f"{IMPACT_RANKING} missing - run the causal ranking first", 404)
    return _json(json.loads(IMPACT_RANKING.read_text(encoding="utf-8")))


@app.post("/api/feedback")
def api_feedback() -> Response:
    body = request.get_json(silent=True) or {}
    required = ("card_id", "episode_id", "decision")
    missing = [k for k in required if not body.get(k)]
    if missing:
        return _fail(f"missing field(s): {', '.join(missing)}")
    try:
        row = store.record(
            card_id=str(body["card_id"]),
            episode_id=str(body["episode_id"]),
            decision=str(body["decision"]),
            reason_code=body.get("reason_code") or None,
            dominant_source=body.get("dominant_source") or None,
        )
    except ValueError as exc:
        return _fail(str(exc))
    return _json({"recorded": row, "stats": _stats_payload()})


@app.get("/api/feedback/stats")
def api_feedback_stats() -> Response:
    return _json(_stats_payload())


def _stats_payload() -> dict[str, Any]:
    """Acceptance stats plus the reason-code vocabulary the reject control must offer."""
    return {
        **store.stats(),
        "decisions_log": store.decisions(),
        "reason_codes": list(store.REASON_CODES),
        "source_types": list(evidence.SOURCE_TYPES),
    }


@app.after_request
def _no_store(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    return response


def main() -> int:
    STATE.build()
    print("dashboard on http://127.0.0.1:8000/  (loopback only, no auth)")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
