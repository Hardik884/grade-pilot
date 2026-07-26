"""Retrieve similar past transitions by position in grade-property space.

Why property space and not grade codes
--------------------------------------
The obvious way to find "similar past transitions" is to look up other episodes with the
same ``grade_from``/``grade_to`` pair. That fails on this dataset, and it fails in a real
mill for the same reason: the pair distribution has a long tail. **30 of the grade pairs
present in ``data/episodes`` occur exactly once.** For those, code matching returns zero
neighbours and the advisor has nothing historical to say - precisely on the unusual
transitions where an operator most wants help.

So a transition is embedded instead as a *vector*: each grade becomes
``[bw, ash, moist, caliper]`` normalised per dimension across the whole catalogue, and the
transition is the difference ``to_props - from_props`` in that normalised space. A
G12 -> G11 change (heavier, less ash, wetter, bulkier) then sits close to G09 -> G10 and
G10 -> G11 regardless of whether that exact code pair has ever run before. The physics the
operator cares about lives in the property deltas, not in the labels.

``nominal_speed_m_min`` is deliberately excluded from the embedding: it is an operating
parameter chosen per grade, not a property of the paper (see the episode-schema contract).
It comes back later as the projection's speed target, not as a similarity dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import loader

__all__ = [
    "GRADE_DIMS",
    "DEFAULT_K",
    "Neighbour",
    "Normaliser",
    "build_normaliser",
    "embed_grade",
    "transition_vector",
    "load_catalogue",
    "find_similar",
    "pair_counts",
    "succeeded",
    "failed",
]

#: The embedding dimensions, in order. Machine speed is not one of them.
GRADE_DIMS: tuple[str, ...] = ("bw", "ash", "moist", "caliper")

#: Neighbourhood size. Nine is small enough that every neighbour can be named in the
#: evidence card and large enough to split into succeeded/failed groups.
DEFAULT_K: int = 9

_FEATURE_CACHE = Path("data/features_90s.parquet")


@dataclass(frozen=True)
class Normaliser:
    """Per-dimension min/max over the grade catalogue."""

    lo: dict[str, float]
    span: dict[str, float]

    def __call__(self, props: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [(float(props[d]) - self.lo[d]) / self.span[d] for d in GRADE_DIMS],
            dtype=float,
        )


@dataclass(frozen=True)
class Neighbour:
    """One retrieved past transition, with the outcome the advisor learns from."""

    episode_id: str
    grade_from: str
    grade_to: str
    distance: float
    off_spec: bool
    max_dev_pct: float
    stab_from_ramp_end_sec: float | None
    vector: np.ndarray = field(default_factory=lambda: np.zeros(len(GRADE_DIMS)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "pair": f"{self.grade_from}->{self.grade_to}",
            "distance": round(self.distance, 4),
            "off_spec": bool(self.off_spec),
            "max_dev_pct": round(float(self.max_dev_pct), 3),
            "stab_from_ramp_end_sec": (
                None
                if self.stab_from_ramp_end_sec is None
                or not np.isfinite(self.stab_from_ramp_end_sec)
                else round(float(self.stab_from_ramp_end_sec), 1)
            ),
        }


def build_normaliser(grades: Mapping[str, Mapping[str, float]] | None = None) -> Normaliser:
    """Min-max normaliser fitted on the grade catalogue, not on the episode sample.

    Fitting on the catalogue keeps the embedding stable as episodes are added: the same
    transition always has the same coordinates.
    """
    grades = grades if grades is not None else loader.load_grades()
    lo: dict[str, float] = {}
    span: dict[str, float] = {}
    for dim in GRADE_DIMS:
        values = [float(props[dim]) for props in grades.values()]
        d_lo, d_hi = min(values), max(values)
        lo[dim] = d_lo
        # A degenerate dimension must not divide by zero; it simply contributes nothing.
        span[dim] = (d_hi - d_lo) if d_hi > d_lo else 1.0
    return Normaliser(lo=lo, span=span)


def embed_grade(props: Mapping[str, float], normaliser: Normaliser) -> np.ndarray:
    """One grade as a point in normalised ``[bw, ash, moist, caliper]`` space."""
    return normaliser(props)


def transition_vector(
    from_props: Mapping[str, float],
    to_props: Mapping[str, float],
    normaliser: Normaliser,
) -> np.ndarray:
    """A transition as ``to - from`` in normalised grade space.

    Sign matters: G07 -> G12 (heavier) and G12 -> G07 (lighter) are opposite vectors and
    must not retrieve each other, because the corrective move is opposite too.
    """
    return embed_grade(to_props, normaliser) - embed_grade(from_props, normaliser)


def _props_from_index_row(row: pd.Series, side: str) -> dict[str, float]:
    return {d: float(row[f"grade_{side}_props.{d}"]) for d in GRADE_DIMS}


def load_catalogue(
    root: str | Path = "data/episodes",
    *,
    grades: Mapping[str, Mapping[str, float]] | None = None,
    feature_cache: str | Path | None = _FEATURE_CACHE,
) -> pd.DataFrame:
    """Every past episode as a transition vector plus its outcome labels.

    Reads ``index.parquet`` (one read, no ground truth) and joins the derived
    ``stab_from_ramp_end_sec`` from the cached feature table when it is available.
    Stabilisation measured from ramp end is the honest number: ``labels`` carries
    ``stabilisation_t_sec``, which is 1.0 on episodes that never left the band.
    """
    index = loader.load_index(root)
    normaliser = build_normaliser(grades)

    vectors = np.vstack(
        [
            transition_vector(
                _props_from_index_row(row, "from"), _props_from_index_row(row, "to"), normaliser
            )
            for _, row in index.iterrows()
        ]
    )

    out = pd.DataFrame(
        {
            "episode_id": index["episode_id"].astype(str),
            "grade_from": index["grade_from"].astype(str),
            "grade_to": index["grade_to"].astype(str),
            "off_spec": index["labels.off_spec"].astype(bool),
            "max_dev_pct": index["labels.max_dev_pct"].astype(float),
        }
    )
    for i, dim in enumerate(GRADE_DIMS):
        out[f"v_{dim}"] = vectors[:, i]

    out["stab_from_ramp_end_sec"] = np.nan
    cache = Path(feature_cache) if feature_cache is not None else None
    if cache is not None and cache.exists():
        feats = pd.read_parquet(cache, columns=["episode_id", "stab_from_ramp_end_sec"])
        merged = out.merge(feats, on="episode_id", how="left", suffixes=("", "_f"))
        out["stab_from_ramp_end_sec"] = merged["stab_from_ramp_end_sec_f"].to_numpy()
    return out


def pair_counts(catalogue: pd.DataFrame) -> pd.Series:
    """How many episodes exist per grade pair. The tail is the point of this module."""
    return catalogue.groupby(["grade_from", "grade_to"]).size().sort_values()


def find_similar(
    meta: Mapping[str, Any],
    k: int = DEFAULT_K,
    *,
    catalogue: pd.DataFrame | None = None,
    grades: Mapping[str, Mapping[str, float]] | None = None,
    root: str | Path = "data/episodes",
) -> list[Neighbour]:
    """The ``k`` nearest past transitions to ``meta``, by grade-space vector.

    The query episode is excluded by ``episode_id``, so this is usable as leave-one-out
    retrieval on historical data as well as live. Distance is Euclidean in the normalised
    delta space; every dimension therefore contributes on a comparable 0-1 scale.
    """
    if catalogue is None:
        catalogue = load_catalogue(root, grades=grades)
    normaliser = build_normaliser(grades)

    query = transition_vector(meta["grade_from_props"], meta["grade_to_props"], normaliser)
    pool = catalogue.loc[catalogue["episode_id"] != str(meta.get("episode_id", ""))]
    if pool.empty:
        return []

    vectors = pool[[f"v_{d}" for d in GRADE_DIMS]].to_numpy(dtype=float)
    distances = np.linalg.norm(vectors - query[None, :], axis=1)
    order = np.argsort(distances, kind="stable")[: max(int(k), 0)]

    rows = pool.iloc[order]
    out: list[Neighbour] = []
    for (_, row), dist in zip(rows.iterrows(), distances[order]):
        stab = row["stab_from_ramp_end_sec"]
        out.append(
            Neighbour(
                episode_id=str(row["episode_id"]),
                grade_from=str(row["grade_from"]),
                grade_to=str(row["grade_to"]),
                distance=float(dist),
                off_spec=bool(row["off_spec"]),
                max_dev_pct=float(row["max_dev_pct"]),
                stab_from_ramp_end_sec=(None if pd.isna(stab) else float(stab)),
                vector=vectors[order][len(out)],
            )
        )
    return out


def succeeded(neighbours: Iterable[Neighbour]) -> list[Neighbour]:
    """Neighbours that stayed in spec - the ones worth imitating."""
    return [n for n in neighbours if not n.off_spec]


def failed(neighbours: Sequence[Neighbour]) -> list[Neighbour]:
    """Neighbours that went off spec - the counterfactual evidence."""
    return [n for n in neighbours if n.off_spec]
