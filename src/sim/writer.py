"""Atomic episode persistence and index derivation.

An episode is written into a temporary directory and renamed into place, so a
reader never sees a directory containing a ``meta.json`` without its
``series.parquet``, or half a parquet file.

``index.parquet`` is derived and regenerated from the ``meta.json`` files. It
deliberately **omits** ``injected_faults``: the index is the fast query surface every
downstream module uses, and simulator ground truth must not be one query away from
M3-M6. The faults stay in ``meta.json``, where only the tests go looking.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["write_episode", "read_episode", "rebuild_index", "INDEX_NAME"]

INDEX_NAME = "index.parquet"
_SERIES_NAME = "series.parquet"
_META_NAME = "meta.json"
_TMP_PREFIX = ".tmp-"

#: Never written to ``index.parquet``.
_GROUND_TRUTH_KEYS = frozenset({"injected_faults"})


def write_episode(root: str | Path, df: pd.DataFrame, meta: dict[str, Any]) -> Path:
    """Write one episode atomically. Returns the episode directory."""
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    episode_id = str(meta["episode_id"])
    dest = root_p / episode_id
    tmp = root_p / f"{_TMP_PREFIX}{episode_id}"

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        df.to_parquet(tmp / _SERIES_NAME)
        (tmp / _META_NAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def read_episode(episode_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one episode back. Round-trips :func:`write_episode` exactly."""
    d = Path(episode_dir)
    df = pd.read_parquet(d / _SERIES_NAME)
    meta = json.loads((d / _META_NAME).read_text(encoding="utf-8"))
    return df, meta


def rebuild_index(root: str | Path) -> pd.DataFrame:
    """Regenerate ``index.parquet`` from the ``meta.json`` files on disk.

    Derived, never hand-edited. Safe to call on a directory that already has one.
    """
    root_p = Path(root)
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(root_p.glob(f"*/{_META_NAME}")):
        if meta_path.parent.name.startswith(_TMP_PREFIX):
            continue
        rows.append(_flatten(json.loads(meta_path.read_text(encoding="utf-8"))))

    index = pd.DataFrame(rows)
    if not index.empty:
        index = index.sort_values("episode_id").reset_index(drop=True)
    dest = root_p / INDEX_NAME
    tmp = root_p / f"{_TMP_PREFIX}{INDEX_NAME}"
    index.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
    return index


def _flatten(meta: dict[str, Any]) -> dict[str, Any]:
    """One flat row per episode: ``labels.off_spec``, ``machine.trim_m``, and each
    recipe limit split into ``_lo``/``_hi`` so the constraint filter can query them
    without unpacking a list."""
    row: dict[str, Any] = {}
    for key, value in meta.items():
        if key in _GROUND_TRUTH_KEYS:
            continue
        if not isinstance(value, dict):
            row[key] = value
            continue
        for sub, sub_value in value.items():
            name = f"{key}.{sub}"
            if isinstance(sub_value, (list, tuple)) and len(sub_value) == 2:
                row[f"{name}_lo"], row[f"{name}_hi"] = (float(sub_value[0]), float(sub_value[1]))
            else:
                row[name] = sub_value
    return row
