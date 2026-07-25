"""Dataset generator CLI.

``python -m src.sim.generate --n 300 --out data/episodes --seed 42``

Grade pairs are drawn **Zipf**, not uniformly, so the dataset looks like a real
production log: a handful of routine transitions dominate and a long tail of pairs
appears once or never. That tail is the point -- M6 has to retrieve by proximity in
grade space for pairs it has never seen.

The Zipf ranking is biased toward *narrow* transitions rather than being a plain
random permutation. Mills mostly step between neighbouring grades; a 45 -> 160 g/m2
change is a scheduling event, not a routine one. The bias is deliberately noisy, so
transition width and rarity are correlated but not interchangeable -- otherwise
"rare" and "hard" would be the same variable and every downstream model could cheat
on one by learning the other.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.sim.episode import SimConfig, episode_id, simulate_episode
from src.sim.grades import GradeProps, build_catalogue, write_grades_json
from src.sim.machine import MachineConfig
from src.sim.writer import rebuild_index, write_episode

__all__ = ["pair_weights", "generate", "main"]

#: Zipf exponent over the ranked pair list.
ZIPF_EXPONENT: float = 1.05
#: Standard deviation of the noise added to the width ranking, in normalised units.
RANK_NOISE: float = 0.40


def pair_weights(
    catalogue: dict[str, GradeProps], rng: np.random.Generator
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Ordered grade pairs and their Zipf sampling probabilities."""
    pairs = [(a, b) for a in sorted(catalogue) for b in sorted(catalogue) if a != b]

    bw = np.array([g.bw for g in catalogue.values()], dtype=float)
    span = float(bw.max() - bw.min())
    width = np.array(
        [
            abs(catalogue[b].bw - catalogue[a].bw) / span
            + abs(catalogue[b].ash - catalogue[a].ash) / 40.0
            for a, b in pairs
        ],
        dtype=float,
    )
    score = width + rng.normal(0.0, RANK_NOISE, size=width.size)
    order = np.argsort(score, kind="stable")

    ranks = np.empty(len(pairs), dtype=float)
    ranks[order] = np.arange(1, len(pairs) + 1, dtype=float)
    weights = ranks ** (-ZIPF_EXPONENT)
    return pairs, weights / weights.sum()


def generate(
    n: int,
    out: str | Path,
    *,
    seed: int = 42,
    machine: MachineConfig | None = None,
    config: SimConfig | None = None,
    grades_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate ``n`` episodes into ``out`` and rebuild the index.

    Deterministic in ``seed``: the pair draw, the per-episode seeds and every
    episode's internals all descend from it, and episodes are written in a fixed
    order. Returns a summary dict.
    """
    cfg = config or SimConfig()
    mac = machine or MachineConfig()
    catalogue = build_catalogue()
    out_p = Path(out)

    write_grades_json(grades_path or (out_p.parent / "grades.json"), catalogue)

    root_rng = np.random.default_rng(seed)
    pairs, probs = pair_weights(catalogue, root_rng)
    chosen = root_rng.choice(len(pairs), size=n, p=probs)
    # Per-episode seeds are drawn from the root generator, so an episode's content
    # depends only on the master seed and its position in the run.
    episode_seeds = root_rng.integers(0, 2**31 - 1, size=n)

    seq_counter: Counter[tuple[str, str]] = Counter()
    summary_rows: list[dict[str, Any]] = []

    for i in range(n):
        a, b = pairs[int(chosen[i])]
        seq_counter[(a, b)] += 1
        seq = seq_counter[(a, b)]
        df, meta = simulate_episode(
            catalogue[a],
            catalogue[b],
            seed=int(episode_seeds[i]),
            seq=seq,
            machine=mac,
            config=cfg,
        )
        assert meta["episode_id"] == episode_id(a, b, seq)
        write_episode(out_p, df, meta)
        summary_rows.append(
            {
                "episode_id": meta["episode_id"],
                "faults": tuple(meta["injected_faults"]),
                "off_spec": meta["labels"]["off_spec"],
                "stabilisation_t_sec": meta["labels"]["stabilisation_t_sec"],
                "max_dev_pct": meta["labels"]["max_dev_pct"],
                "broke_tonnes": meta["labels"]["broke_tonnes"],
            }
        )

    rebuild_index(out_p)
    return _summarise(summary_rows, n_pairs=len(pairs), n_pairs_used=len(seq_counter))


def _summarise(rows: list[dict[str, Any]], *, n_pairs: int, n_pairs_used: int) -> dict[str, Any]:
    n = len(rows)
    off = [r for r in rows if r["off_spec"]]
    stab = [r["stabilisation_t_sec"] for r in rows if r["stabilisation_t_sec"] is not None]
    fault_counts: Counter[str] = Counter()
    clean = 0
    for r in rows:
        if not r["faults"]:
            clean += 1
        for f in r["faults"]:
            fault_counts[f] += 1
    return {
        "n_episodes": n,
        "off_spec_rate": len(off) / n if n else 0.0,
        "mean_stabilisation_t_sec": float(np.mean(stab)) if stab else float("nan"),
        "n_never_stabilised": n - len(stab),
        "mean_max_dev_pct": float(np.mean([r["max_dev_pct"] for r in rows])) if n else 0.0,
        "total_broke_tonnes": float(np.sum([r["broke_tonnes"] for r in rows])) if n else 0.0,
        "fault_counts": dict(sorted(fault_counts.items())),
        "n_clean_episodes": clean,
        "n_pairs_possible": n_pairs,
        "n_pairs_seen": n_pairs_used,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic grade-change episodes.")
    parser.add_argument("--n", type=int, default=300, help="number of episodes")
    parser.add_argument("--out", type=str, default="data/episodes", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="master seed")
    parser.add_argument(
        "--grades", type=str, default=None, help="path for grades.json (default <out>/../grades.json)"
    )
    args = parser.parse_args(argv)

    summary = generate(args.n, args.out, seed=args.seed, grades_path=args.grades)

    print(f"episodes           : {summary['n_episodes']}")
    print(f"off-spec rate      : {summary['off_spec_rate'] * 100:.1f}%  (target 25-45%)")
    print(f"mean stabilisation : {summary['mean_stabilisation_t_sec']:.1f} s")
    print(f"never stabilised   : {summary['n_never_stabilised']}")
    print(f"mean max deviation : {summary['mean_max_dev_pct']:.2f}%")
    print(f"broke              : {summary['total_broke_tonnes']:.1f} t")
    print(f"clean episodes     : {summary['n_clean_episodes']}")
    print(f"grade pairs seen   : {summary['n_pairs_seen']} of {summary['n_pairs_possible']}")
    print("injected faults    :")
    for name, count in summary["fault_counts"].items():
        print(f"    {name:<26} {count}")

    lo, hi = 0.25, 0.45
    if not lo <= summary["off_spec_rate"] <= hi:
        print(
            f"WARNING: off-spec rate {summary['off_spec_rate'] * 100:.1f}% is outside "
            f"{lo * 100:.0f}-{hi * 100:.0f}% - the simulator is miscalibrated",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
