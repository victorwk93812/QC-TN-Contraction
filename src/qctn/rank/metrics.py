"""Table 6/7 metrics, following the authors' `evaluate_tops_and_regrets`.

Within a group, plans with equal predicted scores share a rank, and the next rank
skips past the tied block. The truly best plan is the first row holding the maximum
target. Regret@k is 1 - best_in_top_k / best, on whichever target column is given.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    top1: float
    top3: float
    regret1: float
    regret3: float
    win: float


def competition_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 is the highest score; ties share a rank (1, 2, 2, 4, ...)."""
    return np.array([1 + np.count_nonzero(scores > s) for s in scores])


def evaluate(df: pd.DataFrame, scores, target: str = "speedup") -> Metrics:
    """Score a ranking of the plans in `df`, grouped by `base_id`.

    Win rate compares the plan ranked first with the MinFill reference (speedup 1),
    counting ties as half. It always uses `speedup`, as that is what it's defined on.
    """
    df = df.assign(_score=np.asarray(scores))
    hit1, hit3, reg1, reg3, wins = [], [], [], [], []
    for _, g in df.groupby("base_id", sort=False):
        y = g[target].to_numpy(dtype=float)
        ranks = competition_ranks(g["_score"].to_numpy(dtype=float))
        best = y.max()
        best_rank = ranks[int(np.argmax(y))]
        hit1.append(best_rank == 1)
        hit3.append(best_rank <= 3)
        reg1.append(1 - y[ranks == 1].max() / best)
        reg3.append(1 - y[ranks <= 3].max() / best)
        chosen = g["speedup"].to_numpy(dtype=float)[ranks == 1].max()
        wins.append(1.0 if chosen > 1 else 0.5 if chosen == 1 else 0.0)
    return Metrics(
        top1=100 * float(np.mean(hit1)),
        top3=100 * float(np.mean(hit3)),
        regret1=float(np.mean(reg1)),
        regret3=float(np.mean(reg3)),
        win=100 * float(np.mean(wins)),
    )
