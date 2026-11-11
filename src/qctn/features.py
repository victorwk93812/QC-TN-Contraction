"""The 18 plan features of Table 2.

Reimplements `get_features_v5` in `src/Funcions_de_Features_v_5.jl` of the authors' release (936630a).
Per step i, with ranks r_A, r_B of the inputs and k_i common indices (all dimensions are 2):

    c_i = r_A + r_B - k_i            log2 of the step's flops (indices touched)
    p_i = r_A + r_B - 2 k_i          rank of the result ("out rank", the paper's parallelism)
    d_i = |r_A - r_B| / (p_i + EPS)  asymmetry
    w_i = 2 ** c_i                   cost weight

The release takes k_i = (r_A + r_B - r_C) / 2 from the file's ranks, not from the printed index list
(they agree on every plan in the release).
"""
from __future__ import annotations

import numpy as np

from qctn.plans.authors import AuthorsPlan

# Constants of the release; the paper doesn't give their values.
EPS = 1.0e-6  # calcul_Ci_Pi_Di_Ki(contractions; epsilon=1.0e-6)
TAU = 6  # fracc_tiny_steps(Ci; tau=6)
Q = 5  # mitjana_top_q_percent(v; q=5), top q percent

# Funcions_de_Features_v_5.jl, get_features_v5, and Data/splits/splits.md.
FEATURE_NAMES = [
    "max_cost", "log2_sum_flops_val", "topq_mean", "avg_cost", "std_cost", "n_steps", "frac_tiny",
    "max_out_rank", "avg_out_rank", "costw_out_rank", "p_at_max_cost_val",
    "max_red_rank_val", "costw_red_rank_val", "k_at_max_cost_val",
    "max_asym_val", "avg_asym", "costw_asym_val", "d_at_max_cost_val",
]


def step_arrays(rank_a, rank_b, rank_c) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ra, rb, rc = (np.asarray(x, dtype=np.int64) for x in (rank_a, rank_b, rank_c))
    k2 = ra + rb - rc
    if np.any(k2 % 2):
        raise ValueError("rank_a + rank_b - rank_c must be even")
    k = k2 // 2
    c = (ra + rb - k).astype(float)
    p = (ra + rb - 2 * k).astype(float)
    d = np.abs(ra - rb) / (p + EPS)
    return c, p, d, k.astype(float)


def julia_round(x: float) -> int:
    """`round(Int, x)`: nearest, ties to even (same as Python's round)."""
    return int(round(x))


def features_from_ranks(rank_a, rank_b, rank_c, tau: float = TAU, q: float = Q) -> dict[str, float]:
    c, p, d, k = step_arrays(rank_a, rank_b, rank_c)
    n = len(c)
    w = 2.0 ** c
    sw = w.sum()
    cmax = c.max()
    at_max = c == cmax
    top = np.sort(c)[::-1][: max(1, julia_round(n * q / 100))]
    return {
        "max_cost": cmax,
        "log2_sum_flops_val": np.log2(sw),
        "topq_mean": top.mean(),
        "avg_cost": c.mean(),
        "std_cost": c.std(ddof=1) if n > 1 else float("nan"),  # Statistics.std is the sample std
        "n_steps": float(n),
        "frac_tiny": np.count_nonzero(c <= cmax - tau) / n,
        "max_out_rank": p.max(),
        "avg_out_rank": p.mean(),
        "costw_out_rank": (p * w).sum() / sw,
        "p_at_max_cost_val": p[at_max].mean(),
        "max_red_rank_val": k.max(),
        "costw_red_rank_val": (k * w).sum() / sw,
        "k_at_max_cost_val": k[at_max].mean(),
        "max_asym_val": d.max(),
        "avg_asym": d.mean(),
        "costw_asym_val": (d * w).sum() / sw,
        "d_at_max_cost_val": d[at_max].mean(),
    }


def plan_features(plan: AuthorsPlan, **kw) -> dict[str, float]:
    s = plan.steps
    return features_from_ranks([x.rank_a for x in s], [x.rank_b for x in s], [x.rank_c for x in s], **kw)


def tree_features(tree, rank, **kw) -> dict[str, float]:
    """Features of a PlanTree given `rank(name) -> int` for leaves and intermediates."""
    ch = [tree.children[c] for c in tree.order]
    return features_from_ranks([rank(a) for a, _ in ch], [rank(b) for _, b in ch],
                               [rank(c) for c in tree.order], **kw)
