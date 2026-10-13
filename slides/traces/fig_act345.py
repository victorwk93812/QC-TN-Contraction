"""fig_act345.py — builders for Act III (map of techniques), Act IV (plan shapes, GEMM features, ranking),
Act V and the backup slides. Registered into figures.BUILDERS.

GEMM step features follow arXiv 2608.05819 Sec 4 (Eqs. 5–11): a pairwise step with m legs only on A,
n only on B and k shared is an (2^m × 2^k)(2^k × 2^n) GEMM; c = m+n+k, p = m+n, d = |m−n|/(m+n+ε).
"""
from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np

from . import graphs as gr
from . import paper
from .carving import order_to_tree, tree_record, tree_stats
from .circuits import circuit_tn, default_circuit
from .core import eliminate, index_graph, write_trace
from .fast_elim import best_width_first_seed, eliminate_fast
from .flowcutter_lite import nested_dissection_fc
from .gn import girvan_newman

EPS = 1e-9
TAU = 6          # paper A.6: a step is "tiny" if c_i <= c_max - 6


def _fig():
    from . import figures
    return figures


def tensor_graph(T):
    owner = {}
    for v, legs in T.items():
        for l in legs:
            owner.setdefault(l, []).append(v)
    G = nx.Graph()
    G.add_nodes_from(sorted(T, key=str))
    G.add_edges_from(tuple(o) for o in owner.values() if len(o) == 2)
    return G


def plan_features(T, tree) -> dict:
    """Step primitives and the paper's plan-level features (Table 2) for one contraction tree."""
    st = tree_stats(T, tree)
    steps = st["steps"]
    c = [s["m"] + s["n"] + s["k"] for s in steps]
    p = [s["m"] + s["n"] for s in steps]
    k = [s["k"] for s in steps]
    d = [abs(s["m"] - s["n"]) / (s["m"] + s["n"] + EPS) for s in steps]
    cmax = max(c)
    w = [2.0 ** (ci - cmax) for ci in c]
    W = sum(w)
    w = [x / W for x in w]
    imax = [i for i, ci in enumerate(c) if ci == cmax]
    return {"c": c, "p": p, "k": k, "d": [round(x, 3) for x in d],
            "max_cost": cmax, "log2_sum_flops": round(math.log2(sum(2.0 ** ci for ci in c)), 2),
            "n_steps": len(c), "frac_tiny_steps": round(sum(ci <= cmax - TAU for ci in c) / len(c), 3),
            "max_out_rank": max(p), "avg_out_rank": round(sum(p) / len(p), 2),
            "costw_out_rank": round(sum(wi * pi for wi, pi in zip(w, p)), 2),
            "p_at_max_cost": round(sum(p[i] for i in imax) / len(imax), 2),
            "max_red_rank": max(k), "k_at_max_cost": round(sum(k[i] for i in imax) / len(imax), 2),
            "max_asym": round(max(d), 3), "avg_asym": round(sum(d) / len(d), 3),
            "costw_asym": round(sum(wi * di for wi, di in zip(w, d)), 3),
            "d_at_max_cost": round(sum(d[i] for i in imax) / len(imax), 3),
            "max_rank": st["max_rank"], "log2cost": round(st["log2cost"], 2)}


def critical_path(T, tree):
    """Largest sum of step costs (χ = 2) along a root-to-leaf path, as log2, and that path's depth."""
    def walk(t):
        if not isinstance(t, tuple):
            return 0, 0, frozenset(T[t])
        ca, da, la = walk(t[0])
        cb, db, lb = walk(t[1])
        own = 2 ** len(la | lb)
        best = max((ca, da), (cb, db))
        return best[0] + own, best[1] + 1, la ^ lb
    total, depth, _ = walk(tree)
    return round(math.log2(total), 2), depth


def tree_depth(t):
    return 0 if not isinstance(t, tuple) else 1 + max(tree_depth(t[0]), tree_depth(t[1]))


def circuit_plans():
    """The plans our generators produce for the talk's circuit (pairwise trees over the 33 gate tensors)."""
    T = circuit_tn(default_circuit())
    IG = index_graph(T)
    plans = {}
    seed, _, _ = best_width_first_seed(IG, "minfill", range(10))
    plans["MinFill"] = order_to_tree(T, [s["elim"] for s in eliminate_fast(IG, rule="minfill", seed=seed)["steps"]])
    seed, _, _ = best_width_first_seed(IG, "mindeg", range(10))
    plans["MinDegree"] = order_to_tree(T, [s["elim"] for s in eliminate_fast(IG, rule="mindeg", seed=seed)["steps"]])
    plans["ND / FlowCutter"] = order_to_tree(T, nested_dissection_fc(IG))
    plans["Girvan–Newman"] = girvan_newman(tensor_graph(T))["tree"]
    s44 = json.loads((_fig().DATA / "S44.json").read_text())["rows"]
    plans["Tamaki"] = order_to_tree(T, s44["tamaki"]["order"])
    plans["Jdrasil"] = order_to_tree(T, s44["jdrasil"]["order"])
    return T, plans

# ---------------------------------------------------------------- Act III


def F01():
    """Pipeline of the paper (all values from paper.py)."""
    return _fig().write_doc("F01", "static", {"title": "The paper's pipeline", "source": "arXiv 2608.05819"},
                            {"groups": paper.SPLIT["total"], "families": len(paper.FAMILIES), "family_groups_sum": sum(paper.FAMILIES.values()), "generators": len(paper.GENERATORS),
                             "features_initial": sum(len(v) for v in paper.FEATURES.values()),
                             "features_discarded": sum(1 for v in paper.FEATURES.values() for _, s in v if s == "discarded"),
                             "features_per_model": 13, "metric": "NDCG@3", "gpu": "RTX A6000", "gpu_transfer": "Tesla V100"})


def F02():
    """The seven candidates by graph used and build direction (our classification of Sec 5.1's list)."""
    cells = {"bottom-up": {"index graph": ["MinFill", "MinFill variant (Jégou–Kanso–Terrioux)", "MinFill variant (Jégou–Kanso–Terrioux)"]},
             "top-down": {"index graph": ["FlowCutter"], "tensor graph": ["Girvan–Newman"]},
             "search": {"index graph": ["Tamaki", "Jdrasil"]}}
    n = sum(len(v) for col in cells.values() for v in col.values())
    return _fig().write_doc("F02", "static", {"title": "Seven candidates on one map"}, {"cells": cells, "n": n})


def F06_7():
    """MinFill on the 7x7 grid (for the bottom-up vs top-down comparison)."""
    G = gr.grid_int(7)
    seed, _, _ = best_width_first_seed(G, "minfill", range(10))
    tr = eliminate(G, rule="minfill", seed=seed)
    return write_trace(_fig().DATA / "F06_7.json", "F06_7", G, gr.int_grid_positions(7, 44, 33), tr,
                       {"title": "MinFill on a 7x7 grid", "rule": "minfill", "seed": seed, "label_mode": "none"})


def S21():
    """Equal-width plans with different shapes: MinFill tie-break seeds 0..199 on the circuit, converted to trees;
    among those with the most common max rank, the cheapest, the costliest and the most lopsided."""
    T = circuit_tn(default_circuit())
    IG = index_graph(T)
    cands = []
    for seed in range(200):
        order = [s["elim"] for s in eliminate_fast(IG, rule="minfill", seed=seed)["steps"]]
        f = plan_features(T, order_to_tree(T, order))
        cands.append((seed, f))
    ranks = [f["max_rank"] for _, f in cands]
    r = max(set(ranks), key=lambda x: (ranks.count(x), -x))
    same = [(s, f) for s, f in cands if f["max_rank"] == r]
    picks = {"cheapest": min(same, key=lambda x: (x[1]["log2cost"], x[0])),
             "costliest": max(same, key=lambda x: (x[1]["log2cost"], -x[0])),
             "most lopsided": max(same, key=lambda x: (x[1]["avg_asym"], -x[0]))}
    keep = ["max_cost", "log2_sum_flops", "n_steps", "frac_tiny_steps", "max_out_rank", "avg_asym", "costw_asym", "p_at_max_cost"]
    return _fig().write_doc("S21", "chart", {"title": "Same width, different shapes", "seeds": "0..199"},
                            {"max_rank": r, "n_same": len(same), "n_total": len(cands),
                             "log2cost_range": [min(f["log2cost"] for _, f in same), max(f["log2cost"] for _, f in same)],
                             "avg_asym_range": [min(f["avg_asym"] for _, f in same), max(f["avg_asym"] for _, f in same)],
                             "plans": {name: {"seed": s, **{k: f[k] for k in keep}, "log2cost": f["log2cost"], "c": f["c"]}
                                       for name, (s, f) in picks.items()}})

# ---------------------------------------------------------------- Act IV


def F23():
    """Four plan shapes for the circuit: MinFill, ND/FlowCutter, Girvan–Newman, Tamaki."""
    T, plans = circuit_plans()
    out = {}
    for name in ["MinFill", "ND / FlowCutter", "Girvan–Newman", "Tamaki"]:
        t = plans[name]
        rec = tree_record(T, t, x0=0, dx=16, y0=0, dy=24)
        f = plan_features(T, t)
        cp, depth = critical_path(T, t)
        out[name] = {"tree": rec, "max_rank": f["max_rank"], "log2cost": f["log2cost"], "max_cost": f["max_cost"],
                     "critical_path": cp, "depth": tree_depth(t), "n_steps": f["n_steps"]}
    return _fig().write_doc("F23", "contraction_tree", {"title": "Plan shapes on the circuit"},
                            {"plans": out, "trees": {k: v["tree"] for k, v in out.items()}, "order": list(out)})


def S46():
    """Orders to plans is many-to-one: every index order of the 2x3 tensor grid, grouped by the tree it yields."""
    import itertools
    names = ["T00", "T01", "T02", "T10", "T11", "T12"]
    bonds = [("T00", "T01", "a"), ("T10", "T11", "b"), ("T00", "T10", "c"), ("T01", "T11", "d"),
             ("T02", "T12", "e"), ("T01", "T02", "p"), ("T11", "T12", "q")]
    T = {v: set() for v in names}
    for a, b, l in bonds:
        T[a].add(l)
        T[b].add(l)

    def canon(t):
        return t if not isinstance(t, tuple) else tuple(sorted((canon(t[0]), canon(t[1])), key=str))
    trees, n = set(), 0
    for o in itertools.permutations("abcdepq"):
        trees.add(canon(order_to_tree(T, o)))
        n += 1
    # parallel bonds: two tensors sharing two bonds of dimension 2 = one bond of dimension 4
    return _fig().write_doc("S46", "static", {"title": "From orders to pairwise plans"},
                            {"n_orders": n, "n_trees": len(trees), "parallel": {"bonds": 2, "chi": 2, "merged_dim": 4}})


def S47():
    """The bottleneck step of the MinFill plan on the circuit, as a GEMM."""
    T, plans = circuit_plans()
    st = tree_stats(T, plans["MinFill"])["steps"]
    i = max(range(len(st)), key=lambda j: (st[j]["exp"], -j))
    s = st[i]
    m, n, k = s["m"], s["n"], s["k"]
    return _fig().write_doc("S47", "static", {"title": "A step is a GEMM"},
                            {"step": i + 1, "n_steps": len(st), "m": m, "n": n, "k": k, "c": m + n + k, "p": m + n,
                             "d": round(abs(m - n) / (m + n + EPS), 3), "M": 2 ** m, "N": 2 ** n, "K": 2 ** k,
                             "flops": 2 * 2 ** (m + n + k)})


def S48():
    """Learning to rank, illustrated on our six plans for the circuit with a FLOP proxy (not GPU runtimes):
    Sp_i = 2^(L_MinFill − L_i), y_i = ln(1 + Sp_i); NDCG@3 with gain 2^y − 1 and discount log2(r + 1)."""
    T, plans = circuit_plans()
    feats = {k: plan_features(T, t) for k, t in plans.items()}
    Lmf = feats["MinFill"]["log2cost"]
    y = {k: round(math.log(1 + 2 ** (Lmf - f["log2cost"])), 3) for k, f in feats.items()}

    def dcg(order):
        return sum((2 ** y[p] - 1) / math.log2(r + 2) for r, p in enumerate(order[:3]))
    ideal = sorted(y, key=lambda p: -y[p])
    by_width = sorted(y, key=lambda p: (feats[p]["max_rank"], p))
    mf_first = ["MinFill"] + [p for p in by_width if p != "MinFill"]
    return _fig().write_doc("S48", "chart", {"title": "A ranking group", "proxy": "FLOPs, χ = 2"},
                            {"plans": {k: {"log2cost": feats[k]["log2cost"], "max_rank": feats[k]["max_rank"], "y": y[k]}
                                       for k in plans},
                             "ideal": ideal, "by_width": by_width, "mf_first": mf_first,
                             "ndcg3_by_width": round(dcg(by_width) / dcg(ideal), 3),
                             "ndcg3_mf_first": round(dcg(mf_first) / dcg(ideal), 3),
                             "bars": {"x": ideal, "y": [y[p] for p in ideal]}})


def P49():
    """Results tables 6, 7, 11 of the paper."""
    def tbl(t):
        return {k: {"top1": v[0], "top3": v[1], "regret1": v[2], "regret3": v[3], "win": v[4]} for k, v in t.items()}
    t11 = {f"{m}|{b}": {"top3": v[0], "regret3": v[1], "top1": v[2], "regret1": v[3], "win": v[4]} for (m, b), v in paper.TABLE11.items()}
    return _fig().write_doc("P49", "table", {"title": "Results", "source": "arXiv 2608.05819 Tables 6, 7, 11"},
                            {"id": tbl(paper.TABLE6), "ood": tbl(paper.TABLE7), "transfer": t11,
                             "n_id": paper.TABLE6_N, "n_ood": paper.TABLE7_N, "split": paper.SPLIT,
                             "stability": paper.STABILITY})


def P50():
    """Table 2: which plan features the selected models keep."""
    blocks = [{"block": b, "features": [{"name": n, "status": s} for n, s in fs]} for b, fs in paper.FEATURES.items()]
    return _fig().write_doc("P50", "static", {"title": "Kept and discarded features", "source": "arXiv 2608.05819 Table 2"},
                            {"blocks": blocks, "n_features": sum(len(b["features"]) for b in blocks),
                             "n_discarded": sum(f["status"] == "discarded" for b in blocks for f in b["features"]),
                             "n_per_model": 13})

# ---------------------------------------------------------------- backups


def B5():
    """Long-range couplings across a bond: rank of the off-diagonal block of J sets the MPO bond dimension
    D_n = 2 + rank M^(n). Exponential J has rank 1 (D = 3); a power law needs rank growing like log(N/ε)."""
    xi, alpha, eps = 3.0, 2.0, 1e-8
    out = {"xi": xi, "alpha": alpha, "eps": eps, "rows": []}
    for N in (16, 32, 64, 128, 256):
        i = np.arange(N)
        dist = np.abs(i[:, None] - i[None, :]).astype(float)
        Jexp = np.exp(-dist / xi)
        Jpow = np.where(dist > 0, 1.0 / np.maximum(dist, 1) ** alpha, 0.0)
        n = N // 2
        se = np.linalg.svd(Jexp[:n, n:], compute_uv=False)
        sp = np.linalg.svd(Jpow[:n, n:], compute_uv=False)
        out["rows"].append({"N": N, "rank_exp": int((se > eps * se[0]).sum()), "rank_pow": int((sp > eps * sp[0]).sum()),
                            "D_exp": 2 + int((se > eps * se[0]).sum()), "D_pow": 2 + int((sp > eps * sp[0]).sum())})
        if N == 64:
            out["sv_exp"] = [round(float(x / se[0]), 12) for x in se[:12]]
            out["sv_pow"] = [round(float(x / sp[0]), 12) for x in sp[:12]]
    out["chart"] = {"xlabel": "singular value index", "ylabel": "log₁₀ (σ / σ₁)", "ymin": -17, "ymax": 1,
                    "series": [{"name": "power law", "x": list(range(1, 13)),
                                "y": [round(math.log10(max(x, 1e-16)), 2) for x in out["sv_pow"]], "cls": "accent"},
                               {"name": "exponential", "x": list(range(1, 13)),
                                "y": [round(math.log10(max(x, 1e-16)), 2) for x in out["sv_exp"]], "cls": "side-b"}]}
    return _fig().write_doc("B5", "chart", {"title": "Long-range couplings across a bond"}, out)


def _m1(kind):
    L = 12
    G = gr.grid_int(L)
    pos = gr.int_grid_positions(L, 40, 20)
    if kind == "sweep":
        tr = eliminate(G, order=[r * L + c for c in range(L) for r in range(L)])
        return write_trace(_fig().DATA / "M1_sweep.json", "M1_sweep", G, pos, tr, {"title": "Column sweep, 12x12", "label_mode": "none"})
    seed, _, _ = best_width_first_seed(G, "minfill", range(10))
    tr = eliminate(G, rule="minfill", seed=seed)
    return write_trace(_fig().DATA / "M1_minfill.json", "M1_minfill", G, pos, tr,
                       {"title": "MinFill, 12x12", "rule": "minfill", "seed": seed, "label_mode": "none"})


def M1_sweep():
    """Manim clip M1: column sweep on the 12x12 grid."""
    return _m1("sweep")


def M1_minfill():
    """Manim clip M1: MinFill (seed protocol) on the 12x12 grid."""
    return _m1("minfill")


BUILDERS = {f.__name__: f for f in [M1_sweep, M1_minfill, F01, F02, F06_7, S21, F23, S46, S47, S48, P49, P50, B5]}
