"""fig_dd12.py — figure builders for deep dives 1 (MinFill) and 2 (separators, ND, FlowCutter).
Registered into figures.BUILDERS."""
from __future__ import annotations

import itertools
import statistics

import networkx as nx

from . import graphs as gr
from . import paper
from .carving import grid_tree_stats, order_to_tree, tree_record, tree_stats
from .circuits import circuit_tn, default_circuit
from .core import (column_sweep, eliminate, hybrid_cut_and_sweep, index_graph, nested_dissection,
                   separator_report, write_trace)
from .fast_elim import best_width_first_seed, eliminate_fast, greedy_runs
from .flowcutter_lite import flowcutter, nested_dissection_fc, peripheral_pair
from .separators import ND_VARIANTS


def _fig():
    from . import figures
    return figures


def circuit_ig():
    return index_graph(circuit_tn(default_circuit()))

# ---------------------------------------------------------------- deep dive 1: MinFill


def F06():
    """MinFill on the 6x6 grid (integer labels; first seed of 0..9 reaching the minimum width)."""
    G = gr.grid_int(6)
    seed, _, _ = best_width_first_seed(G, "minfill", range(10))
    tr = eliminate(G, rule="minfill", seed=seed)
    tr["first_fills"] = [len(s["fill"]) for s in tr["steps"][:5]]
    tr["first_elims"] = [s["elim"] for s in tr["steps"][:5]]
    return write_trace(_fig().DATA / "F06.json", "F06", G, gr.int_grid_positions(6), tr,
                       {"title": "MinFill on a 6x6 grid", "rule": "minfill", "seed": seed, "label_mode": "fill"})


def F06_chordal():
    """MinFill on a chordal graph (triangulated ladder): zero fill at every step."""
    G = gr.triangulated_ladder(6)
    tr = eliminate(G, rule="minfill", seed=0)
    tr["total_fill"] = sum(len(s["fill"]) for s in tr["steps"])
    tr["is_chordal"] = nx.is_chordal(G)
    tr["clique_number"] = max(len(c) for c in nx.find_cliques(G))
    return write_trace(_fig().DATA / "F06_chordal.json", "F06_chordal", G, gr.ladder_positions(6), tr,
                       {"title": "MinFill on a chordal graph", "rule": "minfill", "seed": 0, "label_mode": "fill"})


def F08():
    """Elimination time of each vertex under MinFill on the 10x10 grid."""
    L = 10
    G = gr.grid_int(L)
    seed, width, _ = best_width_first_seed(G, "minfill", range(10))
    tr = eliminate_fast(G, rule="minfill", seed=seed)
    t = {s["elim"]: i for i, s in enumerate(tr["steps"])}
    time = [[t[r * L + c] for c in range(L)] for r in range(L)]
    rings = [[time[r][c] for r in range(L) for c in range(L) if min(r, c, L - 1 - r, L - 1 - c) == k] for k in range(L // 2)]
    return _fig().write_doc("F08", "heatmap", {"title": "MinFill elimination time, 10x10", "seed": seed},
                            {"L": L, "time": time, "width": width, "n": L * L,
                             "ring_mean": [round(statistics.mean(x)) for x in rings]})


F07_L = list(range(8, 25, 2))


def F07():
    """Width and log2 cost vs L: MinFill (seed protocol) against the column sweep."""
    mf, sw = [], []
    for L in F07_L:
        _, w, c = best_width_first_seed(gr.grid_int(L), "minfill", range(10))
        mf.append([w, round(c, 2)])
        tr = eliminate_fast(gr.grid(L), order=column_sweep(L), record_steps=False)
        sw.append([tr["width"], round(tr["log2cost"], 2)])
    cross = next(L for L, a, b in zip(F07_L, mf, sw) if a[1] > b[1])
    return _fig().write_doc("F07", "chart", {"title": "MinFill vs column sweep on L x L grids"},
                            {"L": F07_L, "minfill": mf, "sweep": sw, "first_L_minfill_costlier": cross,
                             "width_chart": {"xlabel": "grid side L", "ylabel": "width", "xticks": [8, 12, 16, 20, 24],
                                             "series": [{"name": "MinFill", "x": F07_L, "y": [w for w, _ in mf], "cls": "accent"},
                                                        {"name": "sweep = L", "x": F07_L, "y": [w for w, _ in sw], "cls": "side-b"}]},
                             "cost_gap": [round(a[1] - b[1], 2) for a, b in zip(mf, sw)],
                             "cost_chart": {"xlabel": "grid side L", "ylabel": "log₂ cost: MinFill − sweep", "xticks": [8, 12, 16, 20, 24],
                                            "yticks": [-1.5, -1, -0.5, 0, 0.5], "ymin": -1.6, "ymax": 0.7,
                                            "series": [{"name": "MinFill costlier ↑", "x": F07_L,
                                                        "y": [round(a[1] - b[1], 2) for a, b in zip(mf, sw)], "cls": "accent"}],
                                            "marks": [{"x": cross, "label": f"L = {cross}"}], "zero": True}})


def S25():
    """Ties and chains: width spread over 40 tie-break seeds at L = 20, the bag-size profile of the
    best run, and the random cubic comparison (MinFill, MinDegree, ND variants)."""
    G = gr.grid_int(20)
    runs = greedy_runs(G, "minfill", range(40))
    widths = [w for _, w, _ in runs]
    hist = {w: widths.count(w) for w in range(min(widths), max(widths) + 1)}
    seed, _, _ = best_width_first_seed(G, "minfill", range(10))
    bags = [s["bag"] for s in eliminate_fast(G, rule="minfill", seed=seed, record_steps=False)["steps"]]
    mx = max(bags)
    first_big = next(i for i, b in enumerate(bags) if b >= mx - 4)
    C = gr.cubic(320, seed=1)
    mf = [w for _, w, _ in greedy_runs(C, "minfill", range(20))]
    md = [w for _, w, _ in greedy_runs(C, "mindeg", range(20))]
    nd = {k: eliminate_fast(C, order=f(C), record_steps=False)["width"] for k, f in ND_VARIANTS.items()}
    return _fig().write_doc("S25", "chart", {"title": "Ties, chains and blind spots"},
                            {"tie_hist": {"x": list(hist), "count": list(hist.values())},
                             "tie_min": min(widths), "tie_max": max(widths), "tie_median": statistics.median(widths),
                             "bags": bags, "bags_max": mx, "n_steps": len(bags),
                             "late_chain_start": first_big, "late_chain_len": len(bags) - first_big,
                             "cubic": {"n": 320, "minfill_best": min(mf), "minfill_median": statistics.median(mf),
                                       "mindeg_best": min(md), "mindeg_median": statistics.median(md), "nd": nd}})


def P26():
    """Slide 26: the paper's verdict on MinFill-first ranking (Tables 6 and 7)."""
    t6, t7 = paper.TABLE6, paper.TABLE7
    return _fig().write_doc("P26", "chart", {"title": "MinFill as the default plan", "source": "arXiv 2608.05819 Tables 6, 7"},
                            {"id": {k: {"top1": v[0], "top3": v[1], "regret1": v[2], "win": v[4]} for k, v in t6.items()},
                             "ood": {k: {"top1": v[0], "top3": v[1], "regret1": v[2], "win": v[4]} for k, v in t7.items()},
                             "n_id": paper.TABLE6_N, "n_ood": paper.TABLE7_N})

# ---------------------------------------------------------------- deep dive 2: separators


def F15():
    """7x7 grid: corner cut {(0,1),(1,0)} vs the balanced middle column."""
    G = gr.grid(7)
    panels = {}
    for name, S in {"corner": [(0, 1), (1, 0)], "middle": [(r, 3) for r in range(7)]}.items():
        rep = separator_report(G, S)
        comps = sorted((sorted(c) for c in nx.connected_components(G.subgraph(set(G) - set(S)))), key=len)
        panels[name] = {"S": S, "A": comps[0], "B": comps[-1], "size": len(S), "sizes": rep["sizes"],
                        "balanced_half": rep["balanced_0.500"], "balanced_two_thirds": rep["balanced_0.667"]}
    return _fig().write_doc("F15", "static", {"title": "Separator size vs balance"},
                            {"panels": panels, "n": 49}, nodes=_fig().P(gr.grid_positions(G, 30, 20)), edges=_fig().E(G))


def _subset_key(S):
    return "".join(sorted(S))


def F16():
    """Separator playground: separator_report for every subset S of the 11 vertices, so the
    interactive slide only looks values up (no algorithm in the browser)."""
    G = gr.separator_graph()
    V = sorted(G)
    table = {}
    for r in range(len(V) + 1):
        for S in itertools.combinations(V, r):
            rest = set(G) - set(S)
            comps = sorted((sorted(c) for c in nx.connected_components(G.subgraph(rest))), key=lambda c: (-len(c), c))
            rep = separator_report(G, S)
            ah = "a" not in S and "h" not in S and not any("a" in c and "h" in c for c in comps)
            table[_subset_key(S)] = {"sep": rep["separator"], "min": rep["minimal"], "b2": rep["balanced_0.500"],
                                     "b3": rep["balanced_0.667"], "ah": ah, "comps": ["".join(c) for c in comps]}
    presets = ["c", "xy", "cd", "efg", "xyz"]
    return _fig().write_doc("F16", "playground", {"title": "Separator playground"},
                            {"table": table, "presets": presets, "preset_reports": {p: table[p] for p in presets},
                             "menger_paths": gr.MENGER_PATHS, "min_ah": nx.node_connectivity(G, "a", "h"),
                             "n": G.number_of_nodes(), "m": G.number_of_edges()},
                            nodes=_fig().P(gr.SEPARATOR_POS), edges=_fig().E(G))


def _grid7_sides(order, S_col=3):
    return {"A": [f"{r}_{c}" for r in range(7) for c in range(S_col)],
            "B": [f"{r}_{c}" for r in range(7) for c in range(S_col + 1, 7)]}


def F12_sideA():
    """Separator-last on the 7x7 grid: eliminate side A (3 columns) with S = middle column alive."""
    G = gr.grid(7)
    order = hybrid_cut_and_sweep(7)
    tr = eliminate(G, order=order)
    tr["separators"] = [{"depth": 0, "separator": [(r, 3) for r in range(7)]}]
    tr["sides"] = _grid7_sides(order)
    A = {(r, c) for r in range(7) for c in range(3)}
    S = {(r, 3) for r in range(7)}
    fills_A = [f for s in tr["steps"][:21] for f in s["fill"]]
    tr["side_a_fill_inside"] = all(set(map(tuple, f)) <= (A | S) for f in fills_A)
    tr["width_A"] = max(len(s["nbrs"]) for s in tr["steps"][:21])
    return write_trace(_fig().DATA / "F12_sideA.json", "F12_sideA", G, gr.grid_positions(G, 44, 33), tr,
                       {"title": "Separator last: side A", "label_mode": "separator"})


def F12():
    """Balanced nested dissection on the 7x7 grid (separators labelled by depth)."""
    G = gr.grid(7)
    log = []
    tr = eliminate(G, order=nested_dissection(range(7), range(7), log=log))
    tr["separators"] = log
    return write_trace(_fig().DATA / "F12.json", "F12", G, gr.grid_positions(G, 44, 33), tr,
                       {"title": "Balanced nested dissection, 7x7", "label_mode": "separator"})


F13_L = [7, 15, 23, 31]


def F13():
    """Width vs L for balanced ND, one cut + inward sweeps, column sweep and MinFill."""
    rows = {"nd": [], "hybrid": [], "sweep": [], "minfill": []}
    for L in F13_L:
        G = gr.grid(L)
        rows["nd"].append(eliminate_fast(G, order=nested_dissection(range(L), range(L)), record_steps=False)["width"])
        rows["hybrid"].append(eliminate_fast(G, order=hybrid_cut_and_sweep(L), record_steps=False)["width"])
        rows["sweep"].append(eliminate_fast(G, order=column_sweep(L), record_steps=False)["width"])
        rows["minfill"].append(best_width_first_seed(gr.grid_int(L), "minfill", range(10))[1])
    bis = grid_tree_stats(32)
    return _fig().write_doc("F13", "chart", {"title": "Balance has a price"},
                            {"L": F13_L, **rows, "bisection32": {"max_rank": bis["max_rank"], "max_exp": bis["max_exp"]},
                             "chart": {"xlabel": "grid side L", "ylabel": "width", "xticks": F13_L,
                                       "series": [{"name": "balanced ND", "x": F13_L, "y": rows["nd"], "cls": "accent", "dy": -4},
                                                  {"name": "MinFill", "x": F13_L, "y": rows["minfill"], "cls": "derived", "dy": 4},
                                                  {"name": "sweep, cut + sweeps: L", "x": F13_L, "y": rows["hybrid"], "cls": "side-b"}]}})


def F18():
    """FlowCutter-lite on the 7x7 grid from corner (0,0) to corner (6,6): one entry per cut."""
    G = gr.grid(7)
    fc = flowcutter(G, (0, 0), (6, 6))
    return _fig().write_doc("F18", "flowcutter", {"title": "FlowCutter on the 7x7 grid", "source": "0_0", "target": "6_6"},
                            {"steps": fc["steps"], "pareto": fc["pareto"], "n": 49},
                            nodes=_fig().P(gr.grid_positions(G, 44, 33)), edges=_fig().E(G))


def F19():
    """Pareto fronts (cut size vs smaller side) on the 7x7 grid and the circuit's index graph."""
    out = {}
    for name, G, st in [("grid7", gr.grid(7), ((0, 0), (6, 6))), ("circuit", circuit_ig(), None)]:
        s, t = st or peripheral_pair(G)
        fc = flowcutter(G, s, t)
        n = G.number_of_nodes()
        front = [[fc["steps"][i]["size"], fc["steps"][i]["small_side"]] for i in fc["pareto"]]
        out[name] = {"n": n, "front": front, "source": s, "target": t}
    series = [{"name": "7×7 grid", "x": [a for a, _ in out["grid7"]["front"]],
               "y": [round(b / out["grid7"]["n"], 3) for _, b in out["grid7"]["front"]], "cls": "accent"},
              {"name": "circuit", "x": [a for a, _ in out["circuit"]["front"]],
               "y": [round(b / out["circuit"]["n"], 3) for _, b in out["circuit"]["front"]], "cls": "side-b"}]
    return _fig().write_doc("F19", "chart", {"title": "FlowCutter Pareto fronts"},
                            {**out, "chart": {"xlabel": "separator size |S|", "ylabel": "smaller side / n",
                                              "series": series, "ymax": 0.5}})


def S35():
    """From cuts to a plan, on the 2x3 tensor grid of slide 7: index separator {p, q} last, the
    tree decomposition of that order, and the pairwise contraction tree it induces; a second order
    of equal width gives a different tree (the conversion is not unique)."""
    names = ["T00", "T01", "T02", "T10", "T11", "T12"]
    bonds = [("T00", "T01", "a"), ("T10", "T11", "b"), ("T00", "T10", "c"), ("T01", "T11", "d"),
             ("T02", "T12", "e"), ("T01", "T02", "p"), ("T11", "T12", "q")]
    T = {v: set() for v in names}
    for a, b, l in bonds:
        T[a].add(l)
        T[b].add(l)
    IG = index_graph(T)
    orders = {"nd": list("cabdepq"), "alt": list("abpecdq")}
    out = {}
    hist = {}
    for o in itertools.permutations(sorted(IG)):          # every width-3 order and its tree's cost
        if eliminate(IG, order=o)["width"] == 3:
            c = tree_stats(T, order_to_tree(T, o))["cost"]
            hist[c] = hist.get(c, 0) + 1
    for k, o in orders.items():
        tr = eliminate(IG, order=o)
        tree = order_to_tree(T, o)
        rec = tree_record(T, tree, x0=30, dx=62, y0=30, dy=60)
        out[k] = {"order": o, "width": tr["width"], "bags": [[s["elim"]] + s["nbrs"] for s in tr["steps"]],
                  "tree": rec, "max_rank": rec["max_rank"], "cost": rec["cost"]}
    return _fig().write_doc("S35", "contraction_tree", {"title": "From cuts to a plan"},
                            {**out, "trees": {"nd": out["nd"]["tree"], "alt": out["alt"]["tree"]},
                             "same_width": out["nd"]["width"] == out["alt"]["width"],
                             "width3_orders": sum(hist.values()), "width3_cost_min": min(hist), "width3_cost_max": max(hist),
                             "width3_cost_hist": {"cost": sorted(hist), "count": [hist[c] for c in sorted(hist)]}})


def S32():
    """Slide 32: the theorems' numbers on the 7x7 grid (planar, n = 49, treewidth 7)."""
    n = 49
    import math
    return _fig().write_doc("S32", "static", {"title": "What theory guarantees"},
                            {"n": n, "sqrt_n": int(math.isqrt(n)), "lt_bound": round(2 * math.sqrt(2) * math.sqrt(n), 1),
                             "tw": 7, "rs_sep": 8, "rs_side": (n - 8) // 2, "dn_factor": 15,
                             "middle_column": 7, "middle_sides": [21, 21]})


def SC():
    """Running scorecard: measured values per technique on the shared graphs (χ = 2)."""
    IG = circuit_ig()
    T = circuit_tn(default_circuit())

    def row(order_fn, label):
        g7, g20, gc = gr.grid(7), gr.grid(20), IG
        w7 = eliminate_fast(g7, order=order_fn(g7), record_steps=False)["width"]
        w20 = eliminate_fast(g20, order=order_fn(g20), record_steps=False)["width"]
        trc = eliminate_fast(gc, order=order_fn(gc), record_steps=False)
        st = tree_stats(T, order_to_tree(T, order_fn(gc)))
        return {"label": label, "w7": w7, "w20": w20, "wc": trc["width"], "cost_c": round(trc["log2cost"], 2),
                "tree_rank_c": st["max_rank"], "tree_cost_c": round(st["log2cost"], 2)}

    def minfill_order(G):
        Gi = G if all(isinstance(v, (int, str)) for v in G) else nx.convert_node_labels_to_integers(G, label_attribute="orig")
        seed, _, _ = best_width_first_seed(Gi, "minfill", range(10))
        o = [s["elim"] for s in eliminate_fast(Gi, rule="minfill", seed=seed)["steps"]]
        if Gi is not G:
            o = [Gi.nodes[v]["orig"] for v in o]
        return o

    rows = {"minfill": row(minfill_order, "MinFill"), "nd_fc": row(nested_dissection_fc, "ND with FlowCutter-lite")}
    return _fig().write_doc("SC", "table", {"title": "Scorecard"}, {"rows": rows})


BUILDERS = {f.__name__: f for f in [F06, F06_chordal, F08, F07, S25, P26, F15, F16, F12_sideA, F12, F13, F18, F19,
                                    S35, S32, SC]}
