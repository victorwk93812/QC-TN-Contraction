"""fig_dd34.py — figure builders for deep dives 3 (Girvan–Newman) and 4 (treewidth solvers).
Registered into figures.BUILDERS. (The solver anytime curve is measured by run_solvers.py.)"""
from __future__ import annotations

import math
import random

import networkx as nx

from . import graphs as gr
from .carving import grid_tn, order_to_tree, tree_record, tree_stats
from .circuits import circuit_tn, default_circuit
from .core import carving_dp, index_graph, write_trace
from .fast_elim import best_width_first_seed, eliminate_fast
from .flowcutter_lite import nested_dissection_fc
from .gn import girvan_newman
from .td import improve_once, td_from_order, width


def _fig():
    from . import figures
    return figures


def wheels_tn():
    """Two wheels (hub + 4-cycle rim) joined by two bonds: a network with two obvious communities."""
    bonds = []
    for side in "AB":
        hub, rim = f"{side}0", [f"{side}{i}" for i in range(1, 5)]
        bonds += [(hub, r, f"{side.lower()}h{i}") for i, r in enumerate(rim, 1)]
        bonds += [(rim[i], rim[(i + 1) % 4], f"{side.lower()}r{i + 1}") for i in range(4)]
    bonds += [("A1", "B4", "x1"), ("A2", "B3", "x2")]          # facing rims
    names = [f"{s}{i}" for s in "AB" for i in range(5)]
    T = {v: set() for v in names}
    for a, b, l in bonds:
        T[a].add(l)
        T[b].add(l)
    return T, bonds


def wheels_pos():
    pos = {}
    for side, cx in (("A", 140), ("B", 470)):
        pos[f"{side}0"] = (cx, 150)
        for i in range(4):
            ang = math.radians(-45 + 90 * i)
            pos[f"{side}{i + 1}"] = (cx + 95 * math.cos(ang), 150 + 95 * math.sin(ang))
    return pos


def tensor_graph(T):
    owner = {}
    for v, legs in T.items():
        for l in legs:
            owner.setdefault(l, []).append(v)
    G = nx.Graph()
    G.add_nodes_from(sorted(T, key=str))
    G.add_edges_from(tuple(o) for o in owner.values() if len(o) == 2)
    return G


def F20():
    """Girvan–Newman on the two-wheel network: one step per removed edge; the dendrogram, read bottom-up,
    is a pairwise contraction tree. Internal tree nodes carry the step at which their split happened."""
    T, bonds = wheels_tn()
    G = tensor_graph(T)
    g = girvan_newman(G)
    rec = tree_record(T, g["tree"], x0=30, dx=52, y0=30, dy=58)
    split_step = {}
    for i, s in enumerate(g["steps"]):
        if s["split"]:
            split_step[frozenset(s["split"]["parent"])] = i + 1
    leaves_of = {}

    def collect(nid):
        n = next(x for x in rec["nodes"] if x["id"] == nid)
        if n["kind"] == "leaf":
            return {n["label"]}
        kids = [e["child"] for e in rec["edges"] if e["parent"] == nid]
        out = set().union(*(collect(k) for k in kids))
        leaves_of[nid] = out
        return out

    collect(rec["root"])
    for n in rec["nodes"]:
        if n["kind"] != "leaf":
            n["split_step"] = split_step[frozenset(leaves_of[n["id"]])]
    best_q, best_part = max(((s["modularity"], s["components"]) for s in g["steps"]), key=lambda x: x[0])
    opt = carving_dp(T)
    return _fig().write_doc("F20", "girvan_newman", {"title": "Girvan–Newman on two wheels"},
                            {"steps": g["steps"], "tree": rec, "best_modularity": round(best_q, 3),
                             "best_partition": best_part, "n_steps": len(g["steps"]),
                             "tensor_graph": {"nodes": _fig().P(wheels_pos()), "bonds": [list(b) for b in bonds]},
                             "first_removed": g["steps"][0]["removed"], "second_removed": g["steps"][1]["removed"],
                             "tree_max_rank": rec["max_rank"], "tree_cost": rec["cost"],
                             "opt_max_rank": opt[0], "opt_cost": opt[1]},
                            nodes=_fig().P(wheels_pos()), edges=_fig().E(G))


def _methods(T):
    """Max rank and log2 cost of contraction trees from GN (tensor graph), MinFill and FlowCutter-ND
    (index graph orders converted with order_to_tree), and the exact optimum when the DP is feasible."""
    IG = index_graph(T)
    out = {}
    st = tree_stats(T, girvan_newman(tensor_graph(T))["tree"])
    out["gn"] = (st["max_rank"], round(st["log2cost"], 2))
    seed, _, _ = best_width_first_seed(IG, "minfill", range(10))
    mf_order = [s["elim"] for s in eliminate_fast(IG, rule="minfill", seed=seed)["steps"]]
    st = tree_stats(T, order_to_tree(T, mf_order))
    out["minfill"] = (st["max_rank"], round(st["log2cost"], 2))
    st = tree_stats(T, order_to_tree(T, nested_dissection_fc(IG)))
    out["nd"] = (st["max_rank"], round(st["log2cost"], 2))
    if len(T) <= 12:
        r, c, _ = carving_dp(T)
        out["opt"] = (r, round(math.log2(c), 2))
    return {k: list(v) for k, v in out.items()}


def F21():
    """Width (max intermediate rank) and log2 cost of GN against MinFill, ND and the optimum, χ = 2."""
    def strkeys(T):
        return {f"{k[0]}_{k[1]}" if isinstance(k, tuple) else k: v for k, v in T.items()}
    nets = {"ring": gr.ring_tn(), "bowtie": gr.bowtie_tn(), "two wheels": wheels_tn()[0],
            "3×3 grid": strkeys(grid_tn(3)), "7×7 grid": strkeys(grid_tn(7)), "circuit": circuit_tn(default_circuit())}
    rows = {k: {"tensors": len(T), **_methods(T)} for k, T in nets.items()}
    return _fig().write_doc("F21", "table", {"title": "GN against the other generators"}, {"rows": rows, "order": list(nets)})


def F22():
    """Tamaki-style local improvement on the 5x5 grid from a random elimination order (seed 7):
    alternately show the widest bag, then the region around it that is re-solved exactly."""
    G = gr.grid(5)
    rng = random.Random(7)
    order = sorted(G, key=lambda v: rng.random())
    bags, T = td_from_order(G, order)
    states = []

    def snapshot(region=None):
        wide = max(bags.values(), key=lambda b: (len(b), sorted(map(str, b))))
        states.append({"width": width(bags), "widest": sorted(wide), "region": sorted(region) if region else [],
                       "bag_sizes": sorted((len(b) for b in bags.values()), reverse=True), "n_bags": len(bags)})

    snapshot()
    rounds = []
    while True:
        r = improve_once(G, bags, T)
        if r is None:
            break
        snapshot(r[2]["region"])
        bags, T, info = r
        rounds.append(info)
        snapshot()
    return _fig().write_doc("F22", "td_improve", {"title": "Local improvement around the widest bag", "order_seed": 7},
                            {"states": states, "rounds": [{k: info[k] for k in ("old_width", "region_old", "region_new", "new_width")}
                                                          for info in rounds],
                             "region_sizes": [len(info["region"]) for info in rounds],
                             "start_width": states[0]["width"], "final_width": states[-1]["width"],
                             "tw": 5},   # tw(L x L grid) = L; the exact DP needs 2^25 states here, so it is not rerun
                            nodes=_fig().P(gr.grid_positions(G, 70, 40)), edges=_fig().E(G))


BUILDERS = {f.__name__: f for f in [F20, F21, F22]}
