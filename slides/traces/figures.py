"""figures.py — write every figure's trace to data/<id>.json.

    python -m traces.figures            # all figures
    python -m traces.figures F03 F10    # selected figures

Each builder returns the JSON document; numbers drawn on slides are read from these files
(`data-bind` attributes or steppers) and asserted in tests/.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import networkx as nx

from . import graphs as gr
from .carving import carving_width_dp, caterpillar, grid_tn, open_legs, tree_record, tree_stats
from .core import (SCHEMA_VERSION, best_caterpillar, carving_dp, eliminate, grid_positions, index_graph,
                   nested_dissection, separator_report, to_jsonable, write_trace)
from .circuits import circuit_tn, default_circuit
from .exact import treewidth_exact

DATA = Path(__file__).resolve().parent.parent / "data"


def P(pos: dict) -> dict:
    """Positions as JSON lists (to_jsonable would turn coordinate tuples into strings)."""
    return {to_jsonable(k) if isinstance(k, tuple) else str(k): [float(x) for x in v] for k, v in pos.items()}


def E(G) -> list:
    """Edges in a canonical order."""
    key = lambda v: to_jsonable(v) if isinstance(v, tuple) else str(v)
    return sorted(sorted(map(key, e)) for e in G.edges())


def write_doc(fid: str, kind: str, meta: dict, body: dict, nodes=None, edges=None) -> dict:
    doc = {"schema": SCHEMA_VERSION, "id": fid, "kind": kind, "meta": meta,
           "nodes": nodes or {}, "edges": edges or [], **to_jsonable(body)}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{fid}.json").write_text(json.dumps(doc, indent=1))
    return doc


def all_trees(names):
    """Every rooted binary contraction tree over `names` (unordered children)."""
    names = tuple(sorted(names))
    if len(names) == 1:
        return [names[0]]
    first, rest = names[0], names[1:]
    out = []
    for r in range(len(rest)):
        for comb in itertools.combinations(rest, r):
            A = (first,) + comb
            B = tuple(v for v in rest if v not in comb)
            for a in all_trees(A):
                for b in all_trees(B):
                    out.append((a, b))
    return out


def carving_width_exact(T: dict) -> int:
    return carving_width_dp(T)


def tensor_nx(T: dict) -> nx.Graph:
    """Tensor graph of a network without parallel bonds (vertices = tensors, edges = bonds)."""
    owner = {}
    for v, legs in T.items():
        for l in legs:
            owner.setdefault(l, []).append(v)
    G = nx.Graph()
    G.add_nodes_from(sorted(T, key=str))
    G.add_edges_from(tuple(o) for o in owner.values() if len(o) == 2)
    return G


# ---------------------------------------------------------------- Act I–II figures

def F03():
    """Ring: tensor graph, index graph, eliminating i, j, k, l (χ = 2)."""
    T = gr.ring_tn()
    IG = index_graph(T)
    tr = eliminate(IG, order=["i", "j", "k", "l"])
    rows = [{"elim": s["elim"], "nbrs": s["nbrs"], "bag": s["bag"], "cost": 2 ** s["bag"],
             "size": 2 ** len(s["nbrs"])} for s in tr["steps"]]
    tr["table"] = rows
    tr["total_cost"] = sum(r["cost"] for r in rows)
    tr["peak_size"] = max(r["size"] for r in rows)
    tr["tensor_graph"] = {"nodes": P(gr.RING_TENSOR_POS), "bonds": [list(b) for b in gr.RING_BONDS]}
    pos = {"i": (340, 80), "j": (390, 130), "k": (340, 180), "l": (290, 130)}
    return write_trace(DATA / "F03.json", "F03", IG, pos, tr,
                       {"title": "Ring: eliminating indices i, j, k, l", "label_mode": "name", "chi": 2})


def F04():
    """Two pairwise contraction trees for the ring: ((AB)(CD)) vs ((AC)(BD))."""
    T = gr.ring_tn()
    good = tree_record(T, (("A", "B"), ("C", "D")), x0=65, dx=70, y0=62, dy=67)
    bad = tree_record(T, (("A", "C"), ("B", "D")), x0=405, dx=70, y0=62, dy=67)
    r, c, _ = carving_dp(T)
    return write_doc("F04", "contraction_tree", {"title": "Ring: two contraction trees"},
                     {"trees": {"good": good, "bad": bad}, "best": {"max_rank": r, "cost": c},
                      "carving_width": carving_width_exact(T), "max_degree": max(len(v) for v in T.values())})


def F05():
    """Treewidth spectrum: tree, cycle, grid, clique (exact values on the drawn instances)."""
    inst = {"tree": nx.balanced_tree(2, 2), "cycle": nx.cycle_graph(8),
            "grid": gr.grid(4), "clique": nx.complete_graph(6)}
    return write_doc("F05", "static", {"title": "Treewidth spectrum"},
                     {"tw": {k: treewidth_exact(G) for k, G in inst.items()},
                      "n": {k: G.number_of_nodes() for k, G in inst.items()}, "grid_L": 4})


def F09():
    """Bowtie tensor graph and index graph; the hub's four legs form a K4."""
    T = gr.bowtie_tn()
    IG = index_graph(T)
    best = None
    for p in itertools.permutations(sorted(IG)):
        tr = eliminate(IG, order=p)
        key = (tr["width"], sum(2 ** s["bag"] for s in tr["steps"]))
        if best is None or key < best[0]:
            best = (key, list(p))
    hub_legs = sorted(T["H"])
    ipos = {"u": (400, 80), "v": (400, 220), "x": (430, 150), "p": (470, 100), "q": (470, 200),
            "r": (570, 100), "s": (570, 200), "t": (640, 150)}
    return write_doc("F09", "static", {"title": "Bowtie: tensor graph and index graph"},
                     {"tensor_graph": {"nodes": P(gr.BOWTIE_TENSOR_POS), "bonds": [list(b) for b in gr.BOWTIE_BONDS]},
                      "hub": "H", "hub_degree": len(hub_legs), "hub_legs": hub_legs,
                      "hub_clique": all(IG.has_edge(a, b) for a, b in itertools.combinations(hub_legs, 2)),
                      "index_tw": treewidth_exact(IG),
                      "best_elimination": {"width": best[0][0], "cost": best[0][1], "order": best[1]},
                      "n_orders": len(list(itertools.permutations(sorted(IG))))},
                     nodes=P(ipos), edges=E(IG))


def F10():
    """Bowtie: best pairwise tree (rank 2) vs best list schedule (rank 3)."""
    T = gr.bowtie_tn()
    r, c, _ = carving_dp(T)
    tree = (((("H", ("R", "S")), ("P", "U")), "Q"), "V")
    rec = tree_record(T, tree, x0=40, dx=70, y0=40, dy=62)
    lr, lc, _ = best_caterpillar(T)
    order = ["R", "S", "H", "P", "U", "Q", "V"]          # the list drawn on the slide
    lst = tree_record(T, caterpillar(order))
    lists = list(itertools.permutations(sorted(T)))
    min_list_rank = min(tree_stats(T, caterpillar(p))["max_rank"] for p in lists)
    return write_doc("F10", "contraction_tree", {"title": "Bowtie: tree vs list"},
                     {"tree": rec, "tree_opt": {"max_rank": r, "cost": c}, "list_rec": lst,
                      "list": {"order": order, "max_rank": lst["max_rank"], "cost": lst["cost"],
                               "ranks": [n["rank"] for n in lst["nodes"] if n["kind"] != "leaf"],
                               "legs": [n["legs"] for n in lst["nodes"] if n["kind"] != "leaf"]},
                      "list_opt": {"max_rank": lr, "cost": lc},
                      "n_lists": len(lists), "min_list_rank": min_list_rank,
                      "n_trees": len(all_trees(T)),
                      "carving_width": carving_width_exact(T)})


def F11():
    """Treewidth catalog: exact values on small members of each exactly-stated family."""
    checks = {"forest": treewidth_exact(nx.balanced_tree(2, 3)),
              "cycle_9": treewidth_exact(nx.cycle_graph(9)),
              "K_7": treewidth_exact(nx.complete_graph(7)),
              "K_3_5": treewidth_exact(nx.complete_bipartite_graph(3, 5)),
              "grid_3": treewidth_exact(gr.grid(3)), "grid_4": treewidth_exact(gr.grid(4)),
              "Q_3": treewidth_exact(nx.hypercube_graph(3)), "Q_4": treewidth_exact(nx.hypercube_graph(4))}
    return write_doc("F11", "table", {"title": "Treewidth catalog"}, {"checks": checks})


def F17():
    """2x3 tensor grid: the bond cut {p, q} is a vertex separator of the index graph."""
    names = ["T00", "T01", "T02", "T10", "T11", "T12"]
    bonds = [("T00", "T01", "a"), ("T10", "T11", "b"), ("T00", "T10", "c"), ("T01", "T11", "d"),
             ("T02", "T12", "e"), ("T01", "T02", "p"), ("T11", "T12", "q")]
    T = {v: set() for v in names}
    for a, b, l in bonds:
        T[a].add(l)
        T[b].add(l)
    left = ["T00", "T01", "T10", "T11"]
    cut = sorted(open_legs(T, left))
    IG = index_graph(T)
    rep = separator_report(IG, cut)
    comps = sorted(sorted(c) for c in nx.connected_components(IG.subgraph(set(IG) - set(cut))))
    tpos = {"T00": (70, 80), "T01": (160, 80), "T02": (250, 80), "T10": (70, 180), "T11": (160, 180), "T12": (250, 180)}
    ipos = {"a": (400, 70), "b": (400, 190), "c": (370, 130), "d": (455, 130), "p": (510, 70), "q": (510, 190), "e": (610, 130)}
    return write_doc("F17", "static", {"title": "Bond cut = index separator"},
                     {"tensor_graph": {"nodes": P(tpos), "bonds": [list(b) for b in bonds], "left": left},
                      "cut": cut, "is_separator": rep["separator"], "components": comps},
                     nodes=P(ipos), edges=E(IG))


def S05():
    """Slide 5: how many orders there are. n! elimination orders of n indices and
    (2n-3)!! = (2n-2)!/(2^(n-1) (n-1)!) rooted binary contraction trees over n tensors."""
    def log10_fact(n):
        return sum(math.log10(i) for i in range(2, n + 1))

    def log10_trees(n):
        return sum(math.log10(i) for i in range(1, 2 * n - 2, 2))   # (2n-3)!!

    def sci(name, lg):
        e = math.floor(lg)
        return {f"{name}_m": round(10 ** (lg - e), 1), f"{name}_e": e}

    circ = circuit_tn(default_circuit())
    xs = list(range(2, 41, 2))
    named = {"ring": (4, 4), "bowtie": (7, 8), "circuit": (len(circ), len(index_graph(circ)))}
    return write_doc("S05", "chart", {"title": "Number of contraction orders"},
                     {"chart": {"xlabel": "number of tensors n", "ylabel": "log₁₀ (number of orders)",
                                "series": [{"name": "contraction trees (2n−3)!!", "x": xs,
                                            "y": [round(log10_trees(n), 3) for n in xs], "cls": "accent"},
                                           {"name": "orders n!", "x": xs, "y": [round(log10_fact(n), 3) for n in xs],
                                            "cls": "muted", "dashed": True, "dy": 14}]},
                      "named": {k: {"tensors": t, "indices": i, **sci("trees", log10_trees(t)),
                                    **sci("orders", log10_fact(i))} for k, (t, i) in named.items()}})


def F06_ring():
    """Slide 8: MinFill on the ring's index graph, fill counts as labels."""
    IG = index_graph(gr.ring_tn())
    tr = eliminate(IG, rule="minfill", seed=0)
    pos = {"i": (340, 80), "j": (390, 130), "k": (340, 180), "l": (290, 130)}
    return write_trace(DATA / "F06_ring.json", "F06_ring", IG, pos, tr,
                       {"title": "MinFill on the ring's index graph", "rule": "minfill", "seed": 0, "label_mode": "fill"})


def F14():
    """Separator tree of balanced nested dissection on the 7x7 grid: bag = own separator +
    inherited boundary, read from the elimination trace (per-level maxima)."""
    G = gr.grid(7)
    log = []
    tr = eliminate(G, order=nested_dissection(range(7), range(7), log=log))
    bag = {s["elim"]: s["bag"] for s in tr["steps"]}
    levels = []
    for dpt in sorted({s["depth"] for s in log}):
        seps = [s["separator"] for s in log if s["depth"] == dpt]
        b = max(bag[v] for S in seps for v in S)
        own = max(len(S) for S in seps)
        levels.append({"depth": dpt, "count": len(seps), "own": own, "bag": b, "inherited": b - own})
    sep_vertices = {v for s in log for v in s["separator"]}
    leaves = [v for v in G if v not in sep_vertices]
    return write_doc("F14", "static", {"title": "Separator tree, 7x7 nested dissection"},
                     {"levels": levels, "leaf_max_bag": max(bag[v] for v in leaves),
                      "n_leaf_blocks": 2 ** len(levels), "width": tr["width"]})


def S14():
    """Slide 14: tw(G), tw(L(G)) and cw(G) on the shared small networks (exact)."""
    nets = {"ring": gr.ring_tn(), "bowtie": gr.bowtie_tn(), "grid2x3": grid_tn(2, 3), "grid3x3": grid_tn(3, 3)}
    rows = {}
    for name, T in nets.items():
        TG = tensor_nx(T)
        rows[name] = {"tensors": len(T), "max_degree": max(d for _, d in TG.degree()),
                      "tw_G": treewidth_exact(TG), "tw_LG": treewidth_exact(index_graph(T)),
                      "cw_G": carving_width_dp(T)}
    return write_doc("S14", "table", {"title": "How the widths relate"}, {"rows": rows})


def F00():
    """Narrative arc: acts and deep dives with their slide ranges and minutes (from PROPOSAL.md)."""
    acts = [("I", "Why contraction order matters", (1, 8), 9, 1),
            ("II", "The shared language", (9, 16), 9, 2),
            ("III", "Seven techniques at a glance", (17, 22), 5, 3),
            ("DD1", "Greedy elimination: MinFill", (23, 28), 7, 4),
            ("DD2", "Separators, ND, FlowCutter", (29, 37), 11, 4),
            ("DD3", "Community detection: Girvan–Newman", (38, 41), 5, 4),
            ("DD4", "Treewidth solvers: Jdrasil, Tamaki", (42, 44), 4, 4),
            ("IV", "Head-to-head: a learned ranker", (45, 50), 8, 5),
            ("V", "Conclusion and open questions", (51, 53), 4, 6)]
    rows = [{"id": a, "title": t, "first": f, "last": l, "slides": l - f + 1, "minutes": m, "group": g}
            for a, t, (f, l), m, g in acts]
    return write_doc("F00", "static", {"title": "Narrative arc"},
                     {"acts": rows, "total_slides": sum(r["slides"] for r in rows),
                      "total_minutes": sum(r["minutes"] for r in rows)})


BUILDERS = {f.__name__: f for f in [F00, F03, F04, F05, F09, F10, F11, F17, S05, F06_ring, F14, S14]}


from .fig_dd12 import BUILDERS as _DD12   # noqa: E402  (deep dives 1-2)
BUILDERS.update(_DD12)
from .fig_primer import BUILDERS as _PRIMER   # noqa: E402  (treewidth primer, slides 3a/3b, 10)
BUILDERS.update(_PRIMER)
from .fig_dd34 import BUILDERS as _DD34       # noqa: E402  (deep dives 3-4)
BUILDERS.update(_DD34)
from .fig_act345 import BUILDERS as _A345     # noqa: E402  (Acts III-V, backups)
BUILDERS.update(_A345)


def main(ids):
    for fid in ids or BUILDERS:
        BUILDERS[fid]()
        print("wrote", f"data/{fid}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
