"""fig_primer.py — the treewidth primer (slides 3a, 3b) and the certificate slide (10).
Registered into figures.BUILDERS."""
from __future__ import annotations

import math

import networkx as nx

from . import graphs as gr
from .core import column_sweep, eliminate, write_trace
from .exact import treewidth_exact


def _fig():
    from . import figures
    return figures


def _ring_pos(names, cx=150.0, cy=150.0, r=110.0, start=-90.0):
    n = len(names)
    return {v: (cx + r * math.cos(math.radians(start + 360 * i / n)), cy + r * math.sin(math.radians(start + 360 * i / n)))
            for i, v in enumerate(names)}


def star():
    leaves = [str(i) for i in range(1, 7)]
    G = nx.Graph([("h", v) for v in leaves])
    return G, {**_ring_pos(leaves), "h": (150.0, 150.0)}


def _primer_trace(fid, G, pos, order, title):
    tr = eliminate(G, order=order)
    tr["tw_exact"] = treewidth_exact(G)
    tr["total_fill"] = sum(len(s["fill"]) for s in tr["steps"])
    return write_trace(_fig().DATA / f"{fid}.json", fid, G, pos, tr, {"title": title, "label_mode": "none"})


def PR1_hub():
    """Star K_{1,6}, hub eliminated first: one bag holds everything."""
    G, pos = star()
    return _primer_trace("PR1_hub", G, pos, ["h"] + [str(i) for i in range(1, 7)], "Star, hub first")


def PR1_leaves():
    """Star K_{1,6}, leaves first: every bag is {leaf, hub}."""
    G, pos = star()
    return _primer_trace("PR1_leaves", G, pos, [str(i) for i in range(1, 7)] + ["h"], "Star, leaves first")


def PR2_tree():
    G = nx.balanced_tree(2, 2)                 # 0 root; 1, 2; leaves 3-6
    pos = {0: (150, 40), 1: (80, 140), 2: (220, 140), 3: (45, 240), 4: (115, 240), 5: (185, 240), 6: (255, 240)}
    return _primer_trace("PR2_tree", G, pos, [3, 4, 5, 6, 1, 2, 0], "Tree, leaves first")


def PR2_cycle():
    names = [f"c{i}" for i in range(6)]
    G = nx.cycle_graph(names)
    return _primer_trace("PR2_cycle", G, _ring_pos(names, r=100), names, "Cycle C6")


def PR2_grid():
    G = gr.grid(3)
    return _primer_trace("PR2_grid", G, gr.grid_positions(G, 90, 60), column_sweep(3), "3x3 grid, column sweep")


def PR2_clique():
    names = [f"k{i}" for i in range(5)]
    G = nx.complete_graph(names)
    return _primer_trace("PR2_clique", G, _ring_pos(names, r=100), names, "Clique K5")


def S10():
    """Certificates that treewidth is large: a clique (ω − 1) and a grid (minor-monotone, tw(k x k) = k)."""
    names = [f"k{i}" for i in range(5)]
    K = nx.complete_graph(names)
    G = gr.grid(4)
    kpos = _ring_pos(names, r=100)
    gpos = gr.grid_positions(G, 70, 40)
    P = _fig().P
    return _fig().write_doc("S10", "static", {"title": "Certificates that treewidth is large"},
                            {"clique": {"nodes": P(kpos), "edges": _fig().E(K), "omega": 5, "tw": treewidth_exact(K),
                                        "degeneracy": max(nx.core_number(K).values())},
                             "grid": {"nodes": P(gpos), "edges": _fig().E(G), "k": 4, "tw": treewidth_exact(G),
                                      "omega": max(len(c) for c in nx.find_cliques(G)),
                                      "degeneracy": max(nx.core_number(G).values())}})


BUILDERS = {f.__name__: f for f in [PR1_hub, PR1_leaves, PR2_tree, PR2_cycle, PR2_grid, PR2_clique, S10]}
