"""trace_core.py — compute algorithm traces for slide figures.

Every figure in the deck is drawn from a JSON trace produced here. Renderers
(reveal.js steppers, Manim scenes, PDF frames) never recompute widths or costs.

Usage:
    python trace_core.py selftest            # reproduce the reference numbers
    python trace_core.py demo out_dir/       # write example traces (ring, grid MinFill, 7x7 ND)
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx

SCHEMA_VERSION = 1

# ---------------------------------------------------------------- graph builders

def grid_graph(L: int, W: int | None = None) -> nx.Graph:
    """L x W grid with nodes labelled (row, col)."""
    return nx.grid_2d_graph(L, W or L)


def grid_positions(G: nx.Graph, spacing: float = 60, margin: float = 40) -> dict:
    return {v: (margin + v[1] * spacing, margin + v[0] * spacing) for v in G}


def tensor_network(nodes, bonds) -> dict:
    """bonds: iterable of (tensor_a, tensor_b, label). Returns {tensor: set(labels)}."""
    T = {v: set() for v in nodes}
    for a, b, lab in bonds:
        T[a].add(lab)
        T[b].add(lab)
    return T


def index_graph(T: dict) -> nx.Graph:
    """Vertices = indices; an edge joins two indices carried by one tensor."""
    G = nx.Graph()
    for legs in T.values():
        G.add_nodes_from(sorted(legs, key=str))   # sorted: node order must not depend on hashing
        G.add_edges_from(itertools.combinations(sorted(legs), 2))
    return G

# ---------------------------------------------------------------- elimination game

def _fill(adj, v):
    nb = list(adj[v])
    return sum(1 for i in range(len(nb)) for j in range(i + 1, len(nb)) if nb[j] not in adj[nb[i]])


def eliminate(G: nx.Graph, order=None, rule: str | None = None, seed: int = 0) -> dict:
    """Run the elimination game. Give either a fixed `order` or a greedy `rule`
    ('minfill' or 'mindeg'; ties broken by the other key, then a seeded random value).
    Returns a trace dict with one entry per step."""
    if (order is None) == (rule is None):
        raise ValueError("give exactly one of order= or rule=")
    adj = {v: set(G[v]) for v in G}
    rng = random.Random(seed)
    tie = {v: rng.random() for v in adj}
    seq = list(order) if order is not None else None
    steps, width = [], 0
    for t in range(len(adj)):
        if seq is not None:
            v = seq[t]
        elif rule == "minfill":
            v = min(adj, key=lambda u: (_fill(adj, u), len(adj[u]), tie[u]))
        elif rule == "mindeg":
            v = min(adj, key=lambda u: (len(adj[u]), _fill(adj, u), tie[u]))
        else:
            raise ValueError(rule)
        nb = set(adj[v])
        fill = []
        for a, b in itertools.combinations(sorted(nb, key=str), 2):
            if b not in adj[a]:
                adj[a].add(b); adj[b].add(a); fill.append([a, b])
        for a in nb:
            adj[a].discard(v)
        del adj[v]
        width = max(width, len(nb))
        steps.append({"elim": v, "nbrs": sorted(nb, key=str), "fill": fill,
                      "bag": len(nb) + 1, "width": width})
    costs = [s["bag"] for s in steps]
    return {"kind": "elimination", "steps": steps, "width": width,
            "log2cost": log2sumexp2(costs)}


def log2sumexp2(exps) -> float:
    m = max(exps)
    return m + math.log2(sum(2.0 ** (e - m) for e in exps))


def best_greedy(G, rule="minfill", seeds=range(10)):
    """Best width over seeds (ties broken by lower cost). Returns (trace, seed)."""
    best = None
    for s in seeds:
        tr = eliminate(G, rule=rule, seed=s)
        key = (tr["width"], tr["log2cost"])
        if best is None or key < best[0]:
            best = (key, tr, s)
    return best[1], best[2]

# ---------------------------------------------------------------- fixed orders on grids

def column_sweep(L, W=None):
    W = W or L
    return [(r, c) for c in range(W) for r in range(L)]


def nested_dissection(rows, cols, leaf=2, log=None, depth=0):
    """Geometric ND on a rectangle: split the longer side at its middle; separator LAST."""
    rows, cols = list(rows), list(cols)
    if len(rows) * len(cols) <= leaf * leaf or (len(rows) <= 1 and len(cols) <= 1):
        return [(r, c) for r in rows for c in cols]
    if len(cols) >= len(rows):
        m = len(cols) // 2
        S = [(r, cols[m]) for r in rows]
        A = nested_dissection(rows, cols[:m], leaf, log, depth + 1)
        B = nested_dissection(rows, cols[m + 1:], leaf, log, depth + 1)
    else:
        m = len(rows) // 2
        S = [(rows[m], c) for c in cols]
        A = nested_dissection(rows[:m], cols, leaf, log, depth + 1)
        B = nested_dissection(rows[m + 1:], cols, leaf, log, depth + 1)
    if log is not None:
        log.append({"depth": depth, "separator": S})
    return A + B + S


def hybrid_cut_and_sweep(L):
    """One middle column cut, then each half swept inward toward it."""
    m = L // 2
    left = [(r, c) for c in range(m) for r in range(L)]
    right = [(r, c) for c in range(L - 1, m, -1) for r in range(L)]
    return left + right + [(r, m) for r in range(L)]

# ---------------------------------------------------------------- pairwise contraction (carving model)

def carving_dp(T: dict):
    """Exact best pairwise contraction tree for small networks (<= ~12 tensors).
    Minimizes (max intermediate rank excluding inputs, total cost) lexicographically.
    Step cost = 2^|legs(A) U legs(B)|. Returns (max_rank, cost, tree)."""
    names = sorted(T); n = len(names); legs = [frozenset(T[v]) for v in names]
    OL = []
    for m in range(1 << n):
        cnt = {}
        for i in range(n):
            if m >> i & 1:
                for l in legs[i]:
                    cnt[l] = cnt.get(l, 0) + 1
        OL.append(frozenset(l for l, c in cnt.items() if c == 1))

    @lru_cache(None)
    def f(mask):
        if mask & (mask - 1) == 0:
            return (0, 0, names[mask.bit_length() - 1])
        best, low, sub = None, mask & -mask, (mask - 1) & mask
        while sub:
            if sub & low:
                a, b = f(sub), f(mask ^ sub)
                r = max(a[0], b[0], len(OL[mask]))
                c = a[1] + b[1] + 2 ** len(OL[sub] | OL[mask ^ sub])
                if best is None or (r, c) < best[:2]:
                    best = (r, c, (a[2], b[2]))
            sub = (sub - 1) & mask
        return best
    return f((1 << n) - 1)


def best_caterpillar(T: dict):
    """Best sequential ('list') schedule by brute force: one running tensor absorbs the next."""
    best = None
    for p in itertools.permutations(sorted(T)):
        run, r, cost = set(T[p[0]]), 0, 0
        for v in p[1:]:
            cost += 2 ** len(run | T[v]); run ^= T[v]; r = max(r, len(run))
        if best is None or (r, cost) < best[:2]:
            best = (r, cost, p)
    return best

# ---------------------------------------------------------------- separators

def separator_report(G: nx.Graph, S, alpha=(0.5, 2 / 3)) -> dict:
    S = set(S); n = G.number_of_nodes()
    comps = [set(c) for c in nx.connected_components(G.subgraph(set(G) - S))]
    full = [c for c in comps if all(any(u in c for u in G[s]) for s in S)]
    sizes = sorted((len(c) for c in comps), reverse=True)
    is_sep = bool(S) and len(comps) >= 2
    return {"separator": is_sep, "sizes": sizes, "minimal": is_sep and len(full) >= 2,
            **{f"balanced_{a:.3f}": is_sep and sizes[0] <= a * n for a in alpha}}

# ---------------------------------------------------------------- trace I/O

def to_jsonable(x):
    if isinstance(x, tuple):
        return "_".join(map(str, x))
    if isinstance(x, dict):
        return {to_jsonable(k) if isinstance(k, tuple) else k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, set)):
        return [to_jsonable(v) for v in x]
    return x


def write_trace(path, figure_id, G, positions, trace, meta=None):
    doc = {"schema": SCHEMA_VERSION, "id": figure_id, "meta": meta or {},
           "nodes": {to_jsonable(v): list(map(float, positions[v])) for v in G},
           "edges": [[to_jsonable(a), to_jsonable(b)] for a, b in G.edges()],
           **to_jsonable(trace)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc

# ---------------------------------------------------------------- reference checks

def selftest():
    ok = True
    def check(name, got, want):
        nonlocal ok
        flag = "ok " if got == want else "FAIL"
        ok &= got == want
        print(f"[{flag}] {name}: got {got}, want {want}")

    ring = tensor_network("ABCD", [("A", "B", "i"), ("B", "C", "j"), ("C", "D", "k"), ("D", "A", "l")])
    tr = eliminate(index_graph(ring), order=["i", "j", "k", "l"])
    check("ring width", tr["width"], 2)
    check("ring total cost", sum(2 ** s["bag"] for s in tr["steps"]), 22)

    bow = tensor_network("UVPQHRS", [("U", "P", "u"), ("V", "Q", "v"), ("P", "Q", "x"), ("P", "H", "p"),
                                     ("Q", "H", "q"), ("H", "R", "r"), ("H", "S", "s"), ("R", "S", "t")])
    r, c, _ = carving_dp(bow)
    check("bowtie best tree (rank, cost)", (r, c), (2, 50))
    r, c, _ = best_caterpillar(bow)
    check("bowtie best list (rank, cost)", (r, c), (3, 58))
    IG = index_graph(bow)
    best = None
    for p in itertools.permutations(sorted(IG)):
        tr = eliminate(IG, order=p)
        key = (tr["width"], sum(2 ** s["bag"] for s in tr["steps"]))
        best = key if best is None or key < best else best
    check("bowtie best elimination (width, cost)", best, (3, 62))

    for L, want in [(8, 10), (12, 15)]:
        G = nx.convert_node_labels_to_integers(grid_graph(L))
        trb, _ = best_greedy(G, "minfill", range(10))
        check(f"MinFill grid L={L} best width", trb["width"], want)
    G = grid_graph(7)
    check("balanced ND 7x7 width", eliminate(G, order=nested_dissection(range(7), range(7)))["width"], 9)
    for L in (7, 15):
        G = grid_graph(L)
        check(f"hybrid width L={L}", eliminate(G, order=hybrid_cut_and_sweep(L))["width"], L)
        check(f"column sweep width L={L}", eliminate(G, order=column_sweep(L))["width"], L)

    SG = nx.Graph()
    SG.add_edges_from(itertools.combinations("abcd", 2)); SG.add_edges_from(itertools.combinations("efgh", 2))
    SG.add_edges_from([("c", "x"), ("x", "e"), ("d", "y"), ("y", "z"), ("z", "f"), ("z", "g")])
    check("min a-h separator", nx.node_connectivity(SG, "a", "h"), 2)
    check("{e,f,g} minimal", separator_report(SG, "efg")["minimal"], True)
    check("{x,y,z} minimal", separator_report(SG, "xyz")["minimal"], False)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


def demo(out):
    out = Path(out)
    ring = tensor_network("ABCD", [("A", "B", "i"), ("B", "C", "j"), ("C", "D", "k"), ("D", "A", "l")])
    IG = index_graph(ring)
    pos = {"i": (340, 80), "j": (390, 130), "k": (340, 180), "l": (290, 130)}
    write_trace(out / "F03.json", "F03", IG, pos, eliminate(IG, order=["i", "j", "k", "l"]),
                {"title": "Ring: eliminating indices", "label_mode": "fill"})
    G = grid_graph(6)
    tr, seed = best_greedy(G, "minfill", range(10))
    write_trace(out / "F06.json", "F06", G, grid_positions(G), tr,
                {"title": "MinFill on a 6x6 grid", "rule": "minfill", "seed": seed, "label_mode": "fill"})
    G = grid_graph(7); log = []
    order = nested_dissection(range(7), range(7), log=log)
    tr = eliminate(G, order=order)
    tr["separators"] = log
    write_trace(out / "F12.json", "F12", G, grid_positions(G, 44, 33), tr,
                {"title": "Balanced nested dissection, 7x7", "label_mode": "separator"})
    print("wrote", *sorted(p.name for p in out.glob("*.json")))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "demo":
        demo(sys.argv[2] if len(sys.argv) > 2 else "data")
