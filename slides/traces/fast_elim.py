"""fast_elim.py — incremental elimination game with the same semantics as core.eliminate.

core.eliminate recomputes fill(u) for every live vertex at every step, which is O(n^2 Δ^2)
and too slow for 24x24 grids or 320-vertex cubic graphs over many seeds. Here fill is kept
incrementally as  fill(u) = C(deg u, 2) - tri(u),  where tri(u) counts edges inside N(u).
Adding edge ab raises tri of a, b and every common neighbour; deleting an edge lowers them.

The selection key (fill, degree, tie[u]) and the seeded tie values are generated exactly as
in core.eliminate, so both functions return identical traces (tested in tests/test_fast_elim.py).
"""
from __future__ import annotations

import itertools
import random

import networkx as nx

from .core import log2sumexp2


def _add_edge(adj, tri, a, b):
    common = adj[a] & adj[b]
    for w in common:
        tri[w] += 1
    tri[a] += len(common)
    tri[b] += len(common)
    adj[a].add(b)
    adj[b].add(a)


def _remove_edge(adj, tri, a, b):
    adj[a].discard(b)
    adj[b].discard(a)
    common = adj[a] & adj[b]
    for w in common:
        tri[w] -= 1
    tri[a] -= len(common)
    tri[b] -= len(common)


def eliminate_fast(G: nx.Graph, order=None, rule: str | None = None, seed: int = 0,
                   record_steps: bool = True) -> dict:
    """Drop-in replacement for core.eliminate (same arguments, same trace)."""
    if (order is None) == (rule is None):
        raise ValueError("give exactly one of order= or rule=")
    adj = {v: set(G[v]) for v in G}
    rng = random.Random(seed)
    tie = {v: rng.random() for v in adj}
    tri = {v: sum(1 for a, b in itertools.combinations(adj[v], 2) if b in adj[a]) for v in adj}
    seq = list(order) if order is not None else None

    def fill(u):
        d = len(adj[u])
        return d * (d - 1) // 2 - tri[u]

    if rule == "minfill":
        key = lambda u: (fill(u), len(adj[u]), tie[u])
    elif rule == "mindeg":
        key = lambda u: (len(adj[u]), fill(u), tie[u])
    elif rule is not None:
        raise ValueError(rule)

    steps, bags, width = [], [], 0
    for t in range(len(adj)):
        v = seq[t] if seq is not None else min(adj, key=key)
        nb = set(adj[v])
        fills = []
        for a, b in itertools.combinations(sorted(nb, key=str), 2):
            if b not in adj[a]:
                _add_edge(adj, tri, a, b)
                fills.append([a, b])
        for a in nb:
            _remove_edge(adj, tri, v, a)
        del adj[v], tri[v]
        width = max(width, len(nb))
        bags.append(len(nb) + 1)
        if record_steps:
            steps.append({"elim": v, "nbrs": sorted(nb, key=str), "fill": fills,
                          "bag": len(nb) + 1, "width": width})
    out = {"kind": "elimination", "width": width, "log2cost": log2sumexp2(bags)}
    out["steps"] = steps if record_steps else [{"bag": b} for b in bags]
    return out


def greedy_runs(G, rule="minfill", seeds=range(10)):
    """[(seed, width, log2cost)] for each tie-break seed, without storing the steps."""
    res = []
    for s in seeds:
        tr = eliminate_fast(G, rule=rule, seed=s, record_steps=False)
        res.append((s, tr["width"], tr["log2cost"]))
    return res


def best_greedy_fast(G, rule="minfill", seeds=range(10)):
    """Same selection as core.best_greedy: lowest (width, log2cost), first seed on ties."""
    runs = greedy_runs(G, rule, seeds)
    s, _, _ = min(runs, key=lambda r: (r[1], r[2]))
    return eliminate_fast(G, rule=rule, seed=s), s


def best_width_first_seed(G, rule="minfill", seeds=range(10)):
    """(seed, width, log2cost) of the first seed in order that reaches the minimum width.
    This is the protocol behind the grid numbers in PROPOSAL.md's acceptance table."""
    runs = greedy_runs(G, rule, seeds)
    wmin = min(w for _, w, _ in runs)
    return next(r for r in runs if r[1] == wmin)
