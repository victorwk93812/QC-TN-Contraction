"""Girvan–Newman as the `gn` generator: remove edges of the tensor graph by betweenness, contract in reverse.

The release contains no code for this generator; the paper (Sec. 5.1) only names it, citing Girvan and Newman
(PNAS 2002). This version reproduces the release's `_pla_gn` plans step for step on the circuits we can build
(see `src/notes/deviations.md`, Phase 3):

- The graph is QXTools' `convert_to_graph(tn)` (`contraction_planning.jl`, l. 18–29): one vertex per tensor in
  network order, one edge per pair of tensors sharing at least one bond (parallel bonds collapse).
- Repeatedly remove the edge of largest (unnormalised) edge betweenness, recomputed after every removal.
  Ties go to the first edge in LightGraphs' edge order, i.e. the smallest (min vertex, max vertex).
  Ties are exact float equality by default (`rtol=0`): that reproduces the release, with either backend,
  while any tolerance from 1e-12 up changes some plans (pricingcall_7, portfolioqaoa_10). `rtol > 0` treats
  values within that relative distance of the maximum as tied.
- The contraction order is the reverse of the removal order: each removed tensor-graph edge stands for its bonds
  (in bond order), and QXTools' `order_to_contraction_plan` turns that bond order into the plan.

Betweenness only depends on the edge's own component, so after a removal only that component is recomputed.
Two backends give the same plans (tested): rustworkx (Rust, default) and networkx (pure Python, fallback).
Origin: the Girvan–Newman stepper of `slides/traces/gn.py`, rewritten for this tie rule and for speed.
"""
from __future__ import annotations

import itertools
import sys

from qctn.plans import qxtools as qx

RTOL = 0.0  # default tie tolerance, relative


def tensor_graph(tn, lg: qx.LineGraph | None = None):
    """(vertex number per tensor, edges as sorted vertex pairs, bonds per edge in bond order)."""
    lg = lg or qx.line_graph(tn)
    num = {t: i for i, t in enumerate(tn.tensors)}
    bonds: dict[tuple[int, int], list[int]] = {}
    for pos, ts in enumerate(lg.edges, 1):
        if len(ts) == 2:
            a, b = sorted((num[ts[0]], num[ts[1]]))
            bonds.setdefault((a, b), []).append(pos)
        elif len(ts) > 2:
            raise NotImplementedError("hyperedges")
    return num, bonds


def _betweenness_rx(nodes, edges, parallel_threshold):
    import rustworkx as rx

    g = rx.PyGraph()
    idx = {v: i for i, v in enumerate(nodes)}
    g.add_nodes_from(nodes)
    eids = g.add_edges_from_no_data([(idx[a], idx[b]) for a, b in edges])
    m = rx.edge_betweenness_centrality(g, normalized=False, parallel_threshold=parallel_threshold)
    return {e: m[i] for e, i in zip(edges, eids)}


def _betweenness_nx(nodes, edges, parallel_threshold=None):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    bc = nx.edge_betweenness_centrality(g, normalized=False)
    return {e: bc[e] if e in bc else bc[(e[1], e[0])] for e in edges}


def _best(bc: dict, rtol: float) -> tuple[float, tuple[int, int]]:
    m = max(bc.values())
    return m, min(e for e, c in bc.items() if c >= m - rtol * abs(m))


def removal_order(n: int, edges, backend: str = "rustworkx",
                  parallel_threshold: int = sys.maxsize, rtol: float | None = None) -> list[tuple[int, int]]:
    """Edges of a graph on vertices 0..n-1 in Girvan–Newman removal order (with the tie rule above)."""
    rtol = RTOL if rtol is None else rtol
    between = {"rustworkx": _betweenness_rx, "networkx": _betweenness_nx}[backend]
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def component(v):
        seen, stack = {v}, [v]
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    comps: dict[int, tuple] = {}  # id -> (nodes, betweenness of its edges, (best value, best edge))
    ids = itertools.count()

    def add_component(nodes):
        es = sorted({(min(a, b), max(a, b)) for a in nodes for b in adj[a]})
        if not es:
            return
        bc = between(sorted(nodes), es, parallel_threshold)
        comps[next(ids)] = (nodes, bc, _best(bc, rtol))

    done = set()
    for v in range(n):
        if v not in done and adj[v]:
            c = component(v)
            done |= c
            add_component(c)
    out = []
    while comps:
        top = max(b[0] for _, _, b in comps.values())
        cid, (_, _, (_, e)) = min(((c, x) for c, x in comps.items() if x[2][0] >= top - rtol * abs(top)),
                                  key=lambda cx: cx[1][2][1])
        del comps[cid]
        a, b = e
        adj[a].discard(b)
        adj[b].discard(a)
        out.append(e)
        ca = component(a)
        add_component(ca)
        if b not in ca:
            add_component(component(b))
    return out


def gn_order(tn, lg: qx.LineGraph | None = None, **kw) -> list[int]:
    """Bond order (line-graph labels) for `order_to_contraction_plan`."""
    lg = lg or qx.line_graph(tn)
    num, bonds = tensor_graph(tn, lg)
    removed = removal_order(len(num), list(bonds), **kw)
    return [p for e in reversed(removed) for p in bonds[e]]


def gn_plan(tn, lg: qx.LineGraph | None = None, **kw) -> list[tuple[str, str, str]]:
    lg = lg or qx.line_graph(tn)
    return qx.order_to_contraction_plan(gn_order(tn, lg, **kw), lg)
