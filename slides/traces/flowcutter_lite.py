"""flowcutter_lite.py — an educational FlowCutter (Hamann & Strasser) for vertex separators.

Start with source set S = {s} and target set T = {t}. Repeat:
  1. compute a minimum S–T vertex cut by max-flow (vertices outside S ∪ T have capacity 1);
  2. record the cut and its balance;
  3. grow the smaller side: it absorbs everything it reaches in the residual graph, then
     "pierces" one vertex of the cut, preferring a vertex that does not create an augmenting
     path (not reachable from the other side), then the one farthest from the other terminal.
Cut sizes never decrease and the balance of the cut keeps improving, so the recorded cuts
trace a size-versus-balance trade-off; the non-dominated ones form the Pareto front.

The real FlowCutter augments the flow incrementally; here max-flow is recomputed from scratch,
which gives the same cuts on these small graphs and keeps the code short.
"""
from __future__ import annotations

import networkx as nx

INF = float("inf")


def _split_graph(G, S, T):
    D = nx.DiGraph()
    for v in G:
        D.add_edge(("in", v), ("out", v), capacity=INF if (v in S or v in T) else 1)
    for u, v in G.edges():
        D.add_edge(("out", u), ("in", v), capacity=INF)
        D.add_edge(("out", v), ("in", u), capacity=INF)
    for v in S:
        D.add_edge("SRC", ("in", v), capacity=INF)
    for v in T:
        D.add_edge(("out", v), "SNK", capacity=INF)
    return D


def _residual_reach(R, start, reverse=False):
    """Vertices reachable from `start` in the residual network (or reaching it, if reverse)."""
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        nbrs = R.pred[x] if reverse else R.succ[x]
        for y in nbrs:
            e = R[y][x] if reverse else R[x][y]
            if e["capacity"] - e["flow"] > 0 and y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def min_vertex_cut(G, S, T):
    D = _split_graph(G, S, T)
    R = nx.algorithms.flow.edmonds_karp(D, "SRC", "SNK")
    src = _residual_reach(R, "SRC")
    snk = _residual_reach(R, "SNK", reverse=True)
    side_s = {v for v in G if ("out", v) in src}          # strictly on the source side
    side_t = {v for v in G if ("in", v) in snk}           # strictly on the target side
    cut_s = {v for v in G if ("in", v) in src and v not in side_s}   # min cut closest to S
    cut_t = {v for v in G if ("out", v) in snk and v not in side_t}  # min cut closest to T
    return R.graph["flow_value"], (cut_s, cut_t), side_s, side_t


def flowcutter(G: nx.Graph, s, t, max_iter: int = 500) -> dict:
    n = G.number_of_nodes()
    S, T = {s}, {t}
    ds = nx.single_source_shortest_path_length(G, s)
    dt = nx.single_source_shortest_path_length(G, t)
    steps = []
    for _ in range(max_iter):
        if any(u in T for v in S for u in G[v]):
            break                      # S and T touch: no vertex cut separates them any more
        _, (cut_s, cut_t), side_s, side_t = min_vertex_cut(G, S, T)
        # FlowCutter reports the min cut on the smaller side, then grows that side
        grow_source = len(side_s) <= len(side_t)
        cut = cut_s if grow_source else cut_t
        rest = set(G) - cut
        A = {v for v in rest if v in side_s} if grow_source else rest - side_t
        B = rest - A
        steps.append({"S": sorted(S, key=str), "T": sorted(T, key=str), "cut": sorted(cut, key=str),
                      "A": sorted(A, key=str), "B": sorted(B, key=str), "size": len(cut),
                      "small_side": min(len(A), len(B)), "balance": min(len(A), len(B)) / n,
                      "grew": "S" if grow_source else "T"})
        other = side_t if grow_source else side_s
        cand = [v for v in cut if v not in (S | T)]
        if not cand:
            break
        dist_own, dist_other = (ds, dt) if grow_source else (dt, ds)
        # prefer cut vertices the other side cannot reach (no augmenting path), then far from it
        cand.sort(key=lambda v: (v in other, -(dist_other.get(v, 0) - dist_own.get(v, 0)), str(v)))
        p = cand[0]
        if grow_source:
            S = (S | (side_s - cut)) | {p}
        else:
            T = (T | (side_t - cut)) | {p}
        if S & T or len(S) + len(T) >= n:
            break
    return {"kind": "flowcutter", "source": s, "target": t, "steps": steps,
            "pareto": pareto_front(steps)}


def pareto_front(steps):
    """Indices of non-dominated cuts: no other cut is at most as large and strictly better balanced."""
    best, front = -1, []
    for i in sorted(range(len(steps)), key=lambda i: (steps[i]["size"], -steps[i]["small_side"])):
        if steps[i]["small_side"] > best:
            best = steps[i]["small_side"]
            front.append(i)
    return front


def peripheral_pair(G):
    """(s, t): a vertex farthest from an arbitrary start, and the vertex farthest from it."""
    v0 = min(G, key=str)
    d = nx.single_source_shortest_path_length(G, v0)
    s = max(d, key=lambda v: (d[v], str(v)))
    d = nx.single_source_shortest_path_length(G, s)
    t = max(d, key=lambda v: (d[v], str(v)))
    return s, t


def best_cut(fc, n, max_imbalance=0.6):
    """FlowCutter's ND choice: the minimum-expansion cut |S| / small side among cuts whose larger
    side holds at most max_imbalance of the vertices."""
    ok = [st for st in fc["steps"] if st["small_side"] > 0 and n - st["size"] - st["small_side"] <= max_imbalance * n]
    pool = ok or [st for st in fc["steps"] if st["small_side"] > 0]
    return min(pool, key=lambda st: (st["size"] / st["small_side"], st["size"]))


def nested_dissection_fc(G: nx.Graph, leaf: int = 4, log=None, depth: int = 0) -> list:
    """Elimination order by recursive FlowCutter-lite separators (separator last), on any graph."""
    order = []
    comps = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=lambda c: sorted(map(str, c)))]
    for H in comps:
        n = H.number_of_nodes()
        if n <= leaf or H.number_of_edges() == n * (n - 1) // 2:
            order += sorted(H, key=str)
            continue
        s, t = peripheral_pair(H)
        cut = best_cut(flowcutter(H, s, t), n)
        S = cut["cut"]
        if log is not None:
            log.append({"depth": depth, "separator": S})
        order += nested_dissection_fc(H.subgraph(set(H) - set(S)).copy(), leaf, log, depth + 1)
        order += S
    return order
