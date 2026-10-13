"""exact.py — exact treewidth for small graphs (n <= ~20) by dynamic programming over subsets.

TW(S) = min_{v in S} max(TW(S - v), |Q(S - v, v)|), where Q(S, v) is the set of vertices outside
S ∪ {v} reachable from v through S (Bodlaender, Fomin, Koster, Kratsch, Thilikos 2006).
tw(G) = TW(V). Used to certify the small-graph facts quoted on slides.
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx


def treewidth_exact(G: nx.Graph) -> int:
    nodes = list(G)
    n = len(nodes)
    if n == 0:
        return -1
    idx = {v: i for i, v in enumerate(nodes)}
    nbr = [0] * n
    for u, v in G.edges():
        nbr[idx[u]] |= 1 << idx[v]
        nbr[idx[v]] |= 1 << idx[u]

    def q_size(S, v):
        # vertices outside S ∪ {v} adjacent to the component of v in G[S ∪ {v}]
        seen, frontier = 1 << v, 1 << v
        while frontier:
            x = (frontier & -frontier).bit_length() - 1
            frontier &= frontier - 1
            new = nbr[x] & S & ~seen
            seen |= new
            frontier |= new
        out = 0
        s = seen
        while s:
            x = (s & -s).bit_length() - 1
            s &= s - 1
            out |= nbr[x]
        return bin(out & ~(S | (1 << v))).count("1")

    @lru_cache(None)
    def tw(S):
        if S == 0:
            return -1
        best = n
        s = S
        while s:
            v = (s & -s).bit_length() - 1
            s &= s - 1
            rest = S & ~(1 << v)
            best = min(best, max(tw(rest), q_size(rest, v)))
        return best

    return tw((1 << n) - 1)


def treewidth_exact_order(G: nx.Graph):
    """(treewidth, an optimal elimination order) by the same subset DP, with reconstruction:
    in TW(S) the minimizing v is eliminated last among S, after an optimal order of S − v."""
    nodes = list(G)
    n = len(nodes)
    if n == 0:
        return -1, []
    idx = {v: i for i, v in enumerate(nodes)}
    nbr = [0] * n
    for u, v in G.edges():
        nbr[idx[u]] |= 1 << idx[v]
        nbr[idx[v]] |= 1 << idx[u]

    def q_size(S, v):
        seen, frontier = 1 << v, 1 << v
        while frontier:
            x = (frontier & -frontier).bit_length() - 1
            frontier &= frontier - 1
            new = nbr[x] & S & ~seen
            seen |= new
            frontier |= new
        out, s = 0, seen
        while s:
            x = (s & -s).bit_length() - 1
            s &= s - 1
            out |= nbr[x]
        return bin(out & ~(S | (1 << v))).count("1")

    @lru_cache(None)
    def tw(S):
        if S == 0:
            return (-1, -1)
        best = (n, -1)
        s = S
        while s:
            v = (s & -s).bit_length() - 1
            s &= s - 1
            rest = S & ~(1 << v)
            cand = max(tw(rest)[0], q_size(rest, v))
            if cand < best[0]:
                best = (cand, v)
        return best

    order, S = [], (1 << n) - 1
    width = tw(S)[0]
    while S:
        v = tw(S)[1]
        order.append(nodes[v])
        S &= ~(1 << v)
    return width, order[::-1]
