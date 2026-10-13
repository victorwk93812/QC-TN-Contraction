"""separators.py — nested dissection on general graphs with interchangeable bisection rules.

Each rule splits a connected graph into two vertex sets; the vertex separator is the smaller of
the two boundary sides (vertices with a neighbour across). Separators are eliminated last.
Used for the "how much does the separator finder matter" comparison on the random cubic graph.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .flowcutter_lite import nested_dissection_fc


def nd_by_bisection(G: nx.Graph, split, leaf: int = 4) -> list:
    order = []
    for comp in sorted(nx.connected_components(G), key=lambda c: min(map(str, c))):
        H = G.subgraph(comp).copy()
        if len(H) <= leaf:
            order += sorted(H, key=str)
            continue
        A, B = split(H)
        bA = {a for a in A if any(b in B for b in H[a])}
        bB = {b for b in B if any(a in A for a in H[b])}
        S = bA if len(bA) <= len(bB) else bB
        order += nd_by_bisection(H.subgraph(set(H) - S).copy(), split, leaf) + sorted(S, key=str)
    return order


def split_spectral(H):
    """Median split of the Fiedler vector."""
    f = nx.fiedler_vector(H, seed=0, method="tracemin_lu")
    nodes = list(H)
    idx = np.argsort(f)
    half = len(nodes) // 2
    return {nodes[i] for i in idx[:half]}, {nodes[i] for i in idx[half:]}


def split_bfs(H):
    """Half the vertices by BFS distance from a peripheral vertex."""
    s = min(H, key=str)
    d = nx.single_source_shortest_path_length(H, s)
    s = max(d, key=lambda v: (d[v], str(v)))
    d = nx.single_source_shortest_path_length(H, s)
    nodes = sorted(H, key=lambda v: (d[v], str(v)))
    h = len(nodes) // 2
    return set(nodes[:h]), set(nodes[h:])


def split_kl(seed):
    return lambda H: nx.algorithms.community.kernighan_lin_bisection(H, seed=seed)


ND_VARIANTS = {
    "spectral": lambda G: nd_by_bisection(G, split_spectral),
    "flowcutter": lambda G: nested_dissection_fc(G),
    "bfs": lambda G: nd_by_bisection(G, split_bfs),
    "kl0": lambda G: nd_by_bisection(G, split_kl(0)),
}
