"""gn.py — Girvan–Newman on a tensor graph, and its dendrogram read as a contraction tree.

Each step removes the edge of highest betweenness (ties: smallest edge in string order), then
recomputes betweenness. Removing one edge splits at most one component into two, so the full
run is a binary hierarchy over the tensors; read bottom-up, it is a pairwise contraction tree.
"""
from __future__ import annotations

import networkx as nx


def _ekey(e):
    a, b = sorted(map(str, e))
    return (a, b)


def modularity(G: nx.Graph, parts) -> float:
    return nx.algorithms.community.modularity(G, [set(p) for p in parts])


def girvan_newman(G: nx.Graph) -> dict:
    """Trace: one step per removed edge, with the betweenness that chose it, the components
    afterwards, the split it caused (if any) and the modularity of the current partition."""
    H = G.copy()
    steps, splits = [], []
    while H.number_of_edges():
        bc = nx.edge_betweenness_centrality(H, normalized=False)
        top = max(bc.values())
        e = min((e for e, c in bc.items() if abs(c - top) < 1e-9), key=_ekey)
        before = {frozenset(c) for c in nx.connected_components(H)}
        H.remove_edge(*e)
        after = {frozenset(c) for c in nx.connected_components(H)}
        split = None
        if after != before:
            parent = next(iter(before - after))
            a, b = sorted(after - before, key=lambda c: (-len(c), sorted(map(str, c))))
            split = {"parent": sorted(parent, key=str), "a": sorted(a, key=str), "b": sorted(b, key=str)}
            splits.append(split)
        comps = sorted((sorted(c, key=str) for c in after), key=lambda c: (-len(c), c))
        steps.append({"removed": list(e), "betweenness": top, "split": split,
                      "components": comps, "modularity": modularity(G, comps)})
    return {"kind": "girvan_newman", "steps": steps, "splits": splits,
            "tree": dendrogram_tree(G, splits)}


def dendrogram_tree(G: nx.Graph, splits):
    """Nested pairs: each split of a component C into (A, B) becomes the contraction (A)(B)."""
    child = {frozenset(s["parent"]): (frozenset(s["a"]), frozenset(s["b"])) for s in splits}

    def build(C):
        if len(C) == 1:
            return next(iter(C))
        a, b = child[C]
        return (build(a), build(b))

    roots = [frozenset(c) for c in nx.connected_components(G)]
    tree = build(roots[0])
    for r in roots[1:]:          # disconnected inputs: outer product of the pieces
        tree = (tree, build(r))
    return tree


def best_modularity_partition(trace) -> tuple[float, list]:
    best = max(trace["steps"], key=lambda s: s["modularity"])
    return best["modularity"], best["components"]
