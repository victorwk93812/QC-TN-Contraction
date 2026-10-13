"""td.py — tree decompositions from elimination orders, validation, and one Tamaki-style local
improvement step: re-solve exactly the region of the decomposition around its widest bag.

A tree decomposition here is (bags: {id: frozenset}, tree: nx.Graph on bag ids).
"""
from __future__ import annotations

import itertools

import networkx as nx

from .exact import treewidth_exact_order


def td_from_order(G: nx.Graph, order):
    """Bag of v = {v} ∪ (neighbours of v later in the order, fill included); parent = earliest of those."""
    pos = {v: i for i, v in enumerate(order)}
    adj = {v: set(G[v]) for v in G}
    bags, T = {}, nx.Graph()
    for v in order:
        later = {u for u in adj[v] if pos[u] > pos[v]}
        for a, b in itertools.combinations(later, 2):
            adj[a].add(b)
            adj[b].add(a)
        bags[v] = frozenset({v} | later)
        T.add_node(v)
        if later:
            T.add_edge(v, min(later, key=pos.get))
    roots = [c for c in nx.connected_components(T)]
    for a, b in zip(roots, roots[1:]):                 # disconnected graph: chain the components
        T.add_edge(next(iter(a)), next(iter(b)))
    return compress({k: v for k, v in bags.items()}, T)


def compress(bags, T):
    """Contract tree edges whose one bag contains the other (keeps the decomposition minimal)."""
    bags, T = dict(bags), T.copy()
    changed = True
    while changed:
        changed = False
        for a, b in list(T.edges()):
            if bags[a] <= bags[b] or bags[b] <= bags[a]:
                keep, drop = (b, a) if bags[a] <= bags[b] else (a, b)
                for c in list(T[drop]):
                    if c != keep:
                        T.add_edge(keep, c)
                T.remove_node(drop)
                del bags[drop]
                changed = True
                break
    return bags, T


def width(bags):
    return max(len(b) for b in bags.values()) - 1


def is_valid(G: nx.Graph, bags, T) -> bool:
    if not nx.is_tree(T) or set(T) != set(bags):
        return False
    if set().union(*bags.values()) != set(G):
        return False
    if any(not any(u in B and v in B for B in bags.values()) for u, v in G.edges()):
        return False
    return all(nx.is_connected(T.subgraph([b for b, B in bags.items() if v in B])) for v in G)


def improve_once(G: nx.Graph, bags, T, max_region: int = 16):
    """Take the widest bag and its tree neighbours (the region), build the torso = G on their vertices
    plus a clique on every separator to the rest of the tree, decompose the torso exactly, and splice
    it back. Returns (new bags, new tree, info) or None if the region is too big or nothing improves."""
    wide = max(bags, key=lambda b: (len(bags[b]), str(b)))
    region = {wide} | set(T[wide])
    U = set().union(*(bags[b] for b in region))
    if len(U) > max_region:
        return None
    H = G.subgraph(U).copy()
    seps = []
    for a in region:
        for c in T[a]:
            if c not in region:
                S = bags[a] & bags[c]
                seps.append((c, S))
                H.add_edges_from(itertools.combinations(S, 2))
    w_new, order = treewidth_exact_order(H)
    old_region_width = max(len(bags[b]) for b in region) - 1
    if w_new >= old_region_width:
        return None
    nb, nT = td_from_order(H, order)
    tag = 1 + max((k[1] for k in bags if isinstance(k, tuple) and len(k) == 3 and k[0] == "n"), default=0)
    nb = {("n", tag, k): v for k, v in nb.items()}          # fresh ids: never collide with earlier rounds
    nT = nx.relabel_nodes(nT, {k: ("n", tag, k) for k in list(nT)})
    out_bags = {k: v for k, v in bags.items() if k not in region}
    out_bags.update(nb)
    out_T = T.copy()
    out_T.remove_nodes_from(region)
    out_T.add_nodes_from(nT.nodes())
    out_T.add_edges_from(nT.edges())
    for c, S in seps:                                  # every separator is a clique of H, so some new bag holds it
        host = next(k for k, B in nb.items() if S <= B)
        out_T.add_edge(c, host)
    out_bags, out_T = compress(out_bags, out_T)
    info = {"widest": sorted(bags[wide], key=str), "region": sorted(U, key=str),
            "old_width": width(bags), "region_old": old_region_width, "region_new": w_new, "new_width": width(out_bags)}
    return out_bags, out_T, info


def td_to_order(G: nx.Graph, bag_list) -> list:
    """An elimination order of width <= (largest bag − 1): make every bag a clique (a chordal
    supergraph of G) and repeatedly eliminate a simplicial vertex (perfect elimination order)."""
    H = nx.Graph(G)
    for B in bag_list:
        H.add_edges_from(itertools.combinations(B, 2))
    order = []
    while H.number_of_nodes():
        v = min((u for u in H if all(H.has_edge(a, b) for a, b in itertools.combinations(H[u], 2))),
                key=lambda u: (H.degree(u), str(u)))
        order.append(v)
        H.remove_node(v)
    return order
