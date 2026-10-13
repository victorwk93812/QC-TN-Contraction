"""carving.py — pairwise contraction trees (the carving model).

A tensor network is {tensor: set(bond labels)}; a contraction tree is nested 2-tuples with tensor
names at the leaves. Merging A and B with m legs only on A, n only on B and k shared costs
2^(m+n+k) (χ = 2) and leaves a tensor of rank m+n.
"""
from __future__ import annotations

from .core import log2sumexp2, tensor_network


def leaves(tree):
    return [tree] if not isinstance(tree, tuple) else leaves(tree[0]) + leaves(tree[1])


def open_legs(T: dict, names) -> frozenset:
    cnt = {}
    for v in names:
        for l in T[v]:
            cnt[l] = cnt.get(l, 0) + 1
    return frozenset(l for l, c in cnt.items() if c == 1)


def tree_stats(T: dict, tree) -> dict:
    """Post-order steps with (m, n, k), plus max intermediate rank, max step exponent,
    total cost and the carving width of the tree (max cut over all tree edges, leaves included)."""
    steps = []

    def walk(t):
        if not isinstance(t, tuple):
            return [t], frozenset(T[t])
        la, a = walk(t[0])
        lb, b = walk(t[1])
        k, m, n = len(a & b), len(a - b), len(b - a)
        out = a ^ b
        steps.append({"left": la, "right": lb, "m": m, "n": n, "k": k,
                      "exp": m + n + k, "rank": len(out)})
        return la + lb, out

    walk(tree)
    exps = [s["exp"] for s in steps]
    inter = [s["rank"] for s in steps[:-1]] or [0]
    return {"steps": steps, "max_rank": max(inter), "max_exp": max(exps),
            "cost": sum(2 ** e for e in exps), "log2cost": log2sumexp2(exps),
            "carving_width": max(max(len(T[v]) for v in T), max(inter))}


# ---------------------------------------------------------------- grid tensor networks

def grid_tn(L: int, W: int | None = None) -> dict:
    """One tensor per site (r, c) of an L x W grid, one bond per lattice edge, no open legs."""
    W = W or L
    sites = [(r, c) for r in range(L) for c in range(W)]
    bonds = []
    for r in range(L):
        for c in range(W):
            if c + 1 < W:
                bonds.append(((r, c), (r, c + 1), f"h{r}_{c}"))
            if r + 1 < L:
                bonds.append(((r, c), (r + 1, c), f"v{r}_{c}"))
    return tensor_network(sites, bonds)


def bisection_tree(rows, cols):
    """Balanced bisection: split the longer side at its middle (columns on ties)."""
    rows, cols = list(rows), list(cols)
    if len(rows) == 1 and len(cols) == 1:
        return (rows[0], cols[0])
    if len(cols) >= len(rows):
        m = len(cols) // 2
        return (bisection_tree(rows, cols[:m]), bisection_tree(rows, cols[m:]))
    m = len(rows) // 2
    return (bisection_tree(rows[:m], cols), bisection_tree(rows[m:], cols))


def _is_leaf_site(t):
    return isinstance(t, tuple) and len(t) == 2 and all(isinstance(x, int) for x in t)


def grid_tree_stats(L: int) -> dict:
    """tree_stats for the balanced bisection of the L x L grid TN (leaves are (r, c) sites)."""
    T = grid_tn(L)
    tree = bisection_tree(range(L), range(L))
    # sites are tuples themselves, so wrap leaves as strings before walking
    def wrap(t):
        return f"{t[0]}_{t[1]}" if _is_leaf_site(t) else (wrap(t[0]), wrap(t[1]))
    TS = {f"{r}_{c}": legs for (r, c), legs in T.items()}
    return tree_stats(TS, wrap(tree))


# ---------------------------------------------------------------- layouts for drawing trees

def tree_record(T: dict, tree, x0=0.0, dx=60.0, y0=0.0, dy=60.0) -> dict:
    """Drawable contraction tree: node list with positions (root on top, y = depth), parent-child
    edges in merge (post-order) order, and per-node open legs and rank."""
    nodes, edges = [], []
    counter = [0]

    def walk(t, depth):
        if not isinstance(t, tuple):
            x = x0 + counter[0] * dx
            counter[0] += 1
            nid = f"L{t}"
            nodes.append({"id": nid, "kind": "leaf", "label": str(t), "x": x, "y": y0 + depth * dy,
                          "legs": sorted(T[t]), "rank": len(T[t])})
            return nid, [t], x
        a, la, xa = walk(t[0], depth + 1)
        b, lb, xb = walk(t[1], depth + 1)
        names = la + lb
        from_legs = open_legs(T, names)
        nid = f"N{len(nodes)}"
        legs_a, legs_b = open_legs(T, la), open_legs(T, lb)
        x = (xa + xb) / 2
        nodes.append({"id": nid, "kind": "internal", "x": x, "y": y0 + depth * dy,
                      "legs": sorted(from_legs), "rank": len(from_legs),
                      "m": len(legs_a - legs_b), "n": len(legs_b - legs_a), "k": len(legs_a & legs_b),
                      "exp": len(legs_a | legs_b)})
        edges.append({"parent": nid, "child": a, "legs": sorted(legs_a)})
        edges.append({"parent": nid, "child": b, "legs": sorted(legs_b)})
        return nid, names, x

    root, _, _ = walk(tree, 0)
    for n in nodes:
        if n["id"] == root:
            n["kind"] = "root"
    return {"nodes": nodes, "edges": edges, "root": root, **{k: v for k, v in tree_stats(T, tree).items() if k != "steps"}}


def caterpillar(order):
    """((((a b) c) d) …) for a list schedule."""
    t = order[0]
    for v in order[1:]:
        t = (t, v)
    return t


def carving_width_dp(T: dict) -> int:
    """Exact carving width of a tensor network's graph (bonds = edges, n <= ~14 tensors):
    the minimum over contraction trees of the largest number of bonds crossing any tree edge,
    leaf edges included. Subset DP over bipartitions."""
    from functools import lru_cache
    names = sorted(T, key=str)
    n = len(names)
    legs = [frozenset(T[v]) for v in names]
    OL = {}

    def cut(mask):
        if mask not in OL:
            cnt = {}
            for i in range(n):
                if mask >> i & 1:
                    for l in legs[i]:
                        cnt[l] = cnt.get(l, 0) + 1
            OL[mask] = sum(1 for c in cnt.values() if c == 1)
        return OL[mask]

    @lru_cache(None)
    def cw(mask):
        if mask & (mask - 1) == 0:
            return cut(mask)
        best, low, sub = None, mask & -mask, (mask - 1) & mask
        while sub:
            if sub & low:
                v = max(cut(sub), cut(mask ^ sub), cw(sub), cw(mask ^ sub))
                best = v if best is None else min(best, v)
            sub = (sub - 1) & mask
        return best

    return cw((1 << n) - 1)


def order_to_tree(T: dict, order):
    """Convert an index elimination order into a pairwise contraction tree.
    Eliminating index l merges the (current) tensors carrying l; an index whose tensors are
    already merged is a trace and adds nothing. Remaining pieces are joined at the end."""
    owners = {}
    for v, legs in T.items():
        for l in legs:
            owners.setdefault(l, []).append(v)
    roots = {v: v for v in T}                 # union-find: tensor -> representative

    def find(v):
        while roots[v] != v:
            v = roots[v]
        return v

    trees = {v: v for v in T}
    for l in order:
        rs = []
        for v in owners.get(l, []):
            r = find(v)
            if r not in rs:
                rs.append(r)
        while len(rs) > 1:
            a, b = rs.pop(0), rs.pop(0)
            trees[a] = (trees[a], trees.pop(b))
            roots[b] = a
            rs.insert(0, a)
    left = [r for r in trees]
    t = trees[left[0]]
    for r in left[1:]:
        t = (t, trees[r])
    return t
