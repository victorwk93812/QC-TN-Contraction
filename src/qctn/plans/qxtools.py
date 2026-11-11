"""The QXTools 1.0 planning path, step for step: network → line graph → elimination order → plan.

Sources (Julia depot of the authors' Manifest; QXTools 1.0.0, QXGraphDecompositions 1.0.0, QXTns, LightGraphs,
Julia 1.9.1):

- `convert_to_line_graph` (QXTools `contraction_planning.jl`, l. 47–63): one vertex per bond, in the
  insertion order of QXTns' `bond_map` (an OrderedDict filled as tensors are pushed, index by index); vertices
  i < j are joined when their bonds share a tensor. Labels are `:1, :2, ...`.
- `min_fill` (QXGraphDecompositions `treewidth.jl`, l. 165–182) with `eliminate!(G, v, c_map)` and
  `cliqueness` (`LabeledGraph.jl`, l. 326–410): the next vertex is `argmin(cliqueness_map)` over a
  `Dict{Int, Int}`, i.e. the first minimum in the Dict's slot order, and LightGraphs' `rem_vertex!` moves the
  last vertex into the removed one's place. `JuliaIntDict` reproduces Julia 1.9's Dict slot layout
  (`base/dict.jl`, `hash_64_64` in `base/hashing.jl`) so ties are broken as in Julia.
- `order_from_tree_decomposition` (`treewidth.jl`, l. 198–232) and `graph_to_gr` (`LabeledGraph.jl`, l. 56).
- `order_to_contraction_plan` (`contraction_planning.jl`, l. 386–448): for each bond in the order, contract the
  intermediates holding its two tensors into `I<i>`, i = the bond's position in the order; bonds already
  inside one intermediate are skipped.

Only plain bonds are handled (`hypergraph=false`). With hyperedges QXTools calls netcon for edges of more than
two tensors; none of the authors' plans for the 17 QASM circuits contain netcon's `t<n>` results.
"""
from __future__ import annotations

from bisect import insort
from dataclasses import dataclass

import numpy as np

_M64 = (1 << 64) - 1


def hash_64_64(n: int) -> int:
    """Julia's `hash_64_64` (base/hashing.jl); `hash(x::Int64) == hash_64_64(x)` for h = 0."""
    a = n & _M64
    a = (~a + (a << 21)) & _M64
    a ^= a >> 24
    a = (a + (a << 3) + (a << 8)) & _M64
    a ^= a >> 14
    a = (a + (a << 2) + (a << 4)) & _M64
    a ^= a >> 28
    a = (a + (a << 31)) & _M64
    return a


def _tablesz(x: int) -> int:
    return 16 if x < 16 else 1 << (x - 1).bit_length()


def julia_dict_slots(keys) -> list[int]:
    """Slot (0-based) of each key after `Dict{Int,Int}(k => v for k in keys)` in Julia 1.9.

    `Dict{K,V}(kv)` calls `sizehint!(length)` then inserts in order (`base/dict.jl`, l. 80–86). Insertion
    probes linearly from `hash & (sz-1)`; a rehash happens if the table gets 2/3 full or a probe exceeds
    `max(16, sz >> 6)`, which re-inserts in slot order.
    """
    keys = list(keys)
    sz = _tablesz(-(-3 * len(keys) // 2)) if keys else 16
    slots: list = [None] * sz

    def rehash(newsz):
        new = [None] * newsz
        for k in slots:  # re-inserted in old slot order, plain linear probing
            if k is not None:
                i = hash_64_64(k) & (newsz - 1)
                while new[i] is not None:
                    i = (i + 1) & (newsz - 1)
                new[i] = k
        return new

    count = 0
    for k in keys:
        while True:
            maxallowed = max(16, sz >> 6)
            i = hash_64_64(k) & (sz - 1)
            it = 0
            while slots[i] is not None and it + 1 < maxallowed:  # slots at probe 0 .. maxallowed-1
                i = (i + 1) & (sz - 1)
                it += 1
            if slots[i] is None:
                break
            sz = sz * 2 if count > 64000 else sz * 4  # ht_keyindex2_shorthash!: rehash!, then retry
            slots = rehash(sz)
        slots[i] = k
        count += 1
        if count * 3 > sz * 2:  # _setindex!: rehash! to count*4 (never with the sizehint above)
            sz = _tablesz(count * 2 if count > 64000 else count * 4)
            slots = rehash(sz)
    pos = {k: i for i, k in enumerate(slots) if k is not None}
    return [pos[k] for k in keys]


class SimpleGraph:
    """LightGraphs' SimpleGraph: vertices 1..n, sorted adjacency, `rem_vertex!` swaps in the last vertex."""

    def __init__(self, n: int):
        self.adj: list[set[int]] = [set() for _ in range(n + 1)]  # index 0 unused

    @property
    def n(self) -> int:
        return len(self.adj) - 1

    def add_edge(self, u: int, v: int) -> bool:
        if v in self.adj[u] or u == v:
            return False
        self.adj[u].add(v)
        self.adj[v].add(u)
        return True

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def neighbors(self, v: int) -> list[int]:
        return sorted(self.adj[v])

    def rem_vertex(self, v: int) -> None:
        n = self.n
        for s in self.adj[v]:
            self.adj[s].discard(v)
        self.adj[v] = set()
        if v != n:
            moved = self.adj[n]
            for s in moved:
                self.adj[s].discard(n)
                self.adj[s].add(v)
            self.adj[v] = moved
        self.adj.pop()

    def edges(self):
        """LightGraphs' edge iterator: (s, d) with s < d, s ascending, d ascending."""
        for s in range(1, self.n + 1):
            for d in sorted(self.adj[s]):
                if d > s:
                    yield s, d

    def copy(self) -> "SimpleGraph":
        g = SimpleGraph(0)
        g.adj = [set(a) for a in self.adj]
        return g


@dataclass
class LineGraph:
    graph: SimpleGraph
    labels: list[int]  # label of vertex i (1-based, labels[0] unused): the bond's position in `bonds`
    bonds: list[tuple]  # bond index names, in bond_map order
    edges: list[list[str]]  # tensors on each bond, in push order

    def to_gr(self) -> str:
        """`graph_to_gr`: `p tw n m`, then the edges in LightGraphs order."""
        g = self.graph
        es = list(g.edges())
        return f"p tw {g.n} {len(es)}\n" + "".join(f"{s} {d}\n" for s, d in es)


def bond_order(tn) -> tuple[list[tuple], list[list[str]]]:
    """QXTns `bond_map` order: first appearance, scanning tensors in push order and their indices in order."""
    seen: dict[tuple, list[str]] = {}
    for t, ixs in tn.tensors.items():
        for ix in ixs:
            seen.setdefault(ix, []).append(t)
    return list(seen), list(seen.values())


def line_graph(tn) -> LineGraph:
    """`convert_to_line_graph(tn; use_hyperedges=false)`. Open indices (one tensor) are vertices too."""
    bonds, edges = bond_order(tn)
    g = SimpleGraph(len(bonds))
    by_tensor: dict[str, list[int]] = {}
    for i, ts in enumerate(edges, 1):
        for t in ts:
            by_tensor.setdefault(t, []).append(i)
    for vs in by_tensor.values():
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                g.add_edge(vs[a], vs[b])
    return LineGraph(g, list(range(len(bonds) + 1)), bonds, edges)


def _cliqueness(g: SimpleGraph, v: int) -> int:
    nb = g.neighbors(v)
    c = 0
    for i in range(len(nb)):
        ai = g.adj[nb[i]]
        for j in range(i + 1, len(nb)):
            if nb[j] not in ai:
                c += 1
    return c


def min_fill(lg: LineGraph) -> tuple[int, list[int]]:
    """QXGraphDecompositions `min_fill`: (max degree at elimination, order of labels)."""
    g = lg.graph.copy()
    labels = list(lg.labels)
    n0 = g.n
    keys = list(range(1, n0 + 1))
    slot = julia_dict_slots(keys)
    # value per slot; deleted / empty slots hold +inf so argmin returns the first minimum in slot order
    nslots = max(slot) + 1 if slot else 0
    val = np.full(nslots, np.iinfo(np.int64).max, dtype=np.int64)
    slot_of = {k: s for k, s in zip(keys, slot)}
    key_at = {s: k for k, s in slot_of.items()}
    for v in keys:
        val[slot_of[v]] = _cliqueness(g, v)

    def add(k, d):
        val[slot_of[k]] += d

    max_degree, order = 0, []
    while g.n > 0:
        v = key_at[int(np.argmin(val))]
        nv_ = g.neighbors(v)
        max_degree = max(max_degree, len(nv_))
        order.append(labels[v])
        # eliminate!(G, v, c_map), LabeledGraph.jl l. 326-375
        for i in range(len(nv_)):
            vi = nv_[i]
            for j in range(i + 1, len(nv_)):
                ui = nv_[j]
                if g.add_edge(vi, ui):
                    avi, aui = g.adj[vi], g.adj[ui]
                    for x in avi & aui:
                        add(x, -1)
                    add(vi, sum(1 for x in avi if x != ui and x not in aui))
                    add(ui, sum(1 for x in aui if x != vi and x not in avi))
        av = g.adj[v]
        for x in nv_:
            add(x, -sum(1 for u in g.adj[x] if u != v and u not in av))
        n = g.n
        val[slot_of[v]] = val[slot_of[n]]
        val[slot_of[n]] = np.iinfo(np.int64).max
        del key_at[slot_of[n]]
        del slot_of[n]
        g.rem_vertex(v)
        labels[v] = labels[n]
        labels.pop()
    return max_degree, order


# ---------------------------------------------------------------------------------------------------------
# Tree decompositions (PACE .td) → elimination order


@dataclass
class TreeDecomposition:
    n_bags: int
    width: int  # max bag size - 1, as `s td` reports it
    n_vertices: int
    bags: dict[int, list[int]]  # bag number (1-based) -> vertices (1-based) in file order
    edges: list[tuple[int, int]]
    comments: list[str]


def parse_td(text: str) -> TreeDecomposition | None:
    """`flow_cutter` output parsing (treewidth.jl, l. 65–84). None if there is no `s` line."""
    bags, edges, comments, s = {}, [], [], None
    for line in text.splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "c":
            comments.append(line)
        elif w[0] == "s":
            s = (int(w[2]), int(w[3]) - 1, int(w[4]))
        elif w[0] == "b":
            bags[int(w[1])] = [int(x) for x in w[2:]]
        elif len(w) == 2:
            edges.append((int(w[0]), int(w[1])))
    if s is None:
        return None
    return TreeDecomposition(s[0], s[1], s[2], bags, edges, comments)


def valid_td(td: TreeDecomposition | None, lg: LineGraph) -> bool:
    """A complete tree decomposition of the line graph: all bags present, a tree on them, every vertex and
    every edge in some bag. (A solver killed while printing leaves a truncated file.)"""
    if td is None or td.n_bags == 0 or len(td.bags) != td.n_bags or td.n_vertices != lg.graph.n:
        return False
    if sorted(td.bags) != list(range(1, td.n_bags + 1)) or len(td.edges) != td.n_bags - 1:
        return False
    parent = list(range(td.n_bags + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in td.edges:
        if not (1 <= a <= td.n_bags and 1 <= b <= td.n_bags) or find(a) == find(b):
            return False
        parent[find(a)] = find(b)
    holds: dict[int, set[int]] = {}
    for b, vs in td.bags.items():
        for v in vs:
            holds.setdefault(v, set()).add(b)
    if set(holds) != set(range(1, lg.graph.n + 1)):
        return False
    return all(holds[u] & holds[v] for u, v in lg.graph.edges())


def order_from_tree_decomposition(td: TreeDecomposition, root: int = 1) -> list[int]:
    """QXGraphDecompositions `order_from_tree_decomposition` (treewidth.jl, l. 198–232).

    The tree is a LabeledGraph over bags b_1..b_k (`tree_from_tree_decompostion`), so removing a leaf
    moves the last bag into its vertex slot; leaves are searched in the order of the remaining bag list B.
    `all_neighbors(tree, b)[1]` is the neighbour with the smallest vertex number.
    """
    k = td.n_bags
    tree = SimpleGraph(k)
    for a, b in td.edges:
        tree.add_edge(a, b)
    labels = list(range(k + 1))  # vertex -> bag number
    vertex = {b: b for b in range(1, k + 1)}  # bag number -> vertex
    B = list(range(1, k + 1))
    order: list[int] = []
    while tree.n > 1:
        b = root
        for bi in B:
            if len(tree.adj[vertex[bi]]) == 1 and bi != root:
                b = bi
                break
        p = labels[min(tree.adj[vertex[b]])]
        pb = set(td.bags.get(p, []))
        seen = set()
        for x in td.bags.get(b, []):  # setdiff keeps first occurrences, in order
            if x not in pb and x not in seen:
                order.append(x)
                seen.add(x)
        if b != root:
            B.remove(b)
            v, n = vertex.pop(b), tree.n
            tree.rem_vertex(v)
            if v != n:
                labels[v] = labels[n]
                vertex[labels[v]] = v
            labels.pop()
    order.extend(td.bags.get(root, []))
    return order


# ---------------------------------------------------------------------------------------------------------
# Elimination order → plan


def order_to_contraction_plan(order: list[int], lg: LineGraph) -> list[tuple[str, str, str]]:
    """`order_to_contraction_plan` for plain bonds: `order` holds line-graph labels (1-based bond positions)."""
    parent: dict[str, str] = {}

    def find(t: str) -> str:
        while t in parent:
            t = parent[t]
        return t

    plan = []
    for i, lab in enumerate(order, 1):
        ts = lg.edges[lab - 1]
        if len(ts) == 2:
            a, b = find(ts[0]), find(ts[1])
            if a != b:
                c = f"I{i}"
                plan.append((a, b, c))
                parent[a] = c
                parent[b] = c
        elif len(ts) > 2:
            raise NotImplementedError("hyperedges (netcon) are not reproduced")
    return plan


def plan_from_order(tn, order: list[int], lg: LineGraph | None = None):
    return order_to_contraction_plan(order, lg or line_graph(tn))
