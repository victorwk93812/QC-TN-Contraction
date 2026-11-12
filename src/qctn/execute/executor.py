"""Execute a pairwise contraction plan on a `qctn.tn` network, step by step, on NumPy or CuPy.

A plan is the list of (a, b, c) steps of the authors' files (`AuthorsPlan.steps`), of `qctn.plans.generators`, or
a `PlanTree`. Each step contracts the two named tensors into `c`, in the plan's order; nothing is reordered or
re-optimised (no cotengra optimiser). Tensors left at the end are joined by outer products in the order they were
produced, like the release's `simple_contraction_rank!` (`src/funcions_article_IA.jl` l. 113-130).

Two representations (`hyper=`):

- `False` (default): every tensor dense, one axis per index. A step is one `tensordot` over the common indices;
  the result's indices are A's free ones then B's, the order `qctn.tn.replay` uses.
- `True`: QXTns' hyperindex form. QXTensor stores a tensor that is diagonal along groups of indices with one axis
  per group (`tensor.jl` l. 55-79, `utils.jl` `find_hyper_edges`, `reduce_tensor`), and `contract_pair`
  (`tensor_network.jl` l. 244-263) labels each tensor's axes by the network-wide groups of `find_connected_indices`
  (l. 486-505) through `contraction_indices` (l. 307-348), then calls OMEinsum's `EinCode`. A group shared by A
  and B that survives is a batch label. Here the groups are the connected components of the leaves' diagonal
  relations (union-find): components only lose members as indices are contracted and never split (a surviving
  group's members end up in one group of C), so this is the same grouping. A step is one `einsum`.

`compile_plan` does all index bookkeeping once (the paper excludes plan creation from the timing); `run` only
moves arrays.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from qctn.execute.tensors import diagonal_groups
from qctn.tn import TensorNetwork

LETTERS = string.ascii_letters  # numpy einsum: at most 52 labels per call


def steps_of(plan) -> list[tuple[str, str, str]]:
    """(a, b, c) steps from an AuthorsPlan, a PlanTree, or any iterable of triples."""
    if hasattr(plan, "steps"):
        return [(s.a, s.b, s.c) for s in plan.steps]
    if hasattr(plan, "children") and hasattr(plan, "order"):
        return [(*plan.children[c], c) for c in plan.order]
    return [tuple(s) for s in plan]


@dataclass
class Op:
    a: int  # slot of A
    b: int  # slot of B
    c: int  # slot of the result
    axes: tuple | None = None  # tensordot axes (dense)
    eq: str | None = None  # einsum equation (hyper)
    out_shape: tuple = ()


@dataclass
class Program:
    """A compiled plan: leaf slots, ops, and size statistics."""
    leaves: list[str]  # tensor id of each leaf slot
    leaf_shapes: list[tuple]
    leaf_reduce: list[str | None]  # einsum to reduce a dense leaf to hyperindex form (None: as is)
    ops: list[Op]
    hyper: bool
    n_plan_steps: int
    result_slot: int
    peak_elements: int = 0  # largest sum of live tensor sizes (elements), leaves included
    max_elements: int = 0  # largest single tensor
    flops: float = 0.0  # sum over steps of prod(dims of all labels involved)
    reduced_ranks: list[int] = field(default_factory=list)  # per plan step: rank of C as stored

    def peak_bytes(self, itemsize: int = 8) -> int:
        return self.peak_elements * itemsize


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        p = self.p.setdefault(x, x)
        if p != x:
            p = self.p[x] = self.find(p)
        return p

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx


def compile_plan(tn: TensorNetwork, plan, *, hyper: bool = False,
                 arrays: dict[str, np.ndarray] | None = None) -> Program:
    """Index bookkeeping for `plan` on `tn`. `hyper=True` needs the leaf `arrays` (to find diagonal groups)."""
    steps = steps_of(plan)
    leaves = list(tn.tensors)
    slot = {t: i for i, t in enumerate(leaves)}
    inds: dict[int, list] = {i: list(tn.tensors[t]) for i, t in enumerate(leaves)}
    comp = _UF()
    leaf_reduce: list[str | None] = [None] * len(leaves)
    if hyper:
        if arrays is None:
            raise ValueError("hyper=True needs the leaf arrays")
        for t in leaves:
            ix = tn.tensors[t]
            for g in diagonal_groups(arrays[t]):
                for k in g[1:]:
                    comp.union(ix[g[0]], ix[k])

    def groups(ix):
        """Axes of a tensor with indices `ix`: one per component, in order of first appearance."""
        if not hyper:
            return [(x,) for x in ix]
        seen: dict = {}
        for x in ix:
            seen.setdefault(comp.find(x), []).append(x)
        return [tuple(v) for v in seen.values()]

    def shape(gs):
        return tuple(tn.dims[g[0]] for g in gs)

    leaf_shapes = []
    for i, t in enumerate(leaves):
        gs = groups(inds[i])
        leaf_shapes.append(shape(gs))
        if hyper and len(gs) < len(inds[i]):
            pos = {x: LETTERS[k] for k, g in enumerate(gs) for x in g}
            leaf_reduce[i] = "".join(pos[x] for x in inds[i]) + "->" + "".join(LETTERS[k] for k in range(len(gs)))

    size = {i: int(np.prod(s, dtype=np.int64)) for i, s in enumerate(leaf_shapes)}
    live = dict(size)
    cur = peak = sum(live.values())
    biggest = max(live.values(), default=0)
    ops: list[Op] = []
    flops = 0.0
    nxt = len(leaves)
    reduced = []
    order: list[int] = []  # result slots in order produced, for the final outer products

    def contract(sa, sb):
        nonlocal nxt, flops, cur, peak, biggest
        A, B = inds.pop(sa), inds.pop(sb)
        sB, sA = set(B), set(A)
        C = [x for x in A if x not in sB] + [x for x in B if x not in sA]
        sc = nxt
        nxt += 1
        if not hyper:
            ia = [A.index(x) for x in A if x in sB]
            ib = [B.index(x) for x in A if x in sB]
            cs = shape(groups(C))
            op = Op(sa, sb, sc, axes=(tuple(ia), tuple(ib)), out_shape=cs)
            f = np.prod([tn.dims[x] for x in A], dtype=float) * np.prod(
                [tn.dims[x] for x in B if x not in sA], dtype=float)
        else:
            ga, gb, gc = groups(A), groups(B), groups(C)
            lab: dict = {}
            for g in ga + gb + gc:
                lab.setdefault(comp.find(g[0]), LETTERS[len(lab)] if len(lab) < len(LETTERS) else None)
            if None in lab.values():
                raise ValueError(f"step {sa},{sb}: more than {len(LETTERS)} labels")
            sub = lambda gs: "".join(lab[comp.find(g[0])] for g in gs)
            cs = shape(gc)
            op = Op(sa, sb, sc, eq=f"{sub(ga)},{sub(gb)}->{sub(gc)}", out_shape=cs)
            f = np.prod([tn.dims[next(x for x in ga + gb if comp.find(x[0]) == r)[0]] for r in lab], dtype=float)
        inds[sc] = C
        ops.append(op)
        flops += f
        n = int(np.prod(cs, dtype=np.int64))
        cur += n
        peak = max(peak, cur)
        cur -= live.pop(sa) + live.pop(sb)
        live[sc] = n
        biggest = max(biggest, n)
        order.append(sc)
        return sc, len(cs)

    for k, (a, b, c) in enumerate(steps):
        if a not in slot or b not in slot:
            raise ValueError(f"step {k + 1}: {a} or {b} is not a live tensor")
        if a == b:
            raise ValueError(f"step {k + 1}: {a} contracted with itself")
        sa, sb = slot.pop(a), slot.pop(b)
        sc, r = contract(sa, sb)
        if c in slot:
            raise ValueError(f"step {k + 1}: result name {c} is live already")
        slot[c] = sc
        reduced.append(r)
    rest = sorted(slot.values())
    while len(rest) > 1:
        sc, _ = contract(rest[0], rest[1])
        rest = [sc] + rest[2:]
    return Program(leaves, leaf_shapes, leaf_reduce, ops, hyper, len(steps), rest[0] if rest else -1,
                   peak, biggest, flops, reduced)


def get_backend(name: str):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy

        return cupy
    raise ValueError(f"backend must be 'numpy' or 'cupy', not {name!r}")


def prepare(program: Program, arrays: dict[str, np.ndarray], xp=np, dtype=np.complex64) -> list:
    """Leaf arrays in slot order, cast to `dtype`, reduced to hyperindex form if needed, on `xp`'s device."""
    out = []
    for t, red in zip(program.leaves, program.leaf_reduce):
        a = np.asarray(arrays[t])
        if red is not None:
            a = np.einsum(red, a)
        out.append(xp.asarray(np.ascontiguousarray(a, dtype=dtype)))
    return out


def run(program: Program, leaves: Sequence, xp=np):
    """Contract. `leaves` from `prepare`; they are not modified. Returns the final tensor (0-d for a closed
    network). Intermediates are released as soon as they are used."""
    slots: list = list(leaves) + [None] * len(program.ops)
    if program.hyper:
        einsum = xp.einsum
        for op in program.ops:
            slots[op.c] = einsum(op.eq, slots[op.a], slots[op.b])
            if op.a >= len(leaves):
                slots[op.a] = None
            if op.b >= len(leaves):
                slots[op.b] = None
    else:
        tensordot = xp.tensordot
        for op in program.ops:
            slots[op.c] = tensordot(slots[op.a], slots[op.b], axes=op.axes)
            if op.a >= len(leaves):
                slots[op.a] = None
            if op.b >= len(leaves):
                slots[op.b] = None
    return slots[program.result_slot]


def contract_plan(tn: TensorNetwork, plan, arrays: dict[str, np.ndarray], *, backend: str = "numpy",
                  hyper: bool = False, dtype=np.complex128):
    """Compile, prepare and run once; returns the result as a NumPy array."""
    xp = get_backend(backend)
    prog = compile_plan(tn, plan, hyper=hyper, arrays=arrays)
    out = run(prog, prepare(prog, arrays, xp, dtype), xp)
    return out.get() if hasattr(out, "get") else np.asarray(out)
