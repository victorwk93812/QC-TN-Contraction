"""Circuit to tensor network, following QXTools 1.0 / QXTns conventions.

`convert_to_tnc` (QXTools `tn_conversion.jl`) with its defaults, as the authors call it:

- Each qubit q has a wire index `Qubit{q+1}`; every gate on q primes it once (plev + 1).
- Gate tensors are numbered t1, t2, ... in circuit order. A one-qubit gate is one tensor
  [out, in]. A two-qubit gate on qubits [target, ctrl] is split by SVD (`decompose_gate`, threshold 1e-15)
  into A = [out_t, in_t, v] and B = [v, out_c, in_c], with a virtual index v of dimension chi (the
  operator Schmidt rank). If chi == 1 there is no v and both halves have rank 2.
- Then one input tensor per qubit ([Qubit q, plev 0]), then one output tensor per qubit (last plev).

Wire indices are named `("Qubit{q}", plev)`; virtual indices `("v", k)` in order of creation.
ITensors gives every index a random id, which the plan files print modulo 1000; those aren't reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qctn.circuits import Circuit

SVD_THRESHOLD = 1e-15


@dataclass
class TensorNetwork:
    tensors: dict[str, list[tuple]]  # id -> indices
    dims: dict[tuple, int]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def rank(self, tid: str) -> int:
        return len(self.tensors[tid])

    def cotengra_inputs(self, order=None):
        """(inputs, output, size_dict, leaf_order) for cotengra; index names as strings."""
        order = list(order or self.tensors)
        name = {ix: (f"{ix[0]}'{ix[1]}" if ix[0] != "v" else f"v{ix[1]}") for ix in self.dims}
        inputs = [[name[i] for i in self.tensors[t]] for t in order]
        size = {name[i]: d for i, d in self.dims.items()}
        return inputs, [], size, order


def schmidt_rank(matrix: np.ndarray, threshold: float = SVD_THRESHOLD) -> int:
    """Operator Schmidt rank across the two qubits, as `decompose_gate` counts it."""
    t = np.asarray(matrix).reshape(2, 2, 2, 2)  # (out0, out1, in0, in1)
    m = t.transpose(0, 2, 1, 3).reshape(4, 4)
    return int(np.sum(np.linalg.svd(m, compute_uv=False) > threshold))


def build(circuit: Circuit, inputs: bool = True, outputs: bool = True) -> TensorNetwork:
    n = circuit.n_qubits
    out = [(f"Qubit{q + 1}", 0) for q in range(n)]
    dims = {ix: 2 for ix in out}
    tensors: dict[str, list[tuple]] = {}
    nv = 0

    def push(ixs):
        tensors[f"t{len(tensors) + 1}"] = ixs

    for g in circuit.gates:
        ins = [out[q] for q in g.qubits]
        outs = [(tag, p + 1) for tag, p in ins]
        for q, o in zip(g.qubits, outs):
            out[q] = o
            dims[o] = 2
        if len(g.qubits) == 1:
            push([outs[0], ins[0]])
        elif len(g.qubits) == 2:
            chi = schmidt_rank(g.matrix)
            if chi == 1:
                push([outs[0], ins[0]])
                push([outs[1], ins[1]])
            else:
                nv += 1
                v = ("v", nv)
                dims[v] = chi
                push([outs[0], ins[0], v])
                push([v, outs[1], ins[1]])
        else:
            raise ValueError(f"gate on {len(g.qubits)} qubits")
    tn = TensorNetwork(tensors, dims)
    if inputs:
        for q in range(n):
            push([(f"Qubit{q + 1}", 0)])
            tn.inputs.append(f"t{len(tensors)}")
    if outputs:
        for q in range(n):
            push([out[q]])
            tn.outputs.append(f"t{len(tensors)}")
    return tn


@dataclass
class ReplayResult:
    ok: bool
    n_steps: int
    mismatches: list[str]
    unused: list[str]


def replay(tn: TensorNetwork, plan, max_report: int = 5) -> ReplayResult:
    """Contract `plan` (an AuthorsPlan) symbolically on `tn` and compare each step with the file:
    ranks of A, B, C and the common indices (tag, prime level, dimension; virtual indices by dimension)."""
    live = {t: list(ix) for t, ix in tn.tensors.items()}
    bad = []
    for i, s in enumerate(plan.steps):
        if s.a not in live or s.b not in live:
            bad.append(f"step {i + 1}: {s.a} or {s.b} not live")
            break
        a, b = live.pop(s.a), live.pop(s.b)
        common = [x for x in a if x in b]
        c = [x for x in a if x not in b] + [x for x in b if x not in a]
        live[s.c] = c
        ours = (len(a), len(b), len(c))
        theirs = (s.rank_a, s.rank_b, s.rank_c)
        mine = sorted((x[0] if x[0] != "v" else "", x[1] if x[0] != "v" else 0, tn.dims[x]) for x in common)
        file = sorted((ix.tags, ix.plev if ix.tags else 0, ix.dim) for ix in s.parsed_indices())
        if ours != theirs or mine != file:
            bad.append(f"step {i + 1} {s.a} {s.b}: ranks {ours} vs {theirs}, common {mine} vs {file}")
    unused = [t for t in tn.tensors if t in live]
    return ReplayResult(not bad, len(plan.steps), bad[:max_report], unused)
