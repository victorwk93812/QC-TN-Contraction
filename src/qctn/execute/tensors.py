"""Tensor values for a `qctn.tn` network, as QXTools `convert_to_tnc` fills them.

- Gate tensors hold the gate matrices (QXTools `gate_matrix`, `circuits.jl` l. 75: `Array{ComplexF64}`). A
  two-qubit gate is split as QXTns `decompose_gate` does (`utils.jl` l. 11-35): the matrix as (out0, in0 | out1,
  in1), SVD, singular values above 1e-15 kept, sqrt(s) on each side. The SVD basis is LAPACK's here (numpy);
  Julia's may differ by a unitary on degenerate singular values, which doesn't change any contraction result.
- Input and output tensors are the vector for '0' (or the given bit), `[1., 0.]` (QXTns
  `tensor_network_circuit.jl` l. 8-11, 159-172). With no `bits` the network is the amplitude <0...0|C|0...0>.

QXTns reshapes a two-qubit matrix in Julia's column-major order (`qxtns_matrix`), so the authors' networks apply
each two-qubit gate with its qubits' roles exchanged, and when the operator Schmidt rank is 1 QXTns `push!`
reshapes A instead of B (`tensor_network_circuit.jl` l. 97), so the second half gets the first half's values.
`leaf_arrays(convention="qxtns")` (default) reproduces both; `convention="circuit"` is the circuit as written.
Ranks and shapes are the same either way.

`diagonal_groups` reproduces QXTns `find_hyper_edges` (`utils.jl` l. 42-71): index positions along which a tensor
is diagonal, with the tolerance `eps(Float64)` of `isdiagonal` (l. 89-95).
"""
from __future__ import annotations

import numpy as np

from qctn.circuits import Circuit
from qctn.tn import SVD_THRESHOLD, TensorNetwork

EPS = float(np.finfo(np.float64).eps)
# QASM gates the release's reader registers under one fixed label with QXZoo `create_gate_1q` / `create_gate_2q`
# (`funcions_qasm.jl` l. 671, 699, 727, 796: "nova_porta_1q_u2", "_u1", "_p", "nova_porta_2q_cu1"). QXZoo's
# `cache_gate!` (`gates/gate_map.jl` l. 205-210) keeps the first matrix stored under a label, so every such gate
# gets the matrix of the first one created in the Julia session. We take the first in the circuit.
CACHED_GATES = ("u2", "u1", "p", "cu1")


def split_gate(matrix: np.ndarray, threshold: float = SVD_THRESHOLD):
    """A [out0, in0, v] and B [v, out1, in1] for a two-qubit gate given big-endian on (q0, q1)."""
    t = np.asarray(matrix, dtype=np.complex128).reshape(2, 2, 2, 2)  # out0, out1, in0, in1
    m = t.transpose(0, 2, 1, 3).reshape(4, 4)  # (out0, in0) x (out1, in1)
    u, s, vh = np.linalg.svd(m)
    chi = int(np.sum(s > threshold))
    r = np.sqrt(s[:chi])
    a = (u[:, :chi] * r).reshape(2, 2, chi)
    b = (r[:, None] * vh[:chi]).reshape(chi, 2, 2)
    return a, b, chi


def qxtns_matrix(matrix: np.ndarray) -> np.ndarray:
    """The two-qubit operator QXTns builds from a gate matrix: `reshape(data, (2, 2, 2, 2))` in Julia's column-
    major order makes gate_qubits[1] the least significant bit (`tensor_network_circuit.jl` l. 93), where our
    matrices (and QXZoo's controlled gates) have it as the most significant one. So the matrix acts with the two
    qubits' roles exchanged: for `cx` the control becomes the second qubit."""
    return np.asarray(matrix).reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)


def leaf_arrays(circuit: Circuit, tn: TensorNetwork, bits: str | None = None,
                out_bits: str | None = None, convention: str = "qxtns") -> dict[str, np.ndarray]:
    """complex128 data for every tensor of `tn` (built by `qctn.tn.build` from `circuit`), in the index order
    of `tn.tensors`. `bits` / `out_bits`: input and output basis states, '0' and '1' per qubit (default all 0).

    `convention="qxtns"` (default) gives the values the authors' networks hold: two-qubit matrices as QXTns
    reshapes them (`qxtns_matrix`), and for Schmidt rank 1 the second half a copy of the first, as QXTns `push!`
    does (l. 97). `convention="circuit"` gives the circuit as written (qubits[0] the most significant bit, both
    halves correct), whose amplitudes `statevector` reproduces."""
    if convention not in ("qxtns", "circuit"):
        raise ValueError(f"convention must be 'qxtns' or 'circuit', not {convention!r}")
    n = circuit.n_qubits
    vec = {"0": np.array([1.0, 0.0], complex), "1": np.array([0.0, 1.0], complex)}
    data: list[np.ndarray] = []
    first: dict[str, np.ndarray] = {}
    for g in circuit.gates:
        m = np.asarray(g.matrix, dtype=np.complex128)
        if convention == "qxtns" and g.name in CACHED_GATES:
            m = first.setdefault(g.name, m)
        if len(g.qubits) == 1:
            data.append(m)  # [out, in]
        else:
            a, b, chi = split_gate(qxtns_matrix(m) if convention == "qxtns" else m)
            if chi == 1:
                data += [a[:, :, 0], a[:, :, 0].copy() if convention == "qxtns" else b[0]]
            else:
                data += [a, b]
    if tn.inputs:
        data += [vec[c] for c in (bits or "0" * n)]
    if tn.outputs:
        data += [vec[c] for c in (out_bits or "0" * n)]
    if len(data) != len(tn.tensors):
        raise ValueError(f"{len(data)} arrays for {len(tn.tensors)} tensors: network not built from this circuit")
    out = {}
    for (tid, ixs), d in zip(tn.tensors.items(), data):
        if d.shape != tuple(tn.dims[i] for i in ixs):
            raise ValueError(f"{tid}: shape {d.shape} vs indices {ixs}")
        out[tid] = d
    return out


def _isdiagonal(m: np.ndarray) -> bool:
    off = ~np.eye(m.shape[0], m.shape[1], dtype=bool)
    return bool(np.all(np.abs(m[off]) < EPS))


def diagonal_groups(a: np.ndarray) -> list[list[int]]:
    """QXTns `find_hyper_edges`: groups of 0-based axis positions along which `a` is diagonal."""
    n = a.ndim
    if n == 2:
        return [[0, 1]] if a.shape[0] == a.shape[1] and _isdiagonal(a) else []
    groups: list[list[int]] = []
    if n < 2:
        return groups
    for i in range(n - 1):
        for j in range(i + 1, n):
            if a.shape[i] != a.shape[j]:
                continue
            rest = [k for k in range(n) if k not in (i, j)]
            t = a.transpose([i, j] + rest).reshape(a.shape[i], a.shape[j], -1)
            if all(_isdiagonal(t[:, :, x]) for x in range(t.shape[2])):
                hit = [g for g in groups if i in g or j in g]
                if hit:
                    groups = [g for g in groups if not (i in g or j in g)]
                    merged = sorted(set().union(*map(set, hit)) | {i, j})
                    groups.append(merged)
                else:
                    groups.append([i, j])
    return groups


def statevector(circuit: Circuit) -> np.ndarray:
    """Dense state C|0...0>, qubit 0 the most significant bit. For tests on small circuits."""
    n = circuit.n_qubits
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1
    for g in circuit.gates:
        k = len(g.qubits)
        u = np.asarray(g.matrix, complex).reshape((2,) * (2 * k))
        psi = np.tensordot(u, psi, axes=(list(range(k, 2 * k)), list(g.qubits)))
        psi = np.moveaxis(psi, list(range(k)), list(g.qubits))
    return psi.reshape(-1)
