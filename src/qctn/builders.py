"""The circuits behind the authors' plans, built the way their Julia code builds them.

Generators follow QXTools 1.0 (`src/circuits/circuits.jl`) and QXZoo (`src/algorithms/qft.jl`, `rqc.jl`,
`src/gates/default_gates.jl`, `composite_gates.jl`) as pinned in the authors' Manifest; QASM circuits go
through `qctn.circuits.authors_gates`, after the register rewriting of `replace_qasm_registers*` in the
release's `src/funcions_qasm.jl` where that applies.

QXZoo conventions reproduced here (1-based qubits in QXZoo, 0-based in `Gate.qubits`):

- `GateCall2(symbol, a, b)` stores `a` as `ctrl` and `b` as `target`; QXTools' `gate_qubits` returns
  `[target, ctrl]`. Every two-qubit constructor used below (`c_x`, `c_z`, `c_r_phase`, `swap`) passes its two
  qubit arguments straight through, so `f(a, b)` gives `gate_qubits == [b, a]`.
- The network only depends on the gate order, each gate's qubits in `gate_qubits` order and the operator
  Schmidt rank of two-qubit gates (CX, CZ, CPhase: 2; SWAP: 4). Single-qubit matrices don't change the
  structure (see `rqc` for the random ones).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from qctn.circuits import _ONE, SWAP, Circuit, Gate, authors_gates_text, controlled, phase, rx, ry

_SX = rx(math.pi / 2)  # sqrt(X), sqrt(Y) up to a global phase
_SY = ry(math.pi / 2)


class _QX:
    """A QXZoo `Circ` as a gate list; methods take 1-based qubits like QXZoo."""

    def __init__(self, n: int):
        self.n = n
        self.gates: list[Gate] = []

    def g1(self, name, q, m):
        assert 1 <= q <= self.n
        self.gates.append(Gate(name, (q - 1,), m))

    def g2(self, name, a, b, m):
        """`GateCall2(symbol, a, b)`: gate_qubits [b, a]."""
        assert 1 <= a <= self.n and 1 <= b <= self.n and a != b
        self.gates.append(Gate(name, (b - 1, a - 1), m))

    def h(self, q):
        self.g1("h", q, _ONE["h"])

    def c_x(self, t, c):
        self.g2("c_x", t, c, controlled(_ONE["x"]))

    def c_z(self, t, c):
        self.g2("c_z", t, c, controlled(_ONE["z"]))

    def c_r_phase(self, t, c, theta):
        self.g2("c_r_phase", t, c, controlled(phase(theta)))

    def swap(self, a, b):
        self.g2("swap", a, b, SWAP)

    def circuit(self, name):
        return Circuit(name, self.n, self.gates, [])


# --------------------------------------------------------------------------------------------- GHZ
def ghz(n: int, name: str | None = None) -> Circuit:
    """QXTools `create_ghz_circuit(n)`: h(1), then c_x(i, i+1) for i in 1:n-1."""
    c = _QX(n)
    c.h(1)
    for i in range(1, n):
        c.c_x(i, i + 1)
    return c.circuit(name or f"GHZ_{n}")


# --------------------------------------------------------------------------------------------- QFT
def _swap_idx(idx):
    mp = len(idx) // 2
    lhs = idx[:mp]
    rhs = idx[::-1][:mp]
    return lhs, rhs


def apply_qft(c: _QX, idx):
    """QXZoo `apply_qft!`."""
    for i in range(len(idx)):
        c.h(idx[i])
        for j in range(i + 1, len(idx)):
            c.c_r_phase(idx[i], idx[j], 2 * math.pi / 2 ** (idx[j] - idx[i] + 1))
    for a, b in zip(*_swap_idx(idx)):
        c.swap(a, b)


def apply_iqft(c: _QX, idx):
    """QXZoo `apply_iqft!`."""
    for a, b in zip(*_swap_idx(idx)):
        c.swap(a, b)
    for i in range(len(idx) - 1, -1, -1):
        for j in range(len(idx) - 1, i, -1):
            c.c_r_phase(idx[i], idx[j], -2 * math.pi / 2 ** (idx[j] - idx[i] + 1))
        c.h(idx[i])


def qft(n: int, name: str | None = None, offset: int = 1) -> Circuit:
    """QXTools `create_qft_circuit(n)`: QXZoo's QFT, then c_x(0, 1) and c_x(1, 2).

    QXTools numbers the qubits 0..n-1 there, which `convert_to_tnc` can't index (Julia arrays are 1-based);
    `offset=1` builds the same circuit on qubits 1..n."""
    c = _QX(n)
    idx = list(range(offset, offset + n))
    apply_qft(c, idx)
    c.c_x(offset, offset + 1)
    c.c_x(offset + 1, offset + 2)
    return c.circuit(name or f"QFT_{n}")


def qft_iqft(n: int, name: str | None = None) -> Circuit:
    """QXZoo `apply_qft!` on qubits 1..ceil(n/2), then `apply_iqft!` on qubits 1..n.

    Inferred from the plans; the release has no generator for this family. The leaves' wire indices in
    the `QFT_IQFT_5` plans give this gate sequence, and every `QFT_IQFT_*` plan replays on it."""
    c = _QX(n)
    apply_qft(c, list(range(1, (n + 1) // 2 + 1)))
    apply_iqft(c, list(range(1, n + 1)))
    return c.circuit(name or f"QFT_IQFT_{n}")


# --------------------------------------------------------------------------------------------- RQC
_PATTERN = [3, 1, 6, 8, 5, 7, 2, 4]


def _rqc_patterns(rows, cols):
    """QXZoo RQC `patterns`: dict 1..8 of lists of qubit pairs [[i, j], [u, v]] (1-based grid coordinates)."""
    gp = {}
    key = 1
    N = max(rows, cols)
    for v in (1, -1):
        for shift in (0, 1):
            for flip in (False, True):
                ps = [[[i, j], [i, j + v]] for j in range(1, N + 1, 2)
                      for i in range(1 + ((j + shift) // 2) % 2, N + 1, 2)]
                if flip:
                    ps = [[p[0][::-1], p[1][::-1]] for p in ps]
                ps = [p for p in ps if 0 < p[0][0] <= rows and 0 < p[1][0] <= rows
                      and 0 < p[0][1] <= cols and 0 < p[1][1] <= cols]
                gp[key] = ps
                key += 1
    return gp


def rqc(rows: int, cols: int, depth: int, final_h: bool = False, seed: int | None = None,
        name: str | None = None) -> Circuit:
    """QXZoo `create_RQC(rows, cols, depth, seed; final_Hadamard_layer=final_h)`.

    The seed only picks which of T, sqrt(X), sqrt(Y) a single-qubit gate is: `random_gate!` (rqc.jl l. 71-80)
    sets the qubit's next gate to `(g + rand(rng, [0, 1])) % 3 + 1`, i.e. one of the two gates other than
    the current one; the first is always T. Whether a gate is placed, and where, doesn't depend on the draw,
    so the network's structure is the same for every seed. Here the draws come from numpy's generator
    (`seed`), not Julia's `MersenneTwister`, so the T/sqrt(X)/sqrt(Y) sequence differs from any Julia run.
    QXZoo indexes the CZ qubits as `lin_idx(col, row, rows)` and the H and single-qubit gates as
    `lin_idx(row, col, cols)` (l. 179 vs 163, 188); that is kept as is."""
    n = rows * cols
    c = _QX(n)
    rng = np.random.default_rng(seed)
    one = {1: ("t", _ONE["t"]), 2: ("sx", _SX), 3: ("sy", _SY)}

    def lin(i, j, maxj):
        return (i - 1) * maxj + j

    next_gate = -np.ones((rows + 1, cols + 1), int)  # 1-based; QXZoo starts at -ones
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            c.h(lin(i, j, cols))
    gp = _rqc_patterns(rows, cols)
    for k in range(depth):
        p = _PATTERN[k % 8]
        if not gp[p]:
            continue
        for t in gp[p]:
            c.c_z(lin(t[1][1], t[1][0], rows), lin(t[0][1], t[0][0], rows))
        hit = {tuple(q) for pair in gp[p] for q in pair}
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                g = next_gate[i, j]
                if (i, j) not in hit and g > 0:
                    next_gate[i, j] = -((g + int(rng.integers(2))) % 3 + 1)
                    c.g1(one[g][0], lin(i, j, cols), one[g][1])
        for (a, b), (u, v) in gp[p]:
            next_gate[a, b] = abs(next_gate[a, b])
            next_gate[u, v] = abs(next_gate[u, v])
    if final_h:
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                c.h(lin(i, j, cols))
    return c.circuit(name or f"RQC_{rows}_{cols}_{depth}")


# --------------------------------------------------------------------------------------------- QASM
def replace_registers(text: str) -> str:
    """`replace_qasm_registers` (QWalk: `node`, `coin` -> `q`)."""
    node = re.search(r"qreg\s+node\[(\d+)\];", text)
    coin = re.search(r"qreg\s+coin\[(\d+)\];", text)
    nn = int(node[1]) if node else 0
    nc = int(coin[1]) if coin else 0
    for i in range(nn):
        text = text.replace(f"node[{i}]", f"q[{i}]")
    for i in range(nc):
        text = text.replace(f"coin[{i}]", f"q[{i + nn}]")
    if nn or nc:
        text = re.sub(r"qreg\s+node\[\d+\];\s*qreg\s+coin\[\d+\];", f"qreg q[{nn + nc}];", text)
    return text


def replace_registers_grover(text: str) -> str:
    """`replace_qasm_registers_grover`: `flag[i]` -> `q[i + len(q)]` and the two qreg lines merged.

    The Julia function scans line by line with `elseif`, so each register is found by its own regex; we do
    the same. Its `node`/`coin` substitutions don't occur in the Grover files."""
    nq = re.search(r"qreg\sq\[(\d+)\];", text)
    fl = re.search(r"qreg\s+flag\[(\d+)\];", text)
    node_q = int(nq[1]) if nq else 0
    nf = int(fl[1]) if fl else 0
    for i in range(nf):
        text = text.replace(f"flag[{i}]", f"q[{i + node_q}]")
    if node_q or nf:
        text = re.sub(r"qreg\s+q\[\d+\];\s*qreg\s+flag\[\d+\];", f"qreg q[{node_q + nf}];", text)
    return text


# An earlier version of the authors' QASM reader. Some plans (all `Qwalk_*`, part of `QNN_*`) replay only on
# a network built without u1, u2, p and sx gates and with every two-qubit gate's qubits in the other order
# (`gate_qubits == [target, ctrl]` in QASM terms). The release's reader marks u2, u1 and p as
# "New search patterns" (funcions_qasm.jl l. 645, 674, 702) and says the current version "fixes the bug present
# in the previous version" about control/target order (l. 233-234, 587-588); the earlier reader itself is not
# in the release, so this is inferred from the plans.
EARLY_SKIPPED = frozenset({"u1", "u2", "p", "sx"})


def reverse_two_qubit(c: Circuit) -> Circuit:
    gates = [Gate(g.name, g.qubits[::-1], g.matrix) if len(g.qubits) == 2 else g for g in c.gates]
    return Circuit(c.name, c.n_qubits, gates, c.skipped)


def from_qasm(path: Path, rewrite=None, name: str | None = None, early: bool = False) -> Circuit:
    """A QASM file through the authors' reader, after `rewrite` (one of the `replace_registers*`).
    `early=True`: the earlier reader, see EARLY_SKIPPED."""
    path = Path(path)
    text = path.read_text()
    if rewrite is not None:
        text = rewrite(text)
    c = authors_gates_text(text, name or path.stem, EARLY_SKIPPED if early else frozenset())
    return reverse_two_qubit(c) if early else c
