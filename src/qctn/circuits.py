"""Circuits from the authors' QASM files.

Two readers:

- `load_qiskit`: qiskit's OpenQASM 2 parser, the full circuit.
- `authors_gates`: the gate list the authors' Julia code builds, following `import_from_qasm` and
  `process_gate_line!` in `src/funcions_qasm.jl` (release 936630a). That parser reads lines with regular
  expressions, only on a register named `q`, and skips lines it doesn't recognise (it prints
  "Unprocessed element" and goes on). Gates it skips include `u`, `u3`, `ccx`, `cry` and any gate on
  another register. Two-qubit gates are stored as QXZoo `GateCall2(target, ctrl)`.

Gates are `Gate(name, qubits, matrix)`, with 0-based qubits in the order QXTools' `gate_qubits` returns
them, `[gate.target, gate.ctrl]`, and the matrix in that qubit order, big-endian (row index
`2 * bit(qubits[0]) + bit(qubits[1])`). QXZoo's `GateCall2(symbol, ctrl, target)` stores its first qubit
argument as `ctrl`, while `c_x(q_target, q_ctrl)` passes the target first, so the authors' `c_x(target, ctrl)`
ends up with `gate_qubits == [ctrl, target]`: QASM order. `swap(q1, q2)` and `c_u(g, q1, q2)` (their
`cu1`) end up as `[q2, q1]`, `c_r_phase(target, ctrl)` (their `cp`) as QASM order. Only the order and the
operator Schmidt rank matter for the network's structure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qctn.paths import authors_circuits


@dataclass(frozen=True, eq=False)
class Gate:
    name: str
    qubits: tuple[int, ...]
    matrix: np.ndarray


@dataclass
class Circuit:
    name: str
    n_qubits: int
    gates: list[Gate]
    skipped: list[str]  # lines the authors' parser does not turn into gates (gate lines only)


_S2 = 1 / math.sqrt(2)
_ONE = {
    "x": np.array([[0, 1], [1, 0]], complex),
    "y": np.array([[0, -1j], [1j, 0]], complex),
    "z": np.diag([1, -1]).astype(complex),
    "h": _S2 * np.array([[1, 1], [1, -1]], complex),
    "s": np.diag([1, 1j]),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]),
}


def rx(a):
    c, s = math.cos(a / 2), math.sin(a / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(a):
    c, s = math.cos(a / 2), math.sin(a / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def u2(phi, lam):
    return _S2 * np.array([[1, -np.exp(1j * lam)], [np.exp(1j * phi), np.exp(1j * (phi + lam))]])


def phase(a):
    return np.diag([1, np.exp(1j * a)])


def controlled(u: np.ndarray) -> np.ndarray:
    """Controlled-u on (ctrl, target), big-endian."""
    m = np.eye(4, dtype=complex)
    m[2:, 2:] = u
    return m


SWAP = np.eye(4)[[0, 2, 1, 3]].astype(complex)


def _angle(expr: str) -> float:
    """The authors evaluate angles with Julia's `eval(Meta.parse(...))`; QASM angles are arithmetic in pi."""
    expr = expr.strip()
    if not re.fullmatch(r"[0-9eE.+\-*/() pi]*", expr):
        raise ValueError(f"unsupported angle expression {expr!r}")
    return float(eval(expr, {"__builtins__": {}}, {"pi": math.pi}))


_Q = r"q\[(\d+)\]"
# Order and patterns as in process_gate_line!. Each entry: (test, pattern, builder).
_RULES = [
    (r"^x\s+q\[", rf"x\s+{_Q};", lambda m: Gate("x", (int(m[1]),), _ONE["x"])),
    (r"^y\s+q\[", rf"y\s+{_Q};", lambda m: Gate("y", (int(m[1]),), _ONE["y"])),
    (r"^z\s+q\[", rf"z\s+{_Q};", lambda m: Gate("z", (int(m[1]),), _ONE["z"])),
    (r"^h\s+q\[", rf"h\s+{_Q};", lambda m: Gate("h", (int(m[1]),), _ONE["h"])),
    (r"^s\s+q\[", rf"s\s+{_Q};", lambda m: Gate("s", (int(m[1]),), _ONE["s"])),
    (r"^t\s+q\[", rf"t\s+{_Q};", lambda m: Gate("t", (int(m[1]),), _ONE["t"])),
    (r"^rx\(", rf"rx\((.*)\)\s+{_Q};", lambda m: Gate("rx", (int(m[2]),), rx(_angle(m[1])))),
    (r"^u2\(", rf"u2\(([^,]+),\s*([^)]+)\)\s+{_Q}",
     lambda m: Gate("u2", (int(m[3]),), u2(_angle(m[1]), _angle(m[2])))),
    (r"^u1\(", rf"u1\(([^)]+)\)\s+{_Q};", lambda m: Gate("u1", (int(m[2]),), phase(_angle(m[1])))),
    (r"^p\(", rf"p\(([^)]+)\)\s+{_Q};", lambda m: Gate("p", (int(m[2]),), phase(_angle(m[1])))),
    (r"^sx\s+q\[", rf"sx\s+{_Q};", lambda m: Gate("sx", (int(m[1]),), rx(math.pi / 2))),
    (r"^ry\(", rf"ry\((.*)\)\s+{_Q};", lambda m: Gate("ry", (int(m[2]),), ry(_angle(m[1])))),
    (r"^rz\(", rf"rz\((.*)\)\s+{_Q};", lambda m: Gate("rz", (int(m[2]),), rz(_angle(m[1])))),
    (r"^cx\s+q\[", rf"cx\s+{_Q},\s*{_Q};",
     lambda m: Gate("cx", (int(m[1]), int(m[2])), controlled(_ONE["x"]))),
    (r"^cy\s+q\[", rf"cy\s+{_Q},\s*{_Q};",
     lambda m: Gate("cy", (int(m[1]), int(m[2])), controlled(_ONE["y"]))),
    (r"^cz\s+q\[", rf"cz\s+{_Q},\s*{_Q};",
     lambda m: Gate("cz", (int(m[1]), int(m[2])), controlled(_ONE["z"]))),
    # The matrix is diagonal, so the same in either qubit order.
    (r"^cu1\(", rf"cu1\(([^)]+)\)\s+{_Q},{_Q};",
     lambda m: Gate("cu1", (int(m[3]), int(m[2])), np.diag([1, 1, 1, np.exp(1j * _angle(m[1]))]))),
    (r"^swap\s+q\[", rf"swap\s+{_Q},\s*{_Q};", lambda m: Gate("swap", (int(m[2]), int(m[1])), SWAP)),
    (r"^cp\(", rf"cp\((.*)\)\s+{_Q},\s*{_Q};",
     lambda m: Gate("cp", (int(m[2]), int(m[3])), controlled(phase(_angle(m[1]))))),
]
_RULES = [(re.match(r"\^(\w+)", a)[1], re.compile(a), re.compile(b), f) for a, b, f in _RULES]
# The release says u2/u1 and p were added later ("New search patterns", funcions_qasm.jl l. 645, 674, 702); some
# plans were made with an earlier reader (see builders.EARLY_SKIPPED).
_NOT_GATES = re.compile(r"^(include|qreg|creg|measure|barrier|reset|gate|opaque|if)\b")


def authors_gates(path: Path, without=frozenset()) -> Circuit:
    """The circuit as `import_from_qasm` in the authors' funcions_qasm.jl builds it."""
    path = Path(path)
    return authors_gates_text(path.read_text(), path.stem, without)


def authors_gates_text(text: str, name: str, without=frozenset()) -> Circuit:
    """`authors_gates` on QASM text (e.g. after the release's register rewriting). Gate names in `without`
    (QASM names such as "u2") are treated as lines the parser doesn't recognise."""
    lines = [l for l in text.splitlines()
             if not (l.strip().startswith("//") or l.strip().startswith("#") or not l.strip())]
    n = None
    for l in lines:
        if re.match(r"^qreg q\[", l):
            n = int(re.search(r"qreg q\[(\d+)\];", l)[1])
            break
    if n is None:
        raise ValueError(f"{name}: no 'qreg q[...]' register")
    gates, skipped = [], []
    for l in lines:
        if not re.match(r"^[a-z]", l):
            continue
        s = l.strip()
        for gname, test, pat, build in _RULES:
            if test.search(s):
                if gname in without:
                    skipped.append(s)
                    break
                m = pat.search(s)
                if m is None:
                    skipped.append(s)  # the Julia code warns and skips
                else:
                    gates.append(build(m))
                break
        else:
            if not _NOT_GATES.match(s):
                skipped.append(s)
    return Circuit(name, n, gates, skipped)


def load_qiskit(path: Path):
    """qiskit QuantumCircuit for a QASM 2 file."""
    from qiskit import QuantumCircuit

    return QuantumCircuit.from_qasm_file(str(path))


def qiskit_gate_counts(qc) -> dict[str, int]:
    """Gate counts without measurements and barriers."""
    return {k: v for k, v in qc.count_ops().items() if k not in ("measure", "barrier")}


def circuit_path(name: str) -> Path:
    return authors_circuits() / f"{name}.qasm"
