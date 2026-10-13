"""circuits.py — the talk's one real circuit network: a seeded random 2D brickwork circuit.

Qubits sit on a rows x cols grid. Couplers (grid edges) fall into four patterns A, B, C, D
(horizontal/vertical, even/odd offset) as in Sycamore-style supremacy circuits. Each cycle applies
one pattern chosen at random (seeded, never the same pattern twice in a row), with a two-qubit
gate on every coupler of that pattern. Single-qubit gates are absorbed into neighbouring
two-qubit gates, and the input |0…0> and output <x| into the first and last gate on each qubit,
so the amplitude <x|C|0> is a closed network of rank-≤4 gate tensors with bond dimension 2.
"""
from __future__ import annotations

import random

import networkx as nx

from .core import tensor_network


def coupler_patterns(rows: int, cols: int) -> dict:
    P = {"A": [], "B": [], "C": [], "D": []}
    for r in range(rows):
        for c in range(cols - 1):
            P["A" if (c + r) % 2 == 0 else "B"].append(((r, c), (r, c + 1)))
    for r in range(rows - 1):
        for c in range(cols):
            P["C" if (r + c) % 2 == 0 else "D"].append(((r, c), (r + 1, c)))
    return P


def brickwork_circuit(rows: int = 3, cols: int = 4, cycles: int = 8, seed: int = 0) -> dict:
    rng = random.Random(seed)
    P = coupler_patterns(rows, cols)
    seq, prev = [], None
    for _ in range(cycles):
        p = rng.choice([k for k in "ABCD" if k != prev])
        seq.append(p)
        prev = p
    gates = []                     # (name, cycle, qubit_a, qubit_b)
    for t, p in enumerate(seq):
        for a, b in P[p]:
            gates.append((f"g{len(gates)}", t, a, b))
    return {"rows": rows, "cols": cols, "cycles": cycles, "seed": seed, "patterns": seq, "gates": gates}


def circuit_tn(circ: dict) -> dict:
    """{gate: set(bond labels)}; a bond is the wire segment of one qubit between two gates."""
    last = {}
    bonds = []
    for name, _, a, b in circ["gates"]:
        for q in (a, b):
            if q in last:
                bonds.append((last[q], name, f"w{q[0]}_{q[1]}_{name}"))
            last[q] = name
    return tensor_network([g[0] for g in circ["gates"]], bonds)


def tensor_graph(T: dict) -> nx.MultiGraph:
    """Gates as vertices, one edge per bond (parallel bonds kept)."""
    owner = {}
    G = nx.MultiGraph()
    G.add_nodes_from(T)
    for v, legs in T.items():
        for l in legs:
            owner.setdefault(l, []).append(v)
    for l, (a, b) in ((l, o) for l, o in owner.items() if len(o) == 2):
        G.add_edge(a, b, key=l)
    return G


def simple_tensor_graph(T: dict) -> nx.Graph:
    """Parallel bonds merged into one edge with weight = number of bonds (log2 of the bond dimension)."""
    G = nx.Graph()
    G.add_nodes_from(T)
    for a, b, _ in tensor_graph(T).edges(keys=True):
        w = G[a][b]["weight"] + 1 if G.has_edge(a, b) else 1
        G.add_edge(a, b, weight=w)
    return G


def circuit_positions(circ: dict, dx: float = 70, dy: float = 34, margin: float = 30) -> dict:
    """Gate at x = cycle, y = mean row-major index of its two qubits (a space-time layout)."""
    cols = circ["cols"]
    pos = {}
    for name, t, a, b in circ["gates"]:
        qa, qb = a[0] * cols + a[1], b[0] * cols + b[1]
        pos[name] = (margin + t * dx, margin + (qa + qb) / 2 * dy)
    return pos


def default_circuit():
    """The circuit used throughout the talk."""
    return brickwork_circuit(3, 4, 8, seed=0)
