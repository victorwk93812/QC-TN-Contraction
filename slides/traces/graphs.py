"""graphs.py — the recurring test graphs of the talk, with fixed positions.

Ring of four tensors, bowtie, grids, the 11-vertex separator graph and random cubic graphs.
Positions are SVG user units and are part of each figure's trace, never recomputed by renderers.
"""
from __future__ import annotations

import itertools

import networkx as nx

from .core import grid_graph, grid_positions, tensor_network

# ---------------------------------------------------------------- ring of four tensors

RING_BONDS = [("A", "B", "i"), ("B", "C", "j"), ("C", "D", "k"), ("D", "A", "l")]


def ring_tn() -> dict:
    return tensor_network("ABCD", RING_BONDS)


# Square ring: tensors at corners, each bond index at the midpoint of its edge.
RING_TENSOR_POS = {"A": (60, 60), "B": (220, 60), "C": (220, 220), "D": (60, 220)}
RING_INDEX_POS = {"i": (140, 60), "j": (220, 140), "k": (140, 220), "l": (60, 140)}

# ---------------------------------------------------------------- bowtie

BOWTIE_BONDS = [("U", "P", "u"), ("V", "Q", "v"), ("P", "Q", "x"), ("P", "H", "p"),
                ("Q", "H", "q"), ("H", "R", "r"), ("H", "S", "s"), ("R", "S", "t")]


def bowtie_tn() -> dict:
    """Degree-4 hub H on two triangles (P,Q,H) and (H,R,S); pendants U, V on P, Q."""
    return tensor_network("UVPQHRS", BOWTIE_BONDS)


BOWTIE_TENSOR_POS = {"U": (40, 50), "V": (40, 230), "P": (130, 70), "Q": (130, 210),
                     "H": (230, 140), "R": (330, 70), "S": (330, 210)}

# ---------------------------------------------------------------- separator playground graph

def separator_graph() -> nx.Graph:
    """Two K4s {a,b,c,d}, {e,f,g,h} joined through x (c-x-e) and y-z (d-y-z-{f,g}).
    11 vertices, 18 edges; min a-h separator has size 2."""
    G = nx.Graph()
    G.add_edges_from(itertools.combinations("abcd", 2))
    G.add_edges_from(itertools.combinations("efgh", 2))
    G.add_edges_from([("c", "x"), ("x", "e"), ("d", "y"), ("y", "z"), ("z", "f"), ("z", "g")])
    return G


SEPARATOR_POS = {"a": (60, 90), "b": (60, 210), "c": (170, 90), "d": (170, 210), "x": (300, 50),
                 "y": (255, 245), "z": (345, 255), "e": (440, 80), "f": (440, 200), "g": (555, 200),
                 "h": (555, 80)}                        # from prototypes/F16
MENGER_PATHS = [["a", "c", "x", "e", "h"], ["a", "d", "y", "z", "f", "h"]]

# ---------------------------------------------------------------- grids and cubic graphs

def grid(L: int, W: int | None = None) -> nx.Graph:
    return grid_graph(L, W)


def grid_int(L: int) -> nx.Graph:
    """Grid relabelled to integers in row-major order (the MinFill seed protocol uses this)."""
    return nx.convert_node_labels_to_integers(grid_graph(L))


def cubic(n: int = 320, seed: int = 1) -> nx.Graph:
    return nx.random_regular_graph(3, n, seed=seed)


def triangulated_ladder(k: int = 6) -> nx.Graph:
    """2 x k ladder with one diagonal per square: a 2-tree, hence chordal."""
    G = nx.Graph()
    for i in range(k):
        G.add_edge(f"u{i}", f"l{i}")
        if i + 1 < k:
            G.add_edges_from([(f"u{i}", f"u{i+1}"), (f"l{i}", f"l{i+1}"), (f"u{i}", f"l{i+1}")])
    return G


def ladder_positions(k: int = 6, dx: float = 70, dy: float = 90, m: float = 30) -> dict:
    return {**{f"u{i}": (m + i * dx, m) for i in range(k)}, **{f"l{i}": (m + i * dx, m + dy) for i in range(k)}}


def int_grid_positions(L: int, spacing: float = 60, margin: float = 40) -> dict:
    """Positions for grid_int(L): node v sits at row v // L, column v % L."""
    return {v: (margin + (v % L) * spacing, margin + (v // L) * spacing) for v in range(L * L)}


__all__ = ["ring_tn", "bowtie_tn", "separator_graph", "grid", "grid_int", "cubic", "grid_positions",
           "RING_TENSOR_POS", "RING_INDEX_POS", "BOWTIE_TENSOR_POS", "SEPARATOR_POS"]
