"""solvers.py — run the PACE 2017 treewidth heuristics (Tamaki et al., Jdrasil) with a time budget.

The solvers are cloned into tools/ and compiled with javac (see README). Each run writes the graph in
PACE .gr format, starts the solver, sends SIGTERM after `budget` seconds (the PACE protocol: print the
best decomposition and exit), parses the .td and validates it here before trusting its width.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import networkx as nx

TOOLS = Path(__file__).resolve().parent.parent / "tools"

SOLVERS = {
    # PACE 2017 heuristic track, 1st place (Makii, Ohtsuka, Sato, Tamaki): greedy start, then local improvement
    # around the largest bag (exact tw / exact pw / greedy on that subgraph, round robin).
    "tamaki": lambda seed: ["java", "-Xmx2g", "-Xss1g", "-cp", str(TOOLS / "PACE2017-TrackA"),
                            "tw.heuristic.MainDecomposer", "-s", str(seed)],
    # Jdrasil heuristic (Bannach, Berndt, Ehlers): greedy runs, reduction, stochastic greedy permutations, local search.
    "jdrasil": lambda seed: ["java", "-Xmx2g", "-cp", str(TOOLS / "Jdrasil" / "build" / "classes"),
                             "jdrasil.Heuristic", "-s", str(seed)],
}


def to_gr(G: nx.Graph):
    """PACE .gr text and the vertex numbering (1-based)."""
    idx = {v: i + 1 for i, v in enumerate(sorted(G, key=str))}
    lines = [f"p tw {G.number_of_nodes()} {G.number_of_edges()}"]
    lines += [f"{a} {b}" for a, b in sorted(tuple(sorted((idx[a], idx[b]))) for a, b in G.edges())]
    return "\n".join(lines) + "\n", idx


def parse_td(text: str):
    bags, tree, n_bags = {}, [], 0
    for line in text.splitlines():
        p = line.split()
        if not p or p[0] == "c":
            continue
        if p[0] == "s":
            n_bags = int(p[2])
        elif p[0] == "b":
            bags[int(p[1])] = [int(x) for x in p[2:]]
        else:
            tree.append((int(p[0]), int(p[1])))
    if not n_bags or len(bags) != n_bags:
        return None
    return bags, tree


def valid_td(G: nx.Graph, idx: dict, bags: dict, tree: list) -> bool:
    """The three tree-decomposition axioms, checked independently of the solver."""
    T = nx.Graph()
    T.add_nodes_from(bags)
    T.add_edges_from(tree)
    if bags and not nx.is_tree(T):
        return False
    covered = set().union(*map(set, bags.values())) if bags else set()
    if covered != set(idx.values()):
        return False
    bag_sets = {b: set(vs) for b, vs in bags.items()}
    for a, b in G.edges():
        if not any(idx[a] in s and idx[b] in s for s in bag_sets.values()):
            return False
    for v in idx.values():
        holding = [b for b, s in bag_sets.items() if v in s]
        if not nx.is_connected(T.subgraph(holding)):
            return False
    return True


def run(solver: str, G: nx.Graph, budget: float, seed: int = 1) -> dict:
    """Width found by `solver` within `budget` seconds (SIGTERM at the limit)."""
    gr, idx = to_gr(G)
    with tempfile.TemporaryFile("w+") as fin:
        fin.write(gr)
        fin.seek(0)
        t0 = time.time()
        p = subprocess.Popen(SOLVERS[solver](seed), stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                out, _ = p.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                out, _ = p.communicate()
        elapsed = time.time() - t0
    td = parse_td(out)
    if td is None:
        return {"solver": solver, "budget": budget, "seed": seed, "width": None, "valid": False, "elapsed": elapsed}
    bags, tree = td
    name = {i: v for v, i in idx.items()}
    return {"solver": solver, "budget": budget, "seed": seed, "width": max(len(b) for b in bags.values()) - 1,
            "valid": valid_td(G, idx, bags, tree), "bags": len(bags), "elapsed": round(elapsed, 2),
            "bag_list": [[name[i] for i in b] for b in bags.values()]}
