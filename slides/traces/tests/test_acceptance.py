"""One assertion per row of the acceptance table in PROPOSAL.md ("Numbers the trace package
must reproduce exactly"). χ = 2 throughout; costs are sums of 2^bag (or 2^(m+n+k))."""
import itertools
import statistics

import pytest

from traces import graphs as gr
from traces.carving import grid_tree_stats
from traces.core import (best_caterpillar, carving_dp, column_sweep, eliminate, hybrid_cut_and_sweep,
                         index_graph, nested_dissection, separator_report)
from traces.fast_elim import best_width_first_seed, eliminate_fast, greedy_runs

import networkx as nx


def test_ring_index_elimination():
    tr = eliminate(index_graph(gr.ring_tn()), order=["i", "j", "k", "l"])
    assert tr["width"] == 2
    assert sum(2 ** s["bag"] for s in tr["steps"]) == 22


def test_bowtie_best_elimination_exhaustive():
    IG = index_graph(gr.bowtie_tn())
    best = min((lambda tr: (tr["width"], sum(2 ** s["bag"] for s in tr["steps"])))(eliminate(IG, order=p))
               for p in itertools.permutations(sorted(IG)))
    assert best == (3, 62)


def test_bowtie_best_tree():
    r, c, _ = carving_dp(gr.bowtie_tn())
    assert (r, c) == (2, 50)


def test_bowtie_best_list():
    r, c, _ = best_caterpillar(gr.bowtie_tn())
    assert (r, c) == (3, 58)


# Seed protocol: tie-break seeds 0..9 on the integer-relabelled grid; "best" = the first seed
# (in seed order) that reaches the minimum width, and its log2 cost is the one quoted.
MINFILL = {8: (10, 13.20), 12: (15, 18.35), 16: (21, 23.68), 20: (26, 29.29), 24: (32, 34.59)}
SWEEP = {8: 14.70, 12: 19.95, 16: 24.83, 20: 29.51, 24: 34.06}


@pytest.mark.parametrize("L", sorted(MINFILL))
def test_minfill_grid_best_of_10(L):
    seed, w, c = best_width_first_seed(gr.grid_int(L), "minfill", range(10))
    assert (w, round(c, 2)) == MINFILL[L]


@pytest.mark.parametrize("L", sorted(SWEEP))
def test_column_sweep_grid(L):
    tr = eliminate_fast(gr.grid(L), order=column_sweep(L), record_steps=False)
    assert tr["width"] == L
    assert round(tr["log2cost"], 2) == SWEEP[L]


def test_minfill_tie_spread_L20_40_seeds():
    widths = [w for _, w, _ in greedy_runs(gr.grid_int(20), "minfill", range(40))]
    assert (min(widths), max(widths), statistics.median(widths)) == (26, 30, 28)


def test_balanced_nd_7x7_levels():
    log = []
    tr = eliminate(gr.grid(7), order=nested_dissection(range(7), range(7), log=log))
    assert tr["width"] == 9
    bag = {s["elim"]: s["bag"] for s in tr["steps"]}
    level = [max(bag[v] for s in log if s["depth"] == d for v in s["separator"]) for d in range(3)]
    assert level == [7, 10, 9]
    seps = {v for s in log for v in s["separator"]}
    assert max(b for v, b in bag.items() if v not in seps) <= 8
    assert max(s["depth"] for s in log) == 2


@pytest.mark.parametrize("L,want", [(7, 9), (15, 21), (23, 33), (31, 45)])
def test_balanced_nd_vs_hybrid(L, want):
    G = gr.grid(L)
    assert eliminate_fast(G, order=nested_dissection(range(L), range(L)), record_steps=False)["width"] == want
    assert eliminate_fast(G, order=hybrid_cut_and_sweep(L), record_steps=False)["width"] == L


def test_separator_playground_graph():
    SG = gr.separator_graph()
    assert (SG.number_of_nodes(), SG.number_of_edges()) == (11, 18)
    assert nx.node_connectivity(SG, "a", "h") == 2
    efg = separator_report(SG, "efg")
    assert efg["separator"] and efg["minimal"] and len("efg") > 2      # minimal, not minimum
    assert separator_report(SG, "xyz")["minimal"] is False


def test_random_cubic_320():
    G = gr.cubic(320, seed=1)
    mf = [w for _, w, _ in greedy_runs(G, "minfill", range(20))]
    md = [w for _, w, _ in greedy_runs(G, "mindeg", range(20))]
    assert min(mf) == 56 and statistics.median(mf) == 59.5
    assert statistics.median(md) == 62


def test_balanced_bisection_grid_tn_L32():
    st = grid_tree_stats(32)
    assert (st["max_rank"], st["max_exp"]) == (40, 48)
