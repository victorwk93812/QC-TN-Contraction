"""Phase 3: the seven plan generators, checked against QXTools conventions and the authors' plans."""
import numpy as np
import pytest

from qctn import paths
from qctn.circuits import Circuit, Gate, authors_gates, circuit_path, controlled
from qctn.features import plan_features
from qctn.plans import pace
from qctn.plans import qxtools as qx
from qctn.plans.authors import format_plan, parse_plan, read_plan
from qctn.plans.generators import generate, leaf_sets, to_authors_plan
from qctn.plans.girvan_newman import gn_plan, removal_order
from qctn.tn import build, replay

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)
SMALL = ["dj_indep_qiskit_40", "pricingcall_nativegates_ibm_qiskit_opt3_5"]


def needs_solver(name):
    return pytest.mark.skipif(not pace.available(name), reason="run scripts/setup_solvers.sh first")


def toy():
    x = np.array([[0, 1], [1, 0]], complex)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return build(Circuit("toy", 3, [Gate("h", (0,), h), Gate("cx", (0, 1), controlled(x)),
                                    Gate("cx", (1, 2), controlled(x))], []))


def theirs(name, gen):
    return [(s.a, s.b, s.c) for s in read_plan(paths.authors_plans() / f"resultats_{name}_pla_{gen}.txt").steps]


def test_julia_hash_and_dict_order():
    # Julia 1.9.1: hash(1), hash(12345), collect(keys(Dict{Int,Int}(v => 0 for v in 1:20)))
    assert qx.hash_64_64(1) == 6614235796240398542
    assert qx.hash_64_64(12345) == 7658450573117590115
    slot = qx.julia_dict_slots(range(1, 21))
    assert sorted(range(1, 21), key=lambda k: slot[k - 1]) == [5, 16, 7, 20, 12, 8, 17, 1, 19, 4, 6, 13, 2,
                                                               10, 11, 9, 15, 18, 14, 3]
    slot = qx.julia_dict_slots(range(1, 301))
    assert sorted(range(1, 301), key=lambda k: slot[k - 1])[:8] == [56, 35, 110, 60, 220, 30, 6, 67]


def test_line_graph_and_order_to_plan():
    tn = toy()
    lg = qx.line_graph(tn)
    # bond_map order: first appearance over tensors in push order, indices in order
    assert lg.bonds[:3] == [("Qubit1", 1), ("Qubit1", 0), ("Qubit1", 2)]
    assert all(len(ts) == 2 for ts in lg.edges)
    assert lg.to_gr().splitlines()[0] == f"p tw {len(lg.bonds)} {len(list(lg.graph.edges()))}"
    order = list(range(1, len(lg.bonds) + 1))
    plan = qx.order_to_contraction_plan(order, lg)
    assert len(plan) == len(tn.tensors) - 1 and plan[0] == ("t1", "t2", "I1")
    assert replay(tn, to_authors_plan(tn, plan, "toy")).ok


def test_min_fill_ties_and_swap_removal():
    # a 4-cycle: every vertex has cliqueness 1; Julia's Dict order picks key 4 first (slot order 4, 1, 2, 3)
    lg = qx.LineGraph(qx.SimpleGraph(4), [0, 1, 2, 3, 4], [], [])
    for a, b in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        lg.graph.add_edge(a, b)
    slot = qx.julia_dict_slots(range(1, 5))
    first = min(range(1, 5), key=lambda k: slot[k - 1])
    tw, order = qx.min_fill(lg)
    assert tw == 2 and order[0] == first and sorted(order) == [1, 2, 3, 4]


def test_order_from_tree_decomposition_and_validation():
    lg = qx.LineGraph(qx.SimpleGraph(4), [0, 1, 2, 3, 4], [], [])
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        lg.graph.add_edge(a, b)
    text = "c path\ns td 3 2 4\nb 1 1 2\nb 2 2 3\nb 3 3 4\n1 2\n2 3\n"
    td = qx.parse_td(text)
    assert qx.valid_td(td, lg) and td.width == 1
    # leaves in bag order: b_3 (parent b_2) gives 4, then b_2 (parent b_1) gives 3, then the root b_1: 1 2
    assert qx.order_from_tree_decomposition(td) == [4, 3, 1, 2]
    assert not qx.valid_td(qx.parse_td(text.rsplit("b 3", 1)[0]), lg)  # truncated output
    assert not qx.valid_td(qx.parse_td("s td 1 2 4\nb 1 1 2\n"), lg)
    assert qx.parse_td("c nothing\n") is None


def test_girvan_newman_ties_and_backends():
    # path 0-1-2-3: the middle edge goes first; then (0,1) and (2,3) tie at 1 -> smaller edge first
    assert removal_order(4, [(0, 1), (1, 2), (2, 3)], backend="networkx") == [(1, 2), (0, 1), (2, 3)]
    assert removal_order(4, [(0, 1), (1, 2), (2, 3)], backend="rustworkx") == [(1, 2), (0, 1), (2, 3)]


def test_authors_format_and_leaf_sets():
    tn = toy()
    lg = qx.line_graph(tn)
    plan = qx.order_to_contraction_plan(qx.min_fill(lg)[1], lg)
    ap = to_authors_plan(tn, plan, "toy_pla_mf")
    again = parse_plan(format_plan(ap), "toy_pla_mf")
    assert [(s.a, s.b, s.c, s.rank_a, s.rank_b, s.rank_c) for s in again.steps] == \
           [(s.a, s.b, s.c, s.rank_a, s.rank_b, s.rank_c) for s in ap.steps]
    assert replay(tn, again).ok
    renamed = [(a, b, "X" + c) for a, b, c in plan]
    renamed = [(a if a in tn.tensors else "X" + a, b if b in tn.tensors else "X" + b, c) for a, b, c in renamed]
    assert leaf_sets(renamed) == leaf_sets(plan)


@needs_external
@pytest.mark.parametrize("name", SMALL)
def test_mf_reproduces_authors_plan(name):
    tn = build(authors_gates(circuit_path(name)))
    g = generate(tn, "mf")
    assert g.steps == theirs(name, "mf")
    ours, ref = plan_features(to_authors_plan(tn, g.steps, name)), plan_features(read_plan(
        paths.authors_plans() / f"resultats_{name}_pla_mf.txt"))
    assert ours == pytest.approx(ref, nan_ok=True)


@needs_external
@pytest.mark.parametrize("name", SMALL)
def test_gn_reproduces_authors_plan_with_both_backends(name):
    tn = build(authors_gates(circuit_path(name)))
    a, b = gn_plan(tn, backend="rustworkx"), gn_plan(tn, backend="networkx")
    assert a == b == theirs(name, "gn")


@needs_external
@pytest.mark.parametrize("gen", ["minfill_mrs", "minfillbg_mrs"])
@pytest.mark.parametrize("name", SMALL)
def test_mrs_heuristics_reproduce_authors_plan(gen, name):
    if not pace.available(gen):
        pytest.skip("run scripts/setup_solvers.sh first")
    tn = build(authors_gates(circuit_path(name)))
    g = generate(tn, gen, time_limit=30)
    assert not g.fallback and g.steps == theirs(name, gen)


@needs_external
@needs_solver("fc")
def test_flowcutter_plan_is_valid():
    name = SMALL[1]
    tn = build(authors_gates(circuit_path(name)))
    g = generate(tn, "fc", time_limit=2)
    assert not g.fallback and g.meta["stopped"]
    assert replay(tn, to_authors_plan(tn, g.steps, name)).ok


@needs_external
@needs_solver("minfill_mrs")
def test_solver_stopped_before_output_falls_back_to_min_fill():
    name = "pricingcall_nativegates_ibm_qiskit_opt3_23"  # minfill_mrs needs several seconds here
    tn = build(authors_gates(circuit_path(name)))
    g = generate(tn, "minfill_mrs", time_limit=0.2)
    assert g.fallback and g.steps == theirs(name, "mf")
