"""Phase 4: the plan executor (CPU; the GPU test skips without CuPy and a device)."""
import math

import numpy as np
import pytest

from qctn import catalog, paths
from qctn.builders import ghz, qft
from qctn.circuits import Circuit, Gate, _ONE, controlled, phase
from qctn.execute import compile_plan, contract_plan, diagonal_groups, leaf_arrays, statevector, time_program
from qctn.execute.tensors import split_gate
from qctn.plans import qxtools as qx
from qctn.plans.authors import GENERATORS, read_plan
from qctn.plans.girvan_newman import gn_plan
from qctn.plans.tree import PlanTree
from qctn.tn import build

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)


def _gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no CuPy GPU")


def random_circuit(n=4, depth=12, seed=0):
    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(depth):
        q = int(rng.integers(n))
        k = rng.integers(4)
        if k == 0:
            gates.append(Gate("h", (q,), _ONE["h"]))
        elif k == 1:
            gates.append(Gate("p", (q,), phase(rng.uniform(0, 2 * math.pi))))
        else:
            a, b = rng.choice(n, 2, replace=False)
            u = _ONE["x"] if k == 2 else phase(rng.uniform(0, 2 * math.pi))
            gates.append(Gate("c", (int(a), int(b)), controlled(u)))
    return Circuit("rand", n, gates, [])


def mf_steps(tn):
    lg = qx.line_graph(tn)
    _, order = qx.min_fill(lg)
    return qx.order_to_contraction_plan(order, lg)


def test_split_gate_reproduces_matrix():
    for m in (controlled(_ONE["x"]), controlled(phase(0.3)), np.eye(4), np.eye(4)[[0, 2, 1, 3]]):
        a, b, chi = split_gate(m)
        t = np.einsum("aiv,vbj->abij", a, b).reshape(4, 4)
        assert np.allclose(t, m)
    assert split_gate(np.eye(4))[2] == 1
    assert split_gate(np.eye(4)[[0, 2, 1, 3]])[2] == 4


def test_diagonal_groups_like_find_hyper_edges():
    assert diagonal_groups(np.diag([1, 2j])) == [[0, 1]]
    assert diagonal_groups(_ONE["h"]) == []
    d = np.zeros((2, 2, 2))
    d[0, 0, 0] = d[1, 1, 1] = 1  # a copy tensor: all three axes in one group
    assert diagonal_groups(d) == [[0, 1, 2]]
    assert diagonal_groups(np.array([1.0, 0.0])) == []


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("hyper", [False, True])
def test_amplitudes_match_statevector(seed, hyper):
    c = random_circuit(seed=seed)
    tn = build(c)
    psi = statevector(c)
    steps = mf_steps(tn)
    for bits in ("0000", "1011", "0110"):
        arr = leaf_arrays(c, tn, out_bits=bits, convention="circuit")
        v = contract_plan(tn, steps, arr, hyper=hyper)
        assert v.shape == ()
        assert np.isclose(complex(v), psi[int(bits, 2)], atol=1e-12)


@pytest.mark.parametrize("hyper", [False, True])
def test_plans_agree(hyper):
    """Different orders on the same network give the same amplitude: MinFill, Girvan-Newman, a
    sequential sweep, and the PlanTree form of one of them."""
    c = qft(5)
    tn = build(c)
    arr = leaf_arrays(c, tn, convention="circuit")
    ref = statevector(c)[0]
    plans = [mf_steps(tn), gn_plan(tn)]
    ids = list(tn.tensors)
    seq, cur = [], ids[0]
    for k, t in enumerate(ids[1:]):
        seq.append((cur, t, f"S{k}"))
        cur = f"S{k}"
    plans.append(seq)
    plans.append(PlanTree.from_steps(plans[0], leaves=ids))
    for p in plans:
        assert np.isclose(complex(contract_plan(tn, p, arr, hyper=hyper)), ref, atol=1e-12)


def test_partial_plan_joins_roots_and_complex64():
    c = ghz(4)
    tn = build(c)
    arr = leaf_arrays(c, tn)
    steps = mf_steps(tn)[:5]  # leaves several roots; the executor joins them by outer products
    v = contract_plan(tn, steps, arr, dtype=np.complex64)
    assert v.dtype == np.complex64
    assert np.isclose(complex(v), 1 / math.sqrt(2), atol=1e-6)


def qxtns_circuit(c):
    """The circuit the authors' network computes: two-qubit matrices with the qubit roles exchanged, and every
    u2 / u1 / p / cu1 with the first such gate's matrix."""
    from qctn.execute.tensors import CACHED_GATES, qxtns_matrix

    first, gates = {}, []
    for g in c.gates:
        m = first.setdefault(g.name, g.matrix) if g.name in CACHED_GATES else g.matrix
        gates.append(Gate(g.name, g.qubits, qxtns_matrix(m) if len(g.qubits) == 2 else m))
    return Circuit(c.name, c.n_qubits, gates, [])


def test_qxtns_convention():
    rng = np.random.default_rng(3)
    gates = [Gate("h", (0,), _ONE["h"]), Gate("h", (2,), _ONE["h"]),
             Gate("cx", (0, 1), controlled(_ONE["x"])), Gate("cx", (2, 1), controlled(_ONE["x"]))]
    gates += [Gate("u1", (q,), phase(rng.uniform(0, 6))) for q in (0, 1, 2)]
    gates += [Gate("cy", (1, 0), controlled(_ONE["y"])), Gate("h", (1,), _ONE["h"])]
    c = Circuit("conv", 3, gates, [])
    tn = build(c)
    steps = mf_steps(tn)
    a = complex(contract_plan(tn, steps, leaf_arrays(c, tn)))
    b = complex(contract_plan(tn, steps, leaf_arrays(c, tn, convention="circuit")))
    assert np.isclose(a, statevector(qxtns_circuit(c))[0], atol=1e-12)
    assert np.isclose(b, statevector(c)[0], atol=1e-12)
    assert not np.isclose(a, b)
    # Schmidt rank 1 (a product gate): QXTns gives the second half the first half's values
    c1 = Circuit("prod", 2, [Gate("hx", (0, 1), np.kron(_ONE["h"], _ONE["x"]))], [])
    t1 = build(c1)
    arr = leaf_arrays(c1, t1)
    assert np.allclose(arr["t1"], arr["t2"])


def test_hyper_reduces_diagonal_gates():
    c = Circuit("cz", 2, [Gate("h", (0,), _ONE["h"]), Gate("cz", (0, 1), controlled(_ONE["z"])),
                          Gate("t", (1,), _ONE["t"])], [])
    tn = build(c)
    arr = leaf_arrays(c, tn)
    dense = compile_plan(tn, mf_steps(tn))
    hyper = compile_plan(tn, mf_steps(tn), hyper=True, arrays=arr)
    assert hyper.flops < dense.flops
    assert [len(s) for s in hyper.leaf_shapes][-5] == 1  # the T gate is stored as its diagonal


def test_bad_plan_raises():
    tn = build(ghz(3))
    with pytest.raises(ValueError):
        compile_plan(tn, [("t1", "t1", "X")])
    with pytest.raises(ValueError):
        compile_plan(tn, [("t1", "t2", "X"), ("t1", "t3", "Y")])


def test_time_program_numpy_records_repeats():
    c = ghz(5)
    tn = build(c)
    prog = compile_plan(tn, mf_steps(tn))
    t = time_program(prog, leaf_arrays(c, tn), backend="numpy", warmup=1, repeats=4)
    assert t.status == "ok" and len(t.times) == 4 and len(t.warmup) == 1
    assert np.isclose(t.result, 1 / math.sqrt(2), atol=1e-6)
    assert t.stat("min") <= t.stat("median") <= max(t.times)


@needs_external
def test_release_plans_agree():
    """All seven release plans of a QASM circuit give the same amplitude, dense and hyper: the one QXTns
    `contract_tn!` returns on the authors' network (Julia 1.9.1, QXTools 1.0, mf plan, ComplexF64), which is
    the statevector amplitude of `qxtns_circuit`. The file's ranks are the dense ranks."""
    name = "pricingcall_nativegates_ibm_qiskit_opt3_5"
    c = catalog.candidates(name)[0].build()
    tn = build(c)
    arr = leaf_arrays(c, tn)
    ref = statevector(qxtns_circuit(c))[0]
    assert np.isclose(ref, -0.0012992575137013044 - 0.0010844321449611167j, atol=1e-14)
    for g in GENERATORS:
        plan = read_plan(paths.authors_plans() / f"resultats_{name}_pla_{g}.txt")
        dense = compile_plan(tn, plan)
        assert dense.reduced_ranks == [s.rank_c for s in plan.steps]
        for hyper in (False, True):
            assert np.isclose(complex(contract_plan(tn, plan, arr, hyper=hyper)), ref, atol=1e-12)


@needs_gpu
@pytest.mark.parametrize("hyper", [False, True])
def test_cupy_matches_numpy(hyper):
    c = random_circuit(seed=7)
    tn = build(c)
    arr = leaf_arrays(c, tn)
    steps = mf_steps(tn)
    ref = contract_plan(tn, steps, arr, hyper=hyper)
    v = contract_plan(tn, steps, arr, backend="cupy", hyper=hyper)
    assert np.isclose(complex(v), complex(ref), atol=1e-12)
    prog = compile_plan(tn, steps, hyper=hyper, arrays=arr)
    t = time_program(prog, arr, backend="cupy", warmup=1, repeats=3)
    assert t.status == "ok" and len(t.times) == 3
    assert np.isclose(t.result, complex(ref), atol=1e-5)
