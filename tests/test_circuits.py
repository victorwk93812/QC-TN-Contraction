"""Phase 3a: the circuits behind the authors' plans (qctn.builders, qctn.catalog)."""
import sys
from collections import Counter
from pathlib import Path

import pytest

from qctn import builders as B
from qctn import paths
from qctn.catalog import base_name, candidates
from qctn.circuits import authors_gates_text
from qctn.plans.authors import read_plan
from qctn.tn import build, replay

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)


def test_ghz_and_qft_shapes():
    g = B.ghz(4)
    assert [x.qubits for x in g.gates] == [(0,), (1, 0), (2, 1), (3, 2)]  # c_x(i, i+1): gate_qubits [i+1, i]
    assert len(build(g).tensors) == 4 * 4 - 1
    q = B.qft(5)
    # QFT_5 in Notebook 2 / the plans: 43 tensors (5 H, 10 CPhase, 2 SWAP, 2 CX, 5 inputs, 5 outputs)
    assert Counter(x.name for x in q.gates) == {"h": 5, "c_r_phase": 10, "swap": 2, "c_x": 2}
    assert len(build(q).tensors) == 43
    assert q.gates[1].qubits == (1, 0)  # c_r_phase(1, 2): gate_qubits [2, 1]
    assert [x.qubits for x in q.gates if x.name == "swap"] == [(4, 0), (3, 1)]
    tn = build(q)
    assert max(d for d in tn.dims.values()) == 4  # SWAP has operator Schmidt rank 4


def test_qft_iqft_shape():
    c = B.qft_iqft(5)
    # QFT on qubits 1..3, then IQFT on 1..5 (per-qubit gate counts 10, 9, 9, 6, 6 as in the plans)
    per_qubit = Counter(q for g in c.gates for q in g.qubits)
    assert [per_qubit[i] for i in range(5)] == [10, 9, 9, 6, 6]
    assert len(build(c).tensors) == 50


def test_rqc_structure_does_not_depend_on_seed():
    a, b = B.rqc(3, 4, 12, seed=1), B.rqc(3, 4, 12, seed=2)
    assert [(g.qubits, len(g.qubits)) for g in a.gates] == [(g.qubits, len(g.qubits)) for g in b.gates]
    assert build(a).tensors == build(b).tensors
    assert {g.name for g in a.gates} <= {"h", "c_z", "t", "sx", "sy"}
    # every qubit's first single-qubit gate after H is T, and no gate repeats on a qubit
    last = {}
    for g in a.gates:
        if g.name in ("t", "sx", "sy"):
            q = g.qubits[0]
            assert g.name != last.get(q) and (q in last or g.name == "t")
            last[q] = g.name
    h = B.rqc(3, 4, 12, final_h=True)
    assert len(h.gates) == len(a.gates) + 12


def test_rqc_patterns_match_qxzoo():
    gp = B._rqc_patterns(2, 2)
    assert gp[1] == [[[1, 1], [1, 2]]] and gp[2] == [[[1, 1], [2, 1]]]
    assert all(len(v) <= 2 for v in gp.values())


def test_register_rewriting():
    text = "OPENQASM 2.0;\nqreg q[2];\nqreg flag[1];\ncreg c[3];\nh q[0];\nx flag[0];\ncx q[1],flag[0];\n"
    c = authors_gates_text(B.replace_registers_grover(text), "g")
    assert c.n_qubits == 3 and [g.qubits for g in c.gates] == [(0,), (2,), (1, 2)]
    assert authors_gates_text(text, "g").skipped == ["x flag[0];", "cx q[1],flag[0];"]
    w = "qreg node[2];\nqreg coin[1];\nh node[1];\ncx coin[0],node[0];\n"
    c = authors_gates_text(B.replace_registers(w), "w")
    assert c.n_qubits == 3 and [g.qubits for g in c.gates] == [(1,), (2, 0)]


def test_early_reader():
    text = "qreg q[2];\nu2(0,pi) q[0];\np(pi/4) q[1];\nsx q[0];\nh q[1];\ncx q[0],q[1];\n"
    now = authors_gates_text(text, "x")
    assert [g.name for g in now.gates] == ["u2", "p", "sx", "h", "cx"]
    early = B.reverse_two_qubit(authors_gates_text(text, "x", B.EARLY_SKIPPED))
    assert [(g.name, g.qubits) for g in early.gates] == [("h", (1,)), ("cx", (1, 0))]
    assert len(early.skipped) == 3


def test_catalog_names():
    assert base_name("QNN_15_fils_28_gpu") == "QNN_15"
    assert base_name("RQC_10_10_14_fil_8") == "RQC_10_10_14"
    assert base_name("Qwalk_5_fil_2_gpu") == "Qwalk_5"
    assert candidates("GHZ_999_gpu")[0].params == {"n": 999}
    assert candidates("RQC_5_28_mockpla_fc")[0].params == dict(rows=5, cols=5, depth=28, final_h=True)
    assert candidates("RQC_3_7_50")[0].params["depth"] == 49
    assert candidates("unknown_circuit") == []


@needs_external
@pytest.mark.parametrize("plan,source", [
    ("GHZ_100_pla_mf", "create_ghz_circuit"),
    ("QFT_5_pla_mf", "create_qft_circuit"),
    ("QFT_IQFT_7_pla_tamaki", "apply_qft"),
    ("QFT_qasm_20_pla_gn", "qft_nativegates"),
    ("QNN_10", "earlier reader"),
    ("QNN_10_pla_fc", "qnn_indep_qiskit_10.qasm()"),
    ("VQE_8_pla_fc", "vqe_nativegates"),
    ("GROVER_6_pla_mf", "opt3_5.qasm"),
    ("QWALK_3_pla_mf", "qwalk-noancilla"),
    ("Qwalk_4", "earlier reader"),
    ("Random_IBM_12", "random_nativegates"),
    ("RQC_3_7_20_pla_fc", "no final H"),
    ("RQC_3_7_50_pla_fc", "depth=49"),
    ("RQC_4_4_14", "final H"),
    ("RQC_5_16_pla_mf", "final H"),
    ("RQC_5_16_pla_jdrasil", "no final H"),
    ("RQC_4_6_30_pla_fc", "depth=32"),
    ("dj_indep_qiskit_25_pla_mf", "pricingcall"),
    ("groundstate_large_indep_qiskit_14_14_pla_mf", "groundstate"),
])
def test_plan_replays_on_its_circuit(plan, source):
    p = read_plan(paths.authors_plans() / f"resultats_{plan}.txt")
    for c in candidates(p.circuit):
        tn = build(c.build())
        r = replay(tn, p)
        if r.ok and not r.unused:
            assert source in c.label()
            return
    pytest.fail(f"{plan}: no candidate replays")


@needs_external
def test_every_plan_replays():
    """All 1663 plan files: 1655 replay fully, 6 agree step by step but leave tensors uncontracted (RQC_4_11,
    RQC_4_20, RQC_4_60), 2 don't parse (phase 2)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from p3_circuits import main

    nets, cmap = main()
    assert len(nets) == 1663 and len(cmap) == 332
    assert nets.status.value_counts().to_dict() == {"pass": 1655, "partial": 6, "unparseable": 2}
    assert set(nets[nets.status == "partial"].circuit) == {"RQC_4_11", "RQC_4_20", "RQC_4_60"}
    assert (cmap.status == "pass").sum() == 327
