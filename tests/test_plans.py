"""Phase 2: the authors' plan files, the 18 features, the split scaler, circuits and networks."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from qctn import paths
from qctn.circuits import SWAP, Circuit, Gate, authors_gates, circuit_path, controlled, phase
from qctn.features import EPS, FEATURE_NAMES, features_from_ranks, plan_features
from qctn.plans.authors import (AuthorsPlan, PlanFormatError, format_line, julia_float, parse_line,
                                read_all, read_plan, split_name)
from qctn.plans.tree import PlanError, PlanTree
from qctn.tn import build, replay, schmidt_rank

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)


@pytest.mark.parametrize("x,s", [
    (0.24606134, "0.24606134"), (9.656e-5, "9.656e-5"), (1e-5, "1.0e-5"), (0.0001, "0.0001"),
    (3.0, "3.0"), (12.5, "12.5"), (1.5e-10, "1.5e-10"),
])
def test_julia_float(x, s):
    assert julia_float(x) == s


def test_line_round_trip_and_indices():
    line = 'I1150 t3 I1151 6 6 8  Index[(dim=2|id=80|"Qubit3")\'150, (dim=2|id=811|"Qubit5")\'\'] 9.656e-5'
    s = parse_line(line)
    assert (s.rank_a, s.rank_b, s.rank_c, s.n_common) == (6, 6, 8, 2)
    assert format_line(s) == line
    ix = s.parsed_indices()
    assert [(i.tags, i.plev, i.id) for i in ix] == [("Qubit3", 150, 80), ("Qubit5", 2, 811)]
    assert parse_line("I1 I2 I3 3 2 5  Index{Int64}[] 0.1").parsed_indices() == []
    with pytest.raises(PlanFormatError):
        parse_line(' 2 1  Index[(dim=2|id=691|"Qubit8")\'\'] 0.000479523')


def test_split_name():
    assert split_name("dj_indep_qiskit_40_pla_minfill_mrs") == ("dj_indep_qiskit_40", "minfill_mrs")
    assert split_name("GHZ_10") == ("GHZ_10", None)


def test_plan_tree():
    t = PlanTree.from_steps([("t1", "t2", "I1"), ("t3", "I1", "I2"), ("t4", "t5", "t6")])
    assert t.leaves == ["t1", "t2", "t3", "t4", "t5"]
    assert t.roots == ["I2", "t6"]
    assert t.ssa_path() == [(0, 1), (2, 5), (3, 4)]
    with pytest.raises(PlanError, match="already contracted"):
        PlanTree.from_steps([("t1", "t2", "I1"), ("t1", "t3", "I2")])
    with pytest.raises(PlanError, match="reused"):
        PlanTree.from_steps([("t1", "t2", "I1"), ("t3", "t4", "I1")])
    with pytest.raises(PlanError, match="not a tensor"):
        PlanTree.from_steps([("t1", "t9", "I1")], leaves=["t1", "t2"])


def test_features_by_hand():
    # steps (rA, rB, rC): (2,1,1) k=1; (3,2,3) k=1; (3,3,2) k=2
    f = features_from_ranks([2, 3, 3], [1, 2, 3], [1, 3, 2])
    c, p, k = np.array([2, 4, 4.]), np.array([1, 3, 2.]), np.array([1, 1, 2.])
    d = np.array([1, 1, 0]) / (p + EPS)
    w = 2 ** c
    assert f["max_cost"] == 4 and f["n_steps"] == 3
    assert f["log2_sum_flops_val"] == pytest.approx(math.log2(36))
    assert f["topq_mean"] == 4  # max(1, round(3 * 5 / 100)) = 1 step
    assert f["std_cost"] == pytest.approx(np.std(c, ddof=1))
    assert f["frac_tiny"] == 0  # tau = 6
    assert f["p_at_max_cost_val"] == 2.5 and f["k_at_max_cost_val"] == 1.5
    assert f["costw_out_rank"] == pytest.approx((p * w).sum() / w.sum())
    assert f["max_asym_val"] == pytest.approx(1 / (1 + EPS))
    assert f["d_at_max_cost_val"] == pytest.approx(d[1:].mean())
    assert f["costw_red_rank_val"] == pytest.approx((k * w).sum() / w.sum())


@needs_external
def test_qft5_matches_notebook_2_output():
    # Notebook 2, cell 12 (printed output), for resultats_QFT_5_pla_tamaki.txt.
    printed = [7.0, 10.236014191900084, 7.0, 4.166666666666667, 1.3778149308741858, 42.0,
               0.023809523809523808, 5.0, 2.9285714285714284, 3.9635157545605306, 5.0, 2.0,
               1.550580431177446, 2.0, 0.99999950000025, 0.38015853951069445, 0.2279711743799879,
               0.199999960000008]
    f = plan_features(read_plan(paths.authors_repo() / "Data" / "plans" / "resultats_QFT_5_pla_tamaki.txt"))
    np.testing.assert_allclose([f[n] for n in FEATURE_NAMES], printed, rtol=1e-12)


@pytest.fixture(scope="module")
def all_plans():
    texts = {}
    plans, errors = read_all(texts=texts)
    return plans, errors, texts


@pytest.fixture(scope="module")
def features_xlsx():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = pd.read_excel(paths.authors_repo() / "Data" / "features" / "Features.xlsx")
    return x.set_index("circuit_name")[FEATURE_NAMES].astype(float)


@needs_external
def test_every_plan_parses_and_round_trips(all_plans):
    plans, errors, texts = all_plans
    assert len(plans) == 1661
    assert sorted(errors) == ["Qwalk_5", "RANDOM_11_pla_mf"]  # one truncated line each
    for name, plan in plans.items():
        assert "".join(format_line(s) + "\n" for s in plan.steps) == texts[name], name
    bad_trees = []
    for name, plan in plans.items():
        try:
            PlanTree.from_steps((s.a, s.b, s.c) for s in plan.steps)
        except PlanError:
            bad_trees.append(name)
    assert bad_trees == ["QWALK_3_pla_Flow_cutter", "QWALK_6_pla_flow_cutter"]


@needs_external
def test_features_match_features_xlsx(all_plans, features_xlsx):
    plans = all_plans[0]
    off = []
    for name in features_xlsx.index:
        if name not in plans:
            continue
        f = plan_features(plans[name])
        ref = features_xlsx.loc[name]
        if not all(math.isclose(f[n], ref[n], rel_tol=1e-9, abs_tol=1e-12) for n in FEATURE_NAMES):
            off.append(name)
    # Plan files that aren't the ones behind Features.xlsx (other step counts, outer products).
    assert off == ["RQC_4_11_pla_mf", "RQC_4_20_pla_fc", "RQC_4_20_pla_gn", "RQC_4_20_pla_mf",
                   "RQC_4_60_pla_fc", "RQC_4_60_pla_mf"]
    assert sum(n in plans for n in features_xlsx.index) == 1657


@needs_external
def test_split_scaler_is_one_zscore_over_773_rows(features_xlsx):
    from qctn.rank.data import FEATURES, load_split

    d = pd.concat([load_split(s, "pair") for s in ("train", "id", "ood")], ignore_index=True)
    raw = [f for f in FEATURE_NAMES if f not in FEATURES["pair"]]
    src = []
    for _, r in d.iterrows():
        circ, gen = split_name(r.circuit_name)
        v = r[raw].astype(float).values
        if np.allclose(features_xlsx.loc[r.circuit_name, raw].values, v, rtol=1e-9):
            src.append(r.circuit_name)
        elif gen == "jdrasil" and np.allclose(features_xlsx.loc[f"{circ}_pla_mf", raw].values, v, rtol=1e-9):
            src.append(f"{circ}_pla_mf")
        else:
            src.append(None)
    d["src"] = src
    ok = d.src.notna()
    assert ok.sum() == 1462
    # 213 jdrasil rows carry the MinFill plan's features.
    assert sum(s is not None and s.endswith("_pla_mf") and n.endswith("_pla_jdrasil")
               for s, n in zip(src, d.circuit_name)) == 213
    n = 773
    for f in ("n_steps", "max_out_rank", "max_red_rank_val"):  # integer features
        r, z = features_xlsx.loc[d.src[ok], f].values, d.loc[ok, f].astype(float).values
        a, b = np.polyfit(r, z, 1)
        assert np.abs(a * r + b - z).max() < 1e-12
        mu, sd = -b / a, 1 / a
        assert abs(mu * n - round(mu * n)) < 1e-6  # integer sum over 773 rows
        ss = sd ** 2 * (n - 1) + n * mu ** 2  # integer sum of squares with ddof = 1
        assert abs(ss - round(ss)) < 1e-3


def test_schmidt_ranks():
    assert schmidt_rank(controlled(np.array([[0, 1], [1, 0]]))) == 2
    assert schmidt_rank(SWAP) == 4
    assert schmidt_rank(controlled(phase(0.3))) == 2
    assert schmidt_rank(controlled(phase(0.0))) == 1


def test_network_conventions():
    x = np.array([[0, 1], [1, 0]], complex)
    c = Circuit("toy", 2, [Gate("h", (0,), np.eye(2)), Gate("cx", (0, 1), controlled(x))], [])
    tn = build(c)
    assert tn.tensors == {
        "t1": [("Qubit1", 1), ("Qubit1", 0)],
        "t2": [("Qubit1", 2), ("Qubit1", 1), ("v", 1)],
        "t3": [("v", 1), ("Qubit2", 1), ("Qubit2", 0)],
        "t4": [("Qubit1", 0)], "t5": [("Qubit2", 0)],
        "t6": [("Qubit1", 2)], "t7": [("Qubit2", 1)],
    }
    plan = AuthorsPlan("toy", [parse_line(l) for l in [
        't4 t1 I1 1 2 1  Index[(dim=2|id=1|"Qubit1")] 0.1',
        'I1 t2 I2 1 3 2  Index[(dim=2|id=2|"Qubit1")\'] 0.1',
        't5 t3 I3 1 3 2  Index[(dim=2|id=3|"Qubit2")] 0.1',
        'I2 I3 I4 2 2 2  Index[(dim=2|id=4)] 0.1',
        'I4 t6 I5 2 1 1  Index[(dim=2|id=5|"Qubit1")\'\'] 0.1',
        'I5 t7 I6 1 1 0  Index[(dim=2|id=6|"Qubit2")\'] 0.1',
    ]])
    assert replay(tn, plan).ok
    tree = PlanTree.from_steps(((s.a, s.b, s.c) for s in plan.steps), leaves=list(tn.tensors))
    inputs, output, size, order = tn.cotengra_inputs()
    ct = tree.to_cotengra(inputs, output, size, order)
    assert ct.is_complete() and ct.contraction_width() == 2


@needs_external
def test_authors_network_matches_their_plans():
    name = "dj_indep_qiskit_40"
    c = authors_gates(circuit_path(name))
    assert c.n_qubits == 40 and c.skipped == []
    tn = build(c)
    assert len(tn.tensors) == 237
    for gen in ("fc", "gn", "jdrasil", "mf", "minfill_mrs", "minfillbg_mrs", "tamaki"):
        r = replay(tn, read_plan(paths.authors_plans() / f"resultats_{name}_pla_{gen}.txt"))
        assert r.ok and r.unused == [], (gen, r.mismatches)


@needs_external
def test_authors_parser_skips_gates_qiskit_reads():
    from qctn.circuits import load_qiskit, qiskit_gate_counts

    path = circuit_path("grover-noancilla_indep_qiskit_4")
    g = authors_gates(path)
    # Gates on the 'flag' register and ccx are dropped.
    assert all("flag[" in s or s.startswith("ccx ") for s in g.skipped) and len(g.skipped) == 17
    assert len(g.gates) + len(g.skipped) == sum(qiskit_gate_counts(load_qiskit(path)).values())
