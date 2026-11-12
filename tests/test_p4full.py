"""Phase 4 full run: executor extensions, the run script's skip / shard / restart logic, the comparison (CPU)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qctn import paths
from qctn.builders import ghz
from qctn.execute import compile_plan, leaf_arrays, time_program
from qctn.execute.executor import _UF
from qctn.plans import qxtools as qx
from qctn.tn import TensorNetwork, build

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import p4full_compare as C  # noqa: E402
import p4full_timing as T  # noqa: E402

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)


def test_union_find_deep_chain():
    """GROVER_13 and QWALK_7 made the recursive union-find hit the recursion limit (hyperindex mode)."""
    uf = _UF()
    for i in range(5000):
        uf.union(i + 1, i)  # each union puts a new root on top: a chain 5000 deep
    assert uf.find(0) == 5000 and uf.p[0] == 5000  # path compressed
    assert len({uf.find(i) for i in range(5001)}) == 1


def test_hyper_compile_ghz():
    c = ghz(1000)
    tn = build(c)
    arr = leaf_arrays(c, tn)
    lg = qx.line_graph(tn)
    steps = qx.order_to_contraction_plan(qx.min_fill(lg)[1], lg)
    prog = compile_plan(tn, steps, hyper=True, arrays=arr)
    assert len(prog.ops) == len(tn.tensors) - 1


def test_sizes_exact_above_int64():
    """70 disjoint vectors: the final outer product has 2^70 elements; sizes are Python ints, not wrapped."""
    tn = TensorNetwork({f"t{i}": [("q", i)] for i in range(70)}, {("q", i): 2 for i in range(70)})
    prog = compile_plan(tn, [])
    assert prog.max_elements == 2**70
    assert prog.peak_bytes(8) >= 8 * 2**70


def test_max_warmup_stops_slow_plan():
    c = ghz(4)
    tn = build(c)
    prog = compile_plan(tn, [])
    t = time_program(prog, leaf_arrays(c, tn), backend="numpy", warmup=1, repeats=5, max_warmup=0.0)
    assert t.status == "slow" and t.times == [] and len(t.warmup) == 1
    t = time_program(prog, leaf_arrays(c, tn), backend="numpy", warmup=1, repeats=3, max_warmup=60.0)
    assert t.status == "ok" and len(t.times) == 3


def test_skip_reason():
    st = dict(peak_gb=13.0, pred_run_s=1.0)
    assert T.skip_reason(st, 12, 600).startswith("memory")
    assert T.skip_reason(dict(peak_gb=1.0, pred_run_s=700.0), 12, 600).startswith("predicted_slow")
    assert T.skip_reason(dict(peak_gb=1.0, pred_run_s=1.0), 12, 600) == ""


def test_shard_balances_and_keeps_circuits_whole():
    rng = np.random.default_rng(0)
    rows = [dict(circuit=f"c{i % 30}", plan=f"p{i}", pred_run_s=float(rng.uniform(0, 1)), compile_s=0.1,
                 build_s=1.0, skip="" if i % 7 else "memory: x") for i in range(200)]
    sh = T.shard(pd.DataFrame(rows), 4, 1, 50)
    assert sh.circuit.is_unique and len(sh) == 30
    load = sh.groupby("shard").pred_s.sum()
    assert len(load) == 4 and load.max() - load.min() <= sh.pred_s.max()


def test_compare_one():
    y = np.array([1.0, 2.0, 1.5, 0.5])
    r = C.compare_one(y.copy(), y)
    assert r["spearman"] == pytest.approx(1) and r["kendall"] == pytest.approx(1)
    assert r["best_same"] and r["regret"] == 0
    r = C.compare_one(-y, y)
    assert r["spearman"] == pytest.approx(-1) and not r["best_same"]
    assert r["regret"] == pytest.approx(1 - 0.5 / 2)
    assert r["regret_random"] == pytest.approx(np.mean(1 - y / 2))


def test_per_circuit_uses_times_and_nodup():
    gens = ["mf", "fc", "gn", "tamaki"]
    t = {"mf": 1.0, "fc": 0.5, "gn": 2.0, "tamaki": 1.0}
    lab = {"mf": 1.0, "fc": 3.0, "gn": 0.4, "tamaki": 1.0}
    df = pd.DataFrame([dict(plan=f"c_pla_{g}", circuit="c", generator=g, mode="hyper", status="ok", mean=t[g],
                            median=t[g], min=t[g], mock_time=1.0 / lab[g], release_speedup=lab[g], split="train",
                            family="F", tensors=10, size_bin="<=500", source="s", dup_of_minfill=g == "tamaki",
                            dup_drop=g == "tamaki") for g in gens])
    pc = C.per_circuit(df)
    a = pc[(pc.ours == "mean") & (pc.variant == "all")].iloc[0]
    assert a.plans == 4 and a.best_same and a.regret == 0 and a.dup_of_minfill == 1
    assert a.spearman == pytest.approx(1)
    nd = pc[(pc.ours == "mean") & (pc.variant == "nodup")].iloc[0]
    assert nd.plans == 3
    m = pc[(pc.ours == "mock") & (pc.variant == "all")].iloc[0]
    assert m.spearman == pytest.approx(1)
    df.loc[df.generator == "gn", "status"] = "oom"  # not timed: out of `mock_same`, still in `mock`
    pc = C.per_circuit(df)
    assert pc[(pc.ours == "mock") & (pc.variant == "all")].plans.iloc[0] == 4
    assert pc[(pc.ours == "mock_same") & (pc.variant == "all")].plans.iloc[0] == 3
    assert pc[(pc.ours == "mean") & (pc.variant == "all")].not_timed.iloc[0] == 1
    sm = C.summarise(pc)
    assert set(sm.group_by) == {"all", "split", "family", "size"}


@needs_external
def test_run_writes_per_circuit_files_and_restarts(tmp_path):
    """`run` on CPU: one file per circuit, every plan and mode, and a rerun skips the circuit."""
    circ = "QFT_10"
    shards = tmp_path / "shards.csv"
    pd.DataFrame([dict(circuit=circ, shard=0, pred_s=1.0)]).to_csv(shards, index=False)
    out = tmp_path / "run"
    args = ["run", "--out", str(out), "--shard", "0", "--shards-file", str(shards), "--backend", "numpy",
            "--repeats", "2"]
    T.main(args)
    f = out / "circuits" / f"{circ}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    n_plans = (T.plan_table().circuit == circ).sum()
    assert len(rows) == 2 * n_plans and {r["mode"] for r in rows} == {"hyper", "dense"}
    assert all(r["status"] == "ok" and len(r["times"]) == 2 for r in rows)
    amp = {complex(*r["result"]) for r in rows}
    assert max(abs(a - next(iter(amp))) for a in amp) < 1e-5  # every plan, both modes: the same amplitude
    T.main(args)
    tasks = [json.loads(p.read_text()) for p in (out / "tasks").glob("*.json")]
    assert len(tasks) == 1  # same shard and job tag: the task file is rewritten
    assert tasks[0]["skipped_done"] == [circ] and tasks[0]["circuits"] == []
    assert not list((out / "circuits").glob("*.part"))
    assert math.isfinite(tasks[0]["seconds"])
