"""Phase 4b: what the labels measure (experiments/p4q_*.py, experiments/julia/p4q_qxtns.jl). CPU only."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qctn import paths

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "experiments"))
import p4q_compare  # noqa: E402
import p4q_labels  # noqa: E402

needs_external = pytest.mark.skipif(not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first")


def test_within_circuit():
    df = pd.DataFrame(dict(base_id=list("aaaabbbccd"), x=[1, 2, 3, 4, 3, 2, 1, 5, 5, 1],
                           speedup=[1, 2, 3, 4, 1, 2, 3, 1, 2, 1]))
    pc = p4q_labels.within_circuit(df, "x").set_index("group")
    # c has a constant statistic and d one plan: both left out
    assert list(pc.index) == ["a", "b"]
    assert pc.spearman["a"] == pytest.approx(1) and pc.kendall["a"] == pytest.approx(1)
    assert pc.spearman["b"] == pytest.approx(-1)
    assert pc.best_same.to_dict() == {"a": True, "b": False}


def test_noise_and_replicate():
    # |d| has median 0.2; sigma = 1.4826 * 0.2 / sqrt(2)
    assert p4q_labels.noise_sigma([0.1, -0.2, 0.3]) == pytest.approx(1.4826 * 0.2 / np.sqrt(2))
    df = pd.DataFrame(dict(speedup=[0.5, 1.0, 2.0]))
    assert np.allclose(p4q_labels.replicate(df, 0.0, np.random.default_rng(0)), df.speedup)


def test_flop_pairs():
    df = pd.DataFrame(dict(base_id=["a"] * 3, f=[10.0, 20.0, 40.0], speedup=[2.0, 1.0, 4.0]))
    rows = pd.DataFrame(p4q_labels.flop_pairs(df, "f", "dense", bins=(0, 15, 100), big=36))
    allp = rows[rows.larger_plan_at_least == "any"].set_index("dlog2_flops_from")
    # pairs: (20 vs 10, label agrees), (40 vs 10: disagrees), (40 vs 20: disagrees)
    assert allp.pairs.to_dict() == {0: 1, 15: 2}
    assert allp.share_flops_agree[0] == 1 and allp.share_flops_agree[15] == 0
    assert rows[rows.larger_plan_at_least == "2^36"].pairs.sum() == 2


def test_run_stats():
    t = pd.DataFrame(dict(kind="main", mode="qxtns", device="gpu", plan="c_pla_mf", circuit="c", generator="mf",
                          order=1, run=[-1, 0, 1, 2, 3], seconds=[9.0, 2.0, 1.0, 2.0, 3.0], gc_s=0.0, alloc_mb=1.0,
                          compile_s=[8.0, 0, 0, 0, 0], re=0.5, im=0.0, inner_s=np.nan))
    s = p4q_compare.run_stats(t).iloc[0]
    assert (s.n, s["mean"], s["median"], s["min"], s.cold, s.warmup, s.cold_compile) == (3, 2.0, 2.0, 1.0, 9.0, 2.0, 8.0)


@needs_external
def test_rescaled_minfill_rows():
    """The split speedups of the circuits whose MinFill row isn't 1 are measures.xlsx's times a constant."""
    rs = p4q_labels.rescaled()
    assert len(rs) == 25  # 26 MinFill rows; RQC_3_7_40's is its second listing (first listing kept)
    cov = rs[rs.in_measures > 0]
    assert len(cov) == 21 and cov.constant.all()
    assert np.allclose(cov.ratio_min, cov.mf_speedup_split)


@needs_external
def test_label_results_files():
    """The committed result files agree with the release data they were made from."""
    noise = pd.read_csv(RESULTS / "p4q_label_noise.csv")
    assert (noise.kind == "minfill_copy").sum() == 17 and (noise.kind == "repeated_listing").sum() == 12
    stats = pd.read_csv(RESULTS / "p4q_label_stats.csv").set_index("statistic")
    assert stats.loc["time_sum", "circuits"] == 223


@pytest.mark.skipif(not os.environ.get("QCTN_RUN_JULIA"), reason="set QCTN_RUN_JULIA=1 (loads the Julia stack)")
def test_julia_modes_cpu(tmp_path):
    """All six modes of p4q_qxtns.jl on CPU arrays, one small circuit: same amplitude from qxtns, decomp and the
    release's function; the mock modes return no value."""
    julia = Path.home() / ".juliaup" / "bin" / "julia"
    ext = paths.authors_repo().parent
    if not julia.exists() or not (ext / "julia-env" / "qx").is_dir():
        pytest.skip("no Julia stack")
    env = dict(os.environ, JULIA_DEPOT_PATH=str(ext / "julia-depot"))
    subprocess.run([str(julia), "+1.9.1", f"--project={ext / 'julia-env' / 'qx'}",
                    str(ROOT / "experiments" / "julia" / "p4q_qxtns.jl"), str(paths.authors_repo()), str(tmp_path),
                    "circuits=pricingcall_nativegates_ibm_qiskit_opt3_5", "gens=mf",
                    "modes=qxtns,mock,authors,authors_mock,decomp,ome", "device=cpu", "repeats=1"],
                   check=True, env=env, timeout=1800)
    t = pd.read_csv(tmp_path / "times.csv")
    assert set(t["mode"]) == {"qxtns", "mock", "authors", "authors_mock", "decomp", "ome"}
    assert (t.run != -9).all()
    v = t[t["mode"].isin(["qxtns", "decomp", "authors"])].assign(z=lambda x: x.re + 1j * x.im)
    assert np.allclose(v.z, v.z.iloc[0], rtol=1e-5)
    assert t[t["mode"] == "mock"].re.isna().all()
    p = pd.read_csv(tmp_path / "plans.csv")
    assert p.steps.iloc[0] == 202 and p.reshaped_operands.iloc[0] >= 0
