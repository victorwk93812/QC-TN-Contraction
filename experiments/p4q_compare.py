"""Phase 4b: compare the authors' stack timed on our V100 (`julia/p4q_qxtns.jl`) with the release's labels.

    PYTHONPATH=src .venv/bin/python experiments/p4q_compare.py results/raw/p4q/<job> [<job> ...]

Reads every `times.csv` and `plans.csv` under the given job directories (`main/`, `others/`, `cpu/` and
`fresh/<plan>/` from `scripts/slurm/p4q_julia.sbatch`), the release's labels (split files, RTX A6000), the plan
files' time column, and phase 4's CuPy timings (`results/p4_timing.csv`). Labels are compared within a circuit
with the plan's time (a smaller time should mean a larger speedup), so circuits without a MinFill plan
(`pricingcall_..._25`) count too. Writes

- `results/p4q_timing.csv`: one row per plan, one column per measure (seconds) plus plan statistics;
- `results/p4q_compare.csv`: one row per circuit and measure: Spearman and Kendall correlation of -time with the
  label speedup, and whether the fastest plan is the labels' best;
- `results/p4q_summary.csv`: per measure, medians and means over circuits;
- `results/p4q_steps.csv`: per plan, the release's time column against our run of the release's
  `contract_tn_rank_mock!` (step-by-step Spearman, medians, sums).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4q_labels import replicate, noise_sigma, within_circuit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# measure -> (kind of run directory, mode, device, statistic); statistic over measured runs (run >= 1),
# or "cold" (run -1, the first call in the process) / "warmup" (run 0).
MEASURES = {
    "qxtns_mean": ("main", "qxtns", "gpu", "mean"),
    "qxtns_median": ("main", "qxtns", "gpu", "median"),
    "qxtns_min": ("main", "qxtns", "gpu", "min"),
    "qxtns_cold_in_session": ("main", "qxtns", "gpu", "cold"),
    "qxtns_fresh_process_cold": ("fresh", "qxtns", "gpu", "cold"),
    "qxtns_fresh_process_mean": ("fresh", "qxtns", "gpu", "mean"),
    "mock_mean": ("others", "mock", "gpu", "mean"),
    "authors_mean": ("others", "authors", "gpu", "mean"),
    "authors_mock_mean": ("others", "authors_mock", "gpu", "mean"),
    "decomp_mean": ("others", "decomp", "gpu", "mean"),
    "ome_mean": ("others", "ome", "gpu", "mean"),
    "cpu_qxtns_mean": ("cpu", "qxtns", "cpu", "mean"),
    "cpu_ome_mean": ("cpu", "ome", "cpu", "mean"),
}


def load_runs(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    t, p = [], []
    for d in dirs:
        for f in sorted(Path(d).rglob("times.csv")):
            kind = "fresh" if "fresh" in f.parts else f.parent.name
            if f.stat().st_size:  # a running job may not have flushed its header yet
                t.append(pd.read_csv(f).assign(kind=kind, job=Path(d).name))
        for f in sorted(Path(d).rglob("plans.csv")):
            if f.stat().st_size:
                p.append(pd.read_csv(f).assign(kind=f.parent.name, job=Path(d).name))
    return pd.concat(t, ignore_index=True), (pd.concat(p, ignore_index=True) if p else pd.DataFrame())


def run_stats(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (kind, mode, device, plan): statistics of the measured runs, cold and warm-up runs."""
    rows = []
    for (kind, mode, dev, plan), g in t.groupby(["kind", "mode", "device", "plan"]):
        m = g[g.run >= 1]
        cold = g[g.run == -1]
        rows.append(dict(kind=kind, mode=mode, device=dev, plan=plan, circuit=g.circuit.iloc[0],
                         generator=g.generator.iloc[0], order=g.order.iloc[0], n=len(m),
                         failed=bool((g.run == -9).any()),
                         mean=m.seconds.mean(), median=m.seconds.median(), min=m.seconds.min(),
                         cv=m.seconds.std() / m.seconds.mean() if len(m) > 1 else np.nan,
                         cold=cold.seconds.mean(), cold_compile=cold.compile_s.mean(),
                         warmup=g[g.run == 0].seconds.mean(), gc_mean=m.gc_s.mean(), alloc_mb=m.alloc_mb.mean(),
                         inner_mean=m.inner_s.mean(), re=m.re.mean(), im=m.im.mean()))
    return pd.DataFrame(rows)


def release_rows() -> pd.DataFrame:
    from qctn.plans.authors import read_plan
    from qctn.paths import authors_plans
    from qctn.rank.data import SPLITS, load_split

    d = pd.concat([load_split(s, "ndcg").assign(split=s) for s in SPLITS]).drop_duplicates("circuit_name")
    return d[["circuit_name", "base_id", "speedup", "split"]].rename(columns={"circuit_name": "plan"})


def mock_column(plans) -> pd.DataFrame:
    from qctn.paths import authors_plans
    from qctn.plans.authors import read_plan

    rows = []
    for n in plans:
        f = authors_plans() / f"resultats_{n}.txt"
        if not f.exists():
            continue
        t = np.array([s.time for s in read_plan(f).steps])
        rows.append(dict(plan=n, release_mock_sum=t.sum(), release_mock_median_x_steps=np.median(t) * len(t),
                         release_mock_sum_wo_max=t.sum() - t.max()))
    return pd.DataFrame(rows)


def step_compare(dirs: list[Path]) -> pd.DataFrame:
    from scipy.stats import spearmanr

    from qctn.paths import authors_plans

    rows = []
    for d in dirs:
        for f in sorted(Path(d).rglob("steps/*__authors_mock__gpu__run1.txt")):
            plan = f.name.split("__")[0]
            rel = np.array([float(l.split()[-1]) for l in open(authors_plans() / f"resultats_{plan}.txt")])
            ours = np.array([float(l.split()[-1]) for l in open(f)])
            if len(rel) != len(ours):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(rel, ours).statistic
            rows.append(dict(plan=plan, steps=len(rel), step_spearman=rho, release_median_us=np.median(rel) * 1e6,
                             ours_median_us=np.median(ours) * 1e6, release_sum=rel.sum(), ours_sum=ours.sum(),
                             release_max=rel.max(), ours_max=ours.max(),
                             release_steps_over_10x_median=int((rel > 10 * np.median(rel)).sum())))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dirs = [Path(a) for a in argv]
    t, p = load_runs(dirs)
    st = run_stats(t)
    wide = None
    for name, (kind, mode, dev, stat) in MEASURES.items():
        s = st[(st.kind == kind) & (st["mode"] == mode) & (st.device == dev) & (st.n > 0 if stat != "cold" else True)]
        s = s.set_index("plan")[stat].rename(name)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    base = st[(st.kind == "main") & (st["mode"] == "qxtns")].set_index("plan")[
        ["circuit", "generator", "order", "n", "cv", "cold_compile", "gc_mean", "alloc_mb", "re", "im"]]
    base = base.rename(columns={"n": "qxtns_runs", "cv": "qxtns_cv", "cold_compile": "qxtns_cold_compile_s",
                                "gc_mean": "qxtns_gc_s", "alloc_mb": "qxtns_alloc_mb"})
    fresh = st[st.kind == "fresh"].set_index("plan")[["cold_compile"]].rename(
        columns={"cold_compile": "fresh_cold_compile_s"})
    au = st[(st.kind == "others") & (st["mode"] == "authors")].set_index("plan")[["inner_mean"]].rename(
        columns={"inner_mean": "authors_inner_mean"})
    df = base.join(wide, how="outer").join(fresh).join(au)
    if len(p):
        dp = p[p.kind == "others"].drop_duplicates("plan").set_index("plan")
        df = df.join(dp[["steps", "tensors", "decomp_indices_s", "decomp_data_s", "decomp_einsum_s", "decomp_rest_s",
                         "reshaped_operands", "host_outputs", "distinct_signatures", "distinct_ndims",
                         "distinct_shapes"]])
    df = df.reset_index().rename(columns={"index": "plan"})
    df["circuit"] = df.plan.str.split("_pla_").str[0]
    df["generator"] = df.plan.str.split("_pla_").str[1]
    p4 = pd.read_csv(OUT / "p4_timing.csv")
    for mode in ("dense", "hyper"):
        q = p4[p4["mode"] == mode].set_index("plan")
        df = df.merge(q[["mean"]].rename(columns={"mean": f"cupy_{mode}_mean"}), left_on="plan", right_index=True,
                      how="left")
        df = df.merge(q[["log2_flops"]].rename(columns={"log2_flops": f"log2_flops_{mode}"}), left_on="plan",
                      right_index=True, how="left")
    df = df.merge(mock_column(df.plan), on="plan", how="left")
    df = df.merge(release_rows(), on="plan", how="left")
    df.to_csv(OUT / "p4q_timing.csv", index=False)

    lab = df[df.speedup.notna()]
    measures = list(MEASURES) + ["authors_inner_mean", "cupy_dense_mean", "cupy_hyper_mean", "release_mock_sum",
                                 "release_mock_median_x_steps", "release_mock_sum_wo_max", "decomp_indices_s",
                                 "decomp_data_s", "decomp_einsum_s", "decomp_rest_s", "distinct_signatures",
                                 "distinct_shapes", "log2_flops_dense", "log2_flops_hyper"]
    pcs, summ = [], []
    for m in measures:
        if m not in lab or lab[m].notna().sum() == 0:
            continue
        x = lab.assign(neg=-lab[m])
        pc = within_circuit(x, "neg", "speedup", group="circuit").assign(measure=m)
        pcs.append(pc)
        summ.append(dict(measure=m, circuits=len(pc), spearman_median=pc.spearman.median(),
                         spearman_mean=pc.spearman.mean(), kendall_median=pc.kendall.median(),
                         kendall_mean=pc.kendall.mean(), best_same=int(pc.best_same.sum())))
    # the labels against a simulated re-measurement with the noise of the identical plans (p4q_label_noise.csv)
    noise = pd.read_csv(OUT / "p4q_label_noise.csv")
    sig = noise_sigma(noise.log_diff)
    rng = np.random.default_rng(0)
    reps = [within_circuit(lab.assign(rep=replicate(lab, sig, rng)), "rep", "speedup", group="circuit")
            for _ in range(500)]
    r = pd.concat(reps)
    summ.append(dict(measure=f"label_replicate_sigma_{sig:.3f}", circuits=len(reps[0]),
                     spearman_median=np.mean([x.spearman.median() for x in reps]),
                     spearman_mean=r.spearman.mean(), kendall_median=np.mean([x.kendall.median() for x in reps]),
                     kendall_mean=r.kendall.mean(), best_same=float(np.mean([x.best_same.sum() for x in reps]))))
    pc = pd.concat(pcs, ignore_index=True)
    pc.to_csv(OUT / "p4q_compare.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "p4q_summary.csv", index=False)
    sc = step_compare(dirs)
    sc.to_csv(OUT / "p4q_steps.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(sm.round(3).to_string(index=False))
    if len(sc):
        print(sc.describe().round(3).to_string())


if __name__ == "__main__":
    main()
