"""Phase 4: compare our GPU times with the release's labels and with the plan files' mock times.

    PYTHONPATH=src .venv/bin/python experiments/p4_compare.py results/raw/p4/<job id>

Reads `<run>/times.jsonl` (from `p4_timing.py`) and, if present, `<run>/julia/julia_times.csv` (from
`julia/p4_qxtns.jl`). Speedup of a plan is t(MinFill) / t(plan) within its circuit, as the release's labels are
(MinFill = 1), for each statistic of the repeats (mean as in the paper, median, min). The release's speedups come
from the split files (`Data/splits`, measured on the paper's RTX A6000). Writes

- `results/p4_timing.csv`: one row per plan and mode, with repeat statistics, plan size, our speedups, the
  release's speedup, the mock-time speedup (sum of the plan file's time column) and QXTns' time if run;
- `results/p4_compare.csv`: one row per circuit, mode and statistic: plans compared, Spearman rank correlation
  with the release's speedups, whether the best plan is the same, and the same for the mock times and QXTns;
- `results/p4_estimate.csv`: the estimated GPU time of a run over all 1663 plan files (see `estimate`).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
STATS = ("mean", "median", "min")


def release_speedups() -> pd.Series:
    from qctn.rank.data import SPLITS, load_split

    d = pd.concat([load_split(s, "ndcg") for s in SPLITS])
    # RQC_3_7_40 / _52 repeat plans with different speedups (phase 1); none are in the QASM circuits.
    return d.groupby("circuit_name").speedup.first()


def load_run(run: Path) -> pd.DataFrame:
    rows = []
    for line in (run / "times.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = np.asarray(r.pop("times", None) or [], float)
        r.pop("wall", None)
        w = r.pop("warmup", None) or []
        r.update(n=len(t), warmup_s=w[0] if w else np.nan,
                 **{s: (getattr(np, s)(t) if len(t) else np.nan) for s in STATS},
                 cv=(t.std(ddof=1) / t.mean()) if len(t) > 1 else np.nan)
        rows.append(r)
    df = pd.DataFrame(rows)
    if "result" in df:
        df["result"] = df.result.map(lambda v: complex(*v) if isinstance(v, list) else np.nan)
    return df


def load_julia(run: Path) -> pd.DataFrame | None:
    p = run / "julia" / "julia_times.csv"
    if not p.exists():
        return None
    j = pd.read_csv(p)
    j = j[j.run > 0]
    return j.groupby("plan").seconds.agg(julia_n="size", julia_mean="mean", julia_median="median",
                                          julia_min="min").reset_index()


def add_speedups(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    mf = df[df.generator == "mf"].set_index(["circuit", "mode"])[col]
    key = list(zip(df.circuit, df["mode"]))
    df[out] = [mf.get(k, np.nan) for k in key] / df[col]
    return df


def per_circuit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, mode), g in df.groupby(["circuit", "mode"]):
        g = g[g.release_speedup.notna()]
        for what in [f"sp_{s}" for s in STATS] + ["sp_mock"] + (["sp_julia"] if "sp_julia" in g else []):
            h = g[g[what].notna() & np.isfinite(g[what])]
            row = dict(circuit=c, mode=mode, ours=what[3:], plans=len(h))
            if len(h) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row["spearman_release"] = spearmanr(h[what], h.release_speedup).statistic
                row["best_same"] = h.loc[h[what].idxmax(), "generator"] == h.loc[h.release_speedup.idxmax(), "generator"]
                row["best_ours"] = h.loc[h[what].idxmax(), "generator"]
                row["best_release"] = h.loc[h.release_speedup.idxmax(), "generator"]
            rows.append(row)
    return pd.DataFrame(rows)


def estimate(df: pd.DataFrame, warmup: int = 1, repeats: int = 50) -> pd.DataFrame:
    """GPU seconds for all 1663 plan files: per mode, fit the measured mean per-run time as a + b * steps +
    c * 2**log2_flops (non-negative least squares on this run's plans), predict each plan file's run time from
    its step count and the release's flop count (`qctn.features`, all dimensions 2), and add (warmup + repeats)
    runs per plan, with and without the 5 s per-plan budget of the small run. Compile time is CPU and measured
    separately."""
    from scipy.optimize import nnls

    from qctn.features import plan_features
    from qctn.plans.authors import read_all

    plans, errors = read_all()
    feats = pd.DataFrame([dict(plan=n, steps=len(p.steps), log2_flops=plan_features(p)["log2_sum_flops_val"])
                          for n, p in plans.items()])
    rows = []
    for mode, g in df[df.status.isin(["ok", "budget"])].groupby("mode"):
        x = np.c_[np.ones(len(g)), g.steps, 2.0 ** g.log2_flops]
        coef, _ = nnls(x / g["mean"].to_numpy()[:, None], np.ones(len(g)))  # relative error
        pred = np.c_[np.ones(len(feats)), feats.steps, 2.0 ** feats.log2_flops] @ coef
        feasible = feats.log2_flops < 40
        full = (warmup + repeats) * pred[feasible]
        capped = warmup * pred[feasible] + np.minimum(repeats * pred[feasible], 5.0 + pred[feasible])
        rows.append(dict(mode=mode, plans=len(feats), parse_errors=len(errors), feasible=int(feasible.sum()),
                         a_s=coef[0], b_s_per_step=coef[1], c_s_per_flop=coef[2],
                         fit_median_rel_err=float(np.median(np.abs(x @ coef / g["mean"] - 1))),
                         gpu_hours_50=full.sum() / 3600, gpu_hours_budget5=capped.sum() / 3600,
                         largest_plan_s=float(pred[feasible].max())))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run = Path(argv[0])
    df = load_run(run)
    df["release_speedup"] = df.plan.map(release_speedups())
    for s in STATS:
        df = add_speedups(df, s, f"sp_{s}")
    df = add_speedups(df, "mock_time", "sp_mock")
    j = load_julia(run)
    if j is not None:
        df = df.merge(j, on="plan", how="left")
        df = add_speedups(df, "julia_mean", "sp_julia")
    cols = [c for c in df.columns if c not in ("tensors",)]
    df[cols].to_csv(OUT / "p4_timing.csv", index=False)
    pc = per_circuit(df)
    pc.to_csv(OUT / "p4_compare.csv", index=False)
    env = json.loads((run / "environment.json").read_text())
    print(f"job {env.get('slurm_job')} on {env.get('node')}, {env.get('gpu')}, cupy {env.get('cupy')}")
    print(df.groupby(["mode", "status"]).size().to_string())
    s = pc[pc.plans >= 3].groupby(["mode", "ours"]).agg(
        circuits=("circuit", "size"), spearman_median=("spearman_release", "median"),
        spearman_mean=("spearman_release", "mean"), best_same=("best_same", "mean"))
    print(s.to_string())
    est = estimate(df)
    est.to_csv(OUT / "p4_estimate.csv", index=False)
    print(est.to_string(index=False))


if __name__ == "__main__":
    main()
