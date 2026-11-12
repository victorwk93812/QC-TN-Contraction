"""Phase 4 full run: compare our GPU times with the release's labels, per circuit, by split, family and size.

    PYTHONPATH=src .venv/bin/python experiments/p4full_compare.py results/raw/p4full/<tag>

Reads `<run>/circuits/*.jsonl` and `<run>/tasks/*.json` (from `p4full_timing.py run`) and
`results/p4full_plans.csv` (plans never run are listed with status `not run`). The release's labels are the
speedups of the split files (`Data/splits`, RTX A6000), with `qctn.rank.duplicates` flags (`dups="flag"`). Writes

- `results/p4full_timing.csv`: one row per plan and mode: status, repeats, mean / median / min / CV of the
  repeats, plan size, node and GPU, our speedups t(MinFill) / t(plan) in the same mode (NaN without a timed `_pla_mf`
  plan), the mock-time speedup (sum of the plan file's time column), the release's speedup, split and dup flags;
- `results/p4full_compare.csv`: one row per circuit, mode, statistic (`mean`, `median`, `min`; `mock` for
  the plan files' time column on every labelled plan, `mock_same` on the plans we timed in that mode) and plan set (`all` labelled plans, or `nodup`: without the rows
  `dups="drop"` removes): plans compared, Spearman and Kendall (tau-b) between our speedups and the labels,
  whether the best plan is the same, the regret of our best plan under the labels (1 - label(ours) / max label)
  and the mean regret of a uniformly random pick;
- `results/p4full_summary.csv`: those per-circuit numbers summarised over all circuits with at least 3 plans,
  and by split, family and network size;
- `results/p4full_tasks.csv`: one row per Slurm task: node, GPU, its utilisation and memory at start and end,
  the node's other jobs, circuits done and seconds.

Rank statistics and regret don't need the MinFill plan (speedups share the circuit's MinFill time), so circuits
without a timed MinFill plan are compared too.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
STATS = ("mean", "median", "min")
SIZE_BINS = [0, 500, 1000, 2500, np.inf]
SIZE_LABELS = ["<=500", "501-1000", "1001-2500", ">2500"]


def labels() -> pd.DataFrame:
    """One row per labelled plan: release speedup, split and duplicate flags (first listing of repeated rows)."""
    from qctn.rank.data import SPLITS, load_split

    d = pd.concat([load_split(s, "ndcg", dups="flag").assign(split=s) for s in SPLITS])
    d = d[~d.repeated_row]
    return d.rename(columns={"circuit_name": "plan", "speedup": "release_speedup"})[
        ["plan", "release_speedup", "split", "dup_of_minfill", "dup_cause", "dup_group", "dup_drop"]]


def family(circuit: str) -> str:
    from qctn.catalog import candidates

    c = candidates(circuit)
    return c[0].family if c else ""


def load_run(run: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((run / "circuits").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            t = np.asarray(r.pop("times", None) or [], float)
            r.pop("wall", None)
            w = r.pop("warmup", None) or []
            g = r.pop("gpu_before", None) or {}
            r.update(n=len(t), warmup_s=w[0] if w else np.nan,
                     **{s: (getattr(np, s)(t) if len(t) else np.nan) for s in STATS},
                     cv=(t.std(ddof=1) / t.mean()) if len(t) > 1 else np.nan,
                     gpu_util_before=g.get("utilization.gpu"), gpu_mem_before=g.get("memory.used"))
            res = r.pop("result", None)
            r["result_re"], r["result_im"] = res if isinstance(res, list) else (np.nan, np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def result_spread(df: pd.DataFrame) -> pd.Series:
    """Per plan: |amplitude - median amplitude of its network (all plans, both modes)| / |median|. Every plan of
    a network contracts the same amplitude, so this checks the run (complex64)."""
    key = df.circuit + "|" + df.source.astype(str)
    re = df.result_re.astype(float)
    im = df.result_im.astype(float)
    med = re.groupby(key).transform("median") + 1j * im.groupby(key).transform("median")
    return np.abs(re + 1j * im - med) / np.abs(med)


def with_planned(df: pd.DataFrame) -> pd.DataFrame:
    """Add the inventory's plans that the run has no row for, as `not run`."""
    inv = pd.read_csv(OUT / "p4full_plans.csv")
    key = set(zip(df.plan, df["mode"])) if len(df) else set()
    miss = inv[[(p, m) not in key for p, m in zip(inv.plan, inv["mode"])]].copy()
    miss["status"] = np.where(miss.skip.fillna("") != "", "skipped: " + miss.skip.fillna(""), "not run")
    cols = [c for c in miss.columns if c in ("plan", "circuit", "generator", "source", "mode", "tensors", "steps",
                                              "mock_time", "ops", "log2_flops", "peak_gb", "max_rank", "pred_run_s",
                                              "status")]
    return pd.concat([df, miss[cols]], ignore_index=True)


def add_speedups(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.status.isin(["ok", "budget"])
    mf = df[ok & (df.generator == "mf")].set_index(["circuit", "mode"])
    key = list(zip(df.circuit, df["mode"]))
    for s in STATS:
        ref = np.array([mf[s].get(k, np.nan) for k in key], float)
        df[f"sp_{s}"] = np.where(ok, ref / df[s], np.nan)
    ref = np.array([mf.mock_time.get(k, np.nan) for k in key], float)
    df["sp_mock"] = ref / df.mock_time
    return df


def compare_one(x: np.ndarray, y: np.ndarray) -> dict:
    """x: our score (higher is better), y: release speedups."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(x, y).statistic
        tau = kendalltau(x, y).statistic
    ib, rb = int(np.argmax(x)), int(np.argmax(y))
    return dict(spearman=rho, kendall=tau, best_same=bool(y[ib] == y[rb]), regret=1 - y[ib] / y[rb],
                regret_random=float(np.mean(1 - y / y[rb])), ours_best_label=y[ib], release_best_label=y[rb])


def per_circuit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lab = df[df.release_speedup.notna()]
    for (c, mode), g in lab.groupby(["circuit", "mode"]):
        meta = dict(circuit=c, mode=mode, split=g.split.iloc[0], family=g.family.iloc[0],
                    tensors=int(g.tensors.max()), size=g.size_bin.iloc[0], labelled=len(g),
                    dup_of_minfill=int(g.dup_of_minfill.sum()), has_mf=bool((g.generator == "mf").any()),
                    networks=g.source.nunique())
        for variant, h0 in (("all", g), ("nodup", g[~g.dup_drop.astype(bool)])):
            for ours in STATS + ("mock", "mock_same"):
                col = ours if ours in STATS else "mock_time"
                timed = h0.status.isin(["ok", "budget"])
                ok = h0.mock_time.notna() if ours == "mock" else timed
                h = h0[ok & np.isfinite(h0[col].astype(float))]
                row = dict(meta, ours=ours, variant=variant, plans=len(h), not_timed=len(h0) - len(h))
                if len(h) >= 2:
                    row.update(compare_one(1.0 / h[col].to_numpy(float), h.release_speedup.to_numpy(float)))
                    row["best_ours"] = h.generator.iloc[int(np.argmin(h[col].to_numpy(float)))]
                    row["best_release"] = h.generator.iloc[int(np.argmax(h.release_speedup.to_numpy(float)))]
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(pc: pd.DataFrame, min_plans: int = 3) -> pd.DataFrame:
    pc = pc[pc.plans >= min_plans]
    out = []
    for by in (None, "split", "family", "size"):
        keys = ["mode", "ours", "variant"] + ([by] if by else [])
        for k, g in pc.groupby(keys):
            row = dict(zip(keys, k))
            row.update(group_by=by or "all", group=row.pop(by) if by else "all", circuits=len(g),
                       plans=int(g.plans.sum()),
                       spearman_median=g.spearman.median(), spearman_mean=g.spearman.mean(),
                       spearman_pos=float((g.spearman > 0).mean()),
                       kendall_median=g.kendall.median(), kendall_mean=g.kendall.mean(),
                       best_same=g.best_same.mean(), regret_median=g.regret.median(), regret_mean=g.regret.mean(),
                       regret_random_mean=g.regret_random.mean())
            out.append(row)
    return pd.DataFrame(out)


def _gres(alloc: str | None) -> str | None:
    """'GresUsed=gpu:V100:3(IDX:0-2)' from the node's scontrol line: GPUs allocated on the node, by index."""
    return next((x.split("=", 1)[1] for x in (alloc or "").split() if x.startswith("GresUsed=")), None)


def tasks(run: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((run / "tasks").glob("*.json")):
        t = json.loads(p.read_text())
        e, s, f = t["env"], t.get("start", {}), t.get("end", {})
        g0, g1 = (s.get("gpus") or [{}])[0], (f.get("gpus") or [{}])[0]
        rows.append(dict(shard=t["shard"], job=e.get("slurm_job"), array_job=e.get("array_job"),
                         partition=e.get("partition"), node=e.get("node"), gpu_index_node=e.get("slurm_gpus"),
                         pci_bus=e.get("pci_bus"), gpu=e.get("gpu"),
                         util_start=g0.get("utilization.gpu"), mem_start_mib=g0.get("memory.used"),
                         util_end=g1.get("utilization.gpu"), mem_end_mib=g1.get("memory.used"),
                         gres_used_start=_gres(s.get("node_alloc")), gres_used_end=_gres(f.get("node_alloc")),
                         node_jobs_start=len(s.get("node_jobs") or []),
                         node_jobs_end=len(f.get("node_jobs") or []) if f else np.nan,
                         circuits=len(t.get("circuits", [])), already_done=len(t.get("skipped_done", [])),
                         seconds=t.get("seconds"), finished="end" in t))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run = Path(argv[0])
    df = with_planned(load_run(run))
    df = df.merge(labels(), on="plan", how="left")
    fam = {c: family(c) for c in df.circuit.unique()}
    df["family"] = df.circuit.map(fam)
    size = df.groupby("circuit").tensors.transform("max")
    df["size_bin"] = pd.cut(size, SIZE_BINS, labels=SIZE_LABELS).astype(str)
    df = add_speedups(df)
    df["result_rel_dev"] = result_spread(df)
    for c in ("dup_of_minfill", "dup_drop"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    df = df.sort_values(["circuit", "plan", "mode"])
    tk = tasks(run)
    if len(tk):  # the GPU as the node's index (from the task), not its UUID
        gpu_of = dict(zip(tk.job.astype(str), tk.gpu_index_node))
        df.insert(df.columns.get_loc("node") + 1, "gpu_index", df.job.map(lambda j: gpu_of.get(str(j).split(".")[0])))
    df.drop(columns=["gpu_uuid"], errors="ignore").to_csv(OUT / "p4full_timing.csv", index=False)
    pc = per_circuit(df)
    pc.to_csv(OUT / "p4full_compare.csv", index=False)
    sm = summarise(pc)
    sm.to_csv(OUT / "p4full_summary.csv", index=False)
    tk.to_csv(OUT / "p4full_tasks.csv", index=False)

    st = df.status.fillna("").str.split(":").str[:2].str.join(":")
    print(df.assign(st=st).groupby(["mode", "st"]).size().to_string())
    pd.set_option("display.width", 200)
    cols = ["mode", "ours", "variant", "group", "circuits", "plans", "spearman_median", "spearman_mean",
            "kendall_median", "best_same", "regret_median", "regret_random_mean"]
    print(sm[sm.group_by.isin(["all", "split"])][cols].round(3).to_string(index=False))
    print(f"amplitude spread: max {df.result_rel_dev.max():.2e}, plans over 1e-3: {(df.result_rel_dev > 1e-3).sum()}")
    ok = df[df.status.isin(["ok", "budget"])]
    both = ok.pivot_table(index=["circuit", "plan"], columns="mode", values="mean")
    if {"dense", "hyper"} <= set(both):
        b = both.dropna().reset_index()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = b.groupby("circuit").apply(lambda g: spearmanr(g.dense, g.hyper).statistic if len(g) >= 3
                                             else np.nan, include_groups=False)
        print(f"dense vs hyper over {len(b)} plans: per-circuit Spearman median {rho.median():.3f} "
              f"({rho.notna().sum()} circuits), median hyper/dense time {np.median(b.hyper / b.dense):.2f}")
    if len(tk):
        print(f"tasks {len(tk)} ({tk.finished.sum()} finished), nodes {sorted(tk.node.dropna().unique())}, "
              f"GPU-hours {tk.seconds.fillna(0).sum() / 3600:.2f}")


if __name__ == "__main__":
    main()
