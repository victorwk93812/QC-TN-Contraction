"""Phase 4b: what the release's labels correlate with, from the release's own data (no GPU).

    PYTHONPATH=src .venv/bin/python experiments/p4q_labels.py

Uses every split row (1475 rows, 225 circuits; the 12 repeated listings counted once except in the noise
estimate) and its plan file. Writes

- `results/p4q_label_stats.csv`: for each plan statistic, the within-circuit Spearman and Kendall correlation
  with the measured speedup (median and mean over circuits with at least 3 plans and a non-constant statistic)
  and how often the statistic's best plan is the labels' best plan. Signs: a positive correlation means a
  larger value goes with a larger speedup. Statistics: the 18 features (`qctn.features`, recomputed from the plan
  files), the plan files' time column (sum, median and mean per step, sum without the largest step), and, for
  comparison, simulated replicates of the labels themselves (`replicate`, see `noise`).
- `results/p4q_label_flops.csv`: pairs of plans of one circuit, binned by the difference in log2 flops (dense,
  from the files' ranks, and in QXTns' hyperindex form from `results/p4q_hyperflops.csv`, written by
  `p4q_hyperflops.py`, if present): how often the plan with more flops has the lower speedup, and the median
  speedup ratio.
- `results/p4q_label_noise.csv`: the groups of identical plans (MinFill copies and repeated listings, phase 3b)
  with their measured speedups, and the noise level they imply.
- `results/p4q_label_rescaled.csv`: circuits whose split-file speedups are `measures.xlsx`'s times a constant.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def within_circuit(df: pd.DataFrame, col: str, target: str = "speedup", group: str = "base_id",
                   min_plans: int = 3) -> pd.DataFrame:
    """Per group: Spearman and Kendall correlation of `col` with `target`, and whether the plan with the largest
    `col` is the plan with the largest `target` (first maximum, as `argmax`). Groups with fewer than `min_plans`
    rows or a constant `col` are left out."""
    rows = []
    for g, h in df.groupby(group, sort=True):
        x = h[col].to_numpy(float)
        y = h[target].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < min_plans or np.ptp(x) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(dict(group=g, plans=len(x), spearman=spearmanr(x, y).statistic,
                             kendall=kendalltau(x, y).statistic, best_same=bool(np.argmax(x) == np.argmax(y))))
    return pd.DataFrame(rows, columns=["group", "plans", "spearman", "kendall", "best_same"])


def summarise(pc: pd.DataFrame, **extra) -> dict:
    return dict(extra, circuits=len(pc), spearman_median=pc.spearman.median(), spearman_mean=pc.spearman.mean(),
                kendall_median=pc.kendall.median(), kendall_mean=pc.kendall.mean(), best_same=pc.best_same.mean())


def noise_sigma(log_diffs: np.ndarray) -> float:
    """Noise level of one measurement's log speedup from differences between two measurements of the same plan
    (each difference has variance 2 sigma^2): a robust estimate, 1.4826 * MAD / sqrt(2), with the median taken
    as 0 (the two measurements are exchangeable)."""
    d = np.abs(np.asarray(log_diffs, float))
    return float(1.4826 * np.median(d) / np.sqrt(2)) if len(d) else np.nan


def replicate(df: pd.DataFrame, sigma: float, rng: np.random.Generator, target: str = "speedup") -> np.ndarray:
    """A simulated second measurement of the labels: log speedup plus normal noise of variance 2 sigma^2 (the
    difference of two independent measurement errors). Correlating it with the labels approximates the
    test-retest agreement of the authors' measurement, slightly pessimistically (the labels' own spread already
    contains noise)."""
    return np.exp(np.log(df[target].to_numpy(float)) + rng.normal(0, np.sqrt(2) * sigma, len(df)))


def load_rows() -> pd.DataFrame:
    from qctn.features import plan_features
    from qctn.plans.authors import read_all
    from qctn.rank.data import SPLITS, load_split

    d = pd.concat([load_split(s, "ndcg").assign(split=s) for s in SPLITS])
    d["listing"] = d.groupby("circuit_name").cumcount()
    plans, _ = read_all()
    feats = {}
    for n in d.circuit_name.unique():
        p = plans.get(n)
        if p is None:
            continue
        t = np.array([s.time for s in p.steps])
        f = plan_features(p)
        f.update(time_sum=t.sum(), time_median_step=np.median(t), time_mean_step=t.mean(),
                 time_sum_wo_max=t.sum() - t.max(), time_max_step=t.max(), time_first_step=t[0])
        feats[n] = f
    f = pd.DataFrame.from_dict(feats, orient="index").rename_axis("circuit_name").reset_index()
    d = d[["circuit_name", "base_id", "speedup", "split", "listing"]].merge(f, on="circuit_name", how="left")
    d["generator"] = d.circuit_name.str.split("_pla_").str[1]
    hf = OUT / "p4q_hyperflops.csv"  # experiments/p4q_hyperflops.py
    if hf.exists():
        h = pd.read_csv(hf)
        h = h[h.status == "ok"][["plan", "log2_flops_hyper", "max_rank_hyper", "log2_peak_elements_hyper"]]
        d = d.merge(h.rename(columns={"plan": "circuit_name"}), on="circuit_name", how="left")
    return d


def noise_groups() -> pd.DataFrame:
    """Identical plans with their own measured speedups: MinFill copies (against MinFill's 1, or against each
    other where the release has no MinFill plan) and the second listing of repeated plans (against the first,
    after removing each listing's circuit-wide factor)."""
    from qctn.rank.duplicates import all_flags
    from qctn.rank.data import SPLITS, load_split

    fl = all_flags()
    rows = []
    for g, h in fl[fl.dup_group.notna()].groupby("dup_group"):
        sp = h.speedup.to_numpy(float)
        ref = h[~h.dup_of_minfill]
        base = float(ref.speedup.iloc[0]) if len(ref) else float(np.exp(np.log(sp).mean()))
        for _, r in h[h.dup_of_minfill].iterrows():
            rows.append(dict(kind="minfill_copy", group=g, base_id=r.base_id, plan=r.circuit_name,
                             cause=r.dup_cause, speedup=r.speedup, reference=base,
                             log_diff=np.log(r.speedup / base), has_mf_plan=bool(len(ref))))
    d = pd.concat([load_split(s, "ndcg") for s in SPLITS])
    rep = d[d.duplicated("circuit_name", keep=False)].copy()
    rep["listing"] = rep.groupby("circuit_name").cumcount()
    for b, h in rep.groupby("base_id"):
        w = h.pivot_table(index="circuit_name", columns="listing", values="speedup")
        ld = np.log(w[0] / w[1])
        ld = ld - ld.median()  # each listing has its own MinFill normaliser
        for n, v in ld.items():
            rows.append(dict(kind="repeated_listing", group=b, base_id=b, plan=n, cause="", speedup=w.loc[n, 1],
                             reference=w.loc[n, 0], log_diff=v, has_mf_plan=True))
    return pd.DataFrame(rows)


def rescaled() -> pd.DataFrame:
    """Circuits whose MinFill row in the splits has a speedup other than 1: the ratio split / measures.xlsx per
    plan (constant within a circuit means the same measurement, normalised differently)."""
    from qctn.paths import authors_repo
    from qctn.rank.data import SPLITS, load_split

    d = pd.concat([load_split(s, "ndcg").assign(split=s) for s in SPLITS]).drop_duplicates("circuit_name")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = pd.read_excel(authors_repo() / "Data" / "measurements" / "measures.xlsx")
    j = d.merge(m, on="circuit_name", how="left")
    mf = j[j.circuit_name.str.endswith("_pla_mf") & ((j.speedup - 1).abs() > 1e-9)]
    rows = []
    for b in mf.base_id:
        h = j[j.base_id == b]
        r = (h.speedup / h.speedup_mf).dropna()
        rows.append(dict(base_id=b, split=h.split.iloc[0], plans=len(h), in_measures=len(r),
                         mf_speedup_split=float(h[h.circuit_name.str.endswith("_pla_mf")].speedup.iloc[0]),
                         ratio_min=r.min() if len(r) else np.nan, ratio_max=r.max() if len(r) else np.nan,
                         constant=bool(r.max() - r.min() < 1e-6 * r.max()) if len(r) else None))
    return pd.DataFrame(rows)


def flop_pairs(one: pd.DataFrame, col: str, form: str, bins=(0, 2, 5, 10, 20, 100), big: float = 36) -> list[dict]:
    """All pairs of plans of one circuit with different flops `col` (log2), binned by the difference: the share
    of pairs whose plan with more flops has the lower speedup, and the median log2 speedup ratio (fewer-flops
    plan over more-flops plan; a GPU limited by arithmetic would give about the flop difference). Once for all
    pairs, once for pairs whose larger plan has at least 2^`big` flops."""
    pr = []
    for b, g in one.groupby("base_id"):
        lf, sp = g[col].to_numpy(float), g.speedup.to_numpy(float)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if not (np.isfinite(lf[i]) and np.isfinite(lf[j])) or lf[i] == lf[j]:
                    continue
                hi, lo = (i, j) if lf[i] > lf[j] else (j, i)
                pr.append(dict(base_id=b, dlf=lf[hi] - lf[lo], lf_max=lf[hi],
                               flops_agree=sp[hi] < sp[lo], log2_ratio=np.log2(sp[lo] / sp[hi])))
    pr = pd.DataFrame(pr)
    out = []
    for only_big, h in ((False, pr), (True, pr[pr.lf_max >= big])):
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = h[(h.dlf > lo) & (h.dlf <= hi)]
            out.append(dict(flops=form, larger_plan_at_least=f"2^{big}" if only_big else "any",
                            dlog2_flops_from=lo, dlog2_flops_to=hi, pairs=len(q), circuits=q.base_id.nunique(),
                            share_flops_agree=q.flops_agree.mean() if len(q) else np.nan,
                            median_log2_speedup_ratio=q.log2_ratio.median() if len(q) else np.nan,
                            median_dlog2_flops=q.dlf.median() if len(q) else np.nan))
    return out


STATS = ["time_sum", "time_median_step", "time_mean_step", "time_sum_wo_max", "time_max_step", "time_first_step"]


def main():
    d = load_rows()
    one = d[d.listing == 0]  # repeated listings once
    rows = []
    from qctn.features import FEATURE_NAMES

    hyper = [c for c in ("log2_flops_hyper", "max_rank_hyper", "log2_peak_elements_hyper") if c in one]
    for col in STATS + FEATURE_NAMES + hyper:
        src = ("plan file time column" if col.startswith("time") else
               "hyperindex form (p4q_hyperflops.csv)" if col in hyper else "feature")
        rows.append(summarise(within_circuit(one, col), statistic=col, source=src))
    noise = noise_groups()
    noise.to_csv(OUT / "p4q_label_noise.csv", index=False)
    cases = {"all copies": noise, "copies without pricingcall": noise[~noise.base_id.str.startswith("pricingcall")],
             "pricingcall copies": noise[noise.base_id.str.startswith("pricingcall")]}
    for k, h in cases.items():
        sig = noise_sigma(h.log_diff)
        rng = np.random.default_rng(0)
        reps = [summarise(within_circuit(one.assign(rep=replicate(one, sig, rng)), "rep"))
                for _ in range(200)]
        r = pd.DataFrame(reps).mean(numeric_only=True).to_dict()
        rows.append(dict(r, statistic=f"replicate ({k}, sigma {sig:.3f})", source="simulated label replicate"))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "p4q_label_stats.csv", index=False)

    fl = pd.DataFrame([r for form, col in (("dense", "log2_sum_flops_val"), ("hyper", "log2_flops_hyper"))
                       if col in one for r in flop_pairs(one, col, form)])
    fl.to_csv(OUT / "p4q_label_flops.csv", index=False)
    rs = rescaled()
    rs.to_csv(OUT / "p4q_label_rescaled.csv", index=False)

    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(fl.round(3).to_string(index=False))
    print(noise.groupby(["kind", noise.base_id.str.startswith("pricingcall").rename("pricingcall")])
          .log_diff.agg(["size", "median", lambda x: x.abs().median()]).round(3))
    print(rs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
