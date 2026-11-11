"""Phase 2b: the phase 1 benchmark on the released and the corrected splits, and the steps between them.

Writes, in results/:
  p2b_rows.csv        every split row whose features change from the released to the corrected variant
  p2b_invariance.csv  the Notebook 5 recipe trained on standardised vs raw features: do the predictions differ?
                      Per feature too: un-standardise one column at a time, and check whether its float32
                      values (what the DMatrix stores) keep their order and ties
  p2b_benchmark.csv   one model per objective, rounds by grouped 5-fold CV on train only, 3 fold seeds,
                      both opt_ndcg parameter sets, on each variant; plus MinFill and random baselines
  p2b_summary.csv     min-max of p2b_benchmark.csv over fold seeds and parameter sets, per variant
See src/notes/deviations.md, "Phase 2b".
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from qctn.features import FEATURE_NAMES
from qctn.plans.authors import split_name
from qctn.rank.corrected import build
from qctn.rank.data import FEATURES, SPLITS, load_split
from qctn.rank.metrics import evaluate
from qctn.rank.train import NOTEBOOK, TABLE4, cv_rounds, fit, predict
from t06_t07 import expected_baselines  # experiments/ is on sys.path when run as a script

OUT = Path(__file__).resolve().parents[1] / "results"
MAX_ROUNDS = 300
NTHREAD = 6
SEEDS = (42, 1, 2)
# Each step changes one thing, so a difference between neighbours has one cause.
VARIANTS = {
    "released": None,                        # the split files as published
    "released_raw": ("none", "none"),        # same rows, standardisation undone
    "jdrasil_fix": ("jdrasil", "released"),  # Jdrasil rows get their own features, release's scaler
    "all_files": ("all", "released"),        # every row from its plan file, release's scaler
    "corrected": ("all", "train"),           # ... z-scored on the training rows (ddof 1)
    "corrected_raw": ("all", "none"),        # ... not standardised
}
PARAMS = {"ndcg": {"notebook": NOTEBOOK["ndcg"], "table4": TABLE4["ndcg"]},
          "pair": {"notebook": NOTEBOOK["pair"]}}  # Table 4's opt_pair is Notebook 5's


def variant(name: str, objective: str) -> dict[str, pd.DataFrame]:
    if VARIANTS[name] is None:
        return {s: load_split(s, objective) for s in SPLITS}
    return build(objective, *VARIANTS[name])


def changed_rows() -> pd.DataFrame:
    """Rows of the split files whose 18 raw features differ from their own plan file's."""
    rows = []
    for obj in FEATURES:
        rel, cor = build(obj, "none", "none"), build(obj, "all", "none")
        for s in SPLITS:
            a, b = rel[s], cor[s]
            diff = (np.abs(a[FEATURE_NAMES].to_numpy(float) - b[FEATURE_NAMES].to_numpy(float))
                    / np.maximum(1, np.abs(b[FEATURE_NAMES].to_numpy(float))))
            mf = {split_name(n)[0]: i for i, n in enumerate(a.circuit_name) if n.endswith("_pla_mf")}
            for i in np.flatnonzero(diff.max(axis=1) > 1e-6):
                name = a.circuit_name.iloc[i]
                circ, gen = split_name(name)
                as_mf = (gen == "jdrasil" and circ in mf and np.allclose(
                    a[FEATURE_NAMES].iloc[i].to_numpy(float), a[FEATURE_NAMES].iloc[mf[circ]].to_numpy(float),
                    rtol=1e-6))
                rows.append(dict(objective=obj, split=s, name=name, generator=gen,
                                 released_equals_mf=as_mf, max_rel_diff=diff[i].max(),
                                 features_changed=int((diff[i] > 1e-6).sum()),
                                 feature_source=b.feature_source.iloc[i]))
    return pd.DataFrame(rows)


def invariance() -> pd.DataFrame:
    """Notebook 5 recipe on standardised vs raw features of the same rows, rounds 1-300."""
    out = []
    for obj in FEATURES:
        for a_name, b_name in (("released", "released_raw"), ("corrected", "corrected_raw")):
            a, b = variant(a_name, obj), variant(b_name, obj)
            ma = fit(a["train"], obj, NOTEBOOK[obj], MAX_ROUNDS, nthread=NTHREAD)
            mb = fit(b["train"], obj, NOTEBOOK[obj], MAX_ROUNDS, nthread=NTHREAD)
            for s in ("id", "ood"):
                diffs, same_metrics = [], 0
                for r in range(1, MAX_ROUNDS + 1):
                    pa, pb = predict(ma, a[s], obj, r), predict(mb, b[s], obj, r)
                    diffs.append(np.abs(pa - pb).max())
                    same_metrics += evaluate(a[s], pa, "label") == evaluate(b[s], pb, "label")
                out.append(dict(objective=obj, pair=f"{a_name} vs {b_name}", split=s,
                                max_abs_pred_diff=max(diffs), rounds_identical=int(np.sum(np.array(diffs) == 0)),
                                rounds_same_metrics=same_metrics, rounds=MAX_ROUNDS))
        # One column at a time, released split files, predictions of the 300-round model on ID + OOD.
        z, raw = variant("released", obj), variant("released_raw", obj)
        base = fit(z["train"], obj, NOTEBOOK[obj], MAX_ROUNDS, nthread=NTHREAD)
        p0 = np.concatenate([predict(base, z[s], obj) for s in ("id", "ood")])
        all_z, all_raw = pd.concat(z.values()), pd.concat(raw.values())
        for f in FEATURES[obj]:
            a, b = all_z[f].to_numpy(np.float32), all_raw[f].to_numpy(np.float32)
            kept = (len(np.unique(a)) == len(np.unique(b))
                    and (np.argsort(a, kind="stable") == np.argsort(b, kind="stable")).all())
            d = {s: z[s].assign(**{f: raw[s][f]}) for s in SPLITS}
            m = fit(d["train"], obj, NOTEBOOK[obj], MAX_ROUNDS, nthread=NTHREAD)
            p1 = np.concatenate([predict(m, d[s], obj) for s in ("id", "ood")])
            out.append(dict(objective=obj, pair=f"released vs {f} raw", split="id+ood",
                            max_abs_pred_diff=np.abs(p1 - p0).max(), rounds=MAX_ROUNDS,
                            float32_order_and_ties_kept=kept,
                            distinct_float32_z=len(np.unique(a)), distinct_float32_raw=len(np.unique(b))))
    return pd.DataFrame(out)


def benchmark() -> pd.DataFrame:
    rows = []
    for obj in FEATURES:
        for vname in VARIANTS:
            d = variant(vname, obj)
            for pname, params in PARAMS[obj].items():
                for seed in SEEDS:
                    r, _ = cv_rounds(d["train"], obj, params, MAX_ROUNDS, seed=seed, nthread=NTHREAD)
                    mdl = fit(d["train"], obj, params, r, nthread=NTHREAD)
                    for s in ("id", "ood"):
                        sc = predict(mdl, d[s], obj)
                        for target in ("label", "speedup"):
                            m = evaluate(d[s], sc, target)
                            rows.append(dict(variant=vname, model=f"opt_{obj}", params=pname, seed=seed,
                                             rounds=r, split=s, target=target, **vars(m)))
    for s in ("id", "ood"):
        df = load_split(s, "ndcg")
        is_mf = df.circuit_name.str.endswith("_pla_mf").astype(float).to_numpy()
        for target in ("label", "speedup"):
            rows.append(dict(variant="-", model="minfill", params="-", seed="-", rounds="-", split=s,
                             target=target, **vars(evaluate(df, is_mf, target))))
        rand, mf_rand = expected_baselines(df)  # exact expectations, on the label
        for name, m in (("random_expected", rand), ("mf_rand_expected", mf_rand)):
            rows.append(dict(variant="-", model=name, params="-", seed="-", rounds="-", split=s,
                             target="label", **vars(m)))
    return pd.DataFrame(rows)


def ranges(b: pd.DataFrame) -> pd.DataFrame:
    m = b[b.model.str.startswith("opt_")]
    g = m.groupby(["model", "split", "target", "variant"], sort=False)
    return g.agg(**{f"{k}_{f}": (k, f) for k in ("top1", "top3", "regret1", "regret3", "win", "rounds")
                    for f in ("min", "max")}).reset_index()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    OUT.mkdir(exist_ok=True)
    ch = changed_rows()
    ch.to_csv(OUT / "p2b_rows.csv", index=False)
    print(ch.groupby(["objective", "generator", "released_equals_mf", "feature_source"]).size().to_string())
    inv = invariance()
    inv.to_csv(OUT / "p2b_invariance.csv", index=False)
    print(inv.to_string(index=False))
    b = benchmark()
    b.round(6).to_csv(OUT / "p2b_benchmark.csv", index=False)
    ranges(b).round(6).to_csv(OUT / "p2b_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(ranges(b[b.target == "label"]).round(4).to_string(index=False))
        print(b[~b.model.str.startswith("opt_")].round(4).to_string(index=False))
