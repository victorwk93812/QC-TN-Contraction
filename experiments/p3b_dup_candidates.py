"""Phase 3b: duplicate candidates in the splits, and the phase 2b benchmark with and without them.

Writes, in results/:
  p3b_flags.csv      every split row with a duplicate flag (qctn.rank.duplicates), its speedup and label,
                     and `same_tree_as`: other plans of the circuit with the same contraction tree
  p3b_benchmark.csv  phase 2b's protocol (one model per objective, rounds by grouped 5-fold CV on train only,
                     fold seeds 42, 1, 2, both opt_ndcg parameter sets) on the released and corrected variants,
                     each as it is (dups "none"), without the MinFill and other plan copies ("drop"), and also
                     without the repeated listings of a plan ("drop_all"); plus MinFill and random baselines
  p3b_summary.csv    min-max of p3b_benchmark.csv over fold seeds and parameter sets
See src/notes/deviations.md, "Phase 3b".
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2b_corrected_splits import MAX_ROUNDS, NTHREAD, PARAMS, SEEDS, ranges  # noqa: E402
from t06_t07 import expected_baselines  # noqa: E402

from qctn.plans.authors import split_name  # noqa: E402
from qctn.plans.generators import leaf_sets  # noqa: E402
from qctn.rank.data import FEATURES, SPLITS, load_split  # noqa: E402
from qctn.rank.duplicates import all_flags, step_list  # noqa: E402
from qctn.rank.metrics import evaluate  # noqa: E402
from qctn.rank.train import cv_rounds, fit, predict  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
VARIANTS = ("released", "corrected")
DUPS = {"none": None, "drop": "drop", "drop_all": "drop_all"}


def flagged() -> pd.DataFrame:
    f = all_flags()
    # Weaker than equal step lists: the same tree (leaf sets), whatever the step order and names.
    trees = {n: leaf_sets(step_list(n)) for n in f.circuit_name.unique()}
    circuit = f.circuit_name.map(lambda n: split_name(n)[0])
    same_tree = {}
    for _, names in f.groupby(circuit).circuit_name:
        names = sorted(set(names))
        for n in names:
            same_tree[n] = " ".join(split_name(m)[1] for m in names if m != n and trees[m] == trees[n])
    f["same_tree_as"] = f.circuit_name.map(same_tree)
    keep = f.dup_of_minfill | (f.dup_group != "") | f.repeated_row | (f.same_tree_as != "")
    return f[keep]


def benchmark() -> pd.DataFrame:
    rows = []
    for obj in FEATURES:
        for vname in VARIANTS:
            for dname, dups in DUPS.items():
                d = {s: load_split(s, obj, vname, dups) for s in SPLITS}
                for pname, params in PARAMS[obj].items():
                    for seed in SEEDS:
                        r, _ = cv_rounds(d["train"], obj, params, MAX_ROUNDS, seed=seed, nthread=NTHREAD)
                        mdl = fit(d["train"], obj, params, r, nthread=NTHREAD)
                        for s in ("id", "ood"):
                            sc = predict(mdl, d[s], obj)
                            for target in ("label", "speedup"):
                                rows.append(dict(variant=vname, dups=dname, model=f"opt_{obj}", params=pname,
                                                 seed=seed, rounds=r, split=s, target=target,
                                                 rows_train=len(d["train"]), rows_eval=len(d[s]),
                                                 **vars(evaluate(d[s], sc, target))))
    for dname, dups in DUPS.items():
        for s in ("id", "ood"):
            df = load_split(s, "ndcg", "released", dups)
            is_mf = df.circuit_name.str.endswith("_pla_mf").astype(float).to_numpy()
            base = dict(variant="-", dups=dname, params="-", seed="-", rounds="-", split=s, rows_train="-",
                        rows_eval=len(df))
            for target in ("label", "speedup"):
                rows.append(dict(base, model="minfill", target=target, **vars(evaluate(df, is_mf, target))))
            rand, mf_rand = expected_baselines(df)  # exact expectations, on the label
            for name, m in (("random_expected", rand), ("mf_rand_expected", mf_rand)):
                rows.append(dict(base, model=name, target="label", **vars(m)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    OUT.mkdir(exist_ok=True)
    f = flagged()
    f.to_csv(OUT / "p3b_flags.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f.drop(columns="base_id").to_string(index=False))
        print(f[f.dup_of_minfill].groupby(["split", "dup_cause"]).size().to_string())
    b = benchmark()
    b.round(6).to_csv(OUT / "p3b_benchmark.csv", index=False)
    m = b[b.model.str.startswith("opt_")].assign(variant=lambda x: x.variant + "/" + x.dups)
    ranges(m).round(6).to_csv(OUT / "p3b_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(ranges(m[m.target == "label"]).round(4).to_string(index=False))
        print(b[~b.model.str.startswith("opt_")].round(4).to_string(index=False))
