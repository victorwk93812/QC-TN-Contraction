"""Tables 6 and 7 of arXiv 2608.05819 from the authors' published splits, on CPU.

Writes results/t06_t07.csv: what the authors' recipe gives, how the published numbers
are composed, and what one model per objective gives with rounds picked by grouped
cross-validation on the training split alone. See src/notes/deviations.md.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from qctn.paths import authors_repo
from qctn.rank.data import load_split
from qctn.rank.metrics import Metrics, evaluate
from qctn.rank.train import NOTEBOOK, TABLE4, cv_rounds, fit, predict

OUT = Path(__file__).resolve().parents[1] / "results" / "t06_t07.csv"
MAX_ROUNDS = 300
PAPER = {  # Top-1, Top-3, Regret1, Regret3, Win
    ("ndcg", "id"): (60.00, 96.00, 0.0141, 0.0004, 92.00),
    ("pair", "id"): (60.00, 92.00, 0.0211, 0.0004, 88.00),
    ("ndcg", "ood"): (38.71, 62.90, 0.0520, 0.0181, 73.39),
    ("pair", "ood"): (30.65, 62.90, 0.0800, 0.0234, 87.10),
}
BASELINES = {  # Table 6 and Table 7 rows
    ("random", "id"): (22.95, 58.44, 0.0699, 0.0137, 70.16),
    ("mf_ndcg", "id"): (12.00, 84.00, 0.0735, 0.0025, 50.00),
    ("mf_rand", "id"): (12.00, 28.00, 0.0735, 0.0319, 50.00),
    ("random", "ood"): (27.87, 54.65, 0.0775, 0.0265, 70.97),
    ("mf_ndcg", "ood"): (8.06, 61.29, 0.0784, 0.0189, 50.00),
    ("mf_rand", "ood"): (8.06, 43.55, 0.0784, 0.0273, 50.00),
}
TOL = (0.006, 0.006, 0.00006, 0.00006, 0.006)


def row(recipe, objective, split, rounds, target, m: Metrics, paper=None):
    out = dict(recipe=recipe, objective=objective, split=split, rounds=rounds, target=target,
               top1=m.top1, top3=m.top3, regret1=m.regret1, regret3=m.regret3, win=m.win)
    for name, value in zip(("top1", "top3", "regret1", "regret3", "win"), paper or (None,) * 5):
        out[f"paper_{name}"] = value
    return out


def expected_baselines(df: pd.DataFrame):
    """Exact expectations: uniform random order, and MinFill first then random."""
    rand, mf_rand = [], []
    for _, g in df.groupby("base_id", sort=False):
        y, sp = g["label"].to_numpy(float), g["speedup"].to_numpy(float)
        is_mf = g["circuit_name"].str.endswith("_pla_mf").to_numpy()
        n, best, bi = len(y), y.max(), int(np.argmax(y))
        win = np.mean((sp > 1) + 0.5 * (sp == 1))
        triples = list(itertools.combinations(range(n), min(3, n)))
        rand.append((1 / n, min(3, n) / n, np.mean(1 - y / best),
                     np.mean([1 - y[list(c)].max() / best for c in triples]), win))
        m = int(np.argmax(is_mf))
        rest = [i for i in range(n) if i != m]
        with_mf = [(m, *c) for c in itertools.combinations(rest, min(2, len(rest)))]
        mf_rand.append((float(bi == m), np.mean([bi in c for c in with_mf]), 1 - y[m] / best,
                        np.mean([1 - y[list(c)].max() / best for c in with_mf]), 0.5))
    pack = lambda v: Metrics(100 * np.mean([x[0] for x in v]), 100 * np.mean([x[1] for x in v]),
                             np.mean([x[2] for x in v]), np.mean([x[3] for x in v]),
                             100 * np.mean([x[4] for x in v]))
    return pack(rand), pack(mf_rand)


def rounds_matching(results: list[Metrics], paper, fields: tuple[int, ...]) -> list[int]:
    names = ("top1", "top3", "regret1", "regret3", "win")
    return [r + 1 for r, m in enumerate(results)
            if all(abs(getattr(m, names[i]) - paper[i]) < TOL[i] for i in fields)]


def main() -> None:
    rows, composition = [], []
    for objective, recipe_rounds in (("ndcg", (1, 65)), ("pair", (16, 14))):
        train = load_split("train", objective)
        test = load_split("id", objective)
        ood = load_split("ood", objective)
        model = fit(train, objective, NOTEBOOK[objective], MAX_ROUNDS)

        # The authors' recipe: one model for Top-1/Regret1, another for Top-3/Regret3.
        for r in recipe_rounds:
            m = evaluate(test, predict(model, test, objective, r), "label")
            rows.append(row("notebook", objective, "id", r, "label", m, PAPER[(objective, "id")]))

        # Their stored model against the same recipe fitted on all 225 circuits.
        stored = xgb.Booster()
        stored.load_model(authors_repo() / "models" / f"opt_{objective}.json")
        n_stored = stored.num_boosted_rounds()
        everything = fit(pd.concat([train, test, ood]), objective, NOTEBOOK[objective], n_stored)
        for split, df in (("id", test), ("ood", ood)):
            s = predict(stored, df, objective)
            rows.append(row("stored", objective, split, n_stored, "label", evaluate(df, s, "label"),
                            PAPER[(objective, split)]))
            a = predict(everything, df, objective)
            rows.append(row("fit_on_all", objective, split, n_stored, "label", evaluate(df, a, "label"),
                            PAPER[(objective, split)]))
            composition.append(dict(
                objective=objective, split=split, check="stored_vs_fit_on_all",
                value=f"spearman={pd.Series(s).corr(pd.Series(a), method='spearman'):.6f} "
                      f"max_abs_diff={np.abs(s - a).max():.3g}"))

        # Which rounds and training sets give each published number.
        for split, df, trains in (("id", test, {"train": train}),
                                  ("ood", ood, {"train": train, "train+id": pd.concat([train, test])})):
            paper = PAPER[(objective, split)]
            for name, tr in trains.items():
                for params_name, params in (("notebook", NOTEBOOK[objective]), ("table4", TABLE4[objective])):
                    mdl = fit(tr, objective, params, MAX_ROUNDS)
                    res = [evaluate(df, predict(mdl, df, objective, r), "label") for r in range(1, MAX_ROUNDS + 1)]
                    base = dict(objective=objective, split=split, train=name, params=params_name)
                    for label, fields in (("top1+regret1", (0, 2)), ("top3+regret3", (1, 3)),
                                          ("all_four", (0, 1, 2, 3)), ("win", (4,))):
                        composition.append({**base, "check": label,
                                            "value": str(rounds_matching(res, paper, fields)[:20])})

        # One model, rounds from the training split alone: three fold assignments.
        for params_name, params in (("notebook", NOTEBOOK[objective]), ("table4", TABLE4[objective])):
            for seed in (42, 1, 2):
                r, _ = cv_rounds(train, objective, params, MAX_ROUNDS, seed=seed)
                mdl = fit(train, objective, params, r)
                for split, df in (("id", test), ("ood", ood)):
                    s = predict(mdl, df, objective)
                    for target in ("label", "speedup"):
                        rows.append(row(f"single_cv_{params_name}_seed{seed}", objective, split, r, target,
                                        evaluate(df, s, target), PAPER[(objective, split)]))

    for split in ("id", "ood"):
        df = load_split(split, "ndcg")
        rand, mf_rand = expected_baselines(df)
        rows.append(row("random_expected", "-", split, "-", "label", rand, BASELINES[("random", split)]))
        rows.append(row("mf_rand_expected", "-", split, "-", "label", mf_rand, BASELINES[("mf_rand", split)]))

    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(OUT, index=False)
    pd.DataFrame(composition).to_csv(OUT.with_name("t06_t07_composition.csv"), index=False)


if __name__ == "__main__":
    main()
