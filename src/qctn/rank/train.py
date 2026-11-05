"""Training as in the authors' `Getting_model`.

Pinned to xgboost 2.1.4 (pyproject.toml): predictions then match their Julia stack
(XGBoost_jll 2.1.5) to about 5e-8, while 3.4.1 trains measurably different models.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from qctn.rank.data import FEATURES

# Their notebook 5 (Preliminar_getting_results.jl). opt_ndcg differs from Table 4 of the
# paper, which gives eta 0.01 and subsample 0.7; `TABLE4["ndcg"]` holds those values.
NOTEBOOK = {
    "ndcg": {"objective": "rank:ndcg", "eval_metric": "ndcg@3", "eta": 0.02,
             "max_depth": 6, "subsample": 0.8, "seed": 42},
    "pair": {"objective": "rank:pairwise", "eval_metric": "ndcg@3", "eta": 0.05, "max_depth": 5,
             "subsample": 0.5, "colsample_bytree": 0.5, "min_child_weight": 3, "gamma": 0.5,
             "lambda": 0.0, "seed": 42},
}
TABLE4 = {"ndcg": {**NOTEBOOK["ndcg"], "eta": 0.01, "subsample": 0.7}, "pair": NOTEBOOK["pair"]}


def relevance(df: pd.DataFrame) -> np.ndarray:
    """Integer grade per group: 0 for the worst label up to n-1 for the best."""
    def grade(s: pd.Series) -> np.ndarray:
        order = np.argsort(s.astype("float32").to_numpy(), kind="stable")
        return np.argsort(order, kind="stable")
    return df.groupby("base_id", sort=False)["label"].transform(grade).to_numpy()


def matrix(df: pd.DataFrame, objective: str) -> xgb.DMatrix:
    return xgb.DMatrix(df[FEATURES[objective]].to_numpy(dtype=float))


def ranking_matrix(df: pd.DataFrame, objective: str) -> xgb.DMatrix:
    """Training matrix with groups by `base_id`; `df` must be sorted by it."""
    dm = matrix(df, objective)
    dm.set_label(relevance(df))
    dm.set_group(df.groupby("base_id", sort=False).size().to_numpy())
    return dm


def fit(train: pd.DataFrame, objective: str, params: dict, rounds: int, nthread: int = 8) -> xgb.Booster:
    """Train on `train`, sorted by `base_id` the way their `sort!` does."""
    train = train.sort_values("base_id", kind="stable")
    return xgb.train({**params, "nthread": nthread}, ranking_matrix(train, objective), rounds)


def predict(model: xgb.Booster, df: pd.DataFrame, objective: str, rounds: int | None = None) -> np.ndarray:
    """Scores for each row, optionally from the first `rounds` trees only."""
    limit = (0, rounds) if rounds else (0, 0)
    return model.predict(matrix(df, objective), iteration_range=limit)


def cv_rounds(train: pd.DataFrame, objective: str, params: dict, max_rounds: int = 300,
              folds: int = 5, seed: int = 42, nthread: int = 8) -> tuple[int, pd.Series]:
    """Round count with the best mean NDCG@3 over grouped folds of the training split only."""
    groups = train["base_id"].drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(groups)
    scores = []
    for held_out in np.array_split(groups, folds):
        is_val = train["base_id"].isin(held_out)
        fold_train = train[~is_val].sort_values("base_id", kind="stable")
        fold_val = train[is_val].sort_values("base_id", kind="stable")
        log: dict = {}
        xgb.train(
            {**params, "nthread": nthread}, ranking_matrix(fold_train, objective), max_rounds,
            evals=[(ranking_matrix(fold_val, objective), "val")], evals_result=log, verbose_eval=False,
        )
        scores.append(log["val"]["ndcg@3"])
    mean = pd.Series(np.mean(scores, axis=0), index=range(1, max_rounds + 1))
    return int(mean.idxmax()), mean
