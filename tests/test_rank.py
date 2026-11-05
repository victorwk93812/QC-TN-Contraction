"""Phase 1: the ranker on the authors' published splits."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from pathlib import Path

from qctn import paths
from qctn.rank.data import load_split
from qctn.rank.metrics import competition_ranks, evaluate
from qctn.rank.train import NOTEBOOK, fit, predict, relevance

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_competition_ranks_share_ties_and_skip():
    assert competition_ranks(np.array([0.1, 0.9, 0.9, 0.5])).tolist() == [4, 1, 1, 3]


def test_relevance_grades_within_each_group():
    df = pd.DataFrame({"base_id": ["a", "a", "a", "b", "b"], "label": [0.2, 0.9, 0.5, 0.7, 0.1]})
    assert relevance(df).tolist() == [0, 2, 1, 1, 0]


def test_evaluate_on_a_small_group():
    df = pd.DataFrame({
        "base_id": ["a"] * 4,
        "speedup": [1.0, 2.0, 0.5, 1.5],
        "label": np.log1p([1.0, 2.0, 0.5, 1.5]),
    })
    m = evaluate(df, [4, 3, 2, 1])  # picks speedup 1.0, truly best plan ranked second
    assert (m.top1, m.top3, m.win) == (0, 100, 50)
    assert m.regret1 == pytest.approx(0.5)
    assert m.regret3 == 0


@needs_external
@pytest.mark.parametrize("objective,top1_rounds,top3_rounds,paper", [
    ("ndcg", 1, 65, (60.00, 96.00, 0.0141, 0.0004)),
    ("pair", 16, 14, (60.00, 92.00, 0.0211, 0.0004)),
])
def test_table6_comes_from_two_models_and_log_labels(objective, top1_rounds, top3_rounds, paper):
    train, test = load_split("train", objective), load_split("id", objective)
    model = fit(train, objective, NOTEBOOK[objective], max(top1_rounds, top3_rounds))
    first = evaluate(test, predict(model, test, objective, top1_rounds), "label")
    second = evaluate(test, predict(model, test, objective, top3_rounds), "label")
    assert (round(first.top1, 2), round(first.regret1, 4)) == (paper[0], paper[2])
    assert (round(second.top3, 2), round(second.regret3, 4)) == (paper[1], paper[3])
    # Neither model alone gives the published row.
    assert first.top3 < paper[1]
    assert second.top1 <= paper[0] and second.regret1 > paper[2]


@needs_external
@pytest.mark.parametrize("split", ["id", "ood"])
def test_stored_pair_model_is_the_recipe_fitted_on_all_splits(split):
    frames = [load_split(s, "pair") for s in ("train", "id", "ood")]
    stored = xgb.Booster()
    stored.load_model(paths.authors_repo() / "models" / "opt_pair.json")
    refit = fit(pd.concat(frames), "pair", NOTEBOOK["pair"], stored.num_boosted_rounds())
    df = load_split(split, "pair")
    np.testing.assert_allclose(predict(stored, df, "pair"), predict(refit, df, "pair"), atol=1e-4)


@needs_external
@pytest.mark.parametrize("objective", ["ndcg", "pair"])
def test_python_matches_the_authors_julia_stack(objective):
    """Same metrics as their own code path, run in Julia (results/julia_t06_t07.csv)."""
    julia = pd.read_csv(RESULTS / "julia_t06_t07.csv")
    julia = julia[(julia.objective == objective) & (julia.model == "notebook")]
    train = load_split("train", objective)
    model = fit(train, objective, NOTEBOOK[objective], int(julia["rounds"].astype(int).max()))
    for _, r in julia.iterrows():
        df = load_split(r["split"], objective)
        m = evaluate(df, predict(model, df, objective, int(r["rounds"])), "label")
        assert m.top1 == pytest.approx(r["top1"])
        assert m.top3 == pytest.approx(r["top3"])
        assert m.regret1 == pytest.approx(r["regret1"], abs=1e-9)
        assert m.regret3 == pytest.approx(r["regret3"], abs=1e-9)
