"""Phase 0: the pinned third-party data and the Python stack are in place."""
import subprocess

import numpy as np
import pandas as pd
import pytest
import xgboost

from qctn import paths

GENERATORS = {"fc", "gn", "jdrasil", "mf", "minfill_mrs", "minfillbg_mrs", "tamaki"}

needs_external = pytest.mark.skipif(
    not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first"
)


@needs_external
def test_authors_repo_is_pinned():
    head = subprocess.run(
        ["git", "-C", paths.authors_repo(), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == paths.AUTHORS_COMMIT


@needs_external
def test_unpacked_plans_and_circuits():
    plans = list(paths.authors_plans().glob("*.txt"))
    assert len(plans) == 1663
    assert len(list(paths.authors_circuits().glob("*"))) == 257
    names = {p.stem.rsplit("_pla_", 1)[-1] for p in plans if "_pla_" in p.stem}
    assert GENERATORS <= names


@needs_external
@pytest.mark.parametrize("model,trees", [("opt_ndcg", 2600), ("opt_pair", 1850)])
def test_authors_models_load(model, trees):
    booster = xgboost.Booster()
    booster.load_model(paths.authors_repo() / "models" / f"{model}.json")
    assert booster.num_boosted_rounds() == trees
    assert booster.num_features() == 13


@needs_external
@pytest.mark.parametrize("split,groups", [("df_final", 138), ("df_test_n", 25), ("df_ood_n", 62)])
@pytest.mark.parametrize("objective", ["ndcg", "pair"])
def test_split_group_counts_match_table_3(split, groups, objective):
    path = paths.authors_repo() / "Data" / "splits" / f"{split}_{objective}"
    df = pd.read_excel(path, engine="openpyxl")
    assert df["base_id"].nunique() == groups


@needs_external
def test_labels_are_natural_log():
    """The paper leaves the base of y = log(1 + Sp) open; the authors' data settles it."""
    path = paths.authors_repo() / "Data" / "splits" / "df_test_n_ndcg"
    df = pd.read_excel(path, engine="openpyxl")
    np.testing.assert_allclose(df["label"], np.log1p(df["speedup"]))


def test_cupy_runs_on_gpu():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip(f"no usable GPU: {err}")
    a = cupy.ones((64, 64), dtype=cupy.complex64)
    assert complex(cupy.einsum("ij,jk->ik", a, a)[0, 0]) == 64
