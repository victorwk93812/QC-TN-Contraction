"""Phase 2b: corrected splits (qctn.rank.corrected)."""
import numpy as np
import pytest

from qctn import paths
from qctn.features import FEATURE_NAMES
from qctn.plans.authors import split_name
from qctn.rank import corrected
from qctn.rank.data import FEATURES, SPLITS, load_split

pytestmark = pytest.mark.skipif(not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first")


@pytest.fixture(scope="module", params=["ndcg", "pair"])
def splits(request):
    o = request.param
    return o, {s: load_split(s, o) for s in SPLITS}, {s: load_split(s, o, "corrected") for s in SPLITS}


def test_same_rows_labels_and_assignment(splits):
    _, rel, cor = splits
    for s in SPLITS:
        cols = ["circuit_name", "base_id", "speedup", "label"]
        assert rel[s][cols].equals(cor[s][cols])


def test_jdrasil_rows_no_longer_carry_minfill_features(splits):
    o, rel, cor = splits
    released_as_mf = corrected_as_mf = 0
    for s in SPLITS:
        for d, count in ((rel[s], "rel"), (cor[s], "cor")):
            x = d.set_index("circuit_name")[FEATURE_NAMES].astype(float)
            for name in x.index[x.index.str.endswith("_pla_jdrasil")]:
                mf = split_name(name)[0] + "_pla_mf"
                if mf in x.index and np.allclose(x.loc[name], x.loc[mf], rtol=1e-9):
                    if count == "rel":
                        released_as_mf += 1
                    else:
                        corrected_as_mf += 1
    # 213 carry MinFill's features (one of them, RANDOM_11, has no MinFill row in the splits), plus the
    # six below, plus RQC_4_6_34 and RQC_6_20, whose split rows match no Features.xlsx row.
    assert released_as_mf == 212 + 6 + 2
    # Six circuits whose Jdrasil and MinFill plan files give equal features (GROVER_3, QNN_3, QWALK_3, ...).
    assert corrected_as_mf == 6


def test_scaler_is_fitted_on_train_only(splits):
    o, _, cor = splits
    tr = cor["train"][FEATURES[o]].astype(float)
    np.testing.assert_allclose(tr.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(tr.std(ddof=1), 1, atol=1e-12)
    for s in ("id", "ood"):  # not refitted on the test splits
        assert np.abs(cor[s][FEATURES[o]].astype(float).mean()).max() > 1e-3


def test_unscaled_columns_are_raw_plan_features(splits):
    o, _, cor = splits
    raw = [f for f in FEATURE_NAMES if f not in FEATURES[o]]
    table = corrected.plan_feature_table()
    for s in SPLITS:
        d = cor[s]
        np.testing.assert_allclose(d[raw].to_numpy(float), table.loc[d.circuit_name, raw].to_numpy(float))


def test_released_scaler_round_trip(splits):
    o, rel, _ = splits
    back = corrected.build(o, fix="none", scaler="released")
    for s in SPLITS:
        np.testing.assert_allclose(back[s][FEATURES[o]].to_numpy(float), rel[s][FEATURES[o]].to_numpy(float),
                                   atol=1e-9)


def test_every_row_has_a_recorded_feature_source(splits):
    _, _, cor = splits
    for s in SPLITS:
        assert cor[s].feature_source.isin(["file", "xlsx:file_differs", "xlsx:unparsable", "xlsx:no_file"]).all()


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        load_split("train", "ndcg", "fixed")
