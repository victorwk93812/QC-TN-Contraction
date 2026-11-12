"""Phase 3b: duplicate candidates (qctn.rank.duplicates)."""
from pathlib import Path

import pandas as pd
import pytest

from qctn import paths
from qctn.plans.authors import split_name
from qctn.rank import duplicates
from qctn.rank.data import SPLITS, load_released, load_split

pytestmark = pytest.mark.skipif(not paths.authors_repo().is_dir(), reason="run scripts/setup_external.sh first")
RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def flags():
    return duplicates.all_flags()


def test_counts(flags):
    mf = flags[flags.dup_of_minfill]
    assert mf.groupby("split").size().to_dict() == {"train": 16, "id": 1}
    assert mf.groupby("dup_cause").size().to_dict() == {"timeout": 12, "unexplained": 5}
    assert set(mf[mf.dup_cause == "unexplained"].circuit_name.map(lambda n: split_name(n)[0])) == {"RQC_5_16"}
    assert flags.dup_drop.sum() == 16  # pricingcall_..._25 has no MinFill row, so one of its three copies stays
    assert flags.repeated_row.sum() == 12
    # No copies other than of MinFill: every dup_group holds a MinFill plan or copies of it.
    g = flags[flags.dup_group != ""]
    assert (g.dup_of_minfill | g.circuit_name.str.endswith("_pla_mf")).all()


def test_agrees_with_phase3(flags):
    """Every phase 3 MinFill copy that is a split row is flagged, and nothing else."""
    p3 = pd.read_csv(RESULTS / "p3_generators_all.csv")
    p3 = p3[(p3.choice == "authors") & (p3.theirs_is_minfill == True)]  # noqa: E712
    names = set(p3.circuit + "_pla_" + p3.generator)
    assert set(flags[flags.dup_of_minfill].circuit_name) == names & set(flags.circuit_name)
    assert duplicates.TIMEOUT_LIKE <= names


def test_flags_are_deterministic(flags):
    duplicates._flags.cache_clear()
    again = duplicates.all_flags()
    pd.testing.assert_frame_equal(flags, again)


@pytest.mark.parametrize("variant", ["released", "corrected"])
@pytest.mark.parametrize("objective", ["ndcg", "pair"])
def test_default_unchanged_and_drop_removes_exactly_the_flagged_rows(variant, objective):
    for s in SPLITS:
        base = load_split(s, objective, variant)
        if variant == "released":
            pd.testing.assert_frame_equal(base, load_released(s, objective))
        assert not set(duplicates.FLAG_COLUMNS) & set(base.columns)
        flagged = load_split(s, objective, variant, "flag")
        pd.testing.assert_frame_equal(flagged[base.columns], base)
        drop = load_split(s, objective, variant, "drop")
        pd.testing.assert_frame_equal(drop, flagged[~flagged.dup_drop])
        drop_all = load_split(s, objective, variant, "drop_all")
        pd.testing.assert_frame_equal(drop_all, flagged[~(flagged.dup_drop | flagged.repeated_row)])
        # Every circuit keeps at least one row, and no circuit holds two plans with one step list after "drop".
        assert set(drop.base_id) == set(base.base_id)
        assert not drop.groupby("base_id").dup_group.apply(lambda x: x[x != ""].duplicated().any()).any()


def test_unknown_dups_is_rejected():
    with pytest.raises(ValueError):
        load_split("train", "ndcg", dups="remove")
