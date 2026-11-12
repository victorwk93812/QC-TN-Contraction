"""The authors' train/test/OOD splits (Data/splits in their release)."""
import warnings

import pandas as pd

from qctn.paths import authors_repo

# Data/splits/splits.md in the authors' release.
FEATURES = {
    "ndcg": [
        "std_cost", "n_steps", "frac_tiny", "max_out_rank", "avg_out_rank", "costw_out_rank",
        "p_at_max_cost_val", "max_red_rank_val", "k_at_max_cost_val", "max_asym_val",
        "avg_asym", "costw_asym_val", "d_at_max_cost_val",
    ],
    "pair": [
        "n_steps", "frac_tiny", "max_out_rank", "avg_out_rank", "costw_out_rank",
        "p_at_max_cost_val", "max_red_rank_val", "costw_red_rank_val", "k_at_max_cost_val",
        "max_asym_val", "avg_asym", "costw_asym_val", "d_at_max_cost_val",
    ],
}
SPLIT_FILES = {"train": "df_final", "id": "df_test_n", "ood": "df_ood_n"}
SPLITS = tuple(SPLIT_FILES)


def load_released(split: str, objective: str) -> pd.DataFrame:
    """One split as published: z-scored features plus `speedup`, `base_id`, `label`."""
    path = authors_repo() / "Data" / "splits" / f"{SPLIT_FILES[split]}_{objective}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(path, engine="openpyxl")


def load_split(split: str, objective: str, variant: str = "released", dups: str | None = None) -> pd.DataFrame:
    """One split, "released" as published or "corrected" (qctn.rank.corrected).

    `dups` (qctn.rank.duplicates): None leaves the split as it is; "flag" adds the duplicate-candidate
    columns; "drop" also removes the copies of another plan of the same circuit; "drop_all" also the
    repeated listings of a plan.
    """
    if variant == "released":
        df = load_released(split, objective)
    elif variant == "corrected":
        from qctn.rank.corrected import load_corrected
        df = load_corrected(split, objective)
    else:
        raise ValueError(f"variant must be 'released' or 'corrected', not {variant!r}")
    if dups is None:
        return df
    from qctn.rank.duplicates import apply
    return apply(df, split, dups)
