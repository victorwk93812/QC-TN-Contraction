"""Corrected splits: the published circuits, assignment and labels, with features from the plan files.

The release's split files (Data/splits) contain, for most `_pla_jdrasil` rows, the features of the same
circuit's `_pla_mf` plan, 13 rows that match no `Features.xlsx` row, and features standardised by a fixed
z-score fitted on 773 rows we couldn't identify (src/notes/deviations.md, phase 2). Here every row keeps
its circuit, split, `speedup` and `label`, and gets its 18 features from `qctn.features` on its own plan
file; the model features are then z-scored with the training rows only (sample std, ddof 1).

`build` also gives the intermediate steps, so a change can be attributed to one cause:

    fix="none"     the split files' features, un-standardised with the release's scaler
    fix="jdrasil"  as "none", but `_pla_jdrasil` rows get their own plan's features
    fix="all"      every row from its plan file (Features.xlsx where the file can't be used)

    scaler="released"  the release's fixed mu, sigma (recovered per feature, `released_scaler`)
    scaler="train"     z-score fitted on the training rows of the variant
    scaler="none"      raw values
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from qctn.features import FEATURE_NAMES, plan_features
from qctn.paths import authors_plans, authors_repo
from qctn.plans.authors import PlanFormatError, read_plan, split_name
from qctn.rank.data import FEATURES, SPLITS, load_released

# Plan files that parse but whose features differ from Features.xlsx, with other step counts
# (deviations.md, phase 2): RQC_4_11_pla_mf has a single step, and the RQC_4_20 / RQC_4_60 files
# contain outer products that the release's reader can't handle. Features.xlsx must come from other
# versions of these files, so we take its rows for them.
FILE_DIFFERS_FROM_XLSX = (
    "RQC_4_11_pla_mf", "RQC_4_20_pla_fc", "RQC_4_20_pla_gn", "RQC_4_20_pla_mf",
    "RQC_4_60_pla_fc", "RQC_4_60_pla_mf",
)


def features_xlsx() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = pd.read_excel(authors_repo() / "Data" / "features" / "Features.xlsx")
    return x.set_index("circuit_name")[FEATURE_NAMES].astype(float)


@lru_cache(maxsize=None)
def plan_feature_table() -> pd.DataFrame:
    """The 18 features of every plan in the splits, with `feature_source` saying where each row is from."""
    names = sorted({n for s in SPLITS for o in FEATURES for n in load_released(s, o).circuit_name})
    xlsx = features_xlsx()
    rows = []
    for name in names:
        if name in FILE_DIFFERS_FROM_XLSX:
            rows.append({**xlsx.loc[name].to_dict(), "name": name, "feature_source": "xlsx:file_differs"})
            continue
        try:
            plan = read_plan(authors_plans() / f"resultats_{name}.txt")
            rows.append({**plan_features(plan), "name": name, "feature_source": "file"})
        except (OSError, PlanFormatError) as e:
            if name not in xlsx.index:
                raise ValueError(f"{name}: no usable plan file and no Features.xlsx row") from e
            reason = "unparsable" if isinstance(e, PlanFormatError) else "no_file"
            rows.append({**xlsx.loc[name].to_dict(), "name": name, "feature_source": f"xlsx:{reason}"})
    return pd.DataFrame(rows).set_index("name")


@lru_cache(maxsize=None)
def released_scaler(objective: str) -> pd.DataFrame:
    """The release's mu and sigma per model feature, from the split rows that equal their own xlsx row.

    The split value is an affine map of the raw value (residual <= 1e-14, results/p2_scaler.csv).
    """
    xlsx = features_xlsx()
    d = pd.concat([load_released(s, objective) for s in SPLITS], ignore_index=True)
    raw_cols = [f for f in FEATURE_NAMES if f not in FEATURES[objective]]
    own = xlsx.loc[d.circuit_name]
    same = np.isclose(own[raw_cols].to_numpy(), d[raw_cols].to_numpy(dtype=float), rtol=1e-9).all(axis=1)
    out = {}
    for f in FEATURES[objective]:
        r, z = own[f].to_numpy()[same], d[f].to_numpy(dtype=float)[same]
        a, b = np.polyfit(r, z, 1)
        # Rows that came from another plan (Jdrasil = MinFill) still pass the raw-column test above
        # when those columns coincide, so check the fit on the bulk of the rows, not all of them.
        if np.median(np.abs(a * r + b - z)) > 1e-6:  # max_asym_val: 1.6e-8 (rounding)
            raise AssertionError(f"{objective}/{f}: split values are not an affine map of Features.xlsx")
        out[f] = {"mu": -b / a, "sigma": 1 / a}
    return pd.DataFrame(out).T


def _unscale(d: pd.DataFrame, objective: str) -> pd.DataFrame:
    sc = released_scaler(objective)
    d = d.copy()
    for f in FEATURES[objective]:
        d[f] = d[f].astype(float) * sc.loc[f, "sigma"] + sc.loc[f, "mu"]
    return d


def build(objective: str, fix: str = "all", scaler: str = "train") -> dict[str, pd.DataFrame]:
    """The three splits for one objective. Columns as in the split files, plus `feature_source`."""
    if fix not in ("none", "jdrasil", "all") or scaler not in ("train", "released", "none"):
        raise ValueError(f"fix={fix!r}, scaler={scaler!r}")
    table = plan_feature_table()
    out = {}
    for split in SPLITS:
        d = _unscale(load_released(split, objective), objective)
        d["feature_source"] = "split_file"
        if fix == "none":
            take = np.zeros(len(d), dtype=bool)
        elif fix == "jdrasil":
            take = d.circuit_name.map(lambda n: split_name(n)[1] == "jdrasil").to_numpy()
        else:
            take = np.ones(len(d), dtype=bool)
        src = table.loc[d.circuit_name[take]]
        d.loc[take, FEATURE_NAMES] = src[FEATURE_NAMES].to_numpy()
        d.loc[take, "feature_source"] = src["feature_source"].to_numpy()
        out[split] = d
    if scaler == "none":
        return out
    if scaler == "released":
        sc = released_scaler(objective)
        mu, sigma = sc["mu"], sc["sigma"]
    else:
        tr = out["train"][FEATURES[objective]].astype(float)
        mu, sigma = tr.mean(), tr.std(ddof=1)
    for d in out.values():
        for f in FEATURES[objective]:
            d[f] = (d[f].astype(float) - mu[f]) / sigma[f]
    return out


@lru_cache(maxsize=None)
def _corrected(objective: str) -> dict[str, pd.DataFrame]:
    return build(objective, "all", "train")


def load_corrected(split: str, objective: str) -> pd.DataFrame:
    return _corrected(objective)[split].copy()
