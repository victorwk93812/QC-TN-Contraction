"""Duplicate candidates: split rows whose plan is a copy of another plan of the same circuit.

Phase 3 (src/notes/deviations.md, "Time limits and MinFill fallbacks") found release plans of other generators
that are step for step the MinFill plan of their circuit. Here every row of the split files is compared with
the other plans of its circuit in the splits and with the circuit's `_pla_mf` plan file, on the step list
(the `A B C` of each line, so intermediate names included); the timings make every file differ byte for byte.

Flags, one per split row (`flags`):

    dup_of_minfill  the plan is the circuit's MinFill plan, and the row isn't the `_pla_mf` row
    dup_cause       for those: "timeout" (large network, solver stopped at the budget) or "unexplained"
    dup_group       "<circuit>/<k>" shared by rows of one circuit with identical step lists, "" if none
    repeated_row    the same plan name already appeared earlier in the split (RQC_3_7_40, RQC_3_7_52)
    dup_drop        the copies `dups="drop"` removes: every row of a dup_group but one, kept in this order:
                    the `_pla_mf` row, else the first in `GENERATORS` order

`apply` adds them to a split, and drops rows if asked. The flags depend only on the plan files and the split
membership, so they are the same for both objectives and both variants.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from qctn.paths import authors_plans
from qctn.plans.authors import GENERATORS, split_name
from qctn.rank.data import SPLITS, load_released

FLAG_COLUMNS = ("dup_of_minfill", "dup_cause", "dup_group", "repeated_row", "dup_drop")
DUPS = (None, "flag", "drop", "drop_all")

# The release has no `_pla_mf` plan for this circuit; phase 3's regenerated MinFill plan (which reproduces every
# release `mf` plan) equals these three (results/p3_generators_all.csv, choice "authors", theirs_is_minfill).
MINFILL_WITHOUT_FILE = {
    "pricingcall_nativegates_ibm_qiskit_opt3_25": ("jdrasil", "minfill_mrs", "minfillbg_mrs"),
}
# Phase 3's MinFill copies on large networks (2849 to 29499 tensors), where our run of the same solver is also
# stopped at the 10 s budget. The rest (on 193 to 881 tensors) have no explanation from the release.
TIMEOUT_LIKE = frozenset({
    *(f"{c}_pla_minfillbg_mrs" for c in ("GROVER_9", "GROVER_11", "GROVER_13", "QNN_25", "QWALK_6", "QWALK_7",
                                         "pricingcall_nativegates_ibm_qiskit_opt3_21",
                                         "pricingcall_nativegates_ibm_qiskit_opt3_23",
                                         "pricingcall_nativegates_ibm_qiskit_opt3_25")),
    "pricingcall_nativegates_ibm_qiskit_opt3_25_pla_minfill_mrs",
    "pricingcall_nativegates_ibm_qiskit_opt3_23_pla_jdrasil",
    "pricingcall_nativegates_ibm_qiskit_opt3_25_pla_jdrasil",
})


def step_list(name: str) -> tuple[tuple[str, str, str], ...] | None:
    """The (A, B, C) of every line of a plan file, None if there is no file. Reads files `read_plan` rejects."""
    path = authors_plans() / f"resultats_{name}.txt"
    if not path.exists():
        return None
    return tuple(tuple(line.split(" ", 3)[:3]) for line in path.read_text().splitlines())


def _gen_order(name: str) -> int:
    gen = split_name(name)[1]
    return GENERATORS.index(gen) if gen in GENERATORS else len(GENERATORS)


def _flags_for(names: list[str]) -> pd.DataFrame:
    """Flags for the rows of one split, in order."""
    out = pd.DataFrame({"circuit_name": names})
    out["repeated_row"] = out.circuit_name.duplicated()
    out["dup_of_minfill"] = False
    out["dup_cause"] = ""
    out["dup_group"] = ""
    out["dup_drop"] = False
    circuit = out.circuit_name.map(lambda n: split_name(n)[0])
    for c, idx in out.groupby(circuit, sort=True).groups.items():
        unique = sorted(set(out.circuit_name[idx]), key=lambda n: (_gen_order(n), n))
        steps = {n: step_list(n) for n in unique}
        mf = steps.get(f"{c}_pla_mf") or step_list(f"{c}_pla_mf")
        is_mf_copy = {n: (n != f"{c}_pla_mf" and (
            (mf is not None and steps[n] == mf) or split_name(n)[1] in MINFILL_WITHOUT_FILE.get(c, ())))
            for n in unique}
        classes: dict = {}
        for n in unique:
            classes.setdefault(steps[n], []).append(n)
        k = 0
        group, drop = {}, set()
        for members in classes.values():
            if len(members) < 2:
                continue
            k += 1
            keep = f"{c}_pla_mf" if f"{c}_pla_mf" in members else members[0]
            for n in members:
                group[n] = f"{c}/{k}"
                if n != keep:
                    drop.add(n)
        for i in idx:
            n = out.at[i, "circuit_name"]
            if is_mf_copy[n]:
                out.at[i, "dup_of_minfill"] = True
                out.at[i, "dup_cause"] = "timeout" if n in TIMEOUT_LIKE else "unexplained"
            out.at[i, "dup_group"] = group.get(n, "")
            out.at[i, "dup_drop"] = n in drop
    return out


@lru_cache(maxsize=None)
def _flags(split: str) -> pd.DataFrame:
    return _flags_for(list(load_released(split, "ndcg").circuit_name))


def flags(split: str) -> pd.DataFrame:
    """`circuit_name` and the flag columns for every row of a split, in split-file order."""
    return _flags(split).copy()


def apply(df: pd.DataFrame, split: str, dups: str | None) -> pd.DataFrame:
    """Add the flags to one split (rows in split-file order), and drop rows for "drop" / "drop_all".

    "drop" removes the `dup_drop` rows; "drop_all" also the repeated listings of a plan (`repeated_row`).
    Nothing else changes: the corrected variant keeps the scaler fitted on all its training rows.
    """
    if dups not in DUPS:
        raise ValueError(f"dups must be one of {DUPS}, not {dups!r}")
    if dups is None:
        return df
    f = flags(split)
    if len(f) != len(df) or not (f.circuit_name.to_numpy() == df.circuit_name.to_numpy()).all():
        raise ValueError(f"{split}: rows are not those of the split file")
    df = df.copy()
    for col in FLAG_COLUMNS:
        df[col] = f[col].to_numpy()
    if dups == "drop":
        df = df[~df.dup_drop]
    elif dups == "drop_all":
        df = df[~(df.dup_drop | df.repeated_row)]
    return df


def all_flags() -> pd.DataFrame:
    """Flags of every split row, with `split`, `speedup` and `label`."""
    parts = []
    for s in SPLITS:
        d = load_released(s, "ndcg")[["circuit_name", "base_id", "speedup", "label"]].reset_index(drop=True)
        parts.append(pd.concat([d, flags(s).drop(columns="circuit_name")], axis=1).assign(split=s))
    return pd.concat(parts, ignore_index=True)
