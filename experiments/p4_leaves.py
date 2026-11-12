"""Phase 4 check: our leaf tensors against the authors' (QXTools / QXTns, dumped by `julia/p4_leaves.jl`).

    JULIA_DEPOT_PATH=<external>/julia-depot julia +1.9.1 --project=<external>/julia-env/qx \\
        experiments/julia/p4_leaves.jl "" results/raw/p4/leaves <circuit>...
    PYTHONPATH=src .venv/bin/python experiments/p4_leaves.py results/raw/p4/leaves [<circuit>...]

For every tensor: the hyperindex groups (`qctn.execute.diagonal_groups` against QXTensor's `hyper_indices`) and
the stored data (ours reduced to the groups, column-major, against theirs), up to an overall sign (SVD halves).
Writes `results/p4_leaves.csv`, one row per circuit. Default circuits: every `<circuit>.leaves.csv` in the folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qctn import catalog
from qctn.execute import diagonal_groups, leaf_arrays
from qctn.tn import build

ROOT = Path(__file__).resolve().parents[1]


def reduce(a: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """QXTns `reduce_tensor`: one axis per group (first position), the others removed."""
    rep = list(range(a.ndim))
    for g in groups:
        for k in g[1:]:
            rep[k] = g[0]
    keep = sorted(set(rep))
    letters = "abcdefghijklmnop"
    return np.einsum("".join(letters[r] for r in rep) + "->" + "".join(letters[k] for k in keep), a)


def compare(folder: Path, name: str) -> dict:
    c = catalog.candidates(name)[0].build()
    tn = build(c)
    arr = leaf_arrays(c, tn)
    theirs = pd.read_csv(folder / f"{name}.leaves.csv", keep_default_na=False).set_index("tensor")
    n_groups = n_data = 0
    worst = 0.0
    for t, ix in tn.tensors.items():
        r = theirs.loc[t]
        g = diagonal_groups(arr[t])
        n_groups += ";".join(" ".join(map(str, x)) for x in g) != r.groups or len(ix) != r["rank"]
        jd = np.array([complex(*map(float, x.split(":"))) for x in r.data.split()])
        od = reduce(arr[t], g).flatten(order="F")
        if od.shape != jd.shape:
            n_data += 1
            continue
        e = min(np.abs(od - jd).max(), np.abs(od + jd).max())
        worst = max(worst, float(e))
        n_data += e > 1e-12
    return dict(circuit=name, tensors=len(tn.tensors), theirs=len(theirs),
                with_groups=int(sum(bool(x) for x in theirs.groups)), groups_differ=int(n_groups),
                data_differ=int(n_data), max_abs_diff=worst)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0])
    names = argv[1:] or sorted(p.name[:-len(".leaves.csv")] for p in folder.glob("*.leaves.csv"))
    rows = [compare(folder, n) for n in names]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "p4_leaves.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
