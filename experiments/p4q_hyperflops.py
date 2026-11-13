"""Phase 4b: flops of every split plan in QXTns' hyperindex form, the form the authors' stack contracts.

    PYTHONPATH=src .venv/bin/python experiments/p4q_hyperflops.py [--procs 6]

The plan files' ranks count every index, so `log2_sum_flops_val` is the dense count; QXTns contracts reduced
arrays, one axis per group of hyperindices (phase 4). For each plan of the split files that replays on its
network (`results/p3_networks.csv`, status "pass", network from its `source`), the network is built
(`qctn.tn.build`), its leaf values filled (`qctn.execute.leaf_arrays`) and the plan compiled in both forms
(`qctn.execute.compile_plan`, no GPU). Writes `results/p4q_hyperflops.csv`: plan, log2 flops dense and hyperindex,
largest stored rank, peak live elements (hyperindex).
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def work(task):
    from qctn.catalog import candidates
    from qctn.execute import compile_plan, leaf_arrays
    from qctn.paths import authors_plans
    from qctn.plans.authors import read_plan
    from qctn.tn import build

    # The executor's union-find (`_UF.find`) recurses along a chain of hyperindices; GHZ_999 needs more than
    # Python's default 1000 frames (Python >= 3.11 doesn't use the C stack for these calls).
    sys.setrecursionlimit(1_000_000)
    circuit, label, names = task
    cand = [c for c in candidates(circuit) if c.label() == label]
    rows = []
    if not cand:
        return [dict(plan=n, status="no candidate") for n in names]
    circ = cand[0].build()
    tn = build(circ)
    arrays = leaf_arrays(circ, tn)
    for n in names:
        plan = read_plan(authors_plans() / f"resultats_{n}.txt")
        row = dict(plan=n, network=label, tensors=len(tn.tensors), steps=len(plan.steps))
        try:
            d = compile_plan(tn, plan, hyper=False)
            h = compile_plan(tn, plan, hyper=True, arrays=arrays)
        except (ValueError, RecursionError) as e:
            row["status"] = f"compile: {e}"[:200]
            rows.append(row)
            continue
        row.update(status="ok", log2_flops_dense=float(np.log2(d.flops)), log2_flops_hyper=float(np.log2(h.flops)),
                   max_rank_dense=max(d.reduced_ranks, default=0), max_rank_hyper=max(h.reduced_ranks, default=0),
                   log2_peak_elements_hyper=float(np.log2(max(h.peak_elements, 1))))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)
    from qctn.rank.data import SPLITS, load_split

    names = pd.concat([load_split(s, "ndcg") for s in SPLITS]).circuit_name.unique()
    net = pd.read_csv(OUT / "p3_networks.csv").set_index("name")
    net = net.loc[[n for n in names if n in net.index]]
    missing = [n for n in names if n not in net.index]
    ok = net[net.status == "pass"]
    tasks = [(c, lab, list(g.index)) for (c, lab), g in ok.groupby(["circuit", "source"])]
    tasks.sort(key=lambda t: -len(t[2]))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.imap_unordered(work, tasks) for r in rs]
    rows += [dict(plan=n, status=f"replay {net.status[n]}") for n in net.index if net.status[n] != "pass"]
    rows += [dict(plan=n, status="no plan file") for n in missing]
    df = pd.DataFrame(rows).sort_values("plan")
    df.to_csv(OUT / "p4q_hyperflops.csv", index=False)
    print(df.status.value_counts().to_string())
    ok = df[df.status == "ok"]
    print("dense - hyper log2 flops: median", (ok.log2_flops_dense - ok.log2_flops_hyper).median(),
          "max", (ok.log2_flops_dense - ok.log2_flops_hyper).max())


if __name__ == "__main__":
    main()
