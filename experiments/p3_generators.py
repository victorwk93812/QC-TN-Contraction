"""Phase 3: regenerate the seven plans per circuit and compare them with the authors' plans.

    PYTHONPATH=src .venv/bin/python experiments/p3_generators.py [--budgets 10,60] [--jobs 3]
        [--generators fc,gn,...] [--circuits name,...] [--gn-max-tensors N]

For every circuit with a QASM file in the release (17; the network comes from `qctn.tn.build`) and every
generator, the plan is generated (`qctn.plans.generators.generate`), written in the authors' text format to
`results/raw/p3/<budget>s/` (gitignored), and compared with the release's plan of the same name: identical step
list, identical tree (same leaf sets), log2 of summed flops, max step cost, width (largest result rank), and
the 18 features. The five tree-decomposition solvers run once per time budget (stopped at the budget, MinFill
fallback if no decomposition, as the release's plans indicate); mf and gn are deterministic and run once. Writes `results/p3_generators.csv`.

`plan_rows(name, tn, ...)` takes any network, so circuits without a QASM file in the release can be added once
`qctn.tn` builds them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qctn.circuits import authors_gates, circuit_path
from qctn.features import FEATURE_NAMES, plan_features
from qctn.paths import authors_circuits, authors_plans
from qctn.plans.authors import GENERATORS, PlanFormatError, read_plan, write_plan
from qctn.plans.generators import TD_SOLVERS, generate, leaf_sets, to_authors_plan
from qctn.plans import pace
from qctn.tn import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RAW = OUT / "raw" / "p3"


def qasm_circuits() -> list[str]:
    """Circuits with both a QASM file and at least one plan in the release."""
    have = {p.stem for p in authors_circuits().glob("*.qasm")}
    names = {p.name[len("resultats_"):].rsplit("_pla_", 1)[0] for p in authors_plans().glob("resultats_*_pla_*.txt")}
    return sorted(have & names)


def theirs(circuit: str, gen: str):
    p = authors_plans() / f"resultats_{circuit}_pla_{gen}.txt"
    if not p.exists():
        return None, "no plan in release"
    try:
        return read_plan(p), ""
    except PlanFormatError as e:
        return None, f"unparsable: {e}"


def summary(plan) -> dict:
    f = plan_features(plan)
    return {"steps": len(plan.steps), "log2_flops": f["log2_sum_flops_val"], "max_cost": f["max_cost"],
            "width": f["max_out_rank"]}, f


def one(task) -> dict:
    circuit, gen, budget = task
    tn = build(authors_gates(circuit_path(circuit)))
    return plan_row(circuit, tn, gen, budget)


def plan_row(circuit: str, tn, gen: str, budget: float | None) -> dict:
    """Generate `gen` on `tn` and compare with the release's plan for `circuit` (if any)."""
    row = {"circuit": circuit, "generator": gen, "budget_s": budget, "tensors": len(tn.tensors)}
    try:
        g = generate(tn, gen, time_limit=budget or 10.0)
    except Exception as e:  # report and go on
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    ours = to_authors_plan(tn, g.steps, f"{circuit}_pla_{gen}")
    d = RAW / (f"{budget:g}s" if budget else "det")
    d.mkdir(parents=True, exist_ok=True)
    write_plan(ours, d / f"resultats_{circuit}_pla_{gen}.txt")
    so, fo = summary(ours)
    row.update({f"ours_{k}": v for k, v in so.items()})
    row.update(ours_td_width=g.width, ours_fallback=g.fallback, ours_elapsed_s=round(g.elapsed, 2),
               ours_meta=json.dumps(g.meta, sort_keys=True))
    tp, why = theirs(circuit, gen)
    row["theirs_status"] = why or "ok"
    if tp is not None:
        st, ft = summary(tp)
        row.update({f"theirs_{k}": v for k, v in st.items()})
        a = [(s.a, s.b, s.c) for s in tp.steps]
        row["identical"] = a == g.steps
        row["same_tree"] = leaf_sets(a) == leaf_sets(g.steps)
        diffs = [abs(fo[k] - ft[k]) / max(1.0, abs(ft[k])) for k in FEATURE_NAMES
                 if not (np.isnan(fo[k]) and np.isnan(ft[k]))]
        row["features_max_rel_diff"] = max(diffs) if diffs else 0.0
        row["d_log2_flops"] = so["log2_flops"] - st["log2_flops"]
        row["d_width"] = so["width"] - st["width"]
    return row


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per generator and budget: plans compared, identical, same tree, cost and width differences."""
    d = df[df.get("identical").notna()] if "identical" in df else df.iloc[:0]
    d = d.assign(budget=d.budget_s.fillna(-1), identical=d.identical.astype(bool), same_tree=d.same_tree.astype(bool))
    g = d.groupby(["generator", "budget"])
    out = pd.DataFrame({
        "compared": g.size(), "identical": g.identical.sum(), "same_tree": g.same_tree.sum(),
        "ours_fallback": g.ours_fallback.sum(),
        "cheaper": g.d_log2_flops.apply(lambda x: int((x < -1e-9).sum())),
        "costlier": g.d_log2_flops.apply(lambda x: int((x > 1e-9).sum())),
        "median_d_log2_flops": g.d_log2_flops.median(), "max_abs_d_log2_flops": g.d_log2_flops.apply(lambda x: x.abs().max()),
        "max_abs_d_width": g.d_width.apply(lambda x: x.abs().max()),
    }).reset_index()
    out["budget"] = out.budget.map(lambda b: "" if b < 0 else f"{b:g}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="10,60", help="seconds before the five solvers are stopped")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--generators", default=",".join(GENERATORS))
    ap.add_argument("--circuits", default="")
    ap.add_argument("--gn-max-tensors", type=int, default=10000,
                    help="skip gn above this many tensors (Girvan-Newman is O(m^2 n))")
    ap.add_argument("--out", default=str(OUT / "p3_generators.csv"))
    a = ap.parse_args()
    circuits = a.circuits.split(",") if a.circuits else qasm_circuits()
    gens = a.generators.split(",")
    budgets = [float(b) for b in a.budgets.split(",")]
    sizes = {c: len(build(authors_gates(circuit_path(c))).tensors) for c in circuits}
    tasks, skipped = [], []
    for c in circuits:
        for g in gens:
            if g in TD_SOLVERS:
                if not pace.available(g):
                    skipped.append(dict(circuit=c, generator=g, error="solver not built"))
                    continue
                tasks += [(c, g, b) for b in budgets]
            elif g == "gn" and sizes[c] > a.gn_max_tensors:
                skipped.append(dict(circuit=c, generator=g, tensors=sizes[c],
                                    error=f"skipped: more than {a.gn_max_tensors} tensors"))
            else:
                tasks.append((c, g, None))
    # long tasks first
    tasks.sort(key=lambda t: -(sizes[t[0]] * (50 if t[1] == "gn" else 1) + (t[2] or 0) * 1000))
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, tasks):
            rows.append(r)
            print(r["circuit"], r["generator"], r["budget_s"], r.get("identical"), r.get("same_tree"),
                  r.get("d_log2_flops"), r.get("d_width"), r.get("error", ""), flush=True)
    df = pd.DataFrame(rows + skipped).sort_values(["circuit", "generator", "budget_s"], na_position="first")
    out = Path(a.out)
    if out.exists() and (a.circuits or a.generators != ",".join(GENERATORS)):
        old = pd.read_csv(out)
        key = ["circuit", "generator", "budget_s"]
        old = old[~old.set_index(key).index.isin(df.set_index(key).index)]
        df = pd.concat([old, df]).sort_values(key, na_position="first")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    print(summarise(df).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    sys.exit(main())
