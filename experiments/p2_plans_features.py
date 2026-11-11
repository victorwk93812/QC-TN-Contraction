"""Phase 2 checks against the authors' release; writes results/p2_*.csv.

1. Parse every plan file, write it back, compare bytes; build its contraction tree.
2. Compute the 18 features of every plan and compare with Features.xlsx.
3. Fit the split files' standardisation per feature (raw Features.xlsx value -> split value).
4. Build the network from QASM for every circuit that has both a QASM file and plans, and replay the plans.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from qctn.circuits import authors_gates, circuit_path
from qctn.features import FEATURE_NAMES, plan_features
from qctn.paths import authors_circuits, authors_plans, authors_repo
from qctn.plans.authors import PlanFormatError, format_plan, plan_files, plan_name, read_plan, split_name
from qctn.plans.tree import PlanError, PlanTree
from qctn.rank.data import FEATURES, load_split
from qctn.tn import build, replay

OUT = Path(__file__).resolve().parents[1] / "results"


def plans_and_features():
    xlsx = pd.read_excel(authors_repo() / "Data" / "features" / "Features.xlsx").set_index("circuit_name")
    rows = []
    for p in plan_files():
        name = plan_name(p)
        row = {"name": name, "in_xlsx": name in xlsx.index}
        try:
            plan = read_plan(p)
        except PlanFormatError as e:
            rows.append({**row, "parse": str(e).split(": ", 1)[1]})
            continue
        row.update(parse="ok", steps=len(plan.steps), roundtrip=format_plan(plan) == p.read_text())
        try:
            tree = PlanTree.from_steps((s.a, s.b, s.c) for s in plan.steps)
            row.update(tree="ok", roots=len(tree.roots))
        except PlanError as e:
            row.update(tree=str(e))
        row["outer_products"] = sum(s.n_common == 0 for s in plan.steps)
        if name in xlsx.index:
            ours = plan_features(plan)
            ref = xlsx.loc[name, FEATURE_NAMES].astype(float)
            err = max(abs(ours[f] - ref[f]) / max(1.0, abs(ref[f])) for f in FEATURE_NAMES)
            row.update(max_rel_err=err, xlsx_n_steps=int(ref["n_steps"]))
        rows.append(row)
    return pd.DataFrame(rows)


def scaler():
    """Per feature, the affine map raw -> split value, fitted on rows whose unscaled columns match."""
    x = pd.read_excel(authors_repo() / "Data" / "features" / "Features.xlsx").set_index("circuit_name")
    X = x[FEATURE_NAMES].astype(float)
    out, sources = [], []
    for obj in ("ndcg", "pair"):
        raw = [f for f in FEATURE_NAMES if f not in FEATURES[obj]]
        for split in ("train", "id", "ood"):
            d = load_split(split, obj)
            for _, r in d.iterrows():
                circ, gen = split_name(r.circuit_name)
                v = r[raw].astype(float).values
                if np.allclose(X.loc[r.circuit_name, raw].values, v, rtol=1e-9):
                    src = "self"
                elif gen == "jdrasil" and np.allclose(X.loc[f"{circ}_pla_mf", raw].values, v, rtol=1e-9):
                    src = "mf"
                else:
                    src = None
                sources.append(dict(objective=obj, split=split, name=r.circuit_name, source=src))
        s = pd.DataFrame([z for z in sources if z["objective"] == obj])
        allrows = pd.concat([load_split(sp, obj) for sp in ("train", "id", "ood")], ignore_index=True)
        ok = s.source.notna().values
        src_names = np.where(s.source == "mf", s.name.str.replace("_pla_jdrasil", "_pla_mf"), s.name)
        for f in FEATURES[obj]:
            r = X.loc[src_names[ok], f].values
            z = allrows.loc[ok, f].astype(float).values
            a, b = np.polyfit(r, z, 1)
            out.append(dict(objective=obj, feature=f, mu=-b / a, sigma=1 / a,
                            max_resid=np.abs(a * r + b - z).max(), rows=int(ok.sum())))
    return pd.DataFrame(out), pd.DataFrame(sources)


def networks():
    have = {p.stem for p in authors_circuits().glob("*.qasm")}
    rows = []
    for p in plan_files():
        circ, gen = split_name(plan_name(p))
        if circ not in have:
            continue
        try:
            plan = read_plan(p)
        except PlanFormatError:
            continue
        c = authors_gates(circuit_path(circ))
        tn = build(c)
        r = replay(tn, plan, max_report=1)
        rows.append(dict(name=plan_name(p), tensors=len(tn.tensors), steps=r.n_steps, match=r.ok,
                         unused=len(r.unused), skipped_lines=len(c.skipped),
                         first_mismatch=r.mismatches[0] if r.mismatches else ""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    OUT.mkdir(exist_ok=True)
    pf = plans_and_features()
    pf.to_csv(OUT / "p2_plans.csv", index=False)
    print(f"plans: {len(pf)} files, parsed {(pf.parse == 'ok').sum()}, round trip {pf.roundtrip.sum()}, "
          f"trees ok {(pf.tree == 'ok').sum()}")
    print(pf[pf.parse != "ok"][["name", "parse"]].to_string(index=False))
    print(pf[(pf.parse == "ok") & (pf.tree != "ok")][["name", "tree"]].to_string(index=False))
    cmp = pf[pf.max_rel_err.notna()]
    print(f"features: {len(cmp)} plans compared, {(cmp.max_rel_err < 1e-9).sum()} within 1e-9")
    print(cmp[cmp.max_rel_err >= 1e-9][["name", "steps", "xlsx_n_steps", "outer_products", "max_rel_err"]]
          .to_string(index=False))
    sc, src = scaler()
    sc.to_csv(OUT / "p2_scaler.csv", index=False)
    src.to_csv(OUT / "p2_split_sources.csv", index=False)
    print(sc.to_string(index=False))
    print(src.groupby(["objective", "source"], dropna=False).size().to_string())
    if "--no-tn" not in sys.argv:
        nt = networks()
        nt.to_csv(OUT / "p2_networks.csv", index=False)
        print(f"networks: {len(nt)} plans on {nt.name.map(lambda n: split_name(n)[0]).nunique()} circuits, "
              f"{nt.match.sum()} replay exactly")
        print(nt[~nt.match][["name", "first_mismatch"]].to_string(index=False))
