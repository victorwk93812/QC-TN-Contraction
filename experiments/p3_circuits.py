"""Build the circuit behind every plan in the authors' release and replay each plan on its network.

Writes results/circuits_map.csv (one row per circuit name in the plan files) and results/p3_networks.csv
(one row per plan file). A plan "passes" when every step's ranks and common indices (qubit tag, prime level,
dimension) agree with our network (`qctn.tn.replay`) and every tensor of the network is used; "partial" when the
steps agree but the plan leaves tensors of the network uncontracted.
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from qctn.catalog import candidates
from qctn.plans.authors import PlanFormatError, plan_files, plan_name, read_plan, split_name
from qctn.plans.tree import PlanError, PlanTree
from qctn.tn import build, replay

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    by_circuit = defaultdict(list)
    for p in plan_files():
        by_circuit[split_name(plan_name(p))[0]].append(p)
    rows, cmap = [], []
    for circ in sorted(by_circuit):
        cands = candidates(circ)
        nets = {}
        used = defaultdict(int)
        dropped = {}
        for p in by_circuit[circ]:
            name = plan_name(p)
            row = dict(name=name, circuit=circ, generator=split_name(name)[1] or "")
            try:
                plan = read_plan(p)
            except PlanFormatError as e:
                rows.append({**row, "status": "unparseable", "first_mismatch": str(e).split(": ", 1)[1]})
                continue
            try:
                tree = PlanTree.from_steps((s.a, s.b, s.c) for s in plan.steps)
                leaves, roots = len(tree.leaves), len(tree.roots)
            except PlanError as e:
                leaves, roots = None, None
                row["tree"] = str(e)
            row.update(steps=len(plan.steps), plan_leaves=leaves, roots=roots)
            first = None
            for i, c in enumerate(cands):
                if i not in nets:
                    circuit = c.build()
                    nets[i] = build(circuit)
                    dropped[c.label()] = Counter(re.match(r"[a-z_0-9]+", l)[0] for l in circuit.skipped)
                tn = nets[i]
                r = replay(tn, plan, max_report=1)
                if first is None:
                    first = (c, tn, r)
                if r.ok:
                    status = "pass" if not r.unused else "partial"
                    row.update(status=status, source=c.label(), tensors=len(tn.tensors), unused=len(r.unused),
                               first_mismatch="")
                    used[c.label()] += 1
                    break
            else:
                if first is None:
                    row.update(status="no source")
                else:
                    c, tn, r = first
                    row.update(status="fail", source=c.label(), tensors=len(tn.tensors), unused=len(r.unused),
                               first_mismatch=r.mismatches[0] if r.mismatches else "")
            rows.append(row)
        sub = [r for r in rows if r["circuit"] == circ]
        st = pd.Series([r["status"] for r in sub]).value_counts().to_dict()
        if not cands:
            status = "no source"
        elif st.get("fail", 0) or st.get("no source", 0):
            status = "fail"
        elif st.get("pass", 0) == len(sub):
            status = "pass"
        elif st.get("pass", 0):
            status = "pass, some plans unparseable"
        elif st.get("partial", 0):
            status = "partial only"  # steps agree, but the plans don't cover the whole network
        else:
            status = "no parseable plan"
        c0 = cands[0] if cands else None
        cmap.append(dict(
            name=circ, family=c0.family if c0 else "", source=c0.source if c0 else "",
            params=";".join(f"{k}={v}" for k, v in c0.params.items()) if c0 else "",
            status=status, plans=len(sub), passed=st.get("pass", 0), partial=st.get("partial", 0),
            failed=st.get("fail", 0), unparseable=st.get("unparseable", 0),
            built_as="; ".join(f"{k} x{v}" for k, v in used.items()),
            dropped_by_reader=" ".join(f"{k}:{v}" for k, v in sorted(
                sum((dropped[l] for l in used), Counter()).items())),
            note="; ".join(dict.fromkeys(c.note for c in cands if c.note and c.label() in used)),
        ))
    return pd.DataFrame(rows), pd.DataFrame(cmap)


if __name__ == "__main__":
    nets, cmap = main()
    OUT.mkdir(exist_ok=True)
    cols = ["name", "circuit", "generator", "status", "source", "tensors", "plan_leaves", "roots", "steps",
            "unused", "first_mismatch"]
    ints = ["tensors", "plan_leaves", "roots", "steps", "unused"]
    nets[cols].astype({c: "Int64" for c in ints}).to_csv(OUT / "p3_networks.csv", index=False)
    cmap.to_csv(OUT / "circuits_map.csv", index=False)
    print(f"circuits: {len(cmap)}; by status: {cmap.status.value_counts().to_dict()}")
    print(f"plans: {len(nets)}; by status: {nets.status.value_counts().to_dict()}")
    print(cmap.groupby("family")[["plans", "passed", "partial", "failed", "unparseable"]].sum().to_string())
    bad = cmap[cmap.status != "pass"]
    print(bad.to_string(index=False))
    if "-v" in sys.argv:
        print(nets[nets.status != "pass"][["name", "status", "source", "first_mismatch"]].to_string(index=False))
