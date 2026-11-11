"""Phase 3 on every circuit of the release: the seven generators on the networks of `qctn.catalog`.

    PYTHONPATH=src .venv/bin/python experiments/p3_catalog.py [--budget 10] [--jobs 4] [--gn-max-tensors 5000]
        [--max-tensors N] [--limit K]

Two choices of network, one row per (choice, circuit name, generator) in `results/p3_generators_all.csv`:

- `authors`: each generator runs on the network the release's plan for that generator was made on
  (`results/p3_networks.csv`, `source` of a passing replay). Compares generators, not networks.
- `canonical`: all seven generators run on `catalog.candidates(name)[0]`, the circuit the name says. This is the
  candidate set phase 5 uses. Where the release's plan was made on another network, it isn't compared.

A plan is generated once per (network, generator) and compared with every release plan made on that network.
Circuits are processed largest plan count first; `--limit` stops after K networks, `--max-tensors` skips larger
networks. The five solvers are stopped after `--budget` seconds (MinFill fallback, as the release's plans
indicate); Girvan–Newman is skipped above `--gn-max-tensors`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_generators import minfill_steps, summarise, summary  # noqa: E402

from qctn.catalog import candidates  # noqa: E402
from qctn.features import FEATURE_NAMES  # noqa: E402
from qctn.paths import authors_plans  # noqa: E402
from qctn.plans import pace  # noqa: E402
from qctn.plans.authors import GENERATORS, read_plan, write_plan  # noqa: E402
from qctn.plans.generators import generate, leaf_sets, to_authors_plan  # noqa: E402
from qctn.tn import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
RAW = OUT / "raw" / "p3_all"


def candidate(name: str, label: str):
    for c in candidates(name):
        if c.label() == label:
            return c
    raise KeyError((name, label))


def compare(row: dict, tn, steps, fo: dict, so: dict, plan_name: str) -> dict:
    tp = read_plan(authors_plans() / f"resultats_{plan_name}.txt")
    st, ft = summary(tp)
    a = [(s.a, s.b, s.c) for s in tp.steps]
    diffs = [abs(fo[k] - ft[k]) / max(1.0, abs(ft[k])) for k in FEATURE_NAMES if not (fo[k] != fo[k] and ft[k] != ft[k])]
    row.update({f"theirs_{k}": v for k, v in st.items()})
    row.update(theirs_status="ok", theirs_netcon=any(not s.c.startswith("I") for s in tp.steps),
               theirs_is_minfill=minfill_steps(tn) == a if row["generator"] != "mf" else None,
               identical=a == steps, same_tree=leaf_sets(a) == leaf_sets(steps),
               features_max_rel_diff=max(diffs) if diffs else 0.0,
               d_log2_flops=so["log2_flops"] - st["log2_flops"], d_width=so["width"] - st["width"])
    return row


def work(task) -> list[dict]:
    """task = (build name, network label, generator, budget, [(choice, circuit name, release plan or None, why)])"""
    bname, label, gen, budget, rows_spec = task
    tn = build(candidate(bname, label).build())
    base = dict(network=label, generator=gen, budget_s=budget if gen in pace.ANYTIME or gen in
                ("minfill_mrs", "minfillbg_mrs") else None, tensors=len(tn.tensors))
    try:
        g = generate(tn, gen, time_limit=budget)
    except Exception as e:
        return [dict(base, choice=ch, circuit=n, error=f"{type(e).__name__}: {e}") for ch, n, _, _ in rows_spec]
    ours = to_authors_plan(tn, g.steps, f"{bname}_pla_{gen}")
    d = RAW / "".join(ch if ch.isalnum() or ch in "=,._-" else "_" for ch in label)
    d.mkdir(parents=True, exist_ok=True)
    write_plan(ours, d / f"resultats_pla_{gen}.txt")
    so, fo = summary(ours)
    base.update({f"ours_{k}": v for k, v in so.items()})
    base.update(ours_td_width=g.width, ours_fallback=g.fallback, ours_elapsed_s=round(g.elapsed, 2))
    out = []
    for ch, n, plan_name, why in rows_spec:
        row = dict(base, choice=ch, circuit=n)
        if plan_name:
            try:
                compare(row, tn, g.steps, fo, so, plan_name)
            except Exception as e:  # unparseable release plan etc.
                row["theirs_status"] = f"{type(e).__name__}: {e}"
        else:
            row["theirs_status"] = why
        out.append(row)
    return out


def plan_tasks(budget: float, gn_max: int, max_tensors: int | None, limit: int | None):
    nets = pd.read_csv(OUT / "p3_networks.csv")
    nets = nets[nets.generator.isin(GENERATORS)]
    ok = nets[nets.status == "pass"]
    made_on = {(r.circuit, r.generator): r.source for r in ok.itertuples()}
    plan_of = {(r.circuit, r.generator): r.name for r in ok.itertuples()}
    size = {r.source: r.tensors for r in ok.itertuples()}
    count = nets.groupby("circuit").size().sort_values(ascending=False)
    spec: dict[tuple[str, str], list] = defaultdict(list)
    builder: dict[str, str] = {}
    for circ in count.index:
        cands = candidates(circ)
        if not cands:
            continue
        canon = cands[0].label()
        builder.setdefault(canon, circ)
        for gen in GENERATORS:
            src = made_on.get((circ, gen))
            if src:
                builder.setdefault(src, circ)
                spec[(src, gen)].append(("authors", circ, plan_of[(circ, gen)], ""))
            if src == canon:
                spec[(canon, gen)].append(("canonical", circ, plan_of[(circ, gen)], ""))
            else:
                why = "no plan in release" if (circ, gen) not in set(zip(nets.circuit, nets.generator)) else \
                    (f"release plan made on {src}" if src else "release plan doesn't replay")
                spec[(canon, gen)].append(("canonical", circ, None, why))
    # networks in order of first appearance (circuits by plan count)
    order = list(dict.fromkeys(label for label, _ in spec))
    if limit:
        order = order[:limit]
    tasks, skipped = [], []
    for label in order:
        n = size.get(label)
        if n is None:
            n = len(build(candidate(builder[label], label).build()).tensors)
            size[label] = n
        for gen in GENERATORS:
            rs = spec.get((label, gen))
            if not rs:
                continue
            why = None
            if max_tensors and n > max_tensors:
                why = f"skipped: network has {n} tensors (> {max_tensors})"
            elif gen == "gn" and n > gn_max:
                why = f"skipped: gn above {gn_max} tensors"
            elif gen not in ("mf", "gn") and not pace.available(gen):
                why = "solver not built"
            if why:
                skipped += [dict(choice=ch, circuit=c, network=label, generator=gen, tensors=n, error=why)
                            for ch, c, _, _ in rs]
            else:
                tasks.append((builder[label], label, gen, budget, rs))
    tasks.sort(key=lambda t: -size[t[1]] * (30 if t[2] == "gn" else 1))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gn-max-tensors", type=int, default=5000)
    ap.add_argument("--max-tensors", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(OUT / "p3_generators_all.csv"))
    a = ap.parse_args()
    tasks, skipped = plan_tasks(a.budget, a.gn_max_tensors, a.max_tensors, a.limit)
    print(f"{len(tasks)} generator runs on {len({t[1] for t in tasks})} networks; {len(skipped)} rows skipped",
          flush=True)
    rows, t0 = list(skipped), time.time()
    out = Path(a.out)
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rs = f.result()
            rows += rs
            r = rs[0]
            print(f"[{i}/{len(tasks)} {time.time() - t0:.0f}s] {r['network']} {r['generator']} "
                  f"identical={[x.get('identical') for x in rs]} {r.get('error', '')}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                df = pd.DataFrame(rows).sort_values(["choice", "circuit", "generator"])
                df.to_csv(out, index=False)
    df = pd.DataFrame(rows).sort_values(["choice", "circuit", "generator"])
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    for ch in ("authors", "canonical"):
        print(ch)
        print(summarise(df[df.choice == ch]).to_string(index=False))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    sys.exit(main())
