"""fig_repro.py — builders for the reproduction section (phase 1 of src/README.md), between Acts IV and V.

Numbers come from the repository's own results, never typed by hand:
  results/t06_t07.csv, results/t06_t07_composition.csv   (written by experiments/t06_t07.py)
  results/julia_t06_t07.csv                                (experiments/julia/t06_t07.jl)
  pyproject.toml                                           (pinned versions)
and the paper's tables from paper.py. Findings are described in src/notes/deviations.md.
Registered into figures.BUILDERS.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from . import paper

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
METRICS = ("top1", "top3", "regret1", "regret3", "win")
SPLITS = {"id": (paper.TABLE6, paper.TABLE6_N), "ood": (paper.TABLE7, paper.TABLE7_N)}
OBJECTIVES = {"ndcg": "opt_ndcg", "pair": "opt_pair"}
MAIN = "t06_t07.csv"                 # metrics per recipe / model / split
CHECKS = "t06_t07_composition.csv"   # which round counts match which published pairs of values
JULIA = "julia_t06_t07.csv"          # the same recipe run with the authors' Julia code


def _fig():
    from . import figures
    return figures


def _rows(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def _pin(spec_re: str) -> str:
    m = re.search(spec_re, (REPO / "pyproject.toml").read_text())
    if not m:
        raise ValueError(f"pyproject.toml: no match for {spec_re}")
    return m.group(1)


def _paper_row(objective, split):
    return dict(zip(METRICS, SPLITS[split][0][OBJECTIVES[objective]]))


def _rnd(metric, v):
    """The paper's precision: percentages to 2 decimals, regrets to 4."""
    return round(float(v), 4 if metric.startswith("regret") else 2)


def _rng(metric, lo, hi):
    """Display a range as the deviations note does: % to integers, regrets to 4 decimals."""
    f = (lambda v: f"{v:.4f}") if metric.startswith("regret") else (lambda v: f"{v:.0f}")
    a, b = f(lo), f(hi)
    return a if a == b else f"{a}–{b}"


def R1():
    """Setup of the reproduction: what is pinned and what is only planned (no results here)."""
    return _fig().write_doc("R1", "static", {"title": "Reproduction setup", "source": "pyproject.toml, src/README.md, scripts/README.md, src/notes/deviations.md"}, {
        "package": "qctn",
        "authors_repo": "alfred-miquel/LTR-Quantum-Contraction-Plans",
        "authors_commit": "936630a",
        "julia": "1.9.1",
        "julia_stack": ["QXTools 1.0", "OMEinsum 0.6.13", "CUDA.jl 3.13.1", "XGBoost.jl 2.5.5"],
        "xgboost": _pin(r'"xgboost==([\d.]+)"'),
        "xgboost_jll": "2.1.5",
        "xgboost_match": "5e-8",               # deviations.md: max prediction difference vs their Julia stack
        "xgboost_newer": "3.4.1",
        "xgboost_newer_diff": "0.03–1.06",     # deviations.md: prediction differences with 3.4.1
        "cupy": _pin(r'"(cupy-cuda12x\[ctk\]==[\d.]+)"').replace("==", " "),
        "gpus": "8× Tesla V100-SXM2-16GB",
        "driver": "535",
        "gpu_status": "installed; smoke test only (one einsum); executor planned for phase 4",
    })


def R2():
    """Table 6 rebuilt from the authors' Notebook 5 recipe: each published row takes Top-1/Regret1 from
    one model and Top-3/Regret3 from another (different round counts)."""
    comp = [r for r in _rows(MAIN) if r["recipe"] == "notebook" and r["split"] == "id"]
    checks = _rows(CHECKS)
    julia = [r for r in _rows(JULIA) if r["model"] == "notebook" and r["split"] == "id"]
    rows = {}
    for obj in OBJECTIVES:
        rr = [r for r in comp if r["objective"] == obj]
        assert len(rr) == 2, obj
        r1, r3 = rr                                     # experiments/t06_t07.py writes the Top-1 model first
        models = {r["rounds"]: {m: _rnd(m, r[m]) for m in METRICS} for r in rr}
        composed = {"top1": _rnd("top1", r1["top1"]), "regret1": _rnd("regret1", r1["regret1"]),
                    "top3": _rnd("top3", r3["top3"]), "regret3": _rnd("regret3", r3["regret3"])}
        pub = _paper_row(obj, "id")
        rows[OBJECTIVES[obj]] = {
            "rounds_k1": int(r1["rounds"]), "rounds_k3": int(r3["rounds"]),
            "models": models, "composed": composed, "paper": pub,
            "match": {m: composed[m] == pub[m] for m in composed},
            "win_ours": sorted({_rnd("win", r["win"]) for r in rr}),
            "julia_max_diff": max(abs(float(j[m]) - float(r[m])) for r in rr for j in julia
                                  if j["objective"] == obj and j["rounds"] == r["rounds"] for m in METRICS[:4]),
            "single_model_all_four": [json.loads(c["value"]) for c in checks if c["objective"] == obj and c["check"] == "all_four"
                                      and c["split"] == "id" and c["train"] == "train" and c["params"] == "notebook"][0],
        }
    jd = max(r["julia_max_diff"] for r in rows.values())
    return _fig().write_doc("R2", "table", {"title": "Table 6 from the notebook recipe",
                                            "source": "results/t06_t07.csv, t06_t07_composition.csv, julia_t06_t07.csv; arXiv 2608.05819 Table 6"}, {
        "rows": rows, "julia_max_diff_txt": f"{jd:.0e}".replace("e-0", "e-"),
        "notebook_params": {"eta": 0.02, "subsample": 0.8},     # src/qctn/rank/train.py NOTEBOOK["ndcg"]
        "table4_params": {"eta": 0.01, "subsample": 0.7},       # src/qctn/rank/train.py TABLE4["ndcg"]
        "max_rounds": 300,
    })


def R3():
    """One model per objective, rounds chosen by grouped 5-fold CV on the training split only:
    ranges over three fold assignments and both opt_ndcg parameter sets, against the paper's rows."""
    comp = _rows(MAIN)
    cv = [r for r in comp if r["recipe"].startswith("single_cv_") and r["target"] == "label"]
    base = {(r["recipe"], r["split"]): r for r in comp if r["recipe"] in ("random_expected", "mf_rand_expected")}
    rows = []
    for split in SPLITS:
        for obj in OBJECTIVES:
            rr = [r for r in cv if r["objective"] == obj and r["split"] == split]
            row = {"key": f"{OBJECTIVES[obj]}|{split}", "model": OBJECTIVES[obj], "split": split, "n_runs": len(rr),
                   "rounds": [min(int(r["rounds"]) for r in rr), max(int(r["rounds"]) for r in rr)],
                   "paper": _paper_row(obj, split)}
            for m in METRICS:
                vals = [float(r[m]) for r in rr]
                row[m] = [_rnd(m, min(vals)), _rnd(m, max(vals))]
                row[m + "_txt"] = _rng(m, min(vals), max(vals))
            rows.append(row)
    ref = {split: {"random": {m: _rnd(m, base[("random_expected", split)][m]) for m in METRICS},
                   "mf_first": {m: _rnd(m, base[("mf_rand_expected", split)][m]) for m in METRICS}}
           for split in SPLITS}
    all_rounds = [int(r["rounds"]) for r in cv]
    n_id = paper.TABLE6_N
    return _fig().write_doc("R3", "table", {"title": "One model per objective, rounds by grouped CV",
                                            "source": "results/t06_t07.csv; arXiv 2608.05819 Tables 6, 7"}, {
        "rows": rows, "by_key": {r["key"]: r for r in rows}, "ref": ref,
        "folds": 5, "fold_seeds": sorted({int(r["recipe"].rsplit("seed", 1)[1]) for r in cv}),
        "rounds_range": [min(all_rounds), max(all_rounds)], "rounds_range_txt": f"{min(all_rounds)}–{max(all_rounds)}",
        "n_id": n_id, "n_ood": paper.TABLE7_N, "pts_per_circuit_id": round(100 / n_id, 2),
        "random_ood_top3_txt": f"{float(base[('random_expected', 'ood')]['top3']):.1f}",
        "mf_id_top1_txt": f"{float(base[('mf_rand_expected', 'id')]['top1']):.0f}",
    })


def R4():
    """What phase 1 did not reproduce: baselines, win rate, Table 7; and what the stored JSON models are."""
    comp = _rows(MAIN)
    get = lambda rec, split: next(r for r in comp if r["recipe"] == rec and r["split"] == split and r["objective"] == "-")
    baselines = {}
    for rec, name in (("random_expected", "random"), ("mf_rand_expected", "mf_rand")):
        for split in SPLITS:
            r = get(rec, split)
            baselines[f"{name}|{split}"] = {"ours": {m: _rnd(m, r[m]) for m in METRICS},
                                            "paper": dict(zip(METRICS, SPLITS[split][0][name]))}
    stored = [r for r in comp if r["recipe"] == "stored"]
    s1 = [float(r["top1"]) for r in stored]
    win_ndcg_id = [float(r["win"]) for r in comp if r["objective"] == "ndcg" and r["split"] == "id"
                   and (r["recipe"] == "notebook" or r["recipe"].startswith("single_cv_"))]
    return _fig().write_doc("R4", "table", {"title": "Not reproduced", "source": "results/t06_t07*.csv; src/notes/deviations.md"}, {
        "baselines": baselines,
        "stored_top1_txt": f"{min(s1):.0f}–{max(s1):.0f}",
        "stored_circuits": paper.SPLIT["total"],
        "win_ndcg_id_txt": _rng("win", min(win_ndcg_id), max(win_ndcg_id)), "win_ndcg_id_paper": paper.TABLE6["opt_ndcg"][4],
    })


BUILDERS = {f.__name__: f for f in [R1, R2, R3, R4]}
