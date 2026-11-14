"""fig_repro.py — builders for the reproduction section (phases 1–4b of src/README.md), between Acts IV and V.

Numbers come from the repository's own results, never typed by hand:
  results/t06_t07.csv, results/t06_t07_composition.csv   (written by experiments/t06_t07.py)
  results/julia_t06_t07.csv                                (experiments/julia/t06_t07.jl)
  results/p4q_*.csv                                        (experiments/p4q_compare.py, p4q_labels.py, p4q_hyperflops.py)
  results/p4full_*.csv                                     (experiments/p4full_compare.py)
  results/p4_timing.csv, p2_split_sources.csv              (phases 4 and 2)
  pyproject.toml                                           (pinned versions)
and the paper's tables from paper.py. Findings are described in src/notes/deviations.md and
src/notes/release-issues.md. Registered into figures.BUILDERS.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
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
        "gpu_status": "used: CuPy executor and the authors' Julia stack timed on V100s (phases 4, 4b)",
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


# ---- phases 2–4b: what the release's labels measure ------------------------------------------------------------

THEIR_HW = {"host": "AMD EPYC 7282", "gpu": "RTX A6000"}         # release-issues.md entry 3 (from the paper / release)
OUR_HW = {"host": "Intel Xeon E5-2698 v4", "gpu": "V100-SXM2-16GB"}  # deviations.md Phase 4b (Slurm jobs 224832–224836)
V100_FP32_TFLOPS = 15.7      # NVIDIA V100 SXM2 datasheet, FP32 peak
REAL_OPS_PER_COMPLEX_FMA = 8  # one complex multiply-add = 8 real operations (deviations.md Phase 4b)


def _f2(v) -> str:
    """Two decimals with a typographic minus, as on the slides."""
    s = f"{float(v):.2f}"
    return s.replace("-", "−") if s != "-0.00" else "0.00"


def _span(lo, hi, fmt=_f2) -> str:
    a, b = fmt(lo), fmt(hi)
    return a if a == b else (f"{a} to {b}" if a.startswith("−") else f"{a}–{b}")


def _short(circuit: str) -> str:
    """dj_indep_qiskit_130 → 'DJ indep 130'; pricingcall_nativegates_ibm_qiskit_opt3_5 → 'Pricing call 5'."""
    n = circuit.rsplit("_", 1)[1]
    for pre, name in (("dj_indep", "DJ indep"), ("dj_native", "DJ native"), ("portfolioqaoa", "Portfolio QAOA"),
                      ("pricingcall", "Pricing call")):
        if circuit.startswith(pre):
            return f"{name} {n}"
    raise ValueError(circuit)


def _family_order(circuit: str):
    return (["dj", "portfolioqaoa", "pricingcall"].index(circuit.split("_")[0]), circuit.split("_")[1] != "indep",
            int(circuit.rsplit("_", 1)[1]))


def R5():
    """The release next to the paper: a few headline numbers for the grouped list of mismatches
    (full list: src/notes/release-issues.md)."""
    steps = [float(r["step_spearman"]) for r in _rows("p4q_steps.csv")]
    big = {r["mode"]: r for r in _rows("p4_timing.csv")
           if r["plan"] == "pricingcall_nativegates_ibm_qiskit_opt3_21_pla_jdrasil"}
    copies = [float(r["speedup"]) for r in _rows("p4q_label_noise.csv") if r["kind"] == "minfill_copy"]
    jd = [r for r in _rows("p2_split_sources.csv") if r["objective"] == "ndcg" and r["name"].endswith("_pla_jdrasil")]
    return _fig().write_doc("R5", "static", {"title": "The release next to the paper",
                                             "source": "results/p4q_steps.csv, p4_timing.csv, p4q_label_noise.csv, p2_split_sources.csv; src/notes/release-issues.md"}, {
        "mock_step_spearman_median": round(statistics.median(steps), 2),
        "mock_step_spearman_range_txt": _span(min(steps), max(steps)),
        "mock_plans": len(steps),
        "big_plan": "pricing call 21, Jdrasil",
        "big_dense_rank": int(big["dense"]["max_rank"]), "big_hyper_rank": int(big["hyper"]["max_rank"]),
        "big_dense_status": big["dense"]["status"],
        "big_label_txt": _f2(big["dense"]["release_speedup"]),
        "copies": len(copies), "copies_range_txt": _span(min(copies), max(copies)),
        "jdrasil_mf_features": sum(r["source"] == "mf" for r in jd),
        "rqc_circuits": int(next(r["circuits"] for r in _rows("p4full_summary.csv")
                                 if (r["mode"], r["ours"], r["variant"], r["group_by"], r["group"]) == ("hyper", "mean", "all", "family", "RQC"))),
    })


FOREST = [  # (group, measure in p4q_summary.csv, label on the slide)
    ("host", "qxtns_mean", "QXTns contract_tn!, the paper's protocol"),
    ("host", "decomp_indices_s", "its index phase alone"),
    ("host", "decomp_data_s", "its tensor_data phase alone"),
    ("host", "authors_mock_mean", "release's mock function, our run"),
    ("host", "release_mock_median_x_steps", "release's time column (median step × steps)"),
    ("host", "mock_mean", "bookkeeping loop alone (mock)"),
    ("host", "cpu_qxtns_mean", "contract_tn! on CPU arrays"),
    ("compile", "qxtns_cold_in_session", "first run in the session"),
    ("compile", "qxtns_fresh_process_cold", "first run in a fresh process"),
    ("gpu", "ome_mean", "einsums alone (OMEinsum)"),
    ("gpu", "cupy_dense_mean", "our CuPy executor, dense"),
    ("gpu", "cupy_hyper_mean", "our CuPy executor, hyperindex"),
    ("gpu", "log2_flops_hyper", "flops, hyperindex form"),
]
GROUPS = {"host": "Host-side overhead of the authors' stack", "compile": "Compilation (first runs)",
          "gpu": "GPU work"}


def R6():
    """What the labels track: within-circuit Spearman of each measure (−time, −flops) with the label speedups on
    the 15 labelled QASM circuits, mean with a 95% interval from resampling circuits (p4q_summary.csv); the full
    run's medians over all labelled circuits (p4full_summary.csv); the noise ceiling (p4q_label_stats.csv)."""
    summ = {r["measure"]: r for r in _rows("p4q_summary.csv")}
    rows = []
    for group, key, label in FOREST:
        r = summ[key]
        mean, lo, hi, med = (float(r[k]) for k in ("spearman_mean", "spearman_mean_lo", "spearman_mean_hi", "spearman_median"))
        rows.append({"group": group, "key": key, "label": label, "circuits": int(r["circuits"]),
                     "median": round(med, 2), "mean": round(mean, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                     "median_txt": _f2(med), "mean_txt": _f2(mean), "ci_txt": f"[{_f2(lo)}, {_f2(hi)}]",
                     "best": int(round(float(r["best_same"])))})
    by = {r["key"]: r for r in rows}
    rep = next(r for k, r in summ.items() if k.startswith("label_replicate_sigma_"))
    host_pos = [r for r in rows if r["group"] == "host" and r["lo"] > 0]
    host_rest = [r for r in rows if r["group"] == "host" and r["lo"] <= 0]
    gpu = [r for r in rows if r["group"] == "gpu"]

    full = {(r["mode"], r["ours"]): r for r in _rows("p4full_summary.csv")
            if r["variant"] == "all" and r["group_by"] == "all"}
    stats = {r["statistic"]: r for r in _rows("p4q_label_stats.csv")}
    rep_all = next(r for k, r in stats.items() if k.startswith("replicate (all copies"))
    fr = lambda name, r: {"name": name, "circuits": int(float(r["circuits"])), "median_txt": _f2(r["spearman_median"]),
                          "median": round(float(r["spearman_median"]), 2),
                          "best_pct": round(100 * float(r["best_same"]))}
    full_rows = [fr("our CuPy timings, hyperindex", full[("hyper", "mean")]),
                 fr("our CuPy timings, dense", full[("dense", "mean")]),
                 fr("release's time column, same plans", full[("hyper", "mock_same")]),
                 fr("a re-measurement with the labels' noise", rep_all)]
    n_qasm = rows[0]["circuits"]
    return _fig().write_doc("R6", "table", {"title": "What the labels track",
                                            "source": "results/p4q_summary.csv, p4full_summary.csv, p4q_label_stats.csv; src/notes/deviations.md Phase 4b"}, {
        "rows": rows, "by_key": by, "groups": GROUPS, "circuits": n_qasm,
        "plans": len(_rows("p4q_timing.csv")),
        "ceiling": {"mean": round(float(rep["spearman_mean"]), 2), "median": round(float(rep["spearman_median"]), 2),
                    "mean_txt": _f2(rep["spearman_mean"])},
        "host_pos_txt": _span(min(r["mean"] for r in host_pos), max(r["mean"] for r in host_pos)),
        "host_pos_n": len(host_pos),
        "host_rest_txt": _span(min(r["mean"] for r in host_rest), max(r["mean"] for r in host_rest)),
        "gpu_txt": _span(min(r["mean"] for r in gpu), max(r["mean"] for r in gpu)),
        "full": full_rows,
        "their_hw": THEIR_HW, "our_hw": OUR_HW,
    })


PHASES = ["decomp_indices_s", "decomp_data_s", "decomp_einsum_s", "decomp_rest_s"]


def R7():
    """Where a run of QXTns contract_tn! spends its time (p4q_timing.csv): per circuit, the four phases of the
    step-by-step re-implementation (`decomp`, synchronised after each phase) as shares of their sum over the
    circuit's plans; the einsums alone and the same call on CPU arrays against the GPU run."""
    t = _rows("p4q_timing.csv")
    circuits = sorted({r["circuit"] for r in t}, key=_family_order)
    rows = []
    for c in circuits:
        rr = [r for r in t if r["circuit"] == c]
        tot = {p: sum(float(r[p]) for r in rr) for p in PHASES}
        s = sum(tot.values())
        rows.append({"circuit": c, "short": _short(c), "plans": len(rr),
                     "steps": int(statistics.median(int(r["steps"]) for r in rr)),
                     "shares": {p.split("_")[1]: round(tot[p] / s, 4) for p in PHASES},
                     "ms_per_step": round(statistics.median(1e3 * float(r["qxtns_mean"]) / int(r["steps"]) for r in rr), 2),
                     "cpu_over_gpu": round(statistics.median(float(r["cpu_qxtns_mean"]) / float(r["qxtns_mean"]) for r in rr), 2)})
    pct = lambda v: f"{100 * v:.0f}"
    share_txt = {k: _span(min(r["shares"][k] for r in rows), max(r["shares"][k] for r in rows), pct)
                 for k in ("indices", "data", "einsum", "rest")}
    host = [1 - r["shares"]["einsum"] for r in rows]
    ome = [float(r["ome_mean"]) / float(r["qxtns_mean"]) for r in t]
    cpu = sorted(r["cpu_over_gpu"] for r in rows)
    cpu_le = [v for v in cpu if v <= 1.05]
    cpu_gt = [v for v in cpu if v > 1.05]
    ms = lambda fam: [r["ms_per_step"] for r in rows if r["circuit"].startswith(fam)]
    run_s = [float(r["qxtns_mean"]) for r in t]
    hf = [r for r in _rows("p4q_hyperflops.csv")]
    split_names = {r["name"] for r in _rows("p2_split_sources.csv") if r["objective"] == "ndcg"}
    split_flops = [float(r["log2_flops_hyper"]) for r in hf if r["plan"] in split_names]
    arith_ms = 2 ** 36 * REAL_OPS_PER_COMPLEX_FMA / (V100_FP32_TFLOPS * 1e12) * 1e3
    return _fig().write_doc("R7", "chart", {"title": "Where a contract_tn! run spends its time",
                                            "source": "results/p4q_timing.csv, p4q_hyperflops.csv, p2_split_sources.csv; src/notes/deviations.md Phase 4b"}, {
        "rows": rows, "circuits": len(rows), "plans": len(t),
        "share_txt": share_txt, "host_txt": _span(min(host), max(host), pct),
        "ome_txt": f"{100 * min(ome):.1f}–{100 * max(ome):.0f}",
        "cpu_le_n": len(cpu_le), "cpu_le_txt": _span(min(cpu_le), max(cpu_le)),
        "cpu_gt_txt": " and ".join(f"{v:.1f}×" for v in cpu_gt),
        "ms_pricing_txt": _span(min(ms("pricingcall")), max(ms("pricingcall")), lambda v: f"{v:.1f}"),
        "ms_other_txt": _span(min(ms("dj") + ms("portfolio")), max(ms("dj") + ms("portfolio")), lambda v: f"{v:.1f}"),
        "run_s_txt": _span(min(run_s), max(run_s), lambda v: f"{v:.1f}"),
        "run_s_median_txt": f"{statistics.median(run_s):.1f}",
        "split_plans": len(split_flops), "split_under_2_36": sum(v < 36 for v in split_flops),
        "split_max_log2_txt": f"{max(split_flops):.1f}",
        "arith_ms_2_36_txt": f"{arith_ms:.0f}", "peak_tflops": V100_FP32_TFLOPS,
        "our_hw": OUR_HW, "their_hw": THEIR_HW,
    })


def _noise_sigma(log_diffs) -> float:
    """experiments/p4q_labels.py noise_sigma: 1.4826 · median |d| / √2 (each difference has variance 2σ²)."""
    return 1.4826 * statistics.median(abs(float(d)) for d in log_diffs) / math.sqrt(2)


def R8():
    """Identical plans with different labels: the release's MinFill copies against our timings of the same plans,
    the noise level they imply and what a re-measurement with that noise would reach."""
    noise = _rows("p4q_label_noise.csv")
    copies = [r for r in noise if r["kind"] == "minfill_copy"]
    labels = [{"circuit": r["base_id"], "plan": r["plan"], "cause": r["cause"], "speedup": round(float(r["speedup"]), 4)}
              for r in copies]
    ours = {"hyper": [], "dense": []}
    for r in _rows("p4full_timing.csv"):
        if r["dup_of_minfill"] == "True" and r["status"] == "ok" and r["sp_mean"]:
            ours[r["mode"]].append(round(float(r["sp_mean"]), 4))
    allours = ours["hyper"] + ours["dense"]
    sigma = _noise_sigma(r["log_diff"] for r in noise)
    stats = {r["statistic"]: r for r in _rows("p4q_label_stats.csv")}
    rep_key = next(k for k in stats if k.startswith("replicate (all copies"))
    rep = stats[rep_key]
    sigma_csv = float(re.search(r"sigma ([\d.]+)", rep_key).group(1))
    summ = {r["measure"]: r for r in _rows("p4q_summary.csv")}
    rep15 = next(r for k, r in summ.items() if k.startswith("label_replicate_sigma_"))
    t = [r for r in _rows("p4q_timing.csv") if r["circuit"] == "pricingcall_nativegates_ibm_qiskit_opt3_23"
         and r["generator"] in ("mf", "jdrasil", "minfillbg_mrs")]
    t.sort(key=lambda r: float(r["qxtns_mean"]))
    gen = {"mf": "MinFill", "jdrasil": "Jdrasil", "minfillbg_mrs": "MinFill-BG (MRS)"}
    timeouts = sum(r["cause"] == "timeout" for r in copies)
    return _fig().write_doc("R8", "chart", {"title": "Identical plans, different labels",
                                            "source": "results/p4q_label_noise.csv, p4full_timing.csv, p4q_label_stats.csv, p4q_summary.csv, p4q_timing.csv"}, {
        "labels": labels, "ours": ours,
        "copies": len(copies), "copy_circuits": len({r["base_id"] for r in copies}),
        "timeouts": timeouts, "unexplained": len(copies) - timeouts,
        "label_range_txt": _span(min(l["speedup"] for l in labels), max(l["speedup"] for l in labels)),
        "ours_pairs": len(allours), "ours_range_txt": _span(min(allours), max(allours)),
        "ours_median_txt": f"{statistics.median(allours):.3f}",
        "sigma": round(sigma, 4), "sigma_csv": sigma_csv, "sigma_txt": f"{sigma:.3f}",
        "replicate": {"circuits": int(float(rep["circuits"])), "median_txt": _f2(rep["spearman_median"]),
                      "mean_txt": _f2(rep["spearman_mean"]), "best_pct": round(100 * float(rep["best_same"])),
                      "qasm_mean_txt": _f2(rep15["spearman_mean"])},
        "example": {"circuit": "Pricing call 23",
                    "rows": [{"generator": gen[r["generator"]], "qxtns_s_txt": f"{float(r['qxtns_mean']):.2f}",
                              "label_txt": _f2(r["speedup"])} for r in t]},
    })


BUILDERS = {f.__name__: f for f in [R1, R2, R3, R4, R5, R6, R7, R8]}
