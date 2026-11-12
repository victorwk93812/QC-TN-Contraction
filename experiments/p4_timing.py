"""Phase 4: time the release's plans with the CuPy executor on one GPU.

    PYTHONPATH=src .venv/bin/python experiments/p4_timing.py --out results/raw/p4/<tag>
        [--circuits name,...] [--modes dense,hyper] [--warmup 1] [--repeats 50] [--budget 5]
        [--deadline 1000] [--max-gb 12] [--backend cupy]

For each circuit (default: the 17 with a QASM file in the release, smallest network first) the network is
built with `qctn.tn.build`, its leaves filled by `qctn.execute.leaf_arrays` (the amplitude <0|C|0>), and every
release plan of the seven generators is compiled (`compile_plan`) and timed (`time_program`, complex64) in each
mode. One JSON line per (plan, mode) goes to `<out>/times.jsonl` with every repeat's time, the warm-up time,
plan statistics and the sum of the plan file's time column (the release's mock run). A plan whose live tensors
would exceed `--max-gb` is skipped. `--budget` stops a plan's repeats once that many measured seconds are used;
`--deadline` stops starting new plans after that many seconds. Analysis: `experiments/p4_compare.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qctn import catalog
from qctn.execute import compile_plan, leaf_arrays, time_program
from qctn.paths import authors_plans
from qctn.plans.authors import GENERATORS, read_plan
from qctn.tn import build

ROOT = Path(__file__).resolve().parents[1]


def qasm_circuits() -> list[str]:
    return sorted(pd.read_csv(ROOT / "results" / "p3_generators.csv").circuit.unique())


def environment(backend: str) -> dict:
    env = dict(host_python=platform.python_version(), numpy=np.__version__, backend=backend,
               slurm_job=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURMD_NODENAME"),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if backend == "cupy":
        import cupy

        p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        env.update(cupy=cupy.__version__, gpu=p["name"].decode(), gpu_mem_gb=round(p["totalGlobalMem"] / 2**30, 1),
                   cuda_runtime=cupy.cuda.runtime.runtimeGetVersion(), cuda_driver=cupy.cuda.runtime.driverGetVersion())
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--circuits", default="")
    ap.add_argument("--modes", default="dense,hyper")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--backend", default="cupy")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    env = environment(a.backend) | vars(a)
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env), flush=True)
    names = a.circuits.split(",") if a.circuits else qasm_circuits()
    modes = a.modes.split(",")
    nets = []
    for name in names:
        c = catalog.candidates(name)[0].build()
        nets.append((name, c))
    nets.sort(key=lambda x: len(x[1].gates))
    with open(out / "times.jsonl", "a") as f:
        for name, circ in nets:
            tn = build(circ)
            arrays = leaf_arrays(circ, tn)
            for g in GENERATORS:
                path = authors_plans() / f"resultats_{name}_pla_{g}.txt"
                if not path.exists():
                    continue
                plan = read_plan(path)
                for mode in modes:
                    row = dict(plan=plan.name, circuit=name, generator=g, mode=mode, tensors=len(tn.tensors),
                               steps=len(plan.steps), mock_time=sum(s.time for s in plan.steps))
                    if a.deadline is not None and time.time() - t_start > a.deadline:
                        row["status"] = "deadline"
                        f.write(json.dumps(row) + "\n")
                        continue
                    t0 = time.time()
                    try:
                        prog = compile_plan(tn, plan, hyper=(mode == "hyper"), arrays=arrays)
                    except ValueError as e:
                        row.update(status=f"compile: {e}")
                        f.write(json.dumps(row) + "\n")
                        continue
                    row.update(compile_s=round(time.time() - t0, 3), log2_flops=float(np.log2(prog.flops)),
                               peak_gb=prog.peak_bytes(8) / 2**30, max_rank=max(prog.reduced_ranks, default=0),
                               ops=len(prog.ops))
                    if row["peak_gb"] > a.max_gb:
                        row["status"] = "skipped_memory"
                    else:
                        t0 = time.time()
                        tm = time_program(prog, arrays, backend=a.backend, warmup=a.warmup, repeats=a.repeats,
                                          budget=a.budget)
                        row.update(status=tm.status, times=tm.times, wall=tm.wall, warmup=tm.warmup,
                                   result=None if tm.result is None else [tm.result.real, tm.result.imag],
                                   elapsed_s=round(time.time() - t0, 3))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print(f"{time.time() - t_start:7.1f}s {plan.name} {mode}: {row['status']} "
                          f"n={len(row.get('times', []))} mean={np.mean(row['times']) if row.get('times') else float('nan'):.4g}s",
                          flush=True)
    print(f"total {time.time() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
