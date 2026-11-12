"""Phase 4 full run: time every feasible release plan with the CuPy executor, all plans of a circuit on one GPU.

    # CPU, once: compile every plan in both modes, decide what is feasible, shard the circuits
    PYTHONPATH=src .venv/bin/python experiments/p4full_timing.py inventory [--jobs 8] [--shards 24]
    # GPU, one Slurm array task per shard (scripts/slurm/p4full_array.sbatch)
    PYTHONPATH=src .venv/bin/python experiments/p4full_timing.py run --shard K --out results/raw/p4full/<tag>

Plans: every row of `results/p3_networks.csv` that replays (`pass`) or covers part of a network (`partial`, six
plans; skipped, see below): 1661 parsed plan files, on 331 circuits. Each plan runs on the network it replays on
(column `source`, a `qctn.catalog` candidate), with the leaf values of `qctn.execute.leaf_arrays` (amplitude
<0|C|0>, QXTns convention).

`inventory` writes `results/p4full_plans.csv` (one row per plan and mode: compiled size, predicted run time, and
`skip`, the reason a plan is not timed, empty if it is) and `results/p4full_shards.csv` (circuit -> shard, by
longest-processing-time-first on the predicted GPU seconds). Skip rules, per mode:

- `partial`: the plan contracts part of a network that the release doesn't identify (`qctn.catalog._PARTIAL`);
- `compile: ...`: the executor can't express a step (more than 52 einsum labels in hyperindex form);
- `memory`: largest sum of live tensors over `--max-gb` (complex64, 8 bytes per element) on the 16 GB V100;
- `predicted_slow`: predicted run time over `--max-run` seconds (the paper fails a plan over 10 minutes), from
  the small run's fit (`results/p4_estimate.csv`: a + b * ops + c * flops, with our compiled flops).

`run` times, for each circuit of its shard (largest predicted first), hyperindex mode then dense, every plan not
skipped: 1 warm-up and `--repeats` measured runs (complex64, `qctn.execute.time_program`), all raw times kept. A
warm-up over `--max-run` seconds stops the plan (status `slow`); `--budget` caps a plan's measured seconds
(status `budget`, repeats done recorded). One file per circuit, `<out>/circuits/<circuit>.jsonl`, written as
`.part` and renamed when the circuit is complete; a rerun skips circuits whose file exists. One JSON file per
task, `<out>/tasks/shard<K>_<job>.json`: node, GPU (index, UUID, PCI bus), the node's GPU allocation and
`nvidia-smi` utilisation and memory at task start and end, and the circuits done; each circuit row records the
GPU's utilisation and memory before and after it. Analysis: `experiments/p4full_compare.py`.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODES = ("hyper", "dense")  # hyperindex is the primary mode
ITEMSIZE = 8  # complex64


# ---------------------------------------------------------------- plans and networks

def plan_table() -> pd.DataFrame:
    """Plans to consider: the replaying and partial rows of `p3_networks.csv`."""
    from qctn.plans.authors import split_name

    n = pd.read_csv(RESULTS / "p3_networks.csv")
    n = n[n.status.isin(["pass", "partial"])].copy()
    n["generator"] = [split_name(x)[1] or "" for x in n.name]
    return n[["name", "circuit", "generator", "status", "source", "tensors"]].rename(columns={"name": "plan"})


def candidate(circuit: str, label: str):
    from qctn.catalog import candidates

    for c in candidates(circuit):
        if c.label() == label:
            return c
    raise KeyError((circuit, label))


def networks(circuit: str, sources) -> dict:
    """{source label: (circuit object, network, leaf arrays)} for the networks a circuit's plans use."""
    from qctn.execute import leaf_arrays
    from qctn.tn import build

    out = {}
    for s in dict.fromkeys(sources):
        circ = candidate(circuit, s).build()
        tn = build(circ)
        out[s] = (circ, tn, leaf_arrays(circ, tn))
    return out


def fit_coefficients() -> dict[str, tuple[float, float, float]]:
    e = pd.read_csv(RESULTS / "p4_estimate.csv")
    return {r.mode: (r.a_s, r.b_s_per_step, r.c_s_per_flop) for r in e.itertuples()}


def program_stats(prog, coef) -> dict:
    a, b, c = coef
    return dict(ops=len(prog.ops), log2_flops=math.log2(prog.flops) if prog.flops > 0 else 0.0,
                peak_gb=prog.peak_bytes(ITEMSIZE) / 2**30, max_gb_tensor=prog.max_elements * ITEMSIZE / 2**30,
                max_rank=max(prog.reduced_ranks, default=0), pred_run_s=a + b * len(prog.ops) + c * prog.flops)


def skip_reason(stats: dict, max_gb: float, max_run: float) -> str:
    if stats["peak_gb"] > max_gb:
        return f"memory: peak {stats['peak_gb']:.3g} GB > {max_gb:g} GB"
    if stats["pred_run_s"] > max_run:
        return f"predicted_slow: {stats['pred_run_s']:.3g} s per run > {max_run:g} s"
    return ""


def compile_one(tn, plan, mode, arrays, coef, max_gb, max_run):
    """(program or None, row fields) for one plan and mode."""
    from qctn.execute import compile_plan

    t0 = time.time()
    try:
        prog = compile_plan(tn, plan, hyper=(mode == "hyper"), arrays=arrays)
    except ValueError as e:
        return None, dict(skip=f"compile: {e}", compile_s=round(time.time() - t0, 3))
    except Exception as e:  # noqa: BLE001 - recorded, the other plans go on
        return None, dict(skip=f"compile: {type(e).__name__}: {e}", compile_s=round(time.time() - t0, 3))
    st = program_stats(prog, coef)
    st.update(compile_s=round(time.time() - t0, 3), skip=skip_reason(st, max_gb, max_run))
    return prog, st


# ---------------------------------------------------------------- inventory (CPU)

def inventory_circuit(circuit: str, rows: list[dict], coefs, max_gb, max_run) -> list[dict]:
    from qctn.paths import authors_plans
    from qctn.plans.authors import read_plan

    t0 = time.time()
    ok = [r for r in rows if r["status"] == "pass"]
    try:
        nets = networks(circuit, [r["source"] for r in ok])
    except Exception as e:  # noqa: BLE001
        return [dict(r, mode=m, skip=f"network: {type(e).__name__}: {e}") for r in rows for m in MODES]
    build_s = time.time() - t0
    out = []
    for r in rows:
        plan = read_plan(authors_plans() / f"resultats_{r['plan']}.txt")
        base = dict(r, steps=len(plan.steps), mock_time=sum(s.time for s in plan.steps), build_s=round(build_s, 2))
        for m in MODES:
            if r["status"] == "partial":
                out.append(dict(base, mode=m, skip="partial: plan covers part of an unidentified network"))
                continue
            _, tn, arrays = nets[r["source"]]
            _, st = compile_one(tn, plan, m, arrays, coefs[m], max_gb, max_run)
            out.append(dict(base, mode=m, **st))
    return out


def _quiet_threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def shard(df: pd.DataFrame, n: int, warmup: int, repeats: int) -> pd.DataFrame:
    """Circuits to `n` shards, longest predicted first onto the least loaded shard."""
    t = df[df.skip.fillna("") == ""]
    gpu = t.groupby("circuit").pred_run_s.sum() * (warmup + repeats)
    cpu = df.groupby("circuit").compile_s.sum().fillna(0) + df.groupby("circuit").build_s.first().fillna(0)
    load = (gpu.reindex(cpu.index).fillna(0) + cpu + 5.0).sort_values(ascending=False)  # 5 s per circuit: setup
    heap = [(0.0, k) for k in range(n)]
    out = []
    for c, s in load.items():
        tot, k = heapq.heappop(heap)
        out.append(dict(circuit=c, shard=k, pred_s=round(float(s), 1)))
        heapq.heappush(heap, (tot + s, k))
    return pd.DataFrame(out)


def inventory(a) -> None:
    _quiet_threads()
    plans = plan_table()
    coefs = fit_coefficients()
    groups = {c: g.to_dict("records") for c, g in plans.groupby("circuit")}
    rows = []
    with ProcessPoolExecutor(a.jobs, initializer=_quiet_threads) as ex:
        futs = {ex.submit(inventory_circuit, c, g, coefs, a.max_gb, a.max_run): c for c, g in groups.items()}
        for i, f in enumerate(as_completed(futs)):
            rows += f.result()
            print(f"{i + 1}/{len(futs)} {futs[f]}", flush=True)
    df = pd.DataFrame(rows).sort_values(["circuit", "plan", "mode"])
    df.to_csv(RESULTS / "p4full_plans.csv", index=False)
    sh = shard(df, a.shards, a.warmup, a.repeats)
    sh.to_csv(RESULTS / "p4full_shards.csv", index=False)
    print(df.groupby("mode").skip.apply(lambda s: s.fillna("").str.split(":").str[0].value_counts()).to_string())
    print(sh.groupby("shard").pred_s.sum().describe().to_string())
    t = df[df.skip.fillna("") == ""]
    print("predicted GPU hours:", (t.groupby("mode").pred_run_s.sum() * (a.warmup + a.repeats) / 3600).to_string())


# ---------------------------------------------------------------- run (GPU)

def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"


def gpu_snapshot() -> list[dict]:
    """`nvidia-smi` for the GPUs this process can see (inside a Slurm job: the allocated one)."""
    q = "index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm"
    out = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) == 8:
            rows.append(dict(zip(q.split(","), v)))
    return rows or [dict(raw=out)]


def node_snapshot(node: str | None) -> dict:
    """The whole node: every GPU nvidia-smi can query, the node's allocated TRES and the jobs on it."""
    d = dict(time=time.time(), gpus=gpu_snapshot())
    if node:
        d["node_alloc"] = sh(["scontrol", "show", "node", "-d", node])
        d["node_alloc"] = " ".join(x for x in d["node_alloc"].split() if x.startswith(("AllocTRES", "Gres", "CPUAlloc")))
        d["node_jobs"] = sh(["squeue", "-h", "-w", node, "-o", "%i %P %b %C %T"]).splitlines()
    return d


def environment(backend: str = "cupy") -> dict:
    env = dict(python=platform.python_version(), numpy=np.__version__, backend=backend,
               slurm_job=os.environ.get("SLURM_JOB_ID"), array_job=os.environ.get("SLURM_ARRAY_JOB_ID"),
               array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
               node=os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0],
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), slurm_gpus=os.environ.get("SLURM_JOB_GPUS"))
    if backend == "cupy":
        import cupy

        p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        env.update(cupy=cupy.__version__, gpu=p["name"].decode(), gpu_mem_gb=round(p["totalGlobalMem"] / 2**30, 1),
                   pci_bus=f"{p['pciDomainID']:04x}:{p['pciBusID']:02x}:{p['pciDeviceID']:02x}",
                   cuda_runtime=cupy.cuda.runtime.runtimeGetVersion(),
                   cuda_driver=cupy.cuda.runtime.driverGetVersion())
    return env


def run_circuit(circuit: str, rows: list[dict], a, coefs, env: dict, out_dir: Path) -> dict:
    from qctn.execute import time_program
    from qctn.paths import authors_plans
    from qctn.plans.authors import read_plan

    t0 = time.time()
    todo = [r for r in rows if r["status"] == "pass"]
    nets = networks(circuit, [r["source"] for r in todo]) if todo else {}
    build_s = time.time() - t0
    plans = {r["plan"]: read_plan(authors_plans() / f"resultats_{r['plan']}.txt") for r in rows}
    part = out_dir / f"{circuit}.jsonl.part"
    n_timed = 0
    with open(part, "w") as f:
        for mode in MODES:
            for r in sorted(rows, key=lambda r: r["plan"]):
                plan = plans[r["plan"]]
                row = dict(plan=r["plan"], circuit=circuit, generator=r["generator"], source=r["source"], mode=mode,
                           tensors=r["tensors"], steps=len(plan.steps), mock_time=sum(s.time for s in plan.steps),
                           node=env["node"], gpu_uuid=env.get("gpu_uuid"),
                           job=env["slurm_job"], build_s=round(build_s, 2))
                if r["status"] == "partial":
                    row["status"] = "skipped: partial: plan covers part of an unidentified network"
                    f.write(json.dumps(row) + "\n")
                    continue
                _, tn, arrays = nets[r["source"]]
                prog, st = compile_one(tn, plan, mode, arrays, coefs[mode], a.max_gb, a.max_run)
                row.update(st)
                if st["skip"]:
                    row["status"] = f"skipped: {st['skip']}"
                else:
                    g0 = gpu_snapshot()[0]
                    t1 = time.time()
                    tm = time_program(prog, arrays, backend=a.backend, warmup=a.warmup, repeats=a.repeats,
                                      budget=a.budget, max_warmup=a.max_run)
                    row.update(status=tm.status, times=tm.times, wall=tm.wall, warmup=tm.warmup,
                               result=None if tm.result is None else [tm.result.real, tm.result.imag],
                               elapsed_s=round(time.time() - t1, 3), gpu_before=g0)
                    n_timed += 1
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{time.strftime('%H:%M:%S')} {r['plan']} {mode}: {row['status']} n={len(row.get('times', []))}"
                      f" mean={np.mean(row['times']) if row.get('times') else float('nan'):.4g}s", flush=True)
    part.rename(out_dir / f"{circuit}.jsonl")
    return dict(circuit=circuit, plans=len(rows), timed=n_timed, seconds=round(time.time() - t0, 1))


def run(a) -> None:
    out = Path(a.out)
    (out / "circuits").mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    k = a.shard if a.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    shards = pd.read_csv(Path(a.shards_file))
    mine = shards[shards.shard == k].sort_values("pred_s", ascending=False).circuit.tolist()
    if a.circuits:
        mine = [c for c in mine if c in a.circuits.split(",")]
    env = environment(a.backend)  # fails early without a GPU for cupy
    g = gpu_snapshot()
    env["gpu_uuid"] = g[0].get("uuid")
    env["gpu_index"] = g[0].get("index")
    info = dict(shard=k, args=vars(a), env=env, start=node_snapshot(env["node"]), circuits=[], skipped_done=[])
    tag = f"shard{k:03d}_{env['slurm_job'] or 'local'}"
    task_file = out / "tasks" / f"{tag}.json"
    task_file.write_text(json.dumps(info, indent=1))
    print(json.dumps(env), flush=True)
    coefs = fit_coefficients()
    plan_rows = plan_table()
    t0 = time.time()
    for c in mine:
        if (out / "circuits" / f"{c}.jsonl").exists():
            info["skipped_done"].append(c)
            continue
        if a.deadline is not None and time.time() - t0 > a.deadline:
            info["stopped_at_deadline"] = c
            break
        rows = plan_rows[plan_rows.circuit == c].to_dict("records")
        before = gpu_snapshot()[0]
        res = run_circuit(c, rows, a, coefs, env, out / "circuits")
        res.update(gpu_before=before, gpu_after=gpu_snapshot()[0])
        info["circuits"].append(res)
        task_file.write_text(json.dumps(info, indent=1))
    info["end"] = node_snapshot(env["node"])
    info["seconds"] = round(time.time() - t0, 1)
    task_file.write_text(json.dumps(info, indent=1))
    print(f"shard {k}: {len(info['circuits'])} circuits in {info['seconds']} s, "
          f"{len(info['skipped_done'])} already done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "run"):
        p = sub.add_parser(name)
        p.add_argument("--max-gb", type=float, default=12.0)
        p.add_argument("--max-run", type=float, default=600.0)
        p.add_argument("--warmup", type=int, default=1)
        p.add_argument("--repeats", type=int, default=50)
    p = sub.choices["inventory"]
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--shards", type=int, default=24)
    p = sub.choices["run"]
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    p.add_argument("--shards-file", default=str(RESULTS / "p4full_shards.csv"))
    p.add_argument("--circuits", default="", help="only these circuits of the shard")
    p.add_argument("--budget", type=float, default=600.0, help="measured seconds per plan")
    p.add_argument("--deadline", type=float, default=None, help="no new circuit after this many seconds")
    p.add_argument("--backend", default="cupy", help="numpy: CPU, for tests")
    a = ap.parse_args(argv)
    {"inventory": inventory, "run": run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
