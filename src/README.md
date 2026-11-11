# `src/` — reproducing arXiv 2608.05819

Proposal, agreed 2026-11-05. Nothing here is implemented yet; each phase below lands as its own pull request.

Target: A. M. Pastor, M. Castillo, J. M. Badia, *Learning to Rank Tensor Network Contraction Plans for
GPU-Accelerated Quantum Circuit Simulation*, [arXiv:2608.05819](https://arxiv.org/abs/2608.05819), reproduced in
Python. `src/` is the source of truth for the algorithms; `slides/traces/` becomes a client of it.

## Decisions

- **Start from the authors' release and compare against it**:
  [alfred-miquel/LTR-Quantum-Contraction-Plans](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/tree/936630aa161abb7bcab1557c91137696b3bc5f68)
  pinned at `936630a` (code MIT, data CC BY 4.0). It holds the MQT Bench QASM circuits, all seven plans per
  circuit as text (`t1 t2 result … indices cost` per step), `Features.xlsx`, `measures.xlsx`, the split files
  with labels, and both XGBoost models as JSON. Training and evaluation need no GPU.
- **Their Julia stack is the reference.** Julia 1.9.1 Manifest: QXTools 1.0, OMEinsum 0.6.13, CUDA.jl 3.13.1,
  XGBoost.jl 2.5.5, FlowCutterPACE17_jll. We run it next to our code to check numbers, never as our pipeline.
- **The paper's seven generators make the candidate plans**: `mf`, `fc` (FlowCutter), `gn` (Girvan–Newman),
  `jdrasil`, `tamaki`, `minfill_mrs`, `minfillbg_mrs`. cotengra is used only as the contraction-tree data
  structure and to execute a given tree. CuPy is only the GPU array backend. cotengra's own optimisers are not
  in the paper. If we ever add them, they are an extra candidate labelled as an extension.
- Runtimes depend on the executor (CuPy with cuBLAS/cuTENSOR here, OMEinsum.jl in the paper), so phase 4 times
  both on the same GPU to see whether the labels move.

## Layout

```
pyproject.toml            (repo root) package "qctn", src layout; extras: gpu (cupy-cuda12x), slides (manim), dev
src/qctn/
  circuits.py             QASM / MQT Bench loading (qiskit), family tags
  tn.py                   circuit → tensor network, matching QXTools conventions (tensor ids, open/closed legs)
  plans/
    tree.py               contraction tree (cotengra ContractionTree interop), order ↔ tree, tree stats
    authors.py            read and write the authors' plan text format
    qxtools.py            QXTools' planning path, ported exactly: line graph, min_fill (mf), tree decomposition
                          -> elimination order -> plan (phase 3)
    girvan_newman.py      gn: Girvan-Newman on the tensor graph, reverse removal order (phase 3)
    pace.py               FlowCutter, Tamaki, Jdrasil and the two MinFill PACE heuristics as subprocesses (phase 3)
    generators.py         the seven generators on any qctn.tn network (phase 3)
    elimination.py, separators.py, td.py, exact.py      (to move from slides/traces, phase 6)
  features.py             the 18 features of Table 2 (c, p, k, d per step; cost weights w_i)
  execute/                CuPy executor for a tree; timing: 1 warm-up + 50 runs, synchronised, complex64
  rank/                   labels log(1+Sp), grouped splits, XGBoost rank:ndcg / rank:pairwise,
                          metrics Top-1/3, Regret1/3, win rate, NDCG@3
  julia/                  thin bridge that runs the authors' code headless and exports CSV
experiments/              one script per paper table (t06_id.py, t07_ood.py, t08_kendall.py, t11_zeroshot.py, …)
tests/                    unit tests and cross-checks against the authors' data
external/                 gitignored: authors' repo, solver sources, Julia depot
results/                  small CSV/JSON tracked; raw timings gitignored
```

## Phases

Each phase ends with a check against the authors' numbers.

0. **Setup.** Root `pyproject.toml` with one uv `.venv` (Python 3.12) that the slides also use. Clone the
   authors' repo at `936630a` into `external/` and rejoin `Data/plans/plans_part_*`. Julia 1.9.1 via juliaup
   in the home directory, and `Pkg.instantiate` against their Manifest (it appears to be two concatenated
   blocks; repair if so). A user-local JDK for the PACE solvers.
1. **Ranker on their data (CPU).** Score their JSON models with Python xgboost on their features and splits:
   Tables 6 and 7 exactly. Retrain with the Table 4 hyperparameters: Table 6 within seed noise. Their
   Notebook 5 in Julia as the reference.
2. **Plans and features.** Parse all their plans into `qctn` trees; writing them back reproduces the files.
   Reimplement the 18 features and match `Features.xlsx` (record τ, q, ε once inferred). Build the network
   from their QASM and check tensor and index structure against their plan files.
3. **Generators.** Regenerate the seven plans per circuit and compare cost and width with theirs.
4. **GPU timing on V100.** Time every plan with the CuPy executor and with their OMEinsum/CUDA.jl code on
   the same V100. Compare with each other and with the paper's Volta results (Tables 8, 9, 11).
5. **End to end.** Our circuits, plans, features and timings, then the ranker, reported next to Tables 6–11.
   Ampere (RTX A6000) results are quoted only; there is no A6000 here.
6. **Slides migration**, any time after phase 0. `slides/traces` keeps only the figure builders and imports
   `qctn`; all 161 tests pass and regenerating `slides/data` gives no diff.

## Constraints on DGX106

- 8× Tesla V100-SXM2-16GB, shared: take one free GPU per run (`CUDA_VISIBLE_DEVICES`, check `nvidia-smi`).
- Driver 535 (CUDA 12.2): CUDA 12 builds only (`cupy-cuda12x`); CUDA 13 has dropped Volta.
- No root: Julia, the JDK and solver builds live in the home directory or `external/`.
