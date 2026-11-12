#!/bin/bash
# Phase 4, small run: the release's plans of the 17 QASM circuits, CuPy executor (dense and hyperindex modes),
# then the same stack's QXTns `contract_tn!` on a few plans as a reference. Submit from the repository root
# (a worktree is fine) after `mkdir -p results/raw/p4`:
#     sbatch scripts/slurm/p4_small.sbatch
# Paths come from the checkout: the venv and external/ of the main checkout unless QCTN_PYTHON / QCTN_EXTERNAL
# are set. Output: results/raw/p4/<job id>/ and results/raw/p4/slurm-<job id>.out (gitignored).
#SBATCH --job-name=qctn-p4-small
#SBATCH --partition=nchc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/p4/slurm-%j.out
set -uo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
PY=${QCTN_PYTHON:-$MAIN/.venv/bin/python}
EXT=${QCTN_EXTERNAL:-$MAIN/external}
OUT=$ROOT/results/raw/p4/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export OMP_NUM_THREADS=8 PYTHONPATH=$ROOT/src QCTN_EXTERNAL=$EXT
echo "node $(hostname -s), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
date +%s > "$OUT/t_start"

timeout 1260 "$PY" experiments/p4_timing.py --out "$OUT" --warmup 1 --repeats 50 --budget 5 --deadline 1020
echo "python exit $?"
date +%s > "$OUT/t_python"

PLANS=""
for c in pricingcall_nativegates_ibm_qiskit_opt3_5 dj_indep_qiskit_40 pricingcall_nativegates_ibm_qiskit_opt3_9; do
  for g in mf fc gn jdrasil minfill_mrs minfillbg_mrs tamaki; do PLANS="$PLANS ${c}_pla_$g"; done
done
JULIA_DEPOT_PATH=$EXT/julia-depot timeout 480 "$HOME/.juliaup/bin/julia" +1.9.1 --project="$EXT/julia-env/qx" \
  experiments/julia/p4_qxtns.jl "$EXT/authors" "$OUT/julia" 10 $PLANS
echo "julia exit $?"
date +%s > "$OUT/t_end"
