#!/bin/bash
# Phase 4 full run: one array task per shard of results/p4full_shards.csv (experiments/p4full_timing.py run).
# All plans of a circuit run in one task on one GPU. Submit from the repository root (a worktree is fine) after
# `mkdir -p results/raw/p4full`, with the array range, throttle, partition and time limit on the command line:
#     sbatch --array=0-2 -p nchc,normal -t 01:00:00 scripts/slurm/p4full_array.sbatch     # pilot
#     sbatch --array=3-23%8 -p nchc,normal -t 01:30:00 scripts/slurm/p4full_array.sbatch  # the rest
# Restartable: output goes to results/raw/p4full/$P4FULL_TAG (default run1); a rerun of a shard skips the
# circuits whose file is complete. Paths come from the checkout: the venv and external/ of the main checkout
# unless QCTN_PYTHON / QCTN_EXTERNAL are set. Extra arguments go to the script (e.g. --deadline 3000).
#SBATCH --job-name=qctn-p4full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=results/raw/p4full/slurm-%A_%a.out
set -uo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
PY=${QCTN_PYTHON:-$MAIN/.venv/bin/python}
EXT=${QCTN_EXTERNAL:-$MAIN/external}
OUT=$ROOT/results/raw/p4full/${P4FULL_TAG:-run1}
mkdir -p "$OUT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$ROOT/src QCTN_EXTERNAL=$EXT
echo "node $(hostname -s), job ${SLURM_JOB_ID:-} (array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
"$PY" experiments/p4full_timing.py run --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID:?}" "$@"
echo "python exit $?"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
