#!/bin/bash
# Phase 4b: experiments/julia/p4q_qxtns.jl on one GPU. Submit from the repository root (a worktree is fine) after
# `mkdir -p results/raw/p4q`:
#     sbatch scripts/slurm/p4q_julia.sbatch main  <circuit>,<circuit>,...   # the runs below, in one job
#     sbatch scripts/slurm/p4q_julia.sbatch fresh <circuit>,<circuit>,...   # one Julia process per plan
#     sbatch scripts/slurm/p4q_julia.sbatch custom key=value ...           # one run with these arguments
# `main` runs, in this order and each in its own Julia process:
#   1. QXTns contract_tn! with the paper's protocol: cold run, 1 warm-up, 50 runs (10 min per plan at most);
#   2. the other modes (decomp, mock, authors, authors_mock, ome), 1 warm-up + 10 runs (2 min per plan and mode);
#   3. qxtns and ome with CPU arrays, 1 warm-up + 5 runs (1 min).
# `fresh` starts Julia once per plan and times contract_tn! cold (the first contraction in the process, compile
# included), then 1 warm-up and 2 runs.
# Paths come from the checkout: external/ of the main checkout unless QCTN_EXTERNAL is set. Output:
# results/raw/p4q/<job id>/{main,others,cpu,fresh/<plan>}/ and results/raw/p4q/slurm-<job id>.out (gitignored).
#SBATCH --job-name=qctn-p4q
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/raw/p4q/slurm-%j.out
set -uo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "$ROOT"
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
EXT=${QCTN_EXTERNAL:-$MAIN/external}
OUT=$ROOT/results/raw/p4q/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export JULIA_DEPOT_PATH=$EXT/julia-depot JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8
echo "node $(hostname -s), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}, args: $*"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
lscpu | grep -E "^Model name|^CPU\(s\)|^Thread"
printf '%s\n' "$@" > "$OUT/args.txt"
date +%s > "$OUT/t_start"
jl() { "$HOME/.juliaup/bin/julia" +1.9.1 --project="$EXT/julia-env/qx" experiments/julia/p4q_qxtns.jl "$EXT/authors" "$@"; echo "julia exit $?"; }
GENS=mf,fc,gn,jdrasil,minfill_mrs,minfillbg_mrs,tamaki
case "$1" in
  main)
    jl "$OUT/main" circuits="$2" gens=$GENS modes=qxtns warmup=1 repeats=50 budget=600
    jl "$OUT/others" circuits="$2" gens=$GENS modes=decomp,mock,authors,authors_mock,ome warmup=1 repeats=10 budget=120
    jl "$OUT/cpu" circuits="$2" gens=$GENS modes=qxtns,ome device=cpu warmup=1 repeats=5 budget=60 ;;
  fresh)
    for c in ${2//,/ }; do
      for g in ${GENS//,/ }; do
        [ -f "$EXT/authors-data/plans/resultats/resultats_${c}_pla_${g}.txt" ] || continue
        jl "$OUT/fresh/${c}_pla_${g}" circuits="$c" gens="$g" modes=qxtns warmup=1 repeats=2 steps_dump=0
      done
    done ;;
  custom)
    shift; jl "$OUT" "$@" ;;
  *) echo "unknown recipe $1"; exit 2 ;;
esac
date +%s > "$OUT/t_end"
