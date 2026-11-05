#!/bin/sh
# Pull the toolchain image as a Singularity SIF (no root needed).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIF=${QCTN_SIF:-$ROOT/container/qctn-toolchain.sif}
TAG=${1:-latest}
singularity pull --force "$SIF" "docker://ghcr.io/victorwk93812/qc-tn-toolchain:$TAG"
echo "image: $SIF"
