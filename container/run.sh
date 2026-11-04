#!/bin/sh
# Run one command inside the toolchain image. The repo is mounted at its host path
# and the current directory is kept, so outputs land in the working tree as usual.
# Usage: container/run.sh <command> [args...]
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SIF=${QCTN_SIF:-$ROOT/container/qctn-toolchain.sif}
[ -f "$SIF" ] || { echo "no image at $SIF; run: sh container/pull.sh" >&2; exit 1; }
# The Node scripts import from slides/node_modules; point it at the image's copy.
# (Dangling on the host by design; delete it if you ever install Node on bare metal.)
[ -e "$ROOT/slides/node_modules" ] || [ -L "$ROOT/slides/node_modules" ] \
  || ln -s /opt/node/node_modules "$ROOT/slides/node_modules"
exec singularity exec --cleanenv --bind "$ROOT" --pwd "$PWD" "$SIF" "$@"
