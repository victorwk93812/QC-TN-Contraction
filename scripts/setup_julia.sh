#!/bin/sh
# Instantiate the authors' two Julia environments (reference stack only), with
# the depot under external/. Needs Julia 1.9.1 from juliaup in ~/.juliaup:
#   curl -fsSL https://install.julialang.org -o juliaup-install.sh
#   sh juliaup-install.sh --yes --add-to-path=no --path ~/.juliaup --startup-selfupdate 0
#   ~/.juliaup/bin/juliaup add 1.9.1
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MAIN=$(dirname "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)")
EXT=${QCTN_EXTERNAL:-$MAIN/external}
export JULIA_DEPOT_PATH=$EXT/julia-depot
JULIA="$HOME/.juliaup/bin/julia +1.9.1"
# PyCall gets its own venv so its build path doesn't depend on a worktree.
PY=$EXT/julia-env/py
[ -x "$PY/bin/python" ] || uv venv -q --python 3.12 "$PY"
uv pip install -q --python "$PY/bin/python" qiskit==2.6.1
export PYTHON=$PY/bin/python
"$PYTHON" "$HERE/split_julia_manifest.py" "$EXT/authors" "$EXT/julia-env"
for env in rank qx; do
  echo "== $env"
  $JULIA --project="$EXT/julia-env/$env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()'
done
