#!/bin/sh
# Fetch the authors' release at the pinned commit and unpack its circuits and plans.
# external/ is shared by all worktrees: it lives in the main checkout. Idempotent.
set -e
PIN=936630aa161abb7bcab1557c91137696b3bc5f68
PLANS_SHA=20cc9e9ebd9fa0908a3dea8a956036d2d3e099f2533a9562afdb55990554ed4b
MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
EXT=${QCTN_EXTERNAL:-$MAIN/external}
mkdir -p "$EXT"
if [ ! -d "$EXT/authors/.git" ]; then
  git clone -q https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans.git "$EXT/authors"
fi
git -C "$EXT/authors" checkout -q "$PIN"
D=$EXT/authors-data
mkdir -p "$D/plans" "$D/circuits"
cat "$EXT"/authors/Data/plans/plans_part_* > "$D/plans.tar.gz"
echo "$PLANS_SHA  $D/plans.tar.gz" | sha256sum -c -
tar xzf "$D/plans.tar.gz" -C "$D/plans"
tar xzf "$EXT/authors/Data/circuits/circuits_qasm.tar.gz" -C "$D/circuits"
echo "authors $PIN: $(find "$D/plans" -type f | wc -l) plan files, $(find "$D/circuits" -type f | wc -l) circuits in $D"
