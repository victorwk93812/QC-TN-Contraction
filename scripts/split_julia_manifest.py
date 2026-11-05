#!/usr/bin/env python3
"""Split the authors' Manifest.toml into the two Julia environments it concatenates.

Their file holds two complete manifests (project hashes c11edc… and ea369c…), so it
is not valid TOML. The first is the QXTools/CUDA stack, the second the XGBoost/XLSX
stack. Each gets its own Project.toml, keeping only the direct deps it resolves.

Usage: split_julia_manifest.py <authors repo> <output dir>
"""
import re
import sys
import tomllib
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
blocks = re.split(r"(?m)^(?=julia_version = )", (src / "Manifest.toml").read_text())
blocks = [b for b in blocks if b.strip()]
assert len(blocks) == 2, f"expected two manifests, found {len(blocks)}"

deps = tomllib.loads((src / "Project.toml").read_text())["deps"]
for name, block in zip(("qx", "rank"), blocks):
    known = tomllib.loads(block)["deps"]
    kept = {k: v for k, v in deps.items() if k in known}
    env = out / name
    env.mkdir(parents=True, exist_ok=True)
    (env / "Manifest.toml").write_text(block)
    lines = ["[deps]"] + [f'{k} = "{v}"' for k, v in sorted(kept.items())]
    (env / "Project.toml").write_text("\n".join(lines) + "\n")
    print(f"{name}: {len(known)} packages, direct: {', '.join(sorted(kept))}")
