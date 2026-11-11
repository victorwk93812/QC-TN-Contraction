# Setup for the reproduction

All third-party material is gitignored and pinned. It lives in `external/` in the main
checkout, so every worktree shares one copy (`qctn.paths.external_dir()`, or set `$QCTN_EXTERNAL`).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[gpu,dev]'   # drop gpu on a machine without NVIDIA GPUs
sh scripts/setup_external.sh                               # authors' release at 936630a; unpacks plans and circuits
sh scripts/setup_jdk.sh                                    # Temurin 21 in ~/.local/opt, for the PACE solvers
sh scripts/setup_solvers.sh                                # FlowCutter, Tamaki, Jdrasil, two MinFill heuristics (phase 3)
sh scripts/setup_julia.sh                                  # authors' Julia stack, as a reference; install steps inside
.venv/bin/python -m pytest
```

`setup_julia.sh` splits the authors' `Manifest.toml` first: the file holds two complete manifests, so it
isn't valid TOML. `external/julia-env/qx` is the QXTools/CUDA.jl stack and `external/julia-env/rank` the
XGBoost.jl stack. Run them with
`JULIA_DEPOT_PATH=external/julia-depot ~/.juliaup/bin/julia +1.9.1 --project=external/julia-env/<env>`.

`.venv` also runs the slides' tests (`cd slides && ../.venv/bin/python -m pytest`). Their own scripts still
expect `slides/.venv` until the slides code moves onto `qctn` (phase 6).

CuPy's `ctk` extra provides the CUDA headers, which DGX106 doesn't have. The bundled runtime is CUDA 12.9 and
works on driver 535 through CUDA minor-version compatibility.

`setup_solvers.sh` clones each solver at a pinned commit into `external/solvers/<name>` and builds it there
(g++ and the JDK above). `qctn.plans.pace` runs the binaries; tests that need them skip when they're missing.
