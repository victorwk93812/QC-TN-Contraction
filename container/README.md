# Toolchain container

Bare metal only needs the Python `.venv` (traces, tests, `build_deck.py`, `serve.py`).
The image carries everything else that is a pain to install without root:
Node + npm deps, Chromium, Manim/ffmpeg/pango, TeX Live + biber, Java.

GitHub Actions (`.github/workflows/container.yml`) builds `container/Dockerfile` on
pushes to `main` that touch it or the dependency files, and publishes
`ghcr.io/victorwk93812/qc-tn-toolchain`. Singularity pulls it here without root.

```sh
sh container/pull.sh                       # -> container/qctn-toolchain.sif (gitignored)

cd slides
../container/run.sh sh export/vendor.sh        # deck/vendor (reveal.js, KaTeX, fonts)
../container/run.sh sh export/render_clips.sh  # deck/media (Manim clips + posters)
.venv/bin/python export/build_deck.py          # bare metal, as before
python3 tools/serve.py 8001 &                  # bare metal; the container shares the host network
../container/run.sh npm run pdf                # export/out/deck.pdf via DeckTape
../container/run.sh node export/check_steppers.mjs

cd ../report && ../container/run.sh latexmk    # report/output/
```

`run.sh` mounts the repo at its host path and keeps the working directory, so every
output lands in the normal tree and the bare-metal workflow picks it up unchanged.
It also creates `slides/node_modules` as a symlink to the image's copy; on the host
that link dangles, which is harmless (and gitignored).

Building without CI needs Docker, or Singularity `--fakeroot` (an `/etc/subuid`
entry), neither of which this account has on DGX106.

## Status (2026-11-11)

The CI image is published and `sh container/pull.sh` pulls it on DGX106 without root (no Docker or
`--fakeroot` needed). Checked there on 2026-11-11: `export/vendor.sh`, `export/shots.mjs` and
`export/check_steppers.mjs` run in the container (all 347 stepper states consistent). A worktree can
reuse the main checkout's image with `QCTN_SIF=<main checkout>/container/qctn-toolchain.sif`.
`shots.mjs` and `check_steppers.mjs` read the server port from `PORT` (default 8001), which helps on a
shared node where 8001 may be taken. Manim clips, the PDF export and the report build were not re-run for this note.
