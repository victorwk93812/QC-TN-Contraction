# QC_TN_Contraction

Study of how tensor-network contraction orders are found for quantum-circuit simulation, from treewidth
heuristics (MinFill, nested dissection / FlowCutter, Girvan–Newman, PACE treewidth solvers) to the
learned plan ranker of arXiv 2608.05819. The repository holds a written report, a 60-minute talk, and
(later) a reproduction of the paper's results.

## Layout

```text
QC_TN_Contraction/
├── README.md            this file
├── LICENSE, LICENSE-content   MIT (code) and CC BY 4.0 (content), see below
├── references/          the papers we read (PDFs not tracked — download links below)
├── report/              written report (plain beamer; builds with stock TeX Live)
│   ├── QC_TN_Contraction.tex, preamble.tex, bibliography.bib
│   ├── sections/        one file per section
│   └── output/          latexmk build output (not tracked)
├── slides/              the talk: trace-driven reveal.js deck — see slides/README.md
│   ├── PROPOSAL.md      spec: slide list, figures, acceptance numbers
│   ├── traces/          Python: algorithms + one builder per figure; tests/ assert every number on a slide
│   ├── data/            generated traces (JSON, tracked: the deck reads only these)
│   ├── deck/            reveal.js source: slides/*.html, js/, css/, shell.html
│   ├── manim/           Manim scenes for the five clips
│   ├── export/          build, screenshots, checks, PDF and notes export
│   ├── notes/           paper facts, verified theorem statements, deviations from the proposal
│   ├── prototypes/      chat prototypes of the figures (visual reference only)
│   └── tools/serve.py   static server with HTTP Range support (solver repos are cloned here, not tracked)
├── container/           toolchain image (Node, Chromium, Manim, TeX, Java) for hosts without them — see container/README.md
└── src/                 planned: Python reproduction of arXiv 2608.05819 — see src/README.md
```

`src/` will hold the Python reproduction of arXiv 2608.05819 and is the source of truth for the algorithms;
`slides/traces/` will import it. The agreed layout and phases are in `src/README.md`.

## Tech stack

| Part | Tools |
|---|---|
| Algorithms and figure data | Python 3.12, networkx 3.7, numpy 2.5, pytest 9.1 (`slides/requirements.txt`, `uv`) |
| Treewidth solvers | Java (OpenJDK 26): Tamaki et al. PACE 2017 heuristic, Jdrasil — cloned into `slides/tools/`, built with `javac` |
| Slides | reveal.js 5.2, KaTeX 0.18, plain SVG renderers, Inter / JetBrains Mono (vendored for offline use) |
| Clips | Manim Community 0.21 (needs TeX Live for formulas), ffmpeg |
| Export and checks | DeckTape 3.16 (PDF), Playwright 1.63 with the system Chromium (screenshots, stepper checks) |
| Report | TeX Live 2025: beamer, biblatex (`style=phys`) + biber, latexmk (pdfLaTeX) |

## Quick start

The talk (details, including the full setup of a fresh clone, in `slides/README.md`):

```sh
cd slides
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
PUPPETEER_SKIP_DOWNLOAD=1 PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install
sh export/vendor.sh
.venv/bin/python export/build_deck.py      # generates deck/index.html — required before serving
python3 tools/serve.py                     # http://localhost:8001/deck/
```

No Node, Chromium, Manim or TeX on the host (e.g. a cluster without root)? Pull the toolchain image
and prefix those steps with `container/run.sh` — see `container/README.md`.

The report:

```sh
cd report && latexmk                       # pdfLaTeX + biber, output in report/output/ (see .latexmkrc)
```

## References

Download the PDFs into `references/` (they are not tracked):

| File | Paper |
|---|---|
| `Learning_Rank_Tensor_Network_Contraction_Plans_GPU_Accelerated_Quantum_Circuit_Simulation.pdf` | A. M. Pastor, M. Castillo, J. M. Badia, *Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated Quantum Circuit Simulation*, arXiv:2608.05819 (2026). <https://arxiv.org/abs/2608.05819> |
| `Simulating_Quantum_Computation_Contrating_Tensor_Networks.pdf` | I. L. Markov, Y. Shi, *Simulating quantum computation by contracting tensor networks*, SIAM J. Comput. 38(3):963–981 (2008); arXiv:quant-ph/0511069. <https://arxiv.org/abs/quant-ph/0511069> |

Every other result cited in the talk is listed with its source in `slides/notes/theorem-facts.md`, and every number
quoted from the paper in `slides/notes/paper-facts.md`.

## License

- **Code — MIT** (`LICENSE`): `slides/traces/`, `slides/export/`, `slides/manim/`, `slides/deck/js/`,
  `slides/deck/css/`, `slides/tools/`, and the build configuration files.
- **Content — CC BY 4.0** (`LICENSE-content`): the slide text (`slides/deck/slides/`, `slides/deck/shell.html`),
  `slides/notes/`, `slides/PROPOSAL.md`, `slides/prototypes/`, `report/`, and the generated figure data in `slides/data/`.

Third-party material keeps its own terms and is not redistributed here: the papers in `references/` (not
tracked), and reveal.js, KaTeX, the fonts and the PACE solvers (downloaded at setup, not tracked).
