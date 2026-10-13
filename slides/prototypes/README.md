# Figure prototypes

These are the 16 figures marked **Port** in PROPOSAL.md, exported from the Claude.ai study sessions where they were first drawn.
Open `index.html` (any browser, no server needed) to see them all.

## How Claude Code should use them

- They are **visual references**, not code to copy into the deck. Keep the layout, the colour roles, the labelling and the interaction design.
- They were drawn against the chat's stylesheet; `shim.css` imitates it so they render standalone. In the deck, map classes to `theme.css` tokens: `c-coral → accent`, `c-purple → derived`, `c-teal → side-b`, `c-gray → muted`, `th → 600-weight label`, `ts → secondary label`.
- Any number drawn in a prototype (ranks, bag sizes, widths) was hand-placed or computed in the browser. In the deck it must come from a trace written by `traces/trace_core.py` and covered by a test.
- The three steppers (F06, F12, F16) run their algorithms in JavaScript. Rebuild F06 and F12 on `TraceStepper`; F16 may stay interactive for Q&A, with its readouts checked against `separator_report()`.

## Per-figure notes

| ID | File | Kind | Porting note |
| --- | --- | --- | --- |
| F01 | `F01-pipeline-roadmap.html` | static | Drop the covered/next legend (it tracked our study progress); keep the stage layout. |
| F03 | `F03-ring-elimination.html` | static → stepper | Rebuild as a TraceStepper over data/F03.json; keep the three-panel layout as slide 3's opening frame. |
| F04 | `F04-carving-trees.html` | static + fragments | Reveal each tree edge's open-index set per click; ranks come from the carving DP. |
| F05 | `F05-treewidth-spectrum.html` | static | Fine as is; colours map to theme tokens. |
| F06 | `F06-elimination-stepper.html` | interactive | Replace the in-browser algorithm with TraceStepper + traces from trace_core.py; keep readouts, bar trace, label modes. |
| F09 | `F09-bowtie-graphs.html` | static | Keep; positions move into a trace if the bowtie is animated later. |
| F10 | `F10-bowtie-tree-vs-list.html` | static + fragments | Reveal merges one per click; numbers from carving_dp and best_caterpillar. |
| F11 | `F11-treewidth-catalog.html` | table | Becomes an HTML/KaTeX table on slide 11; highlight the lattice row. |
| F12 | `F12-nested-dissection-stepper.html` | interactive | Replace with TraceStepper over traces from nested_dissection / hybrid_cut_and_sweep / column_sweep. |
| F14 | `F14-separator-tree.html` | static | Bag numbers must come from the ND trace (per-level max bag). |
| F15 | `F15-separator-size-vs-balance.html` | static | Keep; part A/B colours = derived/side-b tokens. |
| F16 | `F16-separator-playground.html` | interactive | Keep interactive for Q&A; verify its readouts against separator_report() in tests. |
| F17 | `F17-edge-cut-index-separator.html` | static | Keep. |
| F24 | `F24-kernel-two-routes.html` | static | Backup slide B1. |
| F25 | `F25-autompo-bond.html` | static | Backup slide B3. |
| F26 | `F26-mpo-automaton.html` | static | Backup slide B4. |
