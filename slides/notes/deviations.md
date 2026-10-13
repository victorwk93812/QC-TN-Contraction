# Where the deck departs from PROPOSAL.md (and why)

## Slide numbering
Two primer slides were added after proposal slide 3 (user request, Oct 1 2026), so **deck number = proposal number + 2**
from proposal slide 4 on (53 slides + B1–B6). Section ids keep the proposal numbers (`s04` … `s51`); the primer slides
are `s03a` (bags and width on a star, two orders) and `s03b` (treewidth on tree / cycle / 3×3 grid / K₅).
Proposal slide 10 (spectrum) became "Certificates that treewidth is large" (K₅ and the 4×4 grid); the spectrum
picture moved into the primer. Act I is 9 min, Act II 9 min (F00).

## Numbers and claims
- **Grid "best of 10 seeds"**: the proposal's log₂ costs (13.20, 18.35, 29.29, …) are the first seed (0–9) that reaches
  the minimum width, not the cheapest seed (13.14, 18.32, 29.04). `fast_elim.best_width_first_seed`.
- **Bowtie tree (F10)**: the prototype tree ((UP)(VQ))(H(RS)) has rank 2 but cost 52; the slide shows the optimum
  ((H(RS))(PU)) Q V with cost 50 (only two optimal trees, mirror images).
- **"Naive ND 68" on the cubic graph**: not reproducible without a definition. Measured ND variants: spectral 57,
  FlowCutter-lite 64, BFS levels 67, Kernighan–Lin (seed 0) 78 (`traces/separators.py`).
- **"Outside-in peeling"**: on the 10×10 grid MinFill works corners-first and in a checkerboard; mean elimination step
  rises from boundary to centre (32, 54, 62, 66, 67) but one boundary vertex is last. Slide title says "on average".
- **Slide 19 / slide 48 (proposal numbering)**: the paper has no runtime data for equal-width plans and no
  feature-importance ranking; slide 19 uses our own equal-width plans with different GEMM shapes, slide 48 shows the
  kept / discarded features of Table 2 (user decision).
- **Theorem attributions corrected** (see `theorem-facts.md`): Markov–Shi runtime uses tw of the circuit graph;
  O*(1.7347ⁿ) is Fomin–Todinca–Villanger 2015 (not Fomin–Villanger 2012); "minimal separators of minimal
  triangulations" is Parra–Scheffler 1997; modularity in A_ij form is Clauset–Newman–Moore 2004; PACE 2017 heuristic
  track was won by Tamaki et al., FlowCutter second.
- **Index graph node order**: `core.index_graph` now sorts legs so traces do not depend on Python's hash seed.
