# Contraction Heuristics Talk — Build Proposal

Oct 1, 2026 · @JCMaxwell

## Purpose and hand-off

Build a 60-minute talk of 51 slides plus 6 backups on how tensor-network contraction orders are found, ending at the learning-to-rank result of arXiv 2608.05819. Every claim is shown twice: once as a picture that moves, once as a formula or theorem that pins it down.

Audience: graduate students and researchers in physics and quantum information. They know tensors and linear algebra; they may not know treewidth, separators or graph heuristics.

Deliverables Claude Code should produce:

- A reveal.js deck with KaTeX math and live SVG widgets, one presenter click per algorithm step.
- A Python package that computes every figure's data as a JSON trace and checks every number quoted on a slide.
- Four to six Manim clips for the animations that need smooth choreography.
- A static PDF of the whole deck, every animation step as its own page, for venues without a working browser.

To hand this over: export this doc as Markdown, save it as `PROPOSAL.md` at the repository root, and install the companion skill `tn-slide-deck` in `.claude/skills/`. Then ask Claude Code to build the deck from `PROPOSAL.md` using that skill, one milestone at a time.

## Narrative arc

The talk moves from a cost problem, to a vocabulary, to a map of seven techniques, to four deep dives, to a learned ranker that beats them all. Slide numbers run 1 to 51, with backups B1 to B6.

&#91;embedded content: narrative arc · 5 acts, 4 deep dives, about 60 min\]

Each deep dive answers the same five questions in the same order, so the audience can compare techniques without a separate summary.

Rules every slide follows:

1. **Picture first, formula second.** A slide opens on its visual; the formula or theorem appears as the last fragment, as the summary of what was just seen.
2. **One change per click.** An algorithm advances one step per presenter click, with the bag size, the width so far and the log₂ cost updating beside it.
3. **Every number is reproducible.** A number shown on a slide comes from a trace file and has a unit test; nothing is typed by hand.
4. **The same test graphs throughout.** The four-tensor ring, the bowtie, the 7×7 and 6×6 grids, the 11-vertex separator graph and one quantum-circuit network recur in every act.

The comparison axes each deep dive must fill, so Act IV can set them side by side:

| Axis | Measured as |
| --- | --- |
| Graph it works on | Index graph or tensor graph |
| Build direction | Bottom-up, top-down or search |
| Width on the shared graphs | Exact value from the trace package |
| Total cost and peak memory | log₂ cost and largest intermediate, χ = 2 |
| Plan shape | Tree depth and how the expensive steps are arranged |
| Guarantee | Theorem it carries, if any |

## Act I — Why contraction order matters (6 slides, 7 min)

The opening claim: the same network costs wildly different amounts depending on the order you contract it, and finding the best order is NP-hard. Figure ids refer to the inventory below.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 1 | Title | Static, ring of four tensors | — |
| 2 | One network, two orders | F04 carving trees: ((AB)(CD)) rank 2 vs ((AC)(BD)) rank 4 | Pairwise step cost 2^(m+n+k) |
| 3 | Contracting = summing out indices | F03 ring: eliminate i, fill edge l–j appears | T\_lj = Σ\_i A\_li B\_ij |
| 4 | Cost, memory, width | F03 continued, the four-step table built row by row | Step cost χ^\|bag\|; peak memory χ^width |
| 5 | Why we can't just search | Counter of orderings vs n, then the theorem | Markov–Shi; NP-hardness |
| 6 | Roadmap of the talk | F00 narrative arc | — |

Key formulas, typeset on slides 3 to 5:

```latex
\text{step cost} = \chi^{\,|N(v)|+1}, \qquad \text{intermediate size} = \chi^{\,|N(v)|}, \qquad \text{total} = \sum_t \chi^{\,|\mathrm{bag}_t|}
```

```latex
\textbf{Markov--Shi (2008).}\quad \mathrm{cc}(G) = \mathrm{tw}\big(L(G)\big), \qquad \text{time} = \mathrm{poly}(|G|)\cdot \exp\!\big(O(\mathrm{tw}(L(G)))\big)
```

```latex
\textbf{Arnborg--Corneil--Proskurowski (1987).}\quad \text{Deciding } \mathrm{tw}(G) \le k \text{ is NP-complete.}
```

Speaker point for slide 5: the hardness result is what makes this a heuristics talk. Bodlaender's linear-time algorithm for fixed k exists, but its constants grow so fast in k that it is impractical beyond very small widths.

## Act II — The shared language (8 slides, 10 min)

Every technique later in the talk is judged with the same three numbers: fill, width and cost. This act defines them on two graphs and shows the two contraction models that use them.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 7 | Two graphs from one network | F03 left and middle panels; F17 edge cut = index separator | Bond cut ↔ vertex separator lemma |
| 8 | The elimination game | F06 MinFill stepper in "one step" mode on the ring | fill(v), width(π), tw as a minimum |
| 9 | Tree decompositions | F14 separator tree used as an example decomposition | Three axioms; chordal completion |
| 10 | What makes treewidth large | F05 spectrum: tree, cycle, grid, clique | Lower bounds; excluded grid theorem |
| 11 | A catalog to remember | Table of graph families, lattice row highlighted | tw(L^d lattice) = Θ(L^(d−1)) |
| 12 | Carving width | F04 carving trees, cut labels revealed edge by edge | cw definition; cost bound 1.5·cw |
| 13 | Three schedules, one network | F09 bowtie graphs, then F10 tree vs list | Exact values: 3/62, 2/50, 3/58 |
| 14 | How the widths relate | Small diagram: tw(G), tw(L(G)), cw(G) with arrows | Degree-dependent bounds |

Definitions and theorems for these slides:

```latex
\operatorname{fill}(v)=\#\{\{a,b\}\subseteq N(v): a\not\sim b\}, \qquad \operatorname{width}(\pi)=\max_t |N(v_t)|, \qquad \operatorname{tw}(G)=\min_\pi \operatorname{width}(\pi)
```

```latex
\text{Tree decomposition } (T,\{B_t\}):\ \ \bigcup_t B_t=V;\ \ \forall uv\in E\ \exists t: \{u,v\}\subseteq B_t;\ \ \{t: v\in B_t\}\text{ is connected};\quad \operatorname{width}=\max_t|B_t|-1
```

```latex
\operatorname{tw}(G)=\min\{\,\omega(H)-1 : H\supseteq G,\ H \text{ chordal}\,\}
```

```latex
\operatorname{tw}(G)\ \ge\ \max\big(\omega(G)-1,\ \operatorname{degeneracy}(G),\ k \text{ if } G \text{ has a } k\times k \text{ grid minor}\big)
```

```latex
\textbf{Excluded grid theorem.}\ \exists f:\ \operatorname{tw}(G)\ge f(k)\Rightarrow G \text{ has a } k\times k \text{ grid minor},\quad f \text{ polynomial (Chekuri--Chuzhoy)}
```

```latex
\operatorname{cw}(G)=\min_{\text{binary trees}}\ \max_{\text{tree edges } e} |\delta(e)|\ \ge\ \Delta(G), \qquad 2(m+n+k)\le 3\,\operatorname{cw}\ \Rightarrow\ \text{step cost}\le \chi^{1.5\,\mathrm{cw}}
```

The last bound follows from m+k ≤ cw, n+k ≤ cw and m+n ≤ cw for the two inputs and the output of a step. Slide 13 must say that a list schedule is a caterpillar tree, and the bowtie proves it can be strictly worse than a branching tree.

## Act III — The seven techniques at a glance (6 slides, 5 min)

The [paper](https://arxiv.org/abs/2608.05819) generates seven structurally different plans per circuit and learns which one runs fastest. This act places them on one map before the deep dives take each family apart.

&#91;embedded content: the seven candidates by graph and build direction\]

Reveal the map one column per click: bottom-up first, since the audience has just seen the elimination game.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 15 | The paper's pipeline | F01: circuit → seven generators → features → ranker | Problem statement as ranking |
| 16 | Seven candidates on one map | F02, revealed column by column | — |
| 17 | Bottom-up against top-down | F06 MinFill frame beside F12 first cut, same grid | Local fill vs global separator |
| 18 | What each family optimizes | Table: fill, cut size and balance, modularity, width itself | Four objectives |
| 19 | Why rank instead of choose | Same-width plans with different runtimes, from F23 | runtime ≠ f(width) alone |
| 20 | The scorecard | Empty comparison-axes table, filled at the end of each deep dive | — |

The four objectives for slide 18, stated precisely:

```latex
\underbrace{\min_v \operatorname{fill}(v)}_{\text{greedy}}\qquad \underbrace{\min |S| \text{ s.t. balance} \ge \beta}_{\text{separators}}\qquad \underbrace{\max\ Q}_{\text{communities}}\qquad \underbrace{\min \operatorname{width}(T)}_{\text{solvers}}
```

## Deep dive 1 — Greedy elimination: MinFill and its PACE 2017 variants (6 slides, 7 min)

MinFill is cheap, exact on chordal graphs and a strong baseline, but it decides from distance-2 information only. Three of the paper's seven candidates belong to this family.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 21 | The rule | F06 stepper, first five steps on the 6×6 grid: corners first | v\* = argmin fill(v), ties by degree |
| 22 | Why it is reasonable | Chordal graph eliminated with zero fill throughout | Simplicial safety; Fulkerson–Gross |
| 23 | Outside-in peeling | F06 played to the end; then F08 elimination-time heat map | Width ≈ 1.3L vs optimal L |
| 24 | Width and cost disagree | F07 chart: width and log₂ cost vs L, MinFill vs column sweep | log₂ cost ≈ max bag + log₂(#steps near max) |
| 25 | Ties, chains and blind spots | Tie histogram at L = 20; bag-size bars showing the late sequential chain | Minimum fill-in is NP-hard |
| 26 | Verdict from the paper | Bar: MinFill Top-1 rate vs learned ranker | 12% Top-1; random ranker beats it \~70% |

```latex
v^\star=\arg\min_v\ \operatorname{fill}(v)\quad(\text{ties: smaller degree, then random})
```

```latex
\textbf{Simplicial safety.}\ N(v)\text{ a clique}\ \Rightarrow\ \operatorname{tw}(G)=\max\big(|N(v)|,\ \operatorname{tw}(G-v)\big)
```

```latex
\textbf{Fulkerson--Gross (1965).}\ G \text{ is chordal} \iff G \text{ has a perfect elimination ordering (zero fill)}
```

```latex
\textbf{Yannakakis (1981).}\ \text{Minimum fill-in is NP-hard;\ MinFill is its one-step greedy approximation.}
```

Speaker points, all measured in our sessions and reproduced by the trace package:

- On L×L grids MinFill's width is about 1.3L (26 at L = 20, 32 at L = 24) against the column sweep's L.
- Below L ≈ 22 MinFill still has the lower total cost, because its large bags occur in a short burst at the end.
- At L = 24 its largest intermediate is 2^32 entries against 2^24 for the sweep, a 256× memory gap.
- Tie-breaking alone moves the width from 26 to 30 across 40 seeds at L = 20.
- On random cubic graphs it is strong: width 56 at n = 320, beating MinDegree (62) and a naive nested dissection (68).
- Each update touches only vertices within distance 2, costing O(|U|Δ²), but Δ grows to about the width after fill.

To verify before the talk: the exact refinements in the two Terrioux et al. PACE 2017 entries. Present them as "MinFill plus refinements" until their descriptions are checked.

## Deep dive 2 — Separators, nested dissection and FlowCutter (9 slides, 11 min)

Separators turn the problem upside down: pick the vertices to eliminate last, from global structure that MinFill cannot see. The act ends on FlowCutter, which searches the size-versus-balance trade-off instead of insisting on perfect halves.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 27 | What a separator is | F15 corner cut vs middle column on a 7×7 grid | (A, S, B) partition; α-balance |
| 28 | Minimal, minimum, balanced | F16 playground with presets {c}, {x,y}, {c,d}, {e,f,g}, {x,y,z} | Full-component test for minimality |
| 29 | Proving a cut is smallest | F16 with the two disjoint a–h paths highlighted | Menger; max-flow min-cut |
| 30 | Why the separator goes last | F12 stepper, A side only: fill stays inside A ∪ S | Separator-last lemma |
| 31 | Nested dissection | F12 balanced ND to the end, then F14 separator tree | Bag = separator + inherited boundary |
| 32 | What theory guarantees | Lattice and planar examples beside the theorems | RS, Dvořák–Norin, Lipton–Tarjan |
| 33 | Balance has a price | F13 chart: width vs L for four orders | ND ≈ 1.5L; hybrid = L |
| 34 | FlowCutter | F18 Manim clip: source and target regions grow, cuts trace a Pareto front | Incremental max-flow; Pareto front |
| 35 | From cuts to a plan | Separator tree → tree decomposition → pairwise contraction tree | Conversion is not unique |

```latex
\textbf{Menger.}\ \min\{|S|: S \text{ separates } u,v\}=\max\{\#\text{ internally vertex-disjoint } u\text{--}v \text{ paths}\}
```

```latex
\textbf{Separator-last lemma.}\ \text{Eliminate } A \text{ before } S\cup B \Rightarrow \text{no fill edge joins } A \text{ and } B;\ \ \operatorname{width}(\pi)=\max(w_A,\,w_B,\,|S|-1)
```

```latex
\operatorname{tw}(G)\le k\ \Rightarrow\ \exists S,\ |S|\le k+1,\ \text{every component of } G-S \text{ has} \le n/2 \text{ vertices}
```

```latex
\textbf{Dvo\v{r}\'ak--Norin.}\ \text{every subgraph has a balanced separator of size} \le s\ \Rightarrow\ \operatorname{tw}(G)\le 15s
```

```latex
\textbf{Lipton--Tarjan.}\ G \text{ planar} \Rightarrow \text{a } \tfrac23\text{-balanced separator of size } O(\sqrt n) \Rightarrow \operatorname{tw}(G)=O(\sqrt n)
```

```latex
\textbf{George (1973).}\ \text{Nested dissection on an } L\times L \text{ grid: fill } O(n\log n),\ \text{operations } O(n^{3/2}),\ n=L^2
```

Speaker points for slide 33, all measured: balanced ND reaches width 45 at L = 31, MinFill 41, the column sweep 31, and one balanced cut plus inward sweeps 31. In the carving picture, balanced bisection gives intermediates up to rank 1.25L and top merges costing 2^(1.5L). The lesson is that George's optimality is for polynomial costs; in tensor networks a factor 1.5 in width is a factor 2^(0.5L) in time and memory.

Still to develop before slide 34 is final: FlowCutter's mechanics from Hamann and Strasser's Pareto-bisection paper. The deck needs a small educational reimplementation (grow source and target, max-flow after each piercing step, record cut size against balance) that runs on the 7×7 grid and on a circuit network.

## Deep dive 3 — Community detection: Girvan–Newman (4 slides, 5 min)

Girvan–Newman also builds the plan top-down, but it splits the network where clusters are weakly linked, not where cuts are small. Its dendrogram is directly a contraction tree over the tensors, so it lives in the carving picture.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 36 | Communities in a tensor network | Two dense blocks joined by a few bonds, coloured by community | Modularity Q |
| 37 | The algorithm | F20 stepper: remove the highest-betweenness edge, recompute, repeat | Edge betweenness |
| 38 | Dendrogram = contraction tree | F20 continued: splits read bottom-up as pairwise contractions | Carving width of the dendrogram |
| 39 | What it does not optimize | F21 table: width and cost of GN vs MinFill vs ND on the shared test graphs | No width guarantee; O(m²n) cost |

```latex
c_B(e)=\sum_{s\ne t}\frac{\sigma_{st}(e)}{\sigma_{st}}\qquad(\sigma_{st}=\#\text{ shortest } s\text{--}t \text{ paths},\ \sigma_{st}(e)=\#\text{ of those using } e)
```

```latex
Q=\frac{1}{2m}\sum_{i,j}\Big[A_{ij}-\frac{k_ik_j}{2m}\Big]\,\delta(c_i,c_j)
```

```latex
\text{Brandes: one betweenness pass } O(nm)\ \Rightarrow\ \text{Girvan--Newman } O(m^2 n) \text{ on unweighted graphs}
```

The key contrast for the audience: betweenness and modularity reward splitting where few shortest paths cross, which often coincides with small cuts but is not the same objective. Nothing bounds the width of the resulting tree, so its quality is an empirical question, answered on slide 39 with numbers the trace package computes.

To verify before the talk: which Girvan–Newman variant and stopping rule the paper's pipeline uses, and whether bond dimensions weight the betweenness.

## Deep dive 4 — Treewidth solvers: Jdrasil and Tamaki (3 slides, 4 min)

These two come from the PACE treewidth competitions and spend a time budget searching for a narrower decomposition, rather than building one in a single pass. Present them at the level of what they optimize and what their output looks like; the paper runs both as black boxes.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 40 | Treewidth as a search problem | Anytime curve: best width found vs time budget, for several solvers on one circuit | FPT and exact exponential bounds |
| 41 | Local improvement (Tamaki) | F22: a tree decomposition, widest bag highlighted, its neighbourhood re-solved, width drops | Width is a max, so fix the worst bag |
| 42 | A modular solver (Jdrasil) | Pipeline sketch: preprocessing, upper-bound heuristics, improvement | Minimal separators and potential maximal cliques |

```latex
\textbf{Bodlaender (1996).}\ \text{For fixed } k,\ \operatorname{tw}(G)\le k \text{ is decidable in } 2^{O(k^3)}\, n \text{ time}
```

```latex
\textbf{Fomin--Villanger (2012).}\ \text{Exact treewidth in } O^*(1.7347^{\,n}) \text{ time}
```

```latex
\textbf{Bouchitt\'e--Todinca.}\ \text{Every minimal triangulation } H\supseteq G \text{ has its separators among the minimal separators of } G
```

The last result ties this act back to deep dive 2: optimal decompositions are assembled from minimal separators, which is why exact solvers enumerate them. Slide 40's anytime curve should be measured by the trace package with each solver given 1 s, 10 s and 60 s.

To verify before the talk: the heuristic modes of Jdrasil and Tamaki that the paper used, their time limits, and both tools' PACE 2017 descriptions.

## Act IV — Head-to-head: from plan shapes to a learned ranker (6 slides, 8 min)

The seven generators produce plans of similar theoretical cost but different shapes, and on a GPU the shape decides the runtime. This act shows the shapes, the GEMM features that describe them, and how the paper learns to rank them.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 43 | Same width, different trees | F23: four contraction trees side by side, coloured by step cost | Critical path vs total work |
| 44 | From orders to pairwise plans | Elimination order → tree decomposition → contraction tree; parallel bonds merged | Conversion is many-to-one |
| 45 | A step is a GEMM | One step drawn as an (M×K)(K×N) product; feature groups as a tree | c, p, k, d primitives |
| 46 | Learning to rank | Seven plans per circuit as one group; label bars; ranked list | NDCG@k; label definition |
| 47 | Results | Top-1 and regret bars: MinFill-first baselines vs learned ranker; QFT hold-out; A6000 → V100 | Numbers from the paper's tables |
| 48 | What the model learned | Feature-importance bars grouped by feature family | — |

```latex
\text{step } i:\ \big(2^{m_i}\times 2^{k_i}\big)\cdot\big(2^{k_i}\times 2^{n_i}\big),\qquad c_i=m_i+n_i+k_i,\quad p_i=m_i+n_i,\quad k_i,\quad d_i=\text{shape imbalance}
```

```latex
y=\log\big(1+\text{speedup over the MinFill plan}\big),\qquad \mathrm{NDCG@}k=\frac{1}{\mathrm{IDCG@}k}\sum_{r=1}^{k}\frac{2^{\,y_{(r)}}-1}{\log_2(r+1)}
```

Speaker points: plans are aggregated into critical-load, granularity, parallelism, reduction-intensity and geometry features. Training splits by circuit so no circuit appears in both training and testing. The out-of-distribution test holds out the whole QFT family, and a separate test moves from an RTX A6000 to a V100.

Numbers already confirmed for slide 47: the baselines that put MinFill first find the fastest plan in only 12% of in-distribution circuit groups, and a random ranker's top pick beats the MinFill plan in about 70% of groups.

To verify before the talk: the exact definition of d\_i, the regret formula, the learned ranker's Top-1, Top-3 and regret values, and the feature-importance ranking, all from the paper's own tables.

## Act V — Conclusion and open questions (3 slides, 4 min)

Close on one chain of reasoning and four open problems. The chain: contraction cost is exponential in width, width is governed by separator size, heuristics differ in how they find separators, and runtime depends on plan shape beyond width.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| 49 | Four takeaways | F00 arc diagram, each act lit in turn | The summary chain below |
| 50 | Open questions | 2×2: theory vs practice, generate vs rank | FHL approximation bound |
| 51 | Thank you | Ring of four tensors contracting to a scalar | — |

```latex
\chi^{\operatorname{tw}(L(G))}\ \le\ \text{peak memory of the best order},\qquad \operatorname{tw}\ \asymp\ \text{balanced separator size},\qquad \text{runtime}=F(\text{plan shape},\ \text{hardware})
```

```latex
\textbf{Feige--Hajiaghayi--Lee (2008).}\ \text{Polynomial-time } O\big(\sqrt{\log \operatorname{tw}}\big)\text{-approximation of treewidth}
```

Takeaways to state on slide 49:

1. The order, not the network, decides the cost: one ring already shows rank 2 against rank 4.
2. Bottom-up greedy rules see only distance-2 structure; top-down separator methods see the global cut, but balance alone does not minimize width.
3. Width, FLOPs, memory and GPU runtime are four different objectives, and they disagree on real graphs.
4. Ranking a diverse pool of plans beats committing to any single heuristic.

Open questions for slide 50:

- **Objective.** Should generators optimize width, FLOPs, memory or GEMM geometry directly, and can a runtime model be put inside the generator instead of after it?
- **Transfer.** Labels are hardware-specific; how far does a ranker trained on one GPU carry to multi-GPU runs or to slicing, where some indices are fixed to fit memory?
- **Pool design.** Seven generators is a choice; adding hypergraph partitioners such as KaHyPar, or randomized restarts of each generator, may matter more than a better ranker.
- **Guarantees.** MinFill has no known approximation guarantee, while separator-based algorithms do; can a practical heuristic carry FHL-style guarantees on circuit networks?

## Backup slides — low rank across a cut

These six slides keep the earlier threads of our study available for questions. Their common message is that a cut decides cost everywhere: in kernels, in MPOs and in contraction orders.

| # | Slide | Visual | Math anchor |
| --- | --- | --- | --- |
| B1 | The kernel trick | F24 two routes to the same scalar | Mercer; tensor-product kernel factorizes |
| B2 | Kernel costs | Table: linear vs kernel SVM, training, memory, prediction | O(M²c\_K) to O(M³) training |
| B3 | AutoMPO, one bond | F25 bond decomposition: prefixes, suffixes, coefficient matrix | D\_n = 2 + rank M^(n) |
| B4 | The MPO as an automaton | F26 three-state automaton with A, B, C, D blocks | Environment induction |
| B5 | Long-range couplings | Singular values of J's off-diagonal block, exponential vs power law | D = 3 vs O(log(N/ε)) |
| B6 | One idea, four places | Four panels: Gram matrix, MPS bond, MPO bond, separator | Rank or size across a cut |

```latex
K \succeq 0:\ \sum_{j,k}c_jc_kK(x_j,x_k)\ge 0\ \ \forall c,\qquad \Lambda(x)=\bigotimes_{i=1}^N\phi(x_i)\ \Rightarrow\ \Lambda(x)^\top\Lambda(x')=\prod_{i=1}^N\phi(x_i)^\top\phi(x'_i)
```

```latex
H=H_{\le n}\otimes\mathbb 1+\mathbb 1\otimes H_{>n}+\sum_{a,b}M^{(n)}_{ab}\,L_a\otimes R_b,\qquad D_n=2+\operatorname{rank}M^{(n)}
```

The numbers for B5 come straight from the decomposition: for J\_ij = e^(−|i−j|/ξ) every off-diagonal block has rank 1, so D\_n = 3 at every bond, while a power law needs a bond dimension growing like log(N/ε).

## Figure inventory

Sixteen figures already exist as chat prototypes and only need porting to the trace-driven stack; the rest are new. Every figure reads its numbers from a trace file, never from hand-typed values.

| ID | Shows | Slides | Type | Status |
| --- | --- | --- | --- | --- |
| F00 | Narrative arc: five acts, lit one at a time | 6, 49 | Static SVG + fragments | New |
| F01 | Paper pipeline: circuit → generators → features → ranker | 15 | Static SVG | Port |
| F02 | Seven techniques mapped by graph used and build direction | 16–20 | Static SVG + fragments | New |
| F03 | Ring: tensor graph, index graph, eliminating i with fill l–j | 3, 4, 7 | Stepper | Port |
| F04 | Carving trees ((AB)(CD)) rank 2 vs ((AC)(BD)) rank 4 | 2, 12 | Static SVG + fragments | Port |
| F05 | Treewidth spectrum: tree, cycle, grid, clique with best cuts | 10 | Static SVG | Port |
| F06 | Elimination-game stepper: MinFill, MinDegree, column sweep on grid or cubic graph | 8, 21, 23 | Stepper | Port |
| F07 | Width and log₂ cost vs L, MinFill vs column sweep | 24 | Chart | New |
| F08 | MinFill elimination-time heat map, 10×10 grid | 23 | Static SVG | New |
| F09 | Bowtie: tensor graph and index graph with the hub's K₄ | 13 | Static SVG | Port |
| F10 | Bowtie: balanced tree (rank 2) vs best list (rank 3) | 13 | Static SVG + fragments | Port |
| F11 | Treewidth catalog of graph families | 11 | Table | Port |
| F12 | Nested-dissection stepper: balanced ND, one cut + sweeps, column sweep | 30, 31 | Stepper | Port |
| F13 | Width vs L: balanced ND, hybrid, column sweep, MinFill | 33 | Chart | New |
| F14 | Separator tree with bags 7, 10, 9, ≤ 8 | 31 | Static SVG | Port |
| F15 | Corner cut vs balanced middle column on 7×7 | 27 | Static SVG | Port |
| F16 | Separator playground: click to build S; presets; Menger paths | 28, 29 | Interactive | Port |
| F17 | Bond cut of tensor graph = vertex separator of index graph | 7 | Static SVG | Port |
| F18 | FlowCutter: regions grow, cuts found, Pareto front traced | 34 | Manim clip | New |
| F19 | FlowCutter Pareto front: cut size vs balance, 7×7 grid and one circuit | 34 | Chart | New |
| F20 | Girvan–Newman stepper: remove top-betweenness edge, dendrogram grows | 37, 38 | Stepper | New |
| F21 | Width and cost of GN, MinFill and ND on shared test graphs | 39 | Table | New |
| F22 | Local improvement: widest bag re-solved, width drops | 41 | Stepper | New |
| F23 | Four plan shapes side by side, coloured by step cost | 43 | Static SVG | New |
| F24 | Kernel trick: two routes to one scalar | B1 | Static SVG | Port |
| F25 | AutoMPO bond decomposition | B3 | Static SVG | Port |
| F26 | MPO automaton with A, B, C, D blocks | B4 | Static SVG | Port |

Manim clips, each under 40 s and also exported as a frame sequence for the PDF:

1. M1 — Column sweep against MinFill's outside-in ring on a 12×12 grid, frontier highlighted, bag-size bars racing.
2. M2 — The separator tree assembling level by level, inherited boundaries flowing down.
3. M3 — FlowCutter on the 7×7 grid (F18).
4. M4 — Girvan–Newman's dendrogram growing into a contraction tree.
5. M5 — Title animation: the four-tensor ring contracting to a scalar.

## Tech stack and architecture

Python computes every figure once as a JSON trace; reveal.js, Manim and the PDF exporter only draw it. This is the single rule that keeps slides, clips and tests consistent.

&#91;embedded content: build architecture · one trace, three renderers\]

The trace package is the only place algorithms live; renderers never recompute a width or a cost.

Repository layout:

```text
talk/
  PROPOSAL.md            this document, exported as Markdown
  traces/                Python package: graphs, elimination, carving DP, separators, GN, FlowCutter-lite
    tests/               one test per number in the acceptance table
  data/                  generated JSON, one file per figure id (F03.json …)
  deck/                  reveal.js project: index.html, slides/*.html, js/stepper.js, css/theme.css
  manim/                 scenes M1–M5 reading data/*.json
  export/                Playwright frame capture, DeckTape config, build scripts
```

A trace is the sequence of states, positions included, so geometry is fixed and identical in every renderer:

```json
{"id": "F03", "nodes": {"i": [340, 80], "j": [390, 130]},
 "edges": [["i", "j"]],
 "steps": [{"elim": "i", "bag": ["i", "j", "l"], "fill": [["l", "j"]], "width": 2, "log2cost": 3.0}]}
```

Renderer contract: `render(k)` draws state k from scratch, with no hidden mutable state. Reveal fragments then give forward, backward and jump for free:

```js
Reveal.on('fragmentshown',  e => e.fragment.dataset.step && fig.render(+e.fragment.dataset.step));
Reveal.on('fragmenthidden', e => e.fragment.dataset.step && fig.render(+e.fragment.dataset.step - 1));
```

Choices and the reasons behind them:

| Layer | Choice | Why |
| --- | --- | --- |
| Algorithms and numbers | Python, networkx, numpy | Same code verified every number in our sessions |
| Live slides | reveal.js, plain SVG, D3 transitions keyed by vertex id | Exact geometry and per-click control; vis.js physics layouts drift |
| Math | KaTeX in slides, LaTeX in Manim | Fast rendering; identical notation in both |
| Smooth choreography | Manim Community Edition | Python, reads the same JSON, native LaTeX |
| Backup and handouts | Playwright frames, DeckTape | A PDF that works without a browser |

Design system, shared by the web theme and Manim:

- Coral marks what matters now: the chosen vertex, the separator, the maximum bag. Purple marks fill and derived structure. Teal marks the second side of a cut. Gray is background.
- Geometry is hand-placed and fixed per figure; only the algorithm's state changes between frames.
- One change per click, with live readouts (bag size, width so far, log₂ cost) and a bar trace beside the graph.
- Thin strokes, dashed fill edges, no gradients or shadows, at least three readable type sizes.

The chat prototypes use the host page's classes (c-purple, th, ts) and CSS variables. Porting means replacing them with `deck/css/theme.css` tokens of the same meaning.

## Build plan and acceptance criteria

Build in nine milestones, each ending with passing tests and a deck that runs. Never start a later milestone while an earlier one fails its checks.

1. **M0 Scaffold.** Repository layout, Python environment with networkx, numpy and Manim Community Edition, a Node project with reveal.js and KaTeX, plus DeckTape and Playwright for export.
2. **M1 Trace package.** Graph builders, elimination with pluggable rules, fixed orders (sweep, nested dissection, hybrid), the exact carving DP, caterpillar search, separator tests, Girvan–Newman and a small FlowCutter. Every number in the table below is a unit test.
3. **M2 Design system and renderer.** Theme tokens, SVG helpers, and a `Stepper` whose `render(k)` draws step k from scratch; reveal fragments drive it forward and back.
4. **M3 Acts I and II.** Port F03, F04, F05, F09, F10, F11 and F17; slides 1–14.
5. **M4 Deep dives 1 and 2.** Port F06, F12, F14, F15 and F16; build F07, F08, F13 and F19; slides 21–35.
6. **M5 Deep dives 3 and 4.** Build F20, F21 and F22; slides 36–42.
7. **M6 Acts III to V and backup.** F00, F01, F02, F23 to F26; slides 15–20 and 43–51; paper numbers entered only after checking the paper.
8. **M7 Manim clips.** M1 to M5, each reading the same trace JSON as the web figure it accompanies.
9. **M8 Export and rehearsal.** One PDF page per animation step, embedded clips replaced by key frames, and a speaker-notes file with every number's source.

Numbers the trace package must reproduce exactly (randomized cases state their seed protocol in the test):

| Check | Expected | Notes |
| --- | --- | --- |
| Ring, index elimination | width 2, total cost 22 | χ = 2 throughout |
| Bowtie, best index-elimination order | width 3, cost 62 | exhaustive over 8! orders |
| Bowtie, best pairwise tree | max intermediate rank 2, cost 50 | subset DP; carving width 4 counts the hub |
| Bowtie, best list | max rank 3, cost 58 | none of 5040 orders reaches rank 2 |
| MinFill on L×L grid, best of 10 seeds | width 10, 15, 21, 26, 32 at L = 8, 12, 16, 20, 24 | log₂ cost 13.20, 18.35, 23.68, 29.29, 34.59 |
| Column sweep on L×L grid | width L | log₂ cost 14.70, 19.95, 24.83, 29.51, 34.06 at the same L |
| MinFill tie spread, L = 20, 40 seeds | width 26 to 30, median 28 | tie-break = degree, then seeded random |
| Balanced ND, 7×7 grid | width 9; level bags 7, 10, 9, ≤ 8 | split the longer side at its middle |
| Balanced ND vs hybrid | ND 9, 21, 33, 45 at L = 7, 15, 23, 31; hybrid = L | hybrid: one middle cut, inward sweeps |
| Separator playground graph | min a–h separator 2; {e,f,g} minimal not minimum; {x,y,z} not minimal | 11 vertices, 18 edges |
| Random 3-regular, n = 320, graph seed 1 | MinFill best 56, median 59.5; MinDegree median 62 | 20 tie-break seeds |
| Balanced bisection, grid tensor network, L = 32 | max intermediate rank 40, max step exponent 48 | carving model |

## References

Only the paper was opened and read during this work; the other entries are cited from memory and should be checked against the originals while building slides.

Primary source:

- [arXiv 2608.05819](https://arxiv.org/abs/2608.05819) — learning to rank tensor-network contraction plans on GPUs.

Theorems used on slides:

- Markov and Shi, *Simulating quantum computation by contracting tensor networks*, SIAM J. Comput. (2008).
- Arnborg, Corneil and Proskurowski, complexity of finding embeddings in a k-tree, SIAM J. Algebraic Discrete Methods (1987).
- Fulkerson and Gross, incidence matrices and interval graphs, Pacific J. Math. (1965).
- Yannakakis, computing the minimum fill-in is NP-complete, SIAM J. Algebraic Discrete Methods (1981).
- Robertson and Seymour, Graph Minors series; Chekuri and Chuzhoy, polynomial bounds for the grid-minor theorem, J. ACM (2016).
- Menger (1927); Ford and Fulkerson, max-flow min-cut (1956).
- Lipton and Tarjan, a separator theorem for planar graphs, SIAM J. Appl. Math. (1979).
- George, nested dissection of a regular finite element mesh, SIAM J. Numer. Anal. (1973).
- Dvořák and Norin, treewidth of graphs with balanced separations, J. Combin. Theory Ser. B (2019).
- Bodlaender, a linear-time algorithm for finding tree-decompositions of small treewidth, SIAM J. Comput. (1996).
- Fomin and Villanger, treewidth computation and extremal combinatorics, Combinatorica (2012).
- Bouchitté and Todinca, treewidth and minimum fill-in via minimal separators, SIAM J. Comput. (2001).
- Feige, Hajiaghayi and Lee, approximation algorithms for minimum-weight vertex separators, SIAM J. Comput. (2008).

Heuristics in the candidate pool:

- Hamann and Strasser, graph bisection with Pareto optimization (FlowCutter), ACM J. Exp. Algorithmics (2018).
- Girvan and Newman, community structure in social and biological networks, PNAS (2002); Brandes, a faster algorithm for betweenness centrality (2001).
- PACE 2016 and 2017 treewidth track reports, for Jdrasil, Tamaki and the Terrioux et al. MinFill variants.

Tools: [reveal.js](https://revealjs.com), [KaTeX](https://katex.org), [Manim Community Edition](https://www.manim.community), [DeckTape](https://github.com/astefanutti/decktape), [Playwright](https://playwright.dev).
