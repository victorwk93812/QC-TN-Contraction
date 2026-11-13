# Release issues: where the paper, the code and the data of the release differ

A consolidated list for checking before contacting the authors. "Paper" is arXiv 2608.05819 (section and page
numbers from the PDF, as in [`code-review-guide.md`](code-review-guide.md)); "release" is
[`alfred-miquel/LTR-Quantum-Contraction-Plans` at `936630a`](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/tree/936630aa161abb7bcab1557c91137696b3bc5f68),
and `file l. N` refers to that commit. Each entry says what we found and how; it doesn't say why the
difference exists, which the release doesn't show. Details and the commands that rerun each check are in the linked
[`deviations.md`](deviations.md) sections. Entries are grouped by what they affect; within a group, the ones with
the largest effect on the published numbers come first.

## Labels and timing

### 1. The release contains no GPU timing code; the plan files' time column is a mock run

- **Paper.** Sec. 5.1 and 5.5: each plan is executed with OMEinsum on CUDA, one warm-up and 50 measured runs, the
  mean reported, Float32, `CUDA.synchronize` around runs, JIT, transfers and allocation excluded; RTX A6000 for the
  labels, a V100 PCIe for the cross-GPU tables.
- **Release.** The only timed contraction is `contract_tn_rank_mock!` (`src/funcions_article_IA.jl` l. 28–71): `@timed`
  around each `contract_pair_rank!` (l. 41), the seconds written as the last column of the plan line (l. 48).
  Notebooks 1_a (cell 18) and 1_b (cell 16) call it with `mock=true`, which makes QXTns build index-only results with
  no data (`contract_pair`, QXTns `tensor_network.jl` l. 256–258); 1_a cell 23 and 1_b cell 21 say so ("we do not
  contract 'really' in a GPU"). Nothing in `src/` or the notebooks uses BenchmarkTools, `CUDA.synchronize`, a warm-up,
  repeats or Float32 (search in deviations.md, Phase 4).
- **Evidence.** Running the release's function with `mock=true` on our machine gives per-step times that follow the
  release's column step by step (on the 109 plans of the 17 QASM circuits, Spearman over the steps of a plan median
  0.93, range 0.73–0.98; median step 286 µs against the file's 312 µs; `results/p4q_steps.csv`). The release's sums contain occasional single steps of 0.1–1.3 s
  (first-call compilation or garbage collection), e.g. `dj_indep_qiskit_40_pla_jdrasil` 2.24 s in total, of which
  one step is 1.34 s.
- **Effect.** The labels (speedups in `Data/splits`) can't be regenerated from the release; the time column isn't a
  GPU time and isn't the labels' numerator or denominator.
- **Sections.** [Phase 4](deviations.md#phase-4-executing-and-timing-plans), [Phase 4b](deviations.md#phase-4b-what-the-labels-measure).

### 2. Labels that can't come from dense contraction

- **Paper.** Plans are executed and timed; those over 10 minutes or out of memory are excluded (Sec. 5.1).
- **Data.** `pricingcall_nativegates_ibm_qiskit_opt3_21_pla_jdrasil` (a netcon plan, entry 15) needs a rank-55 dense
  intermediate (2^55 complex numbers), but has a label (speedup 1.10). In QXTns' hyperindex form its largest stored
  tensor has rank 19.
- **Effect.** The labelled runs used the hyperindex form, as QXTns' non-mock `contract_pair` does; the paper's
  description (a pairwise GEMM per step, Sec. 4.1–4.2, Eq. 5) is of dense contraction. The features are computed
  from the dense ranks.
- **Section.** [Phase 4](deviations.md#phase-4-executing-and-timing-plans).

### 3. The labels track host-side overhead, not GPU work

- **Paper.** The labels are GPU execution times of the plans (Sec. 5.1, 5.5); the features are built to describe GPU
  work (GEMM shapes, flops, ranks; Sec. 4, Table 2), and the paper states that runtime is not a monotone function of
  flops (App. A.7).
- **Evidence (our measurements, stated at the strength they support).** On the 17 QASM circuits (109 plans, a V100,
  `results/p4q_summary.csv`): the authors' library call (QXTns `contract_tn!`, ComplexF32, 1 warm-up + 50 runs) spends
  83–98% of its time in host-side index and hyperindex bookkeeping; the einsums alone take 0.6–10%, and the same call
  on CPU arrays is as fast or faster than on the GPU on 15 of the 17 circuits (0.72–1.02×). The library call, its bookkeeping phases and the release's mock function and
  column rank plans like the labels (mean within-circuit Spearman 0.28–0.46, 95% intervals above 0; other host-bound
  variants 0.10–0.17, intervals including 0); every GPU-work measure (einsums alone, our CuPy runs, flops) doesn't
  (means −0.18 to −0.14). A re-measurement with the labels' own noise (entry 4) would reach about 0.57, so the host-overhead
  timings reproduce about half of the reproducible part. Over all 223 labelled circuits
  (`results/p4q_label_stats.csv`, `results/p4q_label_flops.csv`), the labels are unrelated to flops in dense or
  hyperindex form (median Spearman 0.04 and 0.11, positive meaning more flops go with larger speedups) and correlate
  with the plan files' mock bookkeeping time per step (0.36). In hyperindex form 1379 of the 1463 split plans need
  under 2^36 flops, less than 40 ms of arithmetic at the V100's peak rate, against 0.1 s or more of per-step overhead.
- **Not established.** That the labels are exactly QXTns times: their host CPU (AMD EPYC 7282), GPU (RTX A6000),
  session order and the code that timed them are not in the release. Compilation is not the explanation: first runs
  (up to 125 s of compilation, in a session or in a fresh Julia process) don't rank plans like the labels.
- **Effect.** The rankers learn which plans have cheaper bookkeeping in this stack, not which use the GPU better. A
  different contraction library, or larger circuits where arithmetic dominates, could order the same plans differently.
- **Section.** [Phase 4b](deviations.md#phase-4b-what-the-labels-measure).

### 4. Identical plans with different labels (label noise)

- **Paper.** Seven distinct candidate plans per circuit (Sec. 1, 5.1).
- **Data.** 17 split rows are step-for-step the same plan as their circuit's MinFill plan (phase 3b,
  `results/p3b_flags.csv`): 12 look like solver timeouts falling back to MinFill (`minfillbg_mrs` on `GROVER_9/11/13`,
  `QNN_25`, `QWALK_6/7`, pricing-call 21/23/25; `jdrasil` on pricing-call 23/25; `minfill_mrs` on pricing-call 25),
  5 are unexplained (`fc`, `gn`, `minfill_mrs`, `minfillbg_mrs`, `tamaki` on `RQC_5_16`, 382 tensors). Their
  speedups, which should be 1, range from 0.87 to 1.19 (GROVER, QNN, QWALK, RQC_5_16) and from 0.45 to 0.97 (the
  pricing-call circuits, all below 1). In the training split, `RQC_3_7_40` and `RQC_3_7_52` list 12 plans twice with
  the same features and speedups that differ by up to 0.84 (phase 1).
- **Effect.** From these copies, one measurement's log speedup has a noise level of about 0.1
  (`results/p4q_label_noise.csv`). A re-measurement with that noise would agree with the labels only to a
  within-circuit Spearman of about 0.6 and pick the same best plan in about 40% of circuits
  (`results/p4q_label_stats.csv`, rows `replicate`). That bounds what any timing, ours included, can reproduce. The
  ranker is trained on identical feature rows with different grades.
- **Sections.** [Phase 3](deviations.md#phase-3-the-seven-generators), [Phase 3b](deviations.md#phase-3b-duplicate-candidates).

### 5. MinFill rows with a speedup other than 1

- **Paper.** Eq. 12: Sp_i = T(P_mf) / T(P_i), so MinFill's speedup is 1.
- **Data.** 26 `_pla_mf` rows of the split files have another value: 11 in train (`RQC_3_7_40` and ten `RQC_4_5_*`),
  3 in ID (`RQC_4_5_32`, `_34`, `_36`), 12 in OOD (`QFT_6`–`QFT_19`, up to 686.1 for `QFT_6`). For the 21 of these
  circuits that `Data/measurements/measures.xlsx` covers, every row of the circuit equals the `measures.xlsx` value
  times one constant, the MinFill row's split speedup (`results/p4q_label_rescaled.csv`): the order within the circuit
  is the same, the normalisation isn't. `measures.xlsx` has MinFill at exactly 1 for all 209 of its circuits.
- **Effect.** Regret and win rate use the values as they are; MinFill's ID win rate is 52 instead of 50 and its OOD
  win rate 59.7, and Regret1 of the MinFill baselines depends on these rows.
- **Sections.** [Phase 2b](deviations.md#phase-2b-corrected-splits) (baselines), [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits) (Data).

### 6. `measures.xlsx` and the second GPU

- **Paper.** Sec. 5.1 and 7: measurements on the RTX A6000 and the V100 PCIe; Tables 8–11 compare the two.
- **Data.** `Data/measurements/measures.xlsx` has one speedup column (`speedup_mf`, 1367 rows), no raw times and no
  second GPU (`Data/measurements/measurements.md` l. 1). It lacks 97 split rows and differs from the split files in
  155 rows: the 84 QFT rows and 60 `RQC_4_5_*` rows of entry 5 (constant factor per circuit) and 11 rows of the
  repeated `RQC_3_7_40` / `_52` plans.
- **Effect.** Tables 8–11 (cross-GPU) can't be checked from the release.
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits) (Data).

## Ranker and evaluation

### 7. Each Table 6 row combines two models; rounds and parameters

- **Paper.** Sec. 5.4 (p. 11): two optimised rankers, rounds by early stopping on NDCG@3 in grouped 5-fold CV, refit
  and evaluated once; Table 4: `opt_ndcg` eta 0.01, subsample 0.7.
- **Release.** `src/Preliminar_getting_results.jl` l. 478 `num = (1,65)` and l. 500 `num = (16,14)`: Top-1 and
  Regret1 come from one model, Top-3 and Regret3 from another with a different round count (`test_model`, l.
  440–447). `opt_ndcg` uses eta 0.02 and subsample 0.8 (l. 33, 35, 472, 474). No CV or early-stopping code (search S3
  in code-review-guide.md).
- **Evidence.** With these settings Table 6's four ranking values reproduce exactly, in Python and in the release's
  Julia (`results/t06_t07.csv`, `results/julia_t06_t07.csv`); no single model at rounds 1–300 gives a whole row, and
  Table 4's `opt_ndcg` parameters never give the published Top-3 or regrets.
- **Effect.** With one model per objective and rounds from grouped CV on the training split, ID Top-1 is 56–60
  (`opt_ndcg`) and 44–60 (`opt_pair`), Top-3 88–92 and 68–88; OOD values are lower than Table 7's (phase 1 table).
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits).

### 8. Regret on the label, not on the speedup

- **Paper.** Eq. 14–15: Regret1 = 1 − Sp_1 / Sp*, on speedups.
- **Release.** `src/Preliminar_getting_results.jl` l. 298 `sp = g.label    # label vs speedup`, then l. 370
  `regret1 = 1 - best_in_rank1 / best_real`: the ratio of labels log(1 + Sp).
- **Effect.** The published regrets reproduce on labels only; on speedups MinFill's Regret1 is 0.1015, not 0.0735.
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits).

### 9. No code for Table 7, the win rate or the baselines

- **Paper.** Table 7 (OOD), the win rate (Sec. 6.2), the random, `mf_ndcg` and `mf_rand` baselines (Sec. 5.4).
- **Release.** Notebook 5, cell 1, says it replicates Tables 6 and 7, but no `df_ood_*` file is loaded anywhere
  (cell 9; `Choose_model` l. 466–507). `evaluate_tops_and_regrets` returns only Top-k and regrets (l. 410–412). No
  baseline code (searches S1, S2).
- **Evidence.** Exact expectations of the random baseline on the published splits are 16.13 / 46.40 / 0.0798 /
  0.0204 / 56.95 (ID), against the paper's 22.95 / 58.44 / 0.0699 / 0.0137 / 70.16; MinFill+random Top-3 is 44.93
  against 28. With our reading of the win rate, `opt_ndcg` ID stays at 80–84 over rounds 1–300, against 92.
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits).

### 10. The stored models were fitted on all 225 circuits

- **Paper.** Sec. 5.4: a deployment model may be retrained on all labelled data and isn't evaluated with the test
  metrics.
- **Release.** `models/models.md` l. 1 calls them the models "in production used in the article". `opt_pair.json` is
  the Notebook 5 recipe fitted on train + ID + OOD (predictions within 1e-5 at 1850 rounds). Notebook 4 passes raw
  feature values (cell 10) to models trained on standardised ones.
- **Effect.** They score Top-1 84–97% on the test splits, far above the tables, and 8–16% on raw features. Table 6
  doesn't use them; this matters only for anyone using them as published.
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits).

## Features and splits

### 11. Jdrasil rows carry MinFill's features

- **Paper.** Each row is one plan's 18 features (Table 2).
- **Data.** Of 224 `_pla_jdrasil` rows in the split files, 213 have exactly the features of the same circuit's
  `_pla_mf` plan, where `Data/features/Features.xlsx` and the plan files give different values. Their speedups are
  their own (e.g. `GHZ_100`: Jdrasil 1.027, MinFill 1.000, same features). 13 split rows match no xlsx row
  (`results/p2_split_sources.csv`).
- **Effect.** The ranker sees pairs of identical feature rows with different labels in 213 circuits. Recomputing
  them from the plan files lowers `opt_ndcg` ID Top-1 from 56–60 to 44–52 and OOD Top-1 from 27–34 to 16–24 (phase 2b).
- **Sections.** [Phase 2](deviations.md#phase-2-plans-features-scaler-networks), [Phase 2b](deviations.md#phase-2b-corrected-splits).

### 12. Standardised features with an unknown 773-row scaler

- **Paper.** Features are fed to XGBoost (Sec. 5.3); no standardisation is described.
- **Data.** In each split file the 13 model features are z-scored with one fixed μ, σ per feature (the other five
  columns raw; residual ≤ 1e-14 against `Features.xlsx`). The integer features fit only a fit on N = 773 rows with
  the sample std. None of the tried row sets gives μ and σ (`results/p2_scaler.csv`); the release has no scaling code.
- **Effect.** Trees are invariant to affine maps up to float32 rounding; through that rounding the scaler changes
  the models and the metrics by about the size of phase 2b's noise floor.
- **Sections.** [Phase 2](deviations.md#phase-2-plans-features-scaler-networks), [Phase 2b](deviations.md#phase-2b-corrected-splits).

### 13. Repeated plans and circuits without a MinFill plan in the splits

- **Data.** `RQC_3_7_40` and `RQC_3_7_52` list plans twice in the training split (12 duplicated rows, groups of 14
  and 11), same features, speedups up to 0.84 apart. `RANDOM_11`, `RQC_5_32` and
  `pricingcall_nativegates_ibm_qiskit_opt3_25` have no MinFill row, though speedup is defined against it.
- **Section.** [Phase 1](deviations.md#phase-1-ranker-on-the-published-splits) (Data).

## Plans and circuits

### 14. Plan names and the circuits they were made on

- **Paper.** Circuits from MQT Bench and QXTools (Sec. 5.1).
- **Data.** Only 17 of the 332 circuit names in the plan files have a QASM file of that name. Replaying all plans
  (`results/p3_networks.csv`, `results/circuits_map.csv`): all `RQC_3_7_50` plans replay on depth 49; five of seven
  `RQC_4_6_30` plans on depth 32; `GROVER_6` on the 5-qubit Grover circuit (no 6-qubit file); `dj_indep_qiskit_25`
  on `pricingcall_nativegates_ibm_qiskit_opt3_25`; `RQC_4_20`, `RQC_4_60` and `RQC_4_11` contract only part of their
  network. Within one RQC name the Jdrasil plan was often made without the final Hadamard layer, and the QNN and
  Qwalk plans without generator suffix with an earlier QASM reader that isn't in the release.
- **Effect.** A plan's features and timing belong to the network it replays on, not always the one its name says.
- **Sections.** [`circuits.md`](circuits.md), [Phase 3](deviations.md#phase-3-the-seven-generators) (Networks).

### 15. No code for the seven generators; netcon plans; MinFill fallbacks

- **Paper.** Sec. 5.1 names MinFill and FlowCutter (via QXTools), Girvan–Newman, Jdrasil, Tamaki and two MinFill
  heuristics of Terrioux et al., without parameters or time limits.
- **Release.** No generator code; Notebooks 1_a / 1_b call `flow_cutter_contraction_plan(tnc; time=10)`.
  Reconstructed (phase 3): edge elimination orders on the plain-bond line graph through QXTools
  `order_to_contraction_plan`; the solvers stopped at 10 s with MinFill as fallback. 39 plans of the seven generators
  use hyperedge (netcon) steps (results named `t<n>`), unlike the rest; `QWALK_3_pla_Flow_cutter` and
  `QWALK_6_pla_flow_cutter` reuse a result name.
- **Effect.** `mf`, `gn` and `minfill_mrs` reproduce every non-netcon release plan, `fc` and `minfillbg_mrs` all but
  two; Tamaki and Jdrasil (randomised, seed and version unknown) don't. The netcon plans are up to 42.7 costlier in
  log2 flops (dense count).
- **Section.** [Phase 3](deviations.md#phase-3-the-seven-generators).

### 16. Plan files that don't parse or don't match `Features.xlsx`

- **Data.** `Qwalk_5` (line 1009) and `RANDOM_11_pla_mf` (line 71) are truncated and fail the release's own reader;
  six files (`RQC_4_11_pla_mf`, `RQC_4_20_pla_fc/_gn/_mf`, `RQC_4_60_pla_fc/_mf`) give other features and step counts
  than their `Features.xlsx` rows, and `RQC_4_20`/`_60` contain outer products on which the release's reader indexes
  `comptador[0]` (`src/Funcions_de_Features_v_5.jl` l. 582). None of these is in the splits except through entry 11.
- **Section.** [Phase 2](deviations.md#phase-2-plans-features-scaler-networks).

### 17. Gate values in the release's stack

- **Release.** The QASM reader registers every `u2`, `u1`, `p` and `cu1` under one label (`src/funcions_qasm.jl` l.
  671, 699, 727, 796) and QXZoo keeps the first matrix stored under a label (`gate_map.jl` l. 205–210), so all gates
  of one of these kinds get the first one's matrix in a Julia session. QXTns reshapes two-qubit gate matrices
  column-major, so each two-qubit gate acts with its qubits' roles exchanged; QXTns `push!` copies the wrong half when
  the SVD rank is 1 (`tensor_network_circuit.jl` l. 93, 97). Gate matrices are ComplexF64 (paper: Float32).
- **Effect.** Amplitudes, not structure: plans, features and contraction cost are unchanged. Timings could differ
  through the element type only.
- **Section.** [Phase 4](deviations.md#phase-4-executing-and-timing-plans).
