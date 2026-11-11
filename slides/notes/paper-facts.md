# arXiv 2608.05819 — facts extracted for the deck

Pastor, Castillo, Badia, *Learning to Rank Tensor Network Contraction Plans for GPU-Accelerated
Quantum Circuit Simulation*, arXiv:2608.05819v1, 6 Aug 2026. Extracted from the PDF in
`references/` (printed page = PDF page). Only values that appear in the paper are listed;
"NOT IN PAPER" marks things the proposal assumed but the paper does not state.

## Pipeline (Sec 5.1, p.8–9; Sec 5.5, p.11–12)
- 225 circuit groups, seven candidate plans per circuit (Sec 1, p.2). Sources: MQTBench and QXTools-generated instances.
- Families (Table 10, p.17): GHZ 18, Grover 7, QNN 24, Quantum Walk 5, RQC 86, VQE 7, Deutsch–Jozsa 3, Ground State 1, Pricing Call 11, Portfolio QAOA 1, QFT-based 62.
- Generators (p.9): MinFill and FlowCutter "obtained through QXTools"; Girvan–Newman community detection [6]; Jdrasil heuristic solver; Tamaki local-improvement heuristic; two MinFill-based heuristics by Terrioux et al. (PACE 2017 [41]).
- Generator tools for the other five, their time limits, output types, conversion to pairwise plans: NOT IN PAPER.
- Execution: one warm-up + 50 measured repetitions, mean reported; Float32; CUDA.synchronize around runs; transfers/allocation/plan creation/JIT excluded. Timeout 10 min or OOM ⇒ candidate failed (some groups have < 7 valid plans).
- Software: Julia, QXTools.jl (networks + plans), OMEinsum.jl (contraction), CUDA, XGBoost.jl.
- Host: 2× AMD EPYC 7282 (16 cores, 2.8 GHz), 256 GiB DDR4.
- GPUs (Table 5, p.12): RTX A6000 (Ampere, 10,752 CUDA cores, 48 GB GDDR6, 768 GB/s, 38.71 TFLOPS FP32) is the source backend; Tesla V100 PCIe 32GB (Volta, 5,120 CUDA cores, 32 GB HBM2, 897 GB/s, 14.13 TFLOPS) only for Sec 7 cross-GPU.

## Girvan–Newman, Jdrasil, Tamaki, Terrioux et al.
Variant, stopping rule, bond-dimension weighting, modes, time limits, Terrioux variant names: all NOT IN PAPER.

## Step primitives (Sec 4.1–4.2, p.6–7; App. A.2–A.3)
- O_i = contract_{S_i}(A_i, B_i); k_i = |S_i|; m_i = rank(A_i) − k_i; n_i = rank(B_i) − k_i.
- GEMM (M×K)(K×N), M = 2^{m_i}, K = 2^{k_i}, N = 2^{n_i} (Eq. 5).
- c_i = m_i + n_i + k_i (Eq. 6), "work exponent", FLOPs_i = Θ(2^{1+c_i}) (Eq. 18).
- p_i = m_i + n_i (Eq. 7), output-size exponent.
- k_i (Eq. 8), reduction exponent, proxy for reuse / arithmetic intensity.
- d_i = |m_i − n_i| / (m_i + n_i + ε), ε ≪ 1 (Eq. 9): 0 ≈ square, 1 ≈ tall-skinny.
- c_max = max c_i; I_max = argmax set; w_i = 2^{c_i − c_max} / Σ_ℓ 2^{c_ℓ − c_max} (Eq. 10–11, 19). Tiny step threshold τ = 6.

## Features (Sec 4.3; Table 2, p.8; Sec 5.3, p.9–10)
Blocks: complexity/critical load; sequence/granularity; parallelism dynamics; reduction/intensity; geometry/memory access.
18 features initially; 4 discarded (max_cost, log2_sum_flops, topq_mean_cost, avg_cost); union of selected sets 14; each optimized model uses 13 (12 shared + 1 specific: std_cost only in opt_ndcg, costw_red_rank only in opt_pair).
Kept and shared: n_steps, frac_tiny_steps, max_out_rank, avg_out_rank, costw_out_rank, p_at_max_cost, max_red_rank, k_at_max_cost, max_asym, avg_asym, costw_asym, d_at_max_cost.
Feature-importance ranking / values: NOT IN PAPER (method named: gain and Shapley values, results not reported).

## Labels and metrics
- Sp_i = T(P_mf) / T(P_i) (Eq. 12); y_i = log(1 + Sp_i) (Eq. 13).
- NDCG@3 is the validation metric; its formula: NOT IN PAPER (objective rank:ndcg = LambdaMART; rank:pairwise for opt_pair).
- Top-1 / Top-3: % of groups whose true best plan is ranked first / in the first three (p.13).
- Regret1 = 1 − Sp_1/Sp⋆; Regret3 = 1 − max_{i∈Top-3} Sp_i / Sp⋆ (Eq. 14–15).
- Win rate: % of groups whose top pick beats the MinFill plan, ties = half (p.13); 50% by construction for mf_*.
- Baselines: random; mf_ndcg (MinFill forced first, rest by opt_ndcg); mf_rand (MinFill first, rest random).

## Split (Table 3, p.10)
Development 138 groups (5-fold grouped CV, randomized search on NDCG@3, refit on all 138); Test ID 25 groups; OOD 62 groups = whole QFT-based family. All plans of one circuit in one subset.

## Results
Table 6 — ID test (N = 25):

| Model | Top-1 % | Top-3 % | Regret1 | Regret3 | Win rate |
|---|---|---|---|---|---|
| opt_ndcg | 60.00 | 96.00 | 0.0141 | 0.0004 | 92.00 |
| opt_pair | 60.00 | 92.00 | 0.0211 | 0.0004 | 88.00 |
| random | 22.95 | 58.44 | 0.0699 | 0.0137 | 70.16 |
| mf_ndcg | 12.00 | 84.00 | 0.0735 | 0.0025 | 50.00 |
| mf_rand | 12.00 | 28.00 | 0.0735 | 0.0319 | 50.00 |

How the published rows are composed (our reproduction, `src/notes/deviations.md`; not stated in the paper): in the authors'
release (`936630a`, Notebook 5 / `test_model` in `src/Preliminar_getting_results.jl`) each model row combines **two models**:
Top-1 and Regret1 from one, Top-3 and Regret3 from another with a different round count (opt_ndcg 1 and 65 rounds, opt_pair 16 and 14).
That code also uses opt_ndcg eta 0.02 / subsample 0.8 where the paper's Table 4 gives 0.01 / 0.7, and computes regret on the label
log(1 + Sp) where Eq. 14–15 define it on speedup. With these, Table 6's Top-1/Top-3/Regret1/Regret3 reproduce exactly; the win rate does not.
One model per objective with rounds chosen by grouped CV on the training split gives ranges instead (deck slide 53).
The paper's random and MinFill+random rows differ from exact expectations on the published splits (deck slide 54).

Table 7 — OOD QFT (N = 62):

| Model | Top-1 % | Top-3 % | Regret1 | Regret3 | Win rate |
|---|---|---|---|---|---|
| opt_ndcg | 38.71 | 62.90 | 0.0520 | 0.0181 | 73.39 |
| opt_pair | 30.65 | 62.90 | 0.0800 | 0.0234 | 87.10 |
| random | 27.87 | 54.65 | 0.0775 | 0.0265 | 70.97 |
| mf_ndcg | 8.06 | 61.29 | 0.0784 | 0.0189 | 50.00 |
| mf_rand | 8.06 | 43.55 | 0.0784 | 0.0273 | 50.00 |

Table 11 — zero-shot cross-GPU, ID test (N = 25), trained on Ampere:

| Model | Backend | Top-3 % | Regret3 | Top-1 % | Regret1 | Win % |
|---|---|---|---|---|---|---|
| opt_ndcg | Ampere | 96.0 | 0.0004 | 60.0 | 0.0141 | 92.0 |
| opt_ndcg | Volta | 92.0 | 0.0068 | 64.0 | 0.0254 | 92.0 |
| opt_pair | Ampere | 92.0 | 0.0004 | 60.0 | 0.0211 | 88.0 |
| opt_pair | Volta | 88.0 | 0.0125 | 52.0 | 0.0531 | 88.0 |

Ranking stability A6000 vs V100 over 225 groups (Tables 8–9, p.16): mean Kendall τb 0.845 [0.808, 0.878]; same best plan 84.0% [79.1, 88.4]; mean Top-3 overlap 89.63% [87.1, 92.0]. QFT-based family least stable: τb 0.710, same-best 69.4% (Table 10).

## Proposal claims, checked
- "MinFill-first baselines find the fastest plan in only 12%": Table 6, mf_ndcg / mf_rand Top-1 = 12.00 (OOD 8.06, Table 7). Our exact MinFill Top-1 on the split matches.
- "The ranker finds the fastest plan 5× more often" (slide 49, 60 vs 12): the 60 is the Top-1 of a 1-round model, and the row's Top-3 comes from a second model (above); one model per objective gives 56–60 (deck slide 53).
- "Random top pick beats MinFill in about 70%": random win rate 70.16 (Table 6; ties count half), 70.97 OOD (Table 7).
- "Same-width plans with different runtimes" (slide 19): no table or figure in the paper; only qualitative statements (abstract; Sec 3.2 p.4; Sec 4 p.5; App. A.1 p.22 "Why width alone is not sufficient"; App. A.7 p.25 "runtime is not a monotone function of FLOPs").
- Winner distribution across the seven generators: NOT IN PAPER.

## Limitations stated by the authors
Seven candidates per circuit, one held-out family, two GPUs with related stacks (Sec 1); selects among existing candidates rather than generating new orders (Sec 8). Future work: larger and more varied candidate sets, withholding whole generator families, more GPU backends and hardware-conditioned rankers (Sec 8).
