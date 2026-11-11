# Code review guide: where each phase 1 finding comes from

Complements [`deviations.md`](deviations.md): for each finding there, the paper's statement, the place in the
authors' release that shows what the code does, our code that reproduces or tests it, and a command to rerun it.

**Sources.** Paper: arXiv 2608.05819v2 (section, table and equation numbers from the PDF). Release:
[`alfred-miquel/LTR-Quantum-Contraction-Plans` at `936630a`](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/tree/936630aa161abb7bcab1557c91137696b3bc5f68);
every link below is a permalink at that commit, and `L…` line numbers refer to it.
Notebooks are cited as *file, cell index (0-based, as `json.load(...)["cells"][i]`), line within the cell*;
`?plain=1#Ln` points at the line in the raw `.ipynb` JSON. Our paths are relative to this repo.

To dump a notebook with cell indices and in-cell line numbers:

```sh
.venv/bin/python -I -c "import json,sys;[print(f'[{i}:{j}] {l}') for i,c in enumerate(json.load(open(sys.argv[1]))['cells']) for j,l in enumerate(''.join(c['source']).split('\n'),1)]" external/authors/Notebooks/5_Getting_results.ipynb
```

## The authors' code path for Tables 6 and 7

Everything in Table 6 runs through one notebook and one Julia file:

| Step | Location in the release | What it does |
|---|---|---|
| Notebook header | [`Notebooks/5_Getting_results.ipynb`](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L18), cell 1, line 1 | `This notebook aims to replicate the results shown in tables 6 and 7 from the article` |
| Data loaded | same notebook, cell 9, lines 6–9 ([L281](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L281)) | loads `df_final_{ndcg,pair}` and `df_test_n_{ndcg,pair}` only; no `df_ood_*` file is loaded anywhere in the notebook |
| Run | cells 11 and 12, line 1: `menu_test_model()`; stdin answers `1` and `2` are saved in the outputs | prints Top-1/Regret1 then Top-3/Regret3 for each objective ([L335](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L335), [L340](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L340), [L387](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L387), [L392](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/5_Getting_results.ipynb?plain=1#L392)): `60.00% / 0.0141`, `96.00% / 0.0004`, `60.00% / 0.0211`, `92.00% / 0.0004` |
| Functions | [`src/Preliminar_getting_results.jl`](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl), included from cell 7 | `Choose_model` L463–537, `test_model` L440–447, `Getting_model` L170–204, `evaluate_tops_and_regrets` L282–413 |

Our headless equivalent of this path is `experiments/julia/t06_t07.jl` (writes `results/julia_t06_t07.csv`), and
the Python port is `src/qctn/rank/{data,train,metrics}.py`.

## Table 6: what the paper states and what the code does

| Finding (deviations.md) | (a) Paper | (b) Authors' code | (c) Our code | (d) Rerun |
|---|---|---|---|---|
| **Two models per row**: `opt_ndcg` Top-1/Regret1 from 1 round, Top-3/Regret3 from 65; `opt_pair` 16 and 14 | §5.4 (p. 11): "two optimized XGBoost rankers"; rounds from "early stopping applied independently on the validation fold"; models "retrained on the full development set … evaluated once". Table 6 gives one row per model | [`Preliminar_getting_results.jl` L478](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L478) `num = (1,65),` and [L500](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L500) `num = (16,14),`; [L442–L445](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L442-L445): `m_t1= Getting_model(...;num_round=num[1]...)` / `m_t3= Getting_model(...;num_round = num[2]...)`, then `evaluate_tops_and_regrets(m_t1,…;k=1)` and `(m_t3,…;k=3)`. `k` only selects which line is printed (L385–L398) | `experiments/t06_t07.py:76` (round pairs), `:82-85` (recipe rows); `tests/test_rank.py:46` (both halves match, neither model alone gives the row) | C1, C2, C3 |
| **No single model at rounds 1–300 gives all four values**; the release doesn't show how rounds were chosen | §5.2 (p. 10) and §5.4 (p. 11): round count by early stopping on NDCG@3 in five-fold grouped CV | Only the literal tuples at L478 and L500; no CV, `nfold` or early-stopping call anywhere in the release (search below, S3) | `experiments/t06_t07.py:104-116`; `results/t06_t07_composition.csv`, rows `check=all_four` are `[]` for every objective/split/params | C1 |
| **`opt_ndcg` params**: code eta 0.02, subsample 0.8; paper eta 0.01, subsample 0.7 | Table 4 (p. 12): `eta 0.01`, `subsample 0.7` for opt_ndcg. (§5.3 p. 9 also lists eta 0.01 / subsample 0.7 for the diagnostic model) | [L29–L37](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L29-L37) `params_ndcg` and again in `Choose_model` [L472–L474](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L472-L474): `:eta => 0.02,` / `:max_depth => 6,` / `:subsample => 0.8,`. `opt_pair` (L43–L57, L485–L496) equals Table 4 | `src/qctn/rank/train.py:14` (`NOTEBOOK`), `:21` (`TABLE4`); `experiments/julia/sweep_rounds.jl:7-11` | — |
| **Table 4 values never give the published Top-3 or regrets in rounds 1–300** | Table 4; Table 6 | — (no Table 4 configuration of opt_ndcg exists in the release) | `experiments/t06_t07.py:109` (`table4` params); composition rows `ndcg,id,…,train,table4` all `[]` | C5: for `ndcg table4 label`, matching rounds for Top-3, Regret1, Regret3 are each 0 |
| **Regret on the label, not on speedup** | Eq. (14)–(15), §6.2 (p. 13): `Regret1 = 1 − Sp1/Sp⋆`, `Regret3 = 1 − max_{i∈Top-3} Sp_i / Sp⋆`, "Sp⋆ is the optimal speedup in the group" | [L298](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L298) `sp = g.label    # label vs speedup`; [L370–L371](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L370-L371) `regret1 = 1 - best_in_rank1 / best_real`. The docstring L262 calls it `best_real`, L272 "best real speedup" | `src/qctn/rank/metrics.py:27` (`target` argument), `:42`; speedup rows in `results/t06_t07.csv` (`target=speedup`) | C5 (on `speedup` no round matches Top-1+Regret1 or Top-3+Regret3), C6 |
| **MinFill Regret1 on speedup is 0.1015, not 0.0735** | Table 6, rows `mf_ndcg`/`mf_rand`: Regret1 0.0735 | — (no baseline code, see below) | `experiments/t06_t07.py:46-65`, row `mf_rand_expected,id` gives 0.073498 on labels | C6 prints `0.10147` (speedup) and `0.07350` (label) |
| **Labels `log(1 + Sp)`, natural log, then integer ranks per circuit** | Eq. (12)–(13), §5.1 (p. 7): `Sp_i = T(P_mf)/T(P_i)`, `y_i = log(1 + Sp_i)` | Labels are precomputed in the split files (column `label`); the release has no code computing them (S4). Rank conversion: [L182–L184](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L182-L184) `y_train = Float32.(df_train.label)` / `transforma_a_enters_relevancia(...)`, [L238](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L238) `bloc_ordenat = invperm(sortperm(bloc)) .- 1` | `src/qctn/rank/train.py:24` (`relevance`), `tests/test_rank.py:24` | C7 (max \|label − log1p(speedup)\| ≈ 9e-16) |
| Tie handling and "truly best plan" | §6.2: Top-k is "the true best plan is ranked first or appears within the first three positions" | [L328–L348](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L328-L348): competition ranks on tied predictions; `best_idx = argmax(sp)` (first maximum of the label) | `src/qctn/rank/metrics.py:22`, `:39`; `tests/test_rank.py:20` | C2 |

## Table 6 with one model per objective (our protocol)

Not a claim about the authors' code; it's our alternative protocol. Paper: §5.4 (rounds from grouped 5-fold CV
on the development set), Table 3 (138 / 25 / 62 groups). Our code: `src/qctn/rank/train.py:56` (`cv_rounds`),
`experiments/t06_t07.py:118-127`; numbers in `results/t06_t07.csv`, rows `single_cv_{notebook,table4}_seed{42,1,2}`.
Rerun: C1. Every range in the deviations.md table matches these rows; chosen rounds are 7, 156, 86, 158, 138, 127
(`opt_ndcg`) and 18, 9, 93 (`opt_pair`). The 52.2% OOD random Top-3 is row `random_expected,ood` (52.165899).

## Not reproduced

| Item | (a) Paper | (b) Authors' code | (e) Where we searched | (c) Our code | (d) Rerun |
|---|---|---|---|---|---|
| **Table 7** (OOD) | §6.2.2 (p. 14), Table 7 (p. 15): opt_ndcg 38.71 / 62.90 / 0.0520 / 0.0181 / 73.39; opt_pair 30.65 / 62.90 / 0.0800 / 0.0234 / 87.10. §6.2.2: OOD circuits excluded "entirely from training, validation, and the in-distribution test set" | **No code.** Notebook 5 cell 1 says it replicates Tables 6 and 7, but cell 9 loads no `df_ood_*` file and `Choose_model` (L466–L507) only carries `df_test_n_*`. [`Data/splits/splits.md` L3](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Data/splits/splits.md#L3) says `df_ood` "were used during testing processes" | S1, S2 | `experiments/t06_t07.py:105-116` (train vs train+ID); composition rows `ood`: opt_ndcg `top1+regret1 [11]`, `top3+regret3 [23, 24, 135–138]` on train+id, `[]` on train; opt_pair `top1+regret1 [54]` on train+id, `top3+regret3 [14]` on train | C1 |
| **Win rate** | §6.2 (p. 13): "the fraction of groups in which the model's top-ranked plan achieves a higher speedup than the MinFill plan, with ties counted as half a win for each side" | **No code.** `evaluate_tops_and_regrets` returns only `:top1, :top3, :regret1, :regret3` ([L410–L412](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/Preliminar_getting_results.jl#L410-L412)) | S1 | `src/qctn/rank/metrics.py:44-45` | C5: opt_ndcg ID win is 80–84 over rounds 1–300 with the notebook params (76–80 with Table 4 params); paper 92 |
| **random and mf_rand baselines** | §5.4 (p. 11) defines random, mf_ndcg, mf_rand; Table 6: random 22.95 / 58.44 / 0.0699 / 0.0137 / 70.16, mf_rand Top-3 28.00 | **No code** for any baseline | S1 | `experiments/t06_t07.py:46-65` (exact expectations), rows `random_expected`, `mf_rand_expected` | C1 |
| **`mf_ndcg`** | §5.4; Tables 6–7 | No code | S1 | Not attempted | — |

## Stored models

| Finding | (a) Paper | (b) Authors' code / data | (c) Our code | (d) Rerun |
|---|---|---|---|---|
| `models/opt_pair.json` is the Notebook 5 recipe fitted on all 225 circuits (predictions within 1e-5 at 1850 rounds) | §5.4 (p. 11): "a separate deployment model may be retrained using all labeled data … should not be interpreted with the same reported test/OOD metrics" | [`models/models.md` L1](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/models/models.md#L1): `Here we are the main models in production used in the article in json format`. The JSON files carry no training metadata | `experiments/t06_t07.py:87-102`; `tests/test_rank.py:60`; composition rows `stored_vs_fit_on_all`: pair `max_abs_diff` 2.6e-6 (ID), 6.4e-6 (OOD) | C2 |
| `opt_ndcg.json` similar (OOD Top-3 100%) but parameters not identified | same | same | rows `stored,ndcg,*` (2600 rounds) and `fit_on_all,ndcg,*` in `results/t06_t07.csv`; Spearman 0.61 / 0.55 between them | C1 |
| Stored models score far above the tables (Top-1 84–97%) | Tables 6–7 | — | rows `stored,*` in `results/t06_t07.csv`: 84.0, 85.5, 88.0, 96.8 | C1 |
| They rank well on split features, poorly on raw ones (ID Top-1 8–16%); Notebook 4 passes raw values | — | [`Notebooks/4_model_predictions.ipynb`](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Notebooks/4_model_predictions.ipynb?plain=1#L396), cell 8, line 4: `model_fname = "../models/opt_ndcg.json"`; cell 10, line 5: `[8.0, 10.850186837645774, 8.0, 4.30952…, 1.65997…, 42.0, …]` (raw values, e.g. `n_steps` 42); cell 11: `getting_candidate_plan(...)`. [`src/model_predictions.jl` L200–L207](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/model_predictions.jl#L200-L207) selects columns and calls `XGBoost.predict` ([L266](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/src/model_predictions.jl#L266)) with no transform | — (one-off check) | C11: ID Top-1 8.0 (ndcg), 16.0 (pair) on raw features; 84.0 / 88.0 on split features |

## Data

| Finding | (a) Paper | (b) Release | (d) Rerun |
|---|---|---|---|
| Splits cover 225 circuits in 1475 rows | §5.2, Table 3 (p. 10): 138 / 25 / 62 groups | `Data/splits/df_{final,test_n,ood_n}_{ndcg,pair}`; same rows in both objectives | C8: train 932 rows / 138 groups, ID 166 / 25, OOD 377 / 62 |
| Features are standardised, not as in `Features.xlsx`; a z-score over training rows doesn't recover them | Table 2 defines raw features; no standardisation is described (no "standardi", "z-score", "scal…" in the PDF text) | `Data/features/Features.xlsx` (1658 rows, raw); no scaler in the release (S3) | C10: max \|z − split\| ≈ 4.31 (ddof 0 or 1) |
| `RQC_3_7_40`, `RQC_3_7_52` listed twice in training (12 duplicated rows), same features, speedups differ by up to 0.84; groups of 14 and 11 | §5.1: seven candidate plans per circuit, some fewer | `Data/splits/df_final_*` | C8 |
| Three training circuits have no MinFill plan | Eq. (12): speedup is relative to `P_mf` | `Data/splits/df_final_*` | C8: `RANDOM_11`, `RQC_5_32`, `pricingcall_nativegates_ibm_qiskit_opt3_25`, all in train |
| `measures.xlsx`: speedups vs MinFill only (1367 rows), no raw times, no second GPU; lacks 97 split rows, differs in 155 (max 727.9, QFT) | §5.1 (p. 9): "applied consistently to all contraction plans on both GPU backends"; §5.5: RTX A6000 and Tesla V100 | `Data/measurements/measures.xlsx`, columns `circuit_name, speedup_mf`; [`measurements.md` L1](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Data/measurements/measurements.md#L1): `measured against the minfill plan as a reference` | C9: 1367 rows; 97 missing; 155 differing split rows, 84 in QFT and 71 in RQC circuits (11 of the RQC rows are the duplicated plans); max 727.89 at `QFT_6_pla_minfill_mrs`, largest RQC difference 2.38 |

## Library version

| Finding | Release | Ours | Rerun |
|---|---|---|---|
| xgboost 2.1.4 in Python matches the Julia stack (XGBoost_jll 2.1.5) to ~5e-8 | [`Manifest.toml` L1210–L1214](https://github.com/alfred-miquel/LTR-Quantum-Contraction-Plans/blob/936630aa161abb7bcab1557c91137696b3bc5f68/Manifest.toml#L1210-L1214): `XGBoost_jll` `version = "2.1.5+0"`; XGBoost.jl 2.5.5 (L1190–L1194) | `pyproject.toml:14` `xgboost==2.1.4`; `tests/test_rank.py:71` (metrics equal to `results/julia_t06_t07.csv`) | C4 (rounds 1–5: max difference 4.9e-9) |
| xgboost 3.4.1 gives different models (0.03–1.06) | — | `src/qctn/rank/train.py:1-5` docstring | Not rerun for this guide (needs a separate 3.4.1 environment) |

## Searches for code that isn't in the release

- **S1** (win rate, baselines, OOD evaluation): `git -C external/authors grep -n -i -E "win[ _-]?rate|\bwins?\b|mf_rand|mf_ndcg|df_ood|shuffle|randperm|rand\(" -- '*.jl' '*.ipynb' '*.md'` returns one line, `Data/splits/splits.md:3`. `random` alone only hits `Pkg` install logs in notebook outputs.
- **S2** (Table 7): Notebook 5 cells 9–12 and `Preliminar_getting_results.jl` L463–L566; no OOD split loaded.
- **S3** (scaler, CV, early stopping): `git -C external/authors grep -n -i -E "scal|standardi|zscore|nfold|early_stop" -- '*.jl' '*.ipynb' '*.md'` returns only `StatsBase` in install logs of Notebooks 1a/1b.
- **S4** (label computation): `git -C external/authors grep -n -E "log1p|log\(1" -- '*.jl' '*.ipynb'` returns nothing.

Files searched: all of `src/*.jl`, `Notebooks/*.ipynb`, and every `*.md` in the release at `936630a`.

## Commands

From the main checkout (where `.venv` and `external/` are), or from a worktree with `.venv` and `external` reachable:

```sh
export PYTHONPATH=src PY=.venv/bin/python
JL="env JULIA_DEPOT_PATH=external/julia-depot $HOME/.juliaup/bin/julia +1.9.1 --project=external/julia-env/rank"
```

- **C1** all phase 1 numbers (several minutes, 8 threads): `$PY experiments/t06_t07.py` → `results/t06_t07.csv`, `results/t06_t07_composition.csv`
- **C2** tests: `$PY -m pytest -q tests/test_rank.py`
- **C3** the authors' Julia path, headless (the authors' repo path must be absolute): `$JL experiments/julia/t06_t07.jl $PWD/external/authors /tmp/julia_t06_t07.csv`
- **C4** Julia vs Python predictions:

  ```sh
  $JL experiments/julia/sweep_rounds.jl $PWD/external/authors /tmp/sweep5.csv 5
  $PY -c "import warnings,sys;warnings.simplefilter('ignore');import pandas as pd,numpy as np;from qctn.rank.data import load_split as L;from qctn.rank.train import NOTEBOOK,TABLE4,fit,predict;j=pd.read_csv(sys.argv[1]);P={'notebook':NOTEBOOK,'table4':TABLE4};print(max(np.abs(predict(m,L(s,o),o,r)-h.sort_values('row').score.to_numpy()).max() for (o,p),g in j.groupby(['objective','params']) for m in [fit(L('train',o),o,P[p][o],5)] for (s,r),h in g.groupby(['split','round'])))" /tmp/sweep5.csv
  ```
- **C5** rounds 1–300 on label and on speedup, both parameter sets (about 3 min, 8 threads):

  ```sh
  $PY -c "import warnings;warnings.simplefilter('ignore');from qctn.rank.data import load_split as L;from qctn.rank.metrics import evaluate as E;from qctn.rank.train import NOTEBOOK,TABLE4,fit,predict
  P={'ndcg':(60,96,.0141,.0004),'pair':(60,92,.0211,.0004)}
  for o in P:
   tr,te=L('train',o),L('id',o);p=P[o]
   for n,pp in (('notebook',NOTEBOOK[o]),('table4',TABLE4[o])):
    m=fit(tr,o,pp,300)
    for t in ('speedup','label'):
     R=[E(te,predict(m,te,o,r),t) for r in range(1,301)];k=lambda f:[r+1 for r,x in enumerate(R) if f(x)]
     print(o,n,t,'t1+r1',k(lambda x:abs(x.top1-p[0])<.006 and abs(x.regret1-p[2])<6e-5)[:8],'t3+r3',k(lambda x:abs(x.top3-p[1])<.006 and abs(x.regret3-p[3])<6e-5)[:8],'n(t3,r1,r3)',[len(k(f)) for f in (lambda x:abs(x.top3-p[1])<.006,lambda x:abs(x.regret1-p[2])<6e-5,lambda x:abs(x.regret3-p[3])<6e-5)],'win',min(x.win for x in R),max(x.win for x in R))"
  ```
- **C6** MinFill Regret1, speedup vs label: `$PY -c "import warnings;warnings.simplefilter('ignore');from qctn.rank.data import load_split as L;from qctn.rank.metrics import evaluate as E;d=L('id','ndcg');s=d.circuit_name.str.endswith('_pla_mf').astype(float).to_numpy();print(E(d,s,'speedup').regret1,E(d,s,'label').regret1)"`
- **C7** labels are natural log: `$PY -c "import warnings;warnings.simplefilter('ignore');import numpy as np,pandas as pd;from qctn.rank.data import load_split as L;d=pd.concat([L(s,'ndcg') for s in ('train','id','ood')]);print(np.abs(d.label-np.log1p(d.speedup)).max())"`
- **C8** rows, groups, duplicates, missing MinFill: `$PY -c "import warnings;warnings.simplefilter('ignore');import pandas as pd;from qctn.rank.data import load_split as L;S={s:L(s,'ndcg') for s in ('train','id','ood')};a=pd.concat(S.values());print(len(a),a.base_id.nunique(),{s:(len(d),d.base_id.nunique()) for s,d in S.items()});t=S['train'];u=t[t.duplicated('circuit_name',keep=False)];print(len(t)-t.circuit_name.nunique(),u.base_id.unique().tolist(),u.groupby('circuit_name').speedup.agg(lambda x:x.max()-x.min()).max(),t[t.base_id.isin(u.base_id)].groupby('base_id').size().to_dict());print([b for b,g in a.groupby('base_id') if not g.circuit_name.str.endswith('_pla_mf').any()])"`
- **C9** `measures.xlsx` vs splits: `$PY -c "import warnings;warnings.simplefilter('ignore');import pandas as pd;from qctn.rank.data import load_split as L;from qctn.paths import authors_repo as A;a=pd.concat([L(s,'ndcg') for s in ('train','id','ood')]);m=pd.read_excel(A()/'Data/measurements/measures.xlsx');j=a.merge(m,on='circuit_name',how='left');k=j.dropna();d=(k.speedup-k.speedup_mf).abs();print(len(m),m.columns.tolist(),j.speedup_mf.isna().sum(),(d>1e-6).sum(),k[d>1e-6].circuit_name.nunique(),d.max(),k.circuit_name[d.idxmax()])"`
- **C10** scaler: `$PY -c "import warnings;warnings.simplefilter('ignore');import pandas as pd,numpy as np;from qctn.rank.data import load_split as L,FEATURES as F;from qctn.paths import authors_repo as A;X=pd.read_excel(A()/'Data/features/Features.xlsx').drop_duplicates('circuit_name').set_index('circuit_name');t=L('train','ndcg');r=X.loc[t.circuit_name,F['ndcg']].to_numpy(float);print([np.abs((r-r.mean(0))/r.std(0,ddof=k)-t[F['ndcg']].to_numpy(float)).max() for k in (0,1)])"`
- **C11** stored models on raw vs split features: `$PY -c "import warnings;warnings.simplefilter('ignore');import pandas as pd,xgboost as xgb;from qctn.rank.data import load_split as L,FEATURES as F;from qctn.rank.metrics import evaluate as E;from qctn.paths import authors_repo as A;X=pd.read_excel(A()/'Data/features/Features.xlsx').drop_duplicates('circuit_name').set_index('circuit_name');[print(o,E(d,b.predict(xgb.DMatrix(X.loc[d.circuit_name,F[o]].to_numpy(float))),'label').top1,E(d,b.predict(xgb.DMatrix(d[F[o]].to_numpy(float))),'label').top1) for o in ('ndcg','pair') for b in [xgb.Booster(model_file=str(A()/'models'/f'opt_{o}.json'))] for d in [L('id',o)]]"`
