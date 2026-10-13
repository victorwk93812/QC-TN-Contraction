"""paper.py — numbers quoted from arXiv 2608.05819 (Pastor, Castillo, Badia 2026), copied from the
paper's own tables after reading them. Each block names its table; see notes/paper-facts.md.
These are the only hand-entered numbers in the deck."""

# Table 6 (p.14): ID test, N = 25 circuit groups. (Top-1 %, Top-3 %, Regret1, Regret3, Win rate %)
TABLE6 = {
    "opt_ndcg": (60.00, 96.00, 0.0141, 0.0004, 92.00),
    "opt_pair": (60.00, 92.00, 0.0211, 0.0004, 88.00),
    "random":   (22.95, 58.44, 0.0699, 0.0137, 70.16),
    "mf_ndcg":  (12.00, 84.00, 0.0735, 0.0025, 50.00),
    "mf_rand":  (12.00, 28.00, 0.0735, 0.0319, 50.00),
}
TABLE6_N = 25

# Table 7 (p.15): OOD QFT-based family, N = 62 circuit groups. Same columns.
TABLE7 = {
    "opt_ndcg": (38.71, 62.90, 0.0520, 0.0181, 73.39),
    "opt_pair": (30.65, 62.90, 0.0800, 0.0234, 87.10),
    "random":   (27.87, 54.65, 0.0775, 0.0265, 70.97),
    "mf_ndcg":  (8.06, 61.29, 0.0784, 0.0189, 50.00),
    "mf_rand":  (8.06, 43.55, 0.0784, 0.0273, 50.00),
}
TABLE7_N = 62

# Table 11 (p.18): zero-shot cross-GPU on the ID test set (N = 25); models trained on Ampere (RTX A6000).
# (Top-3 %, Regret3, Top-1 %, Regret1, Win rate %)
TABLE11 = {
    ("opt_ndcg", "Ampere"): (96.0, 0.0004, 60.0, 0.0141, 92.0),
    ("opt_ndcg", "Volta"):  (92.0, 0.0068, 64.0, 0.0254, 92.0),
    ("opt_pair", "Ampere"): (92.0, 0.0004, 60.0, 0.0211, 88.0),
    ("opt_pair", "Volta"):  (88.0, 0.0125, 52.0, 0.0531, 88.0),
}

# Table 3 (p.10): split of the 225 circuit groups.
SPLIT = {"development": 138, "test_id": 25, "ood_qft": 62, "total": 225}

# Table 2 (p.8): plan-level features. Kept by both models unless marked; † discarded after selection.
FEATURES = {
    "Complexity / critical load": [("max_cost", "discarded"), ("log2_sum_flops", "discarded"),
                                   ("topq_mean_cost", "discarded"), ("avg_cost", "discarded"),
                                   ("std_cost", "opt_ndcg only")],
    "Sequence / granularity": [("n_steps", "kept"), ("frac_tiny_steps", "kept")],
    "Parallelism dynamics": [("max_out_rank", "kept"), ("avg_out_rank", "kept"),
                             ("costw_out_rank", "kept"), ("p_at_max_cost", "kept")],
    "Reduction / intensity": [("max_red_rank", "kept"), ("costw_red_rank", "opt_pair only"),
                              ("k_at_max_cost", "kept")],
    "Geometry / memory access": [("max_asym", "kept"), ("avg_asym", "kept"), ("costw_asym", "kept"),
                                 ("d_at_max_cost", "kept")],
}

# Sec 5.1 (p.9): the seven candidate generators.
GENERATORS = ["MinFill", "FlowCutter", "Girvan–Newman", "Jdrasil", "Tamaki",
              "Terrioux et al. MinFill variant 1", "Terrioux et al. MinFill variant 2"]

# Tables 8–9 (p.16): A6000 vs V100 ground-truth ranking stability over 225 groups.
STABILITY = {"kendall_tau_mean": 0.845, "same_best_plan_pct": 84.0, "top3_overlap_pct": 89.63}

# Table 10 (p.17): circuit families and their number of groups.
FAMILIES = {"GHZ": 18, "Grover": 7, "QNN": 24, "Quantum Walk": 5, "RQC": 86, "VQE": 7, "Deutsch–Jozsa": 3,
            "Ground State": 1, "Pricing Call": 11, "Portfolio QAOA": 1, "QFT-based": 62}
