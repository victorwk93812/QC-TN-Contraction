"""Every number a slide reads from data/*.json is asserted here, and every data file must be
exactly what the builders produce now (so a stale trace can never reach the deck)."""
import json

import pytest

from traces import figures

DATA = figures.DATA


def load(fid):
    return json.loads((DATA / f"{fid}.json").read_text())


@pytest.mark.parametrize("fid", sorted(figures.BUILDERS))
def test_data_file_is_current(fid):
    on_disk = load(fid)
    assert json.loads(json.dumps(figures.BUILDERS[fid]())) == on_disk


def test_F03_ring_table():                    # slides 3, 4
    d = load("F03")
    assert [r["elim"] for r in d["table"]] == ["i", "j", "k", "l"]
    assert d["steps"][0]["fill"] == [["j", "l"]]
    assert [r["bag"] for r in d["table"]] == [3, 3, 2, 1]
    assert [r["cost"] for r in d["table"]] == [8, 8, 4, 2]
    assert d["total_cost"] == 22 and d["width"] == 2 and d["peak_size"] == 4


def test_F04_ring_trees():                    # slides 2, 12
    d = load("F04")
    assert d["trees"]["good"]["max_rank"] == 2 and d["trees"]["bad"]["max_rank"] == 4
    assert d["trees"]["good"]["cost"] == 20 and d["trees"]["bad"]["cost"] == 48
    assert d["best"] == {"max_rank": 2, "cost": 20}
    assert d["carving_width"] == 2 == d["max_degree"]


def test_F05_spectrum():                      # slide 10
    assert load("F05")["tw"] == {"tree": 1, "cycle": 2, "grid": 4, "clique": 5}


def test_F09_bowtie_graphs():                 # slide 13
    d = load("F09")
    assert d["hub_degree"] == 4 and d["hub_clique"] and d["index_tw"] == 3
    assert (d["best_elimination"]["width"], d["best_elimination"]["cost"]) == (3, 62)
    assert d["n_orders"] == 40320


def test_F10_tree_vs_list():                  # slide 13
    d = load("F10")
    assert (d["tree"]["max_rank"], d["tree"]["cost"]) == (2, 50) == tuple(d["tree_opt"].values())
    assert (d["list"]["max_rank"], d["list"]["cost"]) == (3, 58) == tuple(d["list_opt"].values())
    assert d["list"]["ranks"] == [2, 2, 3, 2, 1, 0]
    assert d["n_lists"] == 5040 and d["min_list_rank"] == 3
    assert d["carving_width"] == 4


def test_F11_catalog_small_cases():           # slide 11
    c = load("F11")["checks"]
    assert c == {"forest": 1, "cycle_9": 2, "K_7": 6, "K_3_5": 3, "grid_3": 3, "grid_4": 4, "Q_3": 3, "Q_4": 6}


def test_F17_cut_is_separator():              # slide 7
    d = load("F17")
    assert d["cut"] == ["p", "q"] and d["is_separator"]
    assert d["components"] == [["a", "b", "c", "d"], ["e"]]


def test_S05_counts():                         # slide 5
    d = load("S05")["named"]
    assert (d["circuit"]["tensors"], d["circuit"]["indices"]) == (33, 54)
    assert (d["circuit"]["trees_m"], d["circuit"]["trees_e"]) == (1.1, 44)      # 63!! = 1.1e44
    assert (d["circuit"]["orders_m"], d["circuit"]["orders_e"]) == (2.3, 71)    # 54! = 2.3e71
    assert (d["bowtie"]["trees_m"], d["bowtie"]["trees_e"]) == (1.0, 4)         # 11!! = 10395


def test_F06_ring_minfill():                   # slide 8
    d = load("F06_ring")
    assert d["width"] == 2 and [s["bag"] for s in d["steps"]] == [3, 3, 2, 1]
    assert len(d["steps"][0]["fill"]) == 1 and all(not s["fill"] for s in d["steps"][1:])


def test_F14_separator_tree():                 # slide 9
    d = load("F14")
    assert [(l["own"], l["inherited"], l["bag"]) for l in d["levels"]] == [(7, 0, 7), (3, 7, 10), (3, 6, 9)]
    assert d["leaf_max_bag"] == 8 and d["width"] == 9


def test_S14_widths():                         # slide 14
    r = load("S14")["rows"]
    assert {k: (v["tw_G"], v["tw_LG"], v["cw_G"]) for k, v in r.items()} == {
        "ring": (2, 2, 2), "bowtie": (2, 3, 4), "grid2x3": (2, 3, 3), "grid3x3": (3, 5, 4)}
    for v in r.values():                       # the two easy bounds shown on the slide
        assert v["tw_LG"] >= v["max_degree"] - 1 and v["cw_G"] >= v["max_degree"]


def test_F00_arc():                            # slides 6, 49
    d = load("F00")
    assert d["total_slides"] == 53 and [a["slides"] for a in d["acts"]] == [8, 8, 6, 6, 9, 4, 3, 6, 3]
    assert d["total_minutes"] == 62


def test_carving_width_dp_matches_enumeration():
    from traces import graphs as gr
    from traces.carving import carving_width_dp, tree_stats
    for T in (gr.ring_tn(), gr.bowtie_tn()):
        assert carving_width_dp(T) == min(tree_stats(T, t)["carving_width"] for t in figures.all_trees(T))


def test_S14_published_bounds_hold():          # slide 14 math anchor
    for v in load("S14")["rows"].values():
        D, tw, twl, cw = v["max_degree"], v["tw_G"], v["tw_LG"], v["cw_G"]
        assert max(D - 1, tw - 1) <= twl <= D * (tw + 1) - 1         # Harvey–Wood
        assert D <= cw <= twl + 1 <= 1.5 * cw                         # DDV Thm 4; vc <= 1.5 ec


# ---------------------------------------------------------------- deep dives 1-2

def test_F06_corners_first():                  # slide 21
    d = load("F06")
    assert sorted(d["first_elims"][:4]) == [0, 5, 30, 35] and d["first_fills"][:4] == [1, 1, 1, 1]
    assert d["width"] == 6


def test_F06_chordal_zero_fill():              # slide 22
    d = load("F06_chordal")
    assert d["is_chordal"] and d["total_fill"] == 0 and d["width"] == d["clique_number"] - 1 == 2


def test_F07_F08_minfill_grids():              # slides 23, 24
    d = load("F07")
    mf = dict(zip(d["L"], d["minfill"]))
    sw = dict(zip(d["L"], d["sweep"]))
    assert [mf[L][0] for L in (8, 12, 16, 20, 24)] == [10, 15, 21, 26, 32]
    assert [mf[L][1] for L in (8, 12, 16, 20, 24)] == [13.20, 18.35, 23.68, 29.29, 34.59]
    assert [sw[L][1] for L in (8, 12, 16, 20, 24)] == [14.70, 19.95, 24.83, 29.51, 34.06]
    assert d["first_L_minfill_costlier"] == 22 and mf[22] == [29, 32.12] and sw[22][1] == 31.79
    assert load("F08")["width"] == 13


def test_S25_ties_chain_cubic():               # slide 25
    d = load("S25")
    assert (d["tie_min"], d["tie_max"], d["tie_median"]) == (26, 30, 28)
    assert sum(d["tie_hist"]["count"]) == 40 and d["bags_max"] == 27 and d["n_steps"] == 400
    c = d["cubic"]
    assert (c["minfill_best"], c["minfill_median"], c["mindeg_median"]) == (56, 59.5, 62)
    assert c["nd"] == {"spectral": 57, "flowcutter": 64, "bfs": 67, "kl0": 78}


def test_P26_paper_table6_table7():            # slide 26
    d = load("P26")
    assert d["id"]["mf_ndcg"]["top1"] == 12.00 and d["id"]["opt_ndcg"]["top1"] == 60.00
    assert d["id"]["random"]["win"] == 70.16 and d["ood"]["mf_ndcg"]["top1"] == 8.06


def test_F15_corner_vs_middle():               # slides 27, 32
    p = load("F15")["panels"]
    assert (p["corner"]["size"], p["corner"]["sizes"]) == (2, [46, 1])
    assert (p["middle"]["size"], p["middle"]["sizes"]) == (7, [21, 21]) and p["middle"]["balanced_half"]


def test_F16_presets():                        # slides 28, 29
    d = load("F16")
    r = d["preset_reports"]
    assert not r["c"]["sep"]
    assert r["xy"]["sep"] and r["xy"]["min"] and r["xy"]["ah"]
    assert r["cd"]["min"] and not r["cd"]["b2"]
    assert r["efg"]["min"] and r["xyz"]["sep"] and not r["xyz"]["min"]
    assert d["min_ah"] == 2 and len(d["table"]) == 2 ** 11
    from traces import graphs as gr
    from traces.core import separator_report
    G = gr.separator_graph()
    for key, rep in d["table"].items():        # the browser only looks up; the table must equal Python
        py = separator_report(G, key)
        assert (rep["sep"], rep["min"], rep["b2"], rep["b3"]) == (py["separator"], py["minimal"], py["balanced_0.500"], py["balanced_0.667"])
    import networkx as nx
    for p in d["menger_paths"]:
        assert all(G.has_edge(a, b) for a, b in zip(p, p[1:]))
    assert not set(d["menger_paths"][0][1:-1]) & set(d["menger_paths"][1][1:-1])


def test_F12_separator_last_and_nd():          # slides 30, 31
    a = load("F12_sideA")
    assert a["side_a_fill_inside"] and a["width_A"] <= 7 and a["width"] == 7
    assert load("F12")["width"] == 9


def test_F13_balance_price():                  # slide 33
    d = load("F13")
    assert d["nd"] == [9, 21, 33, 45] and d["hybrid"] == d["sweep"] == [7, 15, 23, 31] and d["minfill"][3] == 41
    assert d["bisection32"] == {"max_rank": 40, "max_exp": 48}


def test_F18_F19_flowcutter():                 # slide 34
    d = load("F18")
    front = [(d["steps"][i]["size"], d["steps"][i]["small_side"]) for i in d["pareto"]]
    assert front == [(k, k * (k - 1) // 2) for k in range(2, 8)]
    assert load("F19")["circuit"]["front"] == [[3, 1], [5, 2], [6, 23]]


def test_S35_order_to_tree():                  # slide 35
    d = load("S35")
    assert d["nd"]["width"] == d["alt"]["width"] == 3
    assert (d["nd"]["cost"], d["alt"]["cost"]) == (52, 72)
    assert (d["width3_orders"], d["width3_cost_min"], d["width3_cost_max"]) == (3168, 52, 72)


def test_SC_scorecard_rows():                  # slides 26, 35
    r = load("SC")["rows"]
    assert (r["minfill"]["w7"], r["minfill"]["w20"], r["minfill"]["wc"]) == (8, 26, 10)
    assert (r["nd_fc"]["w7"], r["nd_fc"]["w20"], r["nd_fc"]["wc"]) == (9, 28, 13)



# ---------------------------------------------------------------- treewidth primer

def test_primer_star_two_orders():             # slide 3a
    hub, leaves = load("PR1_hub"), load("PR1_leaves")
    assert hub["width"] == 6 and hub["steps"][0]["bag"] == 7 and hub["total_fill"] == 15
    assert leaves["width"] == 1 and all(s["bag"] <= 2 for s in leaves["steps"]) and leaves["total_fill"] == 0
    assert hub["tw_exact"] == leaves["tw_exact"] == 1


def test_primer_four_families_optimal():       # slide 3b: every shown order reaches the exact treewidth
    tree, cycle, grid, clique = load("PR2_tree"), load("PR2_cycle"), load("PR2_grid"), load("PR2_clique")
    assert [(d["width"], d["tw_exact"]) for d in (tree, cycle, grid, clique)] == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_S10_certificates():                   # slide 10
    d = load("S10")
    assert (d["clique"]["omega"], d["clique"]["tw"], d["clique"]["degeneracy"]) == (5, 4, 4)
    assert (d["grid"]["k"], d["grid"]["tw"], d["grid"]["omega"], d["grid"]["degeneracy"]) == (4, 4, 2, 2)


# ---------------------------------------------------------------- deep dives 3-4

def test_F20_girvan_newman():                  # deck slides 38-40
    d = load("F20")
    assert d["first_removed"] == ["A1", "B4"] and d["second_removed"] == ["A2", "B3"]
    assert d["best_modularity"] == 0.389 and d["best_partition"] == [["A0", "A1", "A2", "A3", "A4"], ["B0", "B1", "B2", "B3", "B4"]]
    assert (d["tree_max_rank"], d["tree_cost"], d["opt_max_rank"], d["opt_cost"]) == (6, 740, 4, 388)
    assert all("split_step" in n for n in d["tree"]["nodes"] if n["kind"] != "leaf")
    assert sorted(n["label"] for n in d["tree"]["nodes"] if n["kind"] == "leaf") == sorted(f"{s}{i}" for s in "AB" for i in range(5))


def test_F21_gn_vs_others():                   # deck slide 41
    r = load("F21")["rows"]
    assert r["ring"] == {"tensors": 4, "gn": [2, 4.32], "minfill": [2, 4.32], "nd": [2, 4.32], "opt": [2, 4.32]}
    assert r["two wheels"]["gn"] == [6, 9.53] and r["two wheels"]["minfill"] == [4, 8.64] and r["two wheels"]["opt"] == [4, 8.6]
    assert r["circuit"]["gn"] == [9, 13.67] and r["circuit"]["minfill"] == [8, 13.28] and r["circuit"]["nd"] == [12, 15.01]
    for row in r.values():                      # never better than the optimum where it is known
        if "opt" in row:
            assert all(row[k][0] >= row["opt"][0] for k in ("gn", "minfill", "nd"))


def test_F22_local_improvement_valid():        # deck slide 43: recompute every round and validate it
    import random
    from traces import graphs as gr
    from traces.td import improve_once, is_valid, td_from_order, width
    d = load("F22")
    assert [s["width"] for s in d["states"]] == [8, 8, 7, 7, 6, 6, 5] and d["final_width"] == d["tw"] == 5
    G = gr.grid(5)
    rng = random.Random(7)
    bags, T = td_from_order(G, sorted(G, key=lambda v: rng.random()))
    assert is_valid(G, bags, T) and width(bags) == 8
    while (r := improve_once(G, bags, T)) is not None:
        bags, T, _ = r
        assert is_valid(G, bags, T)
    assert width(bags) == 5


def test_td_roundtrip_order():
    from traces import graphs as gr
    from traces.fast_elim import eliminate_fast
    from traces.td import td_from_order, td_to_order, width
    G = gr.grid(6)
    order = [s["elim"] for s in eliminate_fast(G, rule="minfill", seed=0)["steps"]]
    bags, _ = td_from_order(G, order)
    o2 = td_to_order(G, list(bags.values()))
    assert eliminate_fast(G, order=o2, record_steps=False)["width"] <= width(bags)


def test_S42_S44_solver_runs():                # deck slides 42, 44 (measured by traces/run_solvers.py)
    s = load("S42")
    assert all(r["valid"] for r in s["runs"]) and len(s["runs"]) == 18 and s["minfill"] == 35 and s["n"] == 295
    for sv, best in s["best"].items():
        for b, w in best.items():
            assert w == min(r["width"] for r in s["runs"] if r["solver"] == sv and r["budget"] == int(b))
    t = load("S44")["rows"]
    assert t["tamaki"]["valid"] and t["jdrasil"]["valid"]
    assert (t["tamaki"]["best_width"], t["jdrasil"]["best_width"]) == (8, 9)
    assert t["jdrasil"]["tree_log2cost"] < t["tamaki"]["tree_log2cost"] < load("SC")["rows"]["minfill"]["tree_cost_c"]


# ---------------------------------------------------------------- Acts III-V, backups

def test_F01_F02_pipeline_and_map():           # deck slides 17, 18
    f = load("F01")
    assert (f["groups"], f["families"], f["family_groups_sum"], f["generators"]) == (225, 11, 225, 7)
    assert (f["features_initial"], f["features_discarded"]) == (18, 4)
    assert load("F02")["n"] == 7


def test_S21_equal_width_plans():              # deck slide 21
    d = load("S21")
    assert (d["max_rank"], d["n_same"], d["n_total"]) == (8, 193, 200)
    assert d["log2cost_range"] == [12.94, 13.69]
    p = d["plans"]
    assert p["cheapest"]["log2cost"] == 12.94 and p["costliest"]["log2cost"] == 13.69
    assert all(len(v["c"]) == v["n_steps"] for v in p.values())


def test_F23_plan_shapes():                    # deck slide 45
    p = load("F23")["plans"]
    assert {k: (v["max_rank"], v["log2cost"], v["depth"]) for k, v in p.items()} == {
        "MinFill": (8, 13.28, 10), "ND / FlowCutter": (12, 15.01, 7), "Girvan–Newman": (9, 13.67, 7), "Tamaki": (8, 13.02, 18)}
    for v in p.values():                        # one root-to-leaf path can never cost more than everything
        assert v["critical_path"] <= v["log2cost"]


def test_S46_S47_conversion_and_gemm():         # deck slides 46, 47
    s = load("S46")
    assert (s["n_orders"], s["n_trees"]) == (5040, 208)
    g = load("S47")
    assert (g["m"], g["n"], g["k"], g["c"], g["p"], g["d"]) == (3, 3, 5, 11, 6, 0.0)
    assert (g["M"], g["K"], g["N"]) == (8, 32, 8)


def test_S48_ranking_group():                  # deck slide 48
    import math
    d = load("S48")
    P = d["plans"]
    for v in P.values():
        assert v["y"] == round(math.log(1 + 2 ** (P["MinFill"]["log2cost"] - v["log2cost"])), 3)
    assert d["ideal"][0] == "Jdrasil" and d["mf_first"][0] == "MinFill"
    assert (d["ndcg3_by_width"], d["ndcg3_mf_first"]) == (0.959, 0.893)


def test_P49_P50_paper_values():               # deck slides 49, 50
    d = load("P49")
    assert d["id"]["opt_ndcg"]["top1"] == 60.0 and d["id"]["mf_ndcg"]["top1"] == 12.0
    assert d["ood"]["opt_ndcg"]["top1"] == 38.71 and d["transfer"]["opt_ndcg|Volta"]["top1"] == 64.0
    f = load("P50")
    assert (f["n_features"], f["n_discarded"]) == (18, 4)
    discarded = [x["name"] for b in f["blocks"] for x in b["features"] if x["status"] == "discarded"]
    assert discarded == ["max_cost", "log2_sum_flops", "topq_mean_cost", "avg_cost"]


def test_B5_bond_dimensions():                 # backup B5
    d = load("B5")
    assert all(r["D_exp"] == 3 for r in d["rows"])
    pw = [r["D_pow"] for r in d["rows"]]
    assert pw == sorted(pw) and pw[0] < pw[-1]


def test_S32_theory_numbers():                 # deck slide 34
    import math
    d, f15 = load("S32"), load("F15")
    assert (d["n"], d["sqrt_n"], d["tw"]) == (49, 7, 7)            # 7x7 grid: tw(L x L) = L
    assert d["middle_column"] == f15["panels"]["middle"]["size"] == d["sqrt_n"]
    assert d["middle_sides"] == f15["panels"]["middle"]["sizes"]
    assert d["lt_bound"] == round(2 * math.sqrt(2) * math.sqrt(49), 1)
    assert d["rs_sep"] == d["tw"] + 1 and d["rs_side"] == (d["n"] - d["rs_sep"]) // 2


def test_R1_setup_pins():                       # deck slide 51 (reproduction setup)
    d = load("R1")
    assert d["xgboost"] == "2.1.4" and d["xgboost_jll"] == "2.1.5"
    assert d["cupy"] == "cupy-cuda12x[ctk] 14.2.0" and d["driver"] == "535"
    assert d["authors_commit"] == "936630a" and d["julia"] == "1.9.1"
    assert "planned" in d["gpu_status"]           # no GPU results are claimed


def test_R2_table6_two_model_composition():     # deck slide 52
    doc = load("R2")
    assert float(doc["julia_max_diff_txt"]) >= max(r["julia_max_diff"] for r in doc["rows"].values())
    d = doc["rows"]
    assert (d["opt_ndcg"]["rounds_k1"], d["opt_ndcg"]["rounds_k3"]) == (1, 65)
    assert (d["opt_pair"]["rounds_k1"], d["opt_pair"]["rounds_k3"]) == (16, 14)
    for row in d.values():
        assert all(row["match"].values())                        # composed row == published row, all four metrics
        assert row["single_model_all_four"] == []                # no single model gives the row
        assert row["win_ours"][0] < row["paper"]["win"]          # win rate not reproduced
        assert row["julia_max_diff"] < 1e-6                      # Python and the authors' Julia code agree
        one, three = (row["models"][str(row[k])] for k in ("rounds_k1", "rounds_k3"))
        assert not (one["top3"] == row["paper"]["top3"] and one["regret3"] == row["paper"]["regret3"])


def test_R3_single_model_ranges():              # deck slide 53
    d = load("R3")
    k = d["by_key"]
    assert all(r["n_runs"] == 6 for r in d["rows"])
    assert k["opt_ndcg|id"]["top1_txt"] == "56–60" and k["opt_ndcg|id"]["top3_txt"] == "88–92"
    assert k["opt_pair|id"]["top1_txt"] == "44–60" and k["opt_ndcg|ood"]["top3_txt"] == "47–55"
    assert k["opt_ndcg|ood"]["top1_txt"] == "27–34" and k["opt_pair|ood"]["top3_txt"] == "45–58"
    assert d["rounds_range"] == [7, 158] and d["rounds_range_txt"] == "7–158" and d["fold_seeds"] == [1, 2, 42]
    assert d["mf_id_top1_txt"] == "12" and d["random_ood_top3_txt"] == "52.2" and d["pts_per_circuit_id"] == 4.0
    assert k["opt_ndcg|id"]["top1"][0] > 4 * d["ref"]["id"]["mf_first"]["top1"]   # ID: well ahead of MinFill first
    lo, hi = k["opt_ndcg|ood"]["top3"]
    assert lo < d["ref"]["ood"]["random"]["top3"] < hi                             # OOD Top-3 brackets random


def test_R4_not_reproduced():                   # deck slide 54
    d = load("R4")
    b = d["baselines"]
    assert (b["random|id"]["ours"]["top1"], b["random|id"]["paper"]["top1"]) == (16.13, 22.95)
    assert (b["mf_rand|id"]["ours"]["top3"], b["mf_rand|id"]["paper"]["top3"]) == (44.93, 28.0)
    assert b["mf_rand|id"]["ours"]["top1"] == b["mf_rand|id"]["paper"]["top1"] == 12.0
    assert d["stored_top1_txt"] == "84–97" and d["stored_circuits"] == 225
    assert (d["win_ndcg_id_txt"], d["win_ndcg_id_paper"]) == ("80", 92.0)
