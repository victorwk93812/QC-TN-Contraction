# The authors' own code path for Tables 6 and 7, run headless (their Notebook 5 asks
# for a menu choice on stdin). Reference only. Writes one CSV row per model and split.
# Usage: julia --project=external/julia-env/rank t06_t07.jl <authors repo> <output csv>
authors, out = ARGS
include(joinpath(authors, "src", "Preliminar_getting_results.jl"))

load(name) = getting_dataframe_excel(joinpath(authors, "Data", "splits", name))
NOTEBOOK = (
    ndcg = (params = [:objective => "rank:ndcg", :eval_metric => "ndcg@3", :maximize => true,
                      :eta => 0.02, :max_depth => 6, :subsample => 0.8, :seed => 42],
            rounds = (1, 65), features = FEATURES_13_ndcg),
    pair = (params = params_pairwise, rounds = (16, 14), features = FEATURES_13_pair),
)

open(out, "w") do io
    println(io, "objective,model,split,rounds,top1,top3,regret1,regret3")
    row(o, m, s, r, x) = println(io, join((o, m, s, r, x[:top1], x[:top3], x[:regret1], x[:regret3]), ","))
    for (o, cfg) in pairs(NOTEBOOK)
        train = load("df_final_$o")
        stored = Booster(DMatrix(rand(Float32, 2, 13), Float32[0, 1]))
        XGBoost.load!(stored, joinpath(authors, "models", "opt_$o.json"))
        for (s, name) in (("train", "df_final_$o"), ("id", "df_test_n_$o"), ("ood", "df_ood_n_$o"))
            d = load(name)
            row(o, "stored", s, "-", evaluate_tops_and_regrets(stored, d, cfg.features; verbose=false))
            s == "train" && continue
            for r in cfg.rounds
                m = Getting_model(train, cfg.features, cfg.params; num_round=r, verbose=false)
                row(o, "notebook", s, r, evaluate_tops_and_regrets(m, d, cfg.features; verbose=false))
            end
        end
    end
end
