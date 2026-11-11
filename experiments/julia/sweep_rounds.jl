# Predictions of the authors' training recipe at every boosting round, for each
# parameter set, so metrics can be computed in one place (qctn.rank.metrics).
# Reference only. Writes long-format CSV: objective,params,round,split,row,score
# Usage: julia --project=external/julia-env/rank sweep_rounds.jl <authors repo, or ""> <output csv> <max rounds>
include(joinpath(@__DIR__, "authors_path.jl"))
authors, out, maxr = authors_path(ARGS[1]), abspath(ARGS[2]), parse(Int, ARGS[3])
include(joinpath(authors, "src", "Preliminar_getting_results.jl"))
load(name) = getting_dataframe_excel(joinpath(authors, "Data", "splits", name))

base_ndcg = [:objective => "rank:ndcg", :eval_metric => "ndcg@3", :maximize => true, :seed => 42, :max_depth => 6]
CONFIGS = [
    (:ndcg, "notebook", [base_ndcg; :eta => 0.02; :subsample => 0.8], FEATURES_13_ndcg),
    (:ndcg, "table4", [base_ndcg; :eta => 0.01; :subsample => 0.7], FEATURES_13_ndcg),
    (:pair, "notebook", params_pairwise, FEATURES_13_pair),   # identical to Table 4
]

open(out, "w") do io
    println(io, "objective,params,round,split,row,score")
    for (o, name, params, features) in CONFIGS
        model = Getting_model(load("df_final_$o"), features, params; num_round=maxr, verbose=false)
        for (s, file) in (("id", "df_test_n_$o"), ("ood", "df_ood_n_$o"))
            X = Matrix{Float64}(load(file)[:, features])
            for r in 1:maxr
                p = XGBoost.predict(model, X; ntree_limit=r)
                for (i, v) in enumerate(p)
                    println(io, join((o, name, r, s, i - 1, v), ","))
                end
            end
        end
    end
end
