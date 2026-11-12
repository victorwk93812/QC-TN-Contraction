# Phase 4 check: dump every leaf tensor of the authors' networks (their QASM reader `import_from_qasm` and QXTools
# `convert_to_tnc`), with QXTns' hyperindex groups and the stored (reduced) data, for `experiments/p4_leaves.py`.
# CPU only. funcions_qasm.jl imports qiskit through PyCall at load time for `load_qasm_circuit`, which isn't used
# here, so those two lines are left out.
#
# Usage: julia --project=<external>/julia-env/qx p4_leaves.jl <authors repo or ""> <out dir> <circuit>...
# One file per circuit: <out dir>/<circuit>.leaves.csv with columns tensor, rank, groups (0-based positions,
# groups separated by ';'), shape, data (column-major, re:im).
include(joinpath(@__DIR__, "authors_path.jl"))
authors, outdir = authors_path(ARGS[1]), abspath(ARGS[2])
mkpath(outdir)
using QXTools, QXTns, QXZoo, ITensors
src = read(joinpath(authors, "src", "funcions_qasm.jl"), String)
src = replace(src, "using PyCall" => "", "qiskit = pyimport(\"qiskit\")" => "")
include_string(Main, src, "funcions_qasm.jl")
for name in ARGS[3:end]
    tnc = redirect_stdout(devnull) do  # the reader prints every line it doesn't turn into a gate
        convert_to_tnc(import_from_qasm(joinpath(dirname(authors), "authors-data", "circuits", "$name.qasm")))
    end
    open(joinpath(outdir, "$name.leaves.csv"), "w") do io
        println(io, "tensor,rank,groups,shape,data")
        for k in sort(collect(keys(tnc.tn.tensor_map)); by=x -> parse(Int, string(x)[2:end]))
            t = tnc.tn.tensor_map[k]
            g = join([join(x .- 1, " ") for x in t.hyper_indices], ";")
            d = join(["$(real(x)):$(imag(x))" for x in vec(t.storage)], " ")
            println(io, "$k,$(t.rank),$g,$(join(size(t.storage), "x")),$d")
        end
    end
    println("$name: $(length(tnc.tn)) tensors")
end
