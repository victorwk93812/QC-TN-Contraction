# Phase 4 reference: time the release's plans with QXTns `contract_tn!` (non-mock; OMEinsum on CUDA arrays).
# The release has no harness for this (only the mock run of Notebook 1_a, cell 18); this is our harness around
# their stack: network from their QASM reader (`import_from_qasm`) and QXTools `convert_to_tnc`, storage moved to
# the GPU as in Notebook 1_a but cast to ComplexF32 (the paper's Float32), then per plan one warm-up and
# `repeats` runs of `CUDA.@sync contract_tn!(copy(tn), plan)`; the copy is made outside the timed region.
# The timed region includes QXTns' per-step bookkeeping on the CPU (hyperindex groups, bond map), as it would in
# any use of `contract_tn!`.
#
# Usage: julia --project=<external>/julia-env/qx p4_qxtns.jl <authors repo or ""> <out dir> <repeats> <plan name>...
const T0 = time()
stamp(msg) = (println("[", round(time() - T0; digits=1), " s] ", msg); flush(stdout))
include(joinpath(@__DIR__, "authors_path.jl"))
authors, outdir, repeats = authors_path(ARGS[1]), abspath(ARGS[2]), parse(Int, ARGS[3])
names = ARGS[4:end]
mkpath(outdir)

using QXTools, QXTns, QXZoo, ITensors, CUDA  # QXTns loads OMEinsum and its CUDA module
const GPU = CUDA.functional()  # without a GPU (a dry run on a login node) the same code runs on CPU arrays
stamp("packages loaded; CUDA functional: $GPU" * (GPU ? "; device: $(CUDA.name(CUDA.device()))" : ""))
GPU && CUDA.allowscalar(true)  # Notebook 1_a runs interactively, where GPUArrays only warns on scalar indexing
sync() = GPU && CUDA.synchronize()

# funcions_qasm.jl imports qiskit through PyCall at load time, for `load_qasm_circuit` only; the QXZoo reader
# (`import_from_qasm`, `process_gate_line!`) doesn't need it, so those two lines are left out.
src = read(joinpath(authors, "src", "funcions_qasm.jl"), String)
src = replace(src, "using PyCall" => "", "qiskit = pyimport(\"qiskit\")" => "")
include_string(Main, src, "funcions_qasm.jl")

circuits = joinpath(dirname(authors), "authors-data", "circuits")
read_steps(path) = [Tuple(Symbol.(split(l)[1:3])) for l in eachline(path)]

function to_gpu!(tn)
    for (k, t) in tn.tensor_map
        t.storage = GPU ? CuArray{ComplexF32}(t.storage) : Array{ComplexF32}(t.storage)
    end
    tn
end

open(joinpath(outdir, "julia_times.csv"), "w") do io
    println(io, "plan,run,seconds,re,im")  # run 0 is the warm-up
    for name in names
        circ_name = split(name, "_pla_")[1]
        tnc = redirect_stdout(devnull) do  # the reader prints every line it doesn't turn into a gate
            convert_to_tnc(import_from_qasm(joinpath(circuits, "$circ_name.qasm")))
        end
        plan = read_steps(joinpath(dirname(authors), "authors-data", "plans", "resultats", "resultats_$name.txt"))
        base = to_gpu!(copy(tnc.tn))
        for r in 0:repeats
            tn = copy(base)
            sync()
            t = @elapsed begin
                v = contract_tn!(tn, plan)
                sync()
            end
            val = ComplexF64(Array(v)[1])
            println(io, "$name,$r,$t,$(real(val)),$(imag(val))")
            flush(io)
        end
        stamp("$name: done")
    end
end
stamp("finished")
