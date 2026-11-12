# Phase 4b: time the release's plans in the authors' own Julia stack, in several modes that keep or remove parts
# of it, to find out what the release's labels measure.
#
#     julia --project=<external>/julia-env/qx p4q_qxtns.jl <authors repo or ""> <out dir> key=value...
#
# Keys (defaults in brackets):
#   circuits=a,b,...     circuit names (QASM files in authors-data/circuits) [required]
#   gens=mf,fc,...       generators, in the order they are run within a circuit [mf,fc,gn,jdrasil,minfill_mrs,
#                        minfillbg_mrs,tamaki]; plans without a file are skipped
#   modes=qxtns,...      what is timed, per plan, in this order [qxtns]:
#     qxtns         QXTns `contract_tn!(tn, plan)`, the library call (OMEinsum on the reduced hyperindex arrays)
#     mock          the same loop with `contract_pair!(...; mock=true)`: QXTns' bookkeeping, no tensor data
#     authors       the release's `contract_tn_rank_mock!(tn, plan; mock=false, verbose=false)`
#                   (src/funcions_article_IA.jl l. 28-71): per-step `@timed` and a line written to a file
#     authors_mock  the same with `mock=true`, as Notebooks 1_a / 1_b run it to write the plan files
#     decomp        `contract_pair!` re-implemented step for step (same calls as QXTns' tensor_network.jl
#                   l. 244-297) with a device synchronisation and a clock after each phase: index bookkeeping
#                   (`contraction_indices`), operand data (`tensor_data`, which may expand and re-reduce a
#                   tensor), the einsum, and building the result + updating the bond map
#     ome           only the einsums: each step's `EinCode` called on operands of that step's shapes, prepared
#                   beforehand (values random, results discarded); no QXTns bookkeeping
#   device=gpu|cpu       storage `CuArray{ComplexF32}` or `Array{ComplexF32}` [gpu]
#   warmup=1 repeats=50  per (plan, mode): one cold run (the first in this process, run -1), `warmup` warm-up
#                        runs (run 0) and up to `repeats` measured runs (1, 2, ...)
#   budget=600           stop measured runs of a (plan, mode) once their sum exceeds this many seconds ...
#   minruns=3            ... but only after this many
#   deadline=0           if > 0, start no new plan after this many seconds since the process started
#   ome_max_gb=4         skip `ome` when its prepared operands would exceed this
#   steps_dump=1         keep the per-step file of `authors` / `authors_mock` for the cold run and run 1
#
# Every run is `copy(base)` (outside the timed region), then the call, then a device synchronisation, timed
# with `@timed` (wall time, GC time, host bytes allocated) and Julia's cumulative compile time. Output in <out dir>:
#   times.csv  one row per run; plans.csv  per plan: steps and the decomposition of the first measured decomp run;
#   steps/     per-step files written by the release's function.
const T0 = time()
stamp(msg) = (println("[", round(time() - T0; digits=1), " s] ", msg); flush(stdout))
include(joinpath(@__DIR__, "authors_path.jl"))
const AUTHORS = authors_path(ARGS[1])
const OUTDIR = abspath(ARGS[2])
const OPT = Dict(String(p[1]) => String(p[2]) for p in (split(a, "="; limit=2) for a in ARGS[3:end]))
opt(k, d) = get(OPT, k, d)
const CIRCUITS = split(opt("circuits", ""), ","; keepempty=false)
const GENS = split(opt("gens", "mf,fc,gn,jdrasil,minfill_mrs,minfillbg_mrs,tamaki"), ","; keepempty=false)
const MODES = split(opt("modes", "qxtns"), ","; keepempty=false)
const DEVICE = opt("device", "gpu")
const WARMUP = parse(Int, opt("warmup", "1"))
const REPEATS = parse(Int, opt("repeats", "50"))
const BUDGET = parse(Float64, opt("budget", "600"))
const MINRUNS = parse(Int, opt("minruns", "3"))
const DEADLINE = parse(Float64, opt("deadline", "0"))
const OME_MAX_GB = parse(Float64, opt("ome_max_gb", "4"))
const STEPS_DUMP = opt("steps_dump", "1") == "1"
isempty(CIRCUITS) && error("circuits=... is required")
mkpath(joinpath(OUTDIR, "steps"))

using QXTools, QXTns, QXZoo, ITensors, CUDA
const EinCode = QXTns.OMEinsum.EinCode
const GPU = DEVICE == "gpu"
GPU && !CUDA.functional() && error("device=gpu but CUDA is not functional")
stamp("packages loaded; device $DEVICE" * (GPU ? " ($(CUDA.name(CUDA.device())))" : "") *
      "; threads $(Threads.nthreads()), BLAS $(QXTns.LinearAlgebra.BLAS.get_num_threads())")
GPU && CUDA.allowscalar(true)  # Notebook 1_a runs interactively, where GPUArrays only warns on scalar indexing
sync() = (GPU && CUDA.synchronize(); nothing)
todev(x) = GPU ? CuArray{ComplexF32}(x) : Array{ComplexF32}(x)
Base.cumulative_compile_timing(true)
compile_s() = Base.cumulative_compile_time_ns()[1] / 1e9  # [2], recompilation, is part of [1]

# funcions_qasm.jl imports qiskit through PyCall at load time, for `load_qasm_circuit` only; the QXZoo reader
# (`import_from_qasm`, `process_gate_line!`) doesn't need it, so those two lines are left out (as in p4_qxtns.jl).
let src = read(joinpath(AUTHORS, "src", "funcions_qasm.jl"), String)
    src = replace(src, "using PyCall" => "", "qiskit = pyimport(\"qiskit\")" => "")
    include_string(Main, src, "funcions_qasm.jl")
end
include(joinpath(AUTHORS, "src", "funcions_article_IA.jl"))  # contract_tn_rank_mock!, as the notebooks use it

const DATA = joinpath(dirname(AUTHORS), "authors-data")
read_steps(path) = [Tuple(Symbol.(split(l)[1:3])) for l in eachline(path)]

function to_dev!(tn)
    for (k, t) in tn.tensor_map
        t.storage = todev(t.storage)
    end
    tn
end

# ---------------------------------------------------------------------------------------------- the modes

run_qxtns!(tn, plan) = contract_tn!(tn, plan)

function run_mock!(tn, plan)
    for (a, b, c) in plan
        contract_pair!(tn, a, b, c; mock=true)
    end
    simple_contraction!(tn)  # remaining tensors are mock, so this is bookkeeping too
end

function run_authors!(tn, plan, mock::Bool, dir)
    t = cd(dir) do
        redirect_stdout(devnull) do
            contract_tn_rank_mock!(tn, plan; mock=mock, verbose=false)
        end
    end
    (store(first(values(tn.tensor_map))), t)
end

"Per-step phases of `contract_pair!` (QXTns tensor_network.jl l. 244-297), synchronised after each."
mutable struct Decomp
    t::Vector{Float64}  # indices, data, einsum, result+bonds (s)
    reshaped::Int       # operands for which `tensor_data` returned other storage than the tensor's own
    host_out::Int       # steps whose result storage is not on the device
    sigs::Set{Any}; ndims::Set{Any}; shapes::Set{Any}
    record::Union{Nothing, Vector{Any}}  # (code, size a, size b) per step, for `ome`
end
Decomp(record::Bool) = Decomp(zeros(4), 0, 0, Set(), Set(), Set(), record ? Any[] : nothing)

function pair_decomp!(tn, a_sym, b_sym, c_sym, d::Decomp)
    t0 = time_ns()
    r = contraction_indices(tn, a_sym, b_sym)
    pos = 1
    c_hyper = map(r.c_indices) do g
        pos += length(g)
        collect(pos-length(g):pos-1)
    end
    c_hyper = convert(Vector{Vector{Int}}, c_hyper)
    filter!(x -> length(x) > 1, c_hyper)
    t1 = time_ns()
    da = tensor_data(tn, a_sym); db = tensor_data(tn, b_sym)
    sync()
    t2 = time_ns()
    code = EinCode((Tuple(r.a_labels), Tuple(r.b_labels)), Tuple(r.c_labels))
    c_data = code(da, db)
    sync()
    t3 = time_ns()
    c = QXTensor(convert(Vector{Index}, vcat(r.c_indices...)), c_hyper, c_data)
    a = tn.tensor_map[a_sym]; b = tn.tensor_map[b_sym]
    common = intersect(inds(a), inds(b))
    for ind in common
        delete!(tn.bond_map, ind)
    end
    for ind in symdiff(inds(a), inds(b))
        tn.bond_map[ind] = replace(tn.bond_map[ind], a_sym => c_sym, b_sym => c_sym)
    end
    tn.tensor_map[c_sym] = c
    delete!(tn.tensor_map, a_sym); delete!(tn.tensor_map, b_sym)
    sync()
    t4 = time_ns()
    d.t .+= (t1 - t0, t2 - t1, t3 - t2, t4 - t3) ./ 1e9
    d.reshaped += (da !== a.storage) + (db !== b.storage)
    d.host_out += GPU && !(c.storage isa CuArray)
    push!(d.sigs, (Tuple(r.a_labels), Tuple(r.b_labels), Tuple(r.c_labels)))
    push!(d.ndims, (ndims(da), ndims(db), ndims(c_data)))
    push!(d.shapes, (size(da), size(db)))
    d.record === nothing || push!(d.record, (code, size(da), size(db)))
    c_sym
end

function run_decomp!(tn, plan, d::Decomp)
    for (a, b, c) in plan
        pair_decomp!(tn, a, b, c, d)
    end
    syms = collect(keys(tn))  # simple_contraction!: any disjoint tensors left, contracted in order
    A = syms[1]
    for B in syms[2:end]
        A = pair_decomp!(tn, A, B, next_tensor_id!(tn), d)
    end
    store(tn[A])
end

function run_ome(ops)
    for (code, x, y) in ops
        code(x, y)
    end
    nothing
end

# ---------------------------------------------------------------------------------------------- timing

function timed_run(f)
    GC.gc(false)  # a young-generation collection between runs, outside the timed region
    sync()
    c0 = compile_s()
    s = @timed begin
        v = f()
        sync()
        v
    end
    (value=s.value, time=s.time, gc=s.gctime, bytes=s.bytes, compile=compile_s() - c0)
end

scalar(v) = v === nothing ? NaN + 0im : (v isa Tuple ? scalar(v[1]) :
            (v isa MockTensor ? NaN + 0im : ComplexF64(Array(v)[1])))

const TIMES = open(joinpath(OUTDIR, "times.csv"), "a")
const PLANS = open(joinpath(OUTDIR, "plans.csv"), "a")
filesize(joinpath(OUTDIR, "times.csv")) == 0 &&
    println(TIMES, "circuit,plan,generator,mode,device,order,run,seconds,gc_s,alloc_mb,compile_s,re,im,inner_s")
filesize(joinpath(OUTDIR, "plans.csv")) == 0 &&
    println(PLANS, "circuit,plan,generator,device,order,steps,tensors,build_s,decomp_indices_s,decomp_data_s," *
                   "decomp_einsum_s,decomp_rest_s,reshaped_operands,host_outputs,distinct_signatures," *
                   "distinct_ndims,distinct_shapes,ome_gb")
flush(TIMES); flush(PLANS)

function time_mode(circ, name, gen, order, mode, base, plan, tmp)
    stepfile = joinpath(tmp, "fitxer_sortida.txt")
    local ops = nothing
    if mode == "ome"
        d = Decomp(true)
        run_decomp!(copy(base), plan, d)
        gb = sum(prod(sa) + prod(sb) for (_, sa, sb) in d.record) * 8 / 2^30
        if gb > OME_MAX_GB
            stamp("$name ome: $(round(gb; digits=2)) GB of operands, skipped")
            return gb
        end
        ops = [(code, todev(randn(ComplexF32, sa)), todev(randn(ComplexF32, sb))) for (code, sa, sb) in d.record]
        d = nothing
    end
    measured = 0.0
    runs = 0
    for r in -1:REPEATS
        r > 0 && runs >= MINRUNS && measured >= BUDGET && break
        r == 0 && WARMUP == 0 && continue
        for w in 1:(r == 0 ? WARMUP : 1)
            tn = mode == "ome" ? nothing : copy(base)
            local inner = NaN
            res = if mode == "qxtns"
                timed_run(() -> run_qxtns!(tn, plan))
            elseif mode == "mock"
                timed_run(() -> run_mock!(tn, plan))
            elseif mode in ("authors", "authors_mock")
                x = timed_run(() -> run_authors!(tn, plan, mode == "authors_mock", tmp))
                inner = x.value[2]
                if STEPS_DUMP && r in (-1, 1)
                    cp(stepfile, joinpath(OUTDIR, "steps", "$(name)__$(mode)__$(DEVICE)__run$(r).txt"); force=true)
                end
                x
            elseif mode == "decomp"
                d = Decomp(false)
                x = timed_run(() -> run_decomp!(tn, plan, d))
                if r == 1
                    write_plan_row(circ, name, gen, order, plan, base, d)
                end
                x
            elseif mode == "ome"
                timed_run(() -> run_ome(ops))
            else
                error("unknown mode $mode")
            end
            v = scalar(res.value)
            println(TIMES, join((circ, name, gen, mode, DEVICE, order, r, res.time, res.gc, res.bytes / 2^20,
                                 res.compile, real(v), imag(v), inner), ","))
            flush(TIMES)
            if r > 0
                measured += res.time
                runs += 1
            end
        end
    end
    stamp("$name $mode: $runs runs, mean $(round(measured / max(runs, 1); sigdigits=4)) s")
    ops = nothing
    nothing
end

const PLAN_INFO = Dict{String, Any}()
function write_plan_row(circ, name, gen, order, plan, base, d)
    info = PLAN_INFO[name]
    println(PLANS, join((circ, name, gen, DEVICE, order, length(plan), length(base.tensor_map), info.build_s,
                         d.t..., d.reshaped, d.host_out, length(d.sigs), length(d.ndims), length(d.shapes), ""), ","))
    flush(PLANS)
end

tmp = mktempdir()
for circ in CIRCUITS
    if DEADLINE > 0 && time() - T0 > DEADLINE
        stamp("deadline reached before $circ"); break
    end
    t_build = @elapsed tnc = redirect_stdout(devnull) do  # the reader prints every line it doesn't turn into a gate
        convert_to_tnc(import_from_qasm(joinpath(DATA, "circuits", "$circ.qasm")))
    end
    base = to_dev!(copy(tnc.tn))
    stamp("$circ: $(length(base.tensor_map)) tensors, built in $(round(t_build; digits=2)) s")
    order = 0
    for gen in GENS
        name = "$(circ)_pla_$(gen)"
        path = joinpath(DATA, "plans", "resultats", "resultats_$name.txt")
        isfile(path) || continue
        if DEADLINE > 0 && time() - T0 > DEADLINE
            stamp("deadline reached before $name"); break
        end
        order += 1
        plan = read_steps(path)
        PLAN_INFO[name] = (build_s=t_build,)
        for mode in MODES
            try
                time_mode(circ, name, gen, order, mode, base, plan, tmp)
            catch e
                stamp("$name $mode failed: " * sprint(showerror, e)[1:min(end, 300)])
                println(TIMES, join((circ, name, gen, mode, DEVICE, order, -9, NaN, NaN, NaN, NaN, NaN, NaN, NaN), ","))
                flush(TIMES)
                GPU && (GC.gc(true); CUDA.reclaim())
            end
        end
        GC.gc(true)
        GPU && CUDA.reclaim()
    end
end
close(TIMES); close(PLANS)
stamp("finished")
