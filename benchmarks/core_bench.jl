#!/usr/bin/env julia

# Benchmark the KPM compute core.  This file deliberately bootstraps its local
# environment instead of using [sources], which keeps it compatible with Julia
# 1.10 while always testing the package adjacent to this benchmarks directory.
import Pkg

const BENCHMARK_DIR = @__DIR__
const PACKAGE_DIR = normpath(joinpath(BENCHMARK_DIR, ".."))
Pkg.develop(path = PACKAGE_DIR)

using BenchmarkTools
using KPM
using LinearAlgebra
using Printf
using Random
using SparseArrays

const SEED = 1234
const BENCH_SAMPLES = 3
const BENCH_SECONDS = 60
# The requested NC=256 2D suite, and then NC=128, exceeded five minutes on
# the baseline laptop; 64 keeps the same path practical while retaining a
# meaningful block-size comparison. The actual size is in every result JSON.
const COND_NC = 64

"""
    square_lattice(Lx, Ly; currents=false)

Complex nearest-neighbor hopping on a periodic square lattice.  For current
vertices, this follows the package examples' minimum-image construction:
`Jα[i, j] = H[i, j] * (rᵢ - rⱼ)_α` with boundary-crossing bonds displaced by
one lattice spacing, rather than by the full coordinate jump around the torus.
"""
function square_lattice(Lx::Int, Ly::Int; currents::Bool = false)
    NH = Lx * Ly
    site(x, y) = x + 1 + Lx * y
    minimum_image(delta, L) = mod(delta + fld(L, 2), L) - fld(L, 2)

    rows = Int[]
    cols = Int[]
    hvals = ComplexF64[]
    jxvals = ComplexF64[]
    jyvals = ComplexF64[]

    # Add each directed nearest-neighbor entry explicitly.  This makes H
    # Hermitian and lets the current use the matching directed displacement.
    for y = 0:(Ly - 1), x = 0:(Lx - 1)
        i = site(x, y)
        for (xn, yn) in ((mod(x + 1, Lx), y), (mod(x - 1, Lx), y),
                         (x, mod(y + 1, Ly)), (x, mod(y - 1, Ly)))
            j = site(xn, yn)
            dx = minimum_image(x - xn, Lx)
            dy = minimum_image(y - yn, Ly)
            hij = -1.0 + 0.0im
            push!(rows, i); push!(cols, j); push!(hvals, hij)
            currents && (push!(jxvals, hij * dx); push!(jyvals, hij * dy))
        end
    end

    H = sparse(rows, cols, hvals, NH, NH)
    currents || return H
    return H, sparse(rows, cols, jxvals, NH, NH), sparse(rows, cols, jyvals, NH, NH)
end

function normalized_square(Lx::Int, Ly::Int; currents::Bool = false)
    lattice = square_lattice(Lx, Ly; currents = currents)
    H = currents ? lattice[1] : lattice
    a, Hn = KPM.normalizeH(H)
    return currents ? (Hn, lattice[2], lattice[3]) : Hn
end

function benchmark_seconds(f)
    # Interpolate the closure so benchmark setup does not include a global
    # lookup.  BenchmarkTools returns the minimum sampled elapsed time.
    return @belapsed $f() samples = BENCH_SAMPLES evals = 1 seconds = BENCH_SECONDS
end

function optional_benchmark(f)
    try
        return benchmark_seconds(f)
    catch err
        @warn "optional arr_size benchmark unavailable" exception = (err, catch_backtrace())
        return nothing
    end
end

json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", '"' => "\\\"")
json_number_or_na(x) = x === nothing ? "\"n/a\"" : @sprintf("%.9g", x)

function git_revision()
    try
        return readchomp(`git -C $PACKAGE_DIR rev-parse --short HEAD`)
    catch
        return "unknown"
    end
end

function write_results(path, label, revision, times, workloads)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"label\": \"$(json_escape(label))\",")
        println(io, "  \"julia_version\": \"$(VERSION)\",")
        println(io, "  \"nthreads\": $(Threads.nthreads()),")
        println(io, "  \"git_rev\": \"$(json_escape(revision))\",")
        println(io, "  \"cases\": {")
        names = collect(keys(times))
        for (i, name) in enumerate(names)
            suffix = i == length(names) ? "" : ","
            println(io, "    \"$(name)\": $(json_number_or_na(times[name]))$(suffix)")
        end
        println(io, "  },")
        println(io, "  \"workloads\": {")
        names = collect(keys(workloads))
        for (i, name) in enumerate(names)
            suffix = i == length(names) ? "" : ","
            println(io, "    \"$(name)\": \"$(workloads[name])\"$(suffix)")
        end
        println(io, "  }")
        println(io, "}")
    end
end

function main()
    revision = git_revision()
    label = isempty(ARGS) ? revision : ARGS[1]
    times = Dict{String,Union{Nothing,Float64}}()
    workloads = Dict{String,String}()

    # DOS, 256 × 256.  Probes are made before timing and are never regenerated
    # by the measured function.
    H1 = normalized_square(256, 256)
    psi1 = KPM.random_phase_vectors(Xoshiro(SEED), 1 << 16, 8)
    name = "kpm_1d_dos_NH65536_NC1024_NR8"
    workloads[name] = "NH=65536 (256x256 periodic square), NC=1024, NR=8"
    times[name] = benchmark_seconds(() -> KPM.kpm_1d(H1, Int64(1024), Int64(8), Int64(1 << 16); psi_in = psi1))

    H2 = normalized_square(512, 512)
    psi2 = KPM.random_phase_vectors(Xoshiro(SEED), 1 << 18, 4)
    name = "kpm_1d_dos_NH262144_NC512_NR4"
    workloads[name] = "NH=262144 (512x512 periodic square), NC=512, NR=4"
    times[name] = benchmark_seconds(() -> KPM.kpm_1d(H2, Int64(512), Int64(4), Int64(1 << 18); psi_in = psi2))

    Hc, Jx, Jy = normalized_square(128, 128; currents = true)
    psic = KPM.random_phase_vectors(Xoshiro(SEED), 1 << 14, 4)
    for (suffix, kwargs) in (("default", NamedTuple()), ("arr_size3", (arr_size = 3,)), ("arr_size64", (arr_size = 64,)))
        name = "kpm_2d_cond_NH16384_NC$(COND_NC)_NR4_$(suffix)"
        workloads[name] = "NH=16384 (128x128 periodic square), NC=$(COND_NC) (shrunk from 256), NR=4, $(suffix)"
        run = () -> KPM.kpm_2d(Hc, Jx, Jy, Int64(COND_NC), Int64(4), Int64(1 << 14); psi_in = psic, kwargs...)
        times[name] = suffix == "arr_size64" ? optional_benchmark(run) : benchmark_seconds(run)
    end

    Ha = normalized_square(256, 256)
    Va = KPM.random_phase_vectors(Xoshiro(SEED), 1 << 16, 8)
    # `fermi_coefficients` is already Jackson-damped and includes the complete
    # Chebyshev-series weights expected verbatim by chebyshev_action!.
    C = hcat(
        KPM.fermi_coefficients(1.0, 0.0, 0.0; beta = Inf, NC = 1024),
        KPM.fermi_coefficients(1.0, 0.0, 0.25; beta = Inf, NC = 1024),
    )
    out = zeros(ComplexF64, 1 << 16, 8, 2)
    name = "chebyshev_action_NH65536_NC1024_NR8_K2"
    workloads[name] = "NH=65536 (256x256 periodic square), NC=1024, NR=8, K=2"
    times[name] = benchmark_seconds(() -> KPM.chebyshev_action!(out, Ha, Va, C; check_every = 0))

    Hi = normalized_square(512, 512)
    Vpp = KPM.random_phase_vectors(Xoshiro(SEED), 1 << 18, 8)
    Vp = copy(Vpp)
    # This intentionally measures only one raw recurrence update; the
    # preallocated workspaces are mutated between samples.
    name = "chebyshev_iter_single_NH262144_NR8"
    workloads[name] = "NH=262144 (512x512 periodic square), NR=8, one raw recurrence step"
    times[name] = benchmark_seconds(() -> KPM.chebyshev_iter_single(Hi, Vpp, Vp))

    println("| case | seconds |")
    println("| --- | ---: |")
    for name in keys(times)
        value = times[name]
        value === nothing ? println("| $(name) | n/a |") : @printf("| %s | %.6f |\n", name, value)
    end
    println("Peak RSS: $(Sys.maxrss()) bytes")

    result_path = joinpath(BENCHMARK_DIR, "results", "$(label).json")
    write_results(result_path, label, revision, times, workloads)
    println("Wrote $(result_path)")
end

main()
