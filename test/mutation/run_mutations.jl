#!/usr/bin/env julia
#
# Mutation-testing harness. MANUAL SCRIPT — deliberately not registered in
# test/runtests.jl: it forks a Julia process per mutation and takes minutes.
#
#   julia --project=. test/mutation/run_mutations.jl            # all mutations
#   julia --project=. test/mutation/run_mutations.jl kubo dos   # by name prefix
#
# What it does: for each mutation, copy Project.toml/Manifest.toml/src/ext/test
# into a fresh temporary package directory, apply a small text substitution to
# the *copy*, run the named test file against the mutated copy, and check that
# the test file FAILS. The working tree is never touched.
#
# A mutation whose substitution does not match (source moved) is a hard error:
# a silently-skipped mutation would look like a passing sensitivity check.
#
# A mutation that is NOT caught is a finding about the test suite, not a
# licence to weaken anything.

using Printf

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())

struct Mutation
    name::String
    description::String
    file::String                       # path relative to the repo root
    edits::Vector{Pair{String,String}} # old => new, each must match exactly once
    tests::Vector{String}              # test files that must fail
end

const MUTATIONS = [
    Mutation(
        "kubo_sign",
        "flip the overall sign of the Kubo-Bastin conductivity",
        "src/applications/conductivity.jl",
        ["return -2 * NH / (area * a^2) * real(acc)" =>
            "return +2 * NH / (area * a^2) * real(acc)"],
        ["kubo_bastin_test.jl"],
    ),
    Mutation(
        "dos_factor2",
        "multiply the DOS reconstruction by 2",
        "src/applications/dos.jl",
        ["        rhoE ./= denom\n" => "        rhoE ./= denom\n        rhoE .*= 2\n"],
        ["dos_test.jl"],
    ),
    Mutation(
        "probe_centering",
        "make normalize_by_col mean-center probe columns by default",
        "src/utils/vectors.jl",
        ["function normalize_by_col(psi_in, NR; centering = false)" =>
            "function normalize_by_col(psi_in, NR; centering = true)"],
        ["dos_test.jl"],
    ),
    Mutation(
        "jackson_identity",
        "replace the Jackson kernel by the identity kernel g_n = 1",
        "src/kernels/jackson_kernel.jl",
        ["(1/(N+1)*((N+1-n)*cos(pi*n/(N+1)) + sin(pi*n/(N+1))*cot(pi/(N+1)))) * (n < N)" =>
            "1.0 * (n < N)"],
        ["dos_test.jl", "kubo_bastin_test.jl"],
    ),
    Mutation(
        "marker_sign",
        "flip the sign of the Bianco-Resta Chern marker",
        "src/frontend.jl",
        [
            "markers[lo:hi] .= (4π) .* r" => "markers[lo:hi] .= (-4π) .* r",
            "estimates[lo:hi] .= (4π * length(region)) .* r" =>
                "estimates[lo:hi] .= (-4π * length(region)) .* r",
        ],
        ["chern_marker_test.jl"],
    ),
    Mutation(
        "constant_lorentz_width",
        "broaden with a constant η instead of the position-dependent η(E) " *
        "the θ-uniform kernel damping produces",
        "src/applications/greens.jl",
        [
            "    g = kernel === nothing ? ones(dt_real, NC) : kernel.(0:(NC-1), NC)" =>
                "    g = ones(dt_real, NC)",
            "    η = eta === nothing ? 0.0 : float(eta)" =>
                "    η = eta === nothing ? a * 4.0 / NC : float(eta)",
        ],
        ["greens_test.jl"],
    ),
]

function stage_package(dest)
    mkpath(dest)
    for f in ("Project.toml", "Manifest.toml")
        src = joinpath(REPO, f)
        isfile(src) && cp(src, joinpath(dest, f))
    end
    for d in ("src", "ext", "test")
        src = joinpath(REPO, d)
        isdir(src) && cp(src, joinpath(dest, d))
    end
    isfile(joinpath(dest, "Manifest.toml")) || @warn(
        "no Manifest.toml in the repo root; the mutated copy may need `Pkg.instantiate()`"
    )
    return dest
end

function apply_edits!(pkgdir, m::Mutation)
    path = joinpath(pkgdir, m.file)
    isfile(path) || error("mutation $(m.name): $(m.file) does not exist")
    text = read(path, String)
    for (old, new) in m.edits
        n = length(findall(old, text))
        n == 1 || error(
            "mutation $(m.name): pattern occurs $n times (expected exactly 1) in " *
            "$(m.file); the source moved — fix the harness, do not skip it.\n" *
            "  pattern: $(repr(old))",
        )
        text = replace(text, old => new)
    end
    write(path, text)
    return nothing
end

"Run one test file against `pkgdir`; return true if it FAILED."
function test_fails(pkgdir, testfile)
    testdir = joinpath(pkgdir, "test")
    code = string("using Test, KPM; include(\"", testfile, "\")")
    cmd = Cmd(`$JULIA --project=$pkgdir --color=no -e $code`; dir = testdir)
    out = IOBuffer()
    ok = try
        success(pipeline(cmd; stdout = out, stderr = out))
    catch
        false
    end
    return (!ok, String(take!(out)))
end

function main(args)
    selected = isempty(args) ? MUTATIONS :
        filter(m -> any(a -> startswith(m.name, a), args), MUTATIONS)
    isempty(selected) && error("no mutation matches $(args)")

    rows = NamedTuple[]
    for m in selected
        for testfile in m.tests
            tmp = mktempdir(; prefix = "kpm_mut_")
            try
                pkgdir = stage_package(joinpath(tmp, "KPM"))
                apply_edits!(pkgdir, m)
                @info "running $(m.name) → $(testfile)"
                failed, log = test_fails(pkgdir, testfile)
                push!(
                    rows,
                    (
                        mutation = m.name,
                        test = testfile,
                        expected = "FAIL",
                        observed = failed ? "FAIL" : "PASS (NOT CAUGHT)",
                        caught = failed,
                        log = log,
                    ),
                )
            finally
                rm(tmp; recursive = true, force = true)
            end
        end
    end

    println()
    println("mutation testing: each mutation must make the named test file fail")
    println(repeat("-", 92))
    @printf("%-24s %-26s %-10s %-24s\n", "mutation", "test file", "expected", "observed")
    println(repeat("-", 92))
    for r in rows
        @printf("%-24s %-26s %-10s %-24s\n", r.mutation, r.test, r.expected, r.observed)
    end
    println(repeat("-", 92))
    missed = filter(r -> !r.caught, rows)
    if isempty(missed)
        println("all $(length(rows)) mutations caught")
    else
        println("$(length(missed)) mutation(s) NOT caught — this is a finding:")
        for r in missed
            println("  * $(r.mutation) survives $(r.test)")
            println(join("      " .* split(strip(r.log), '\n')[max(1, end - 12):end], "\n"))
        end
        exit(1)
    end
    return nothing
end

main(ARGS)
