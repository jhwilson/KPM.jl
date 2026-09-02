#!/usr/bin/env julia
#
# Mutation-testing harness. MANUAL SCRIPT — deliberately not registered in
# test/runtests.jl: it forks a Julia process per mutation and takes minutes.
#
#   julia --startup-file=no --project=. test/mutation/run_mutations.jl            # all mutations
#   julia --startup-file=no --project=. test/mutation/run_mutations.jl kubo dos   # by name prefix
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

"Run `cmd`, retaining its combined output in `logpath`."
function run_logged(cmd, logpath)
    open(logpath, "w") do io
        proc = run(pipeline(ignorestatus(cmd), stdout = io, stderr = io))
        return success(proc)
    end
end

"Run one test file against `pkgdir`; return its exit status and captured output."
function run_test(pkgdir, testfile, logpath)
    testdir = joinpath(pkgdir, "test")
    code = string("using Test, KPM; include(\"", testfile, "\")")
    cmd = Cmd(`$JULIA --startup-file=no --project=$pkgdir --color=no -e $code`; dir = testdir)
    return run_logged(cmd, logpath), read(logpath, String)
end

function run_load(pkgdir, logpath)
    cmd = Cmd(`$JULIA --startup-file=no --project=$pkgdir --color=no -e "using KPM"`; dir = pkgdir)
    return run_logged(cmd, logpath), read(logpath, String)
end

function last_lines(log; n = 30)
    lines = split(chomp(log), '\n')
    isempty(lines) && return "(no output)"
    return join(lines[max(1, length(lines) - n + 1):end], "\n")
end

is_test_failure(log) = occursin("Test Failed", log) || occursin("Some tests did not pass", log)

function validate_unmutated!(selected)
    baseline_tmp = mktempdir(; prefix = "kpm_mut_baseline_", cleanup = false)
    pkgdir = stage_package(joinpath(baseline_tmp, "KPM"))
    testfiles = unique(Iterators.flatten(m.tests for m in selected))
    for testfile in testfiles
        logpath = joinpath(baseline_tmp, "baseline_$(testfile).log")
        @info "validating unmutated $(testfile)"
        ok, log = run_test(pkgdir, testfile, logpath)
        ok || error(
            "unmutated staged copy fails $(testfile); aborting mutation harness. " *
            "See $(logpath):\n$(last_lines(log))",
        )
    end
    return baseline_tmp
end

function main(args)
    selected = isempty(args) ? MUTATIONS :
        filter(m -> any(a -> startswith(m.name, a), args), MUTATIONS)
    isempty(selected) && error("no mutation matches $(args)")

    baseline_tmp = validate_unmutated!(selected)
    @info "unmutated validation passed; logs retained in $(baseline_tmp)"
    rows = NamedTuple[]
    for m in selected
        for testfile in m.tests
            tmp = mktempdir(; prefix = "kpm_mut_", cleanup = false)
            pkgdir = stage_package(joinpath(tmp, "KPM"))
            apply_edits!(pkgdir, m)
            load_logpath = joinpath(tmp, "load.log")
            load_ok, load_log = run_load(pkgdir, load_logpath)
            if !load_ok
                push!(
                    rows,
                    (
                        mutation = m.name,
                        test = testfile,
                        expected = "FAIL",
                        observed = "HARNESS ERROR (load failure)",
                        caught = false,
                        harness_error = true,
                        log = load_log,
                        logpath = load_logpath,
                    ),
                )
                continue
            end

            @info "running $(m.name) → $(testfile)"
            logpath = joinpath(tmp, "$(testfile).log")
            ok, log = run_test(pkgdir, testfile, logpath)
            caught = !ok && is_test_failure(log)
            observed = caught ? "FAIL (CAUGHT)" :
                ok ? "PASS (NOT CAUGHT)" : "HARNESS ERROR (non-Test failure)"
            push!(
                rows,
                (
                    mutation = m.name,
                    test = testfile,
                    expected = "FAIL",
                    observed = observed,
                    caught = caught,
                    harness_error = !ok && !caught,
                    log = log,
                    logpath = logpath,
                ),
            )
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
    println("summary: $(length(selected)) mutations, $(length(rows)) (mutation, test file) runs")
    problems = filter(r -> !r.caught, rows)
    if isempty(problems)
        println("all $(length(rows)) mutation runs caught")
    else
        println("$(length(problems)) mutation run(s) need attention:")
        for r in problems
            label = r.harness_error ? "harness error" : "not caught"
            println("  * $(label): $(r.mutation) → $(r.test)")
            println("      log: $(r.logpath)")
            println(join("      " .* split(last_lines(r.log), '\n'), "\n"))
        end
        exit(1)
    end
    for r in rows
        println("caught: $(r.mutation) → $(r.test); log: $(r.logpath)")
    end
    return nothing
end

main(ARGS)
