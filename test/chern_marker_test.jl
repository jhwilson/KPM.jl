using Test
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using KPM

# Fermi projector and Bianco–Resta local Chern marker. Anchors: exact dense
# projectors (ED), and the FHS Chern number of the Haldane model — the same
# sign convention pinned by kubo_bastin_test.jl (σ_xy = +C e²/h).

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

# Evaluate a Chebyshev series with coefficients c at x ∈ (-1, 1).
_cheb_eval(c, x) = sum(c[n+1] * cos(n * acos(x)) for n = 0:(length(c)-1))

@testset "fermi_coefficients: closed form, quadrature, and error paths" begin
    a, b, Ef, NC = 2.3, 0.4, 0.7, 256
    xF = (Ef - b) / a

    # closed form (beta = Inf) is the large-beta quadrature limit; the
    # integrand is then a step, so node discretization limits agreement to
    # O(1/Np) — assert the level and its shrinkage with Np
    c_step = KPM.fermi_coefficients(a, b, Ef; NC = NC)
    c_quad = KPM.fermi_coefficients(a, b, Ef; beta = 1e8, NC = NC)
    @test maximum(abs, c_step .- c_quad) < 2e-3
    c_quad_fine = KPM.fermi_coefficients(a, b, Ef; beta = 1e8, NC = NC, Np = 8192)
    @test maximum(abs, c_step .- c_quad_fine) < 1e-4

    # undamped series reproduces the step away from x̃_F (Gibbs at the edge)
    c_bare = KPM.fermi_coefficients(a, b, Ef; NC = NC, kernel = (n, N) -> 1.0)
    for x in (-0.9, -0.5, xF - 0.2, xF + 0.2, 0.6, 0.9)
        step = x < xF ? 1.0 : 0.0
        @test _cheb_eval(c_bare, x) ≈ step atol = 1e-2
    end
    # Jackson damping: accurate outside a ~π/NC resolution window
    for x in (-0.9, -0.5, xF - 0.2, xF + 0.2, 0.6, 0.9)
        step = x < xF ? 1.0 : 0.0
        @test _cheb_eval(c_step, x) ≈ step atol = 1e-3
    end

    # finite beta matches the exact Fermi factor (smooth: quadrature is exact
    # to machine precision at moderate NC)
    beta = 20.0
    c_T = KPM.fermi_coefficients(a, b, Ef; beta = beta, NC = NC)
    for x in (-0.9, -0.3, xF, 0.5, 0.9)
        f = 1 / (exp((a * x + b - Ef) * beta) + 1)
        @test _cheb_eval(c_T, x) ≈ f atol = 1e-3
    end

    # Ef outside the rescaled spectral window throws
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, b + 1.5 * a; NC = NC)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, b - a; NC = NC)
    # argument validation
    @test_throws ArgumentError KPM.fermi_coefficients(-a, b, Ef; NC = NC)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = 0)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = NC, beta = 0.0)
    @test_throws ArgumentError KPM.fermi_coefficients(a, b, Ef; NC = NC, Np = 0)
    # NC is deliberately required
    @test_throws UndefKeywordError KPM.fermi_coefficients(a, b, Ef)
end
