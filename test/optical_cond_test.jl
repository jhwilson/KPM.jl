using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

# Optical conductivity (Eq. 44) and the 1D current moments that feed it.
#
# UNITS: the `optical_cond1`/`optical_cond2` docstrings quote -ie²/(ħ²ω) and
# state explicitly that, unlike `kubo_bastin_cond`, this normalization has
# **not** been validated against exact diagonalization. There is therefore no
# absolute ED reference to compare against, and a dense spectral-sum reference
# would additionally require matching the ϵ-integration rule — which cannot be
# justified here (see the note in "optical_cond1/2 reconstruction" below).
# What is tested is:
#
#  * `kpm_1d_current` moments against dense Chebyshev matrices (exact trace);
#  * the Eq.-41/42 energy integrals Λn, Λnm assembled from Δn/gn_R/gn_A;
#  * the Eq.-44 reconstruction — kernel/hₙ damping, contraction, prefactor —
#    given exact moments;
#  * agreement between the `optical_cond*` route and the independent
#    `d_optical_cond*` (broadcast) route on the same quadrature nodes.

# Chebyshev polynomials of the first (T) and second (U) kind by recurrence.
function _cheb_TU(z, n)
    n == 0 && return (one(z), zero(z))          # (T_0, U_{-1})
    n == 1 && return (z, one(z))                # (T_1, U_0)
    Tm, Tc = one(z), z
    Um, Uc = one(z), 2z
    for _ = 2:n
        Tm, Tc = Tc, 2z * Tc - Tm
        Um, Uc = Uc, 2z * Uc - Um
    end
    return (Tc, Um)
end

function _chebyshev_matrices(A, NC)
    D = size(A, 1)
    T = Vector{Matrix{ComplexF64}}(undef, NC)
    T[1] = Matrix{ComplexF64}(I, D, D)
    NC >= 2 && (T[2] = Matrix{ComplexF64}(A))
    for k = 3:NC
        T[k] = 2 * Matrix(A) * T[k-1] - T[k-2]
    end
    return T
end

# --- fixture: small Haldane torus ------------------------------------------
Hop, Jx, Jy, area = haldane_model(3, 3; t = 1.0, t2 = 0.2, ϕ = π/2, m = 0.0)
D = size(Hop, 1)               # 18 sites

# Centered rescaling from the dense spectrum: `normalizeH(...; center=true)`
# goes through Arpack, whose default subspace size exceeds this 18×18 matrix.
# The arithmetic is exactly normalizeH's with the default eps = 0.1 margin.
let λ = eigvals(Hermitian(Matrix(Hop)))
    global b_r = (maximum(λ) + minimum(λ)) / 2
    global a_r = (maximum(λ) - minimum(λ)) / (2 - 0.1)
end
H_norm = sparse((Matrix(Hop) - b_r * I) / a_r)

# Bond displacements are user data. `haldane_model` hands back the bond
# currents (J_α)_ij = H_ij (r_i - r_j)_α; the displacement of each bond is
# recovered from them so that the *diamagnetic* operator
# (J_αα)_ij = H_ij (r_i - r_j)_α² can be built by the same rule. J_αα is
# Hermitian (d_ij² is even under i↔j), unlike the anti-Hermitian J_α.
Hm = Matrix(Hop)
dx = [abs(Hm[i, j]) > 0 ? Matrix(Jx)[i, j] / Hm[i, j] : 0.0 + 0im for i = 1:D, j = 1:D]
Jxx = sparse(Hm .* dx .^ 2)

NC = 16
psi = Matrix{ComplexF64}(I, D, D)     # exact trace, no stochastic error
Tmat = _chebyshev_matrices(H_norm, NC)

@testset "kpm_1d_current moments vs dense Chebyshev matrices" begin
    @test norm(Jxx - Jxx') == 0                # diamagnetic operator: Hermitian
    @test norm(Jx + Jx') == 0                  # bond current: anti-Hermitian
    @test maximum(abs, eigvals(Hermitian(Matrix(H_norm)))) < 1

    mu = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi)
    ref = [real(tr(Jxx * Tmat[n]) / D) for n = 1:NC]
    @test mu ≈ ref rtol = 1e-12

    # NR_parallel = false takes a different loop; same numbers
    mu_ser = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi, NR_parallel = false)
    @test mu_ser ≈ mu rtol = 1e-12

    # per-probe output sums back to the average
    mu_all = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi, avg_output = false)
    @test size(mu_all) == (D, NC)
    @test vec(real.(sum(mu_all, dims = 1) ./ D)) ≈ mu rtol = 1e-12

    # force_norm on unnormalized columns == pre-normalizing them
    rng = Xoshiro(4242)
    NRs = 4
    psi_u =
        KPM.random_phase_vectors(rng, D, NRs) .* reshape([1.0, 3.0, 0.25, 7.0], 1, NRs)
    p_forced = copy(psi_u)
    mu_forced =
        KPM.kpm_1d_current(H_norm, Jxx, NC, NRs, D; psi_in = p_forced, force_norm = true)
    p_pre = copy(psi_u)
    KPM.normalize_by_col(p_pre, NRs)
    mu_pre = KPM.kpm_1d_current(H_norm, Jxx, NC, NRs, D; psi_in = p_pre)
    @test mu_forced ≈ mu_pre rtol = 1e-12
end

@testset "kpm_1d_current: implemented operator convention (Jα†, real part)" begin
    # PINNED BEHAVIOUR, NOT A DERIVED RESULT. The `kpm_1d_current` docstring
    # says "Γ_n^α = Tr[J_α T_n(H)]", but the moment is formed as
    # dot(J_α ψ, T_n ψ) — `LinearAlgebra.dot` conjugates its first argument —
    # so the implemented quantity is ⟨ψ|J_α† T_n(H)|ψ⟩, and `avg_output=true`
    # additionally returns only its real part. For the package's own bond
    # current J_α = iħv_α (anti-Hermitian) Tr[J_α T_n] is purely imaginary, so
    # the default output is identically zero. This is checked below on a flux
    # ring, whose persistent current makes Tr[J_x T_n] demonstrably nonzero;
    # if this test starts failing, the source convention changed — read this
    # comment before "fixing" the test.
    N = 12
    hring, ringpos, ringdisp = flux_ring_model(N; t = 1.0, phi = 0.35)
    Jring = sparse([hring[i, j] * ringdisp(i, j)[1] for i = 1:N, j = 1:N])
    @test norm(Jring + Jring') == 0

    ar = 2 * maximum(abs, eigvals(Hermitian(Matrix(hring)))) / (2 - 0.1)
    hn_ring = sparse(Matrix(hring) / ar)
    NCr = 12
    Tr_ = _chebyshev_matrices(hn_ring, NCr)
    psir = Matrix{ComplexF64}(I, N, N)

    tr_J = [tr(Jring * Tr_[n]) / N for n = 1:NCr]
    @test maximum(abs, tr_J) > 0.5              # the ring really does carry current
    @test maximum(abs, real.(tr_J)) < 1e-14     # ...purely imaginary

    mu_default = KPM.kpm_1d_current(hn_ring, Jring, NCr, N, N; psi_in = psir)
    @test maximum(abs, mu_default) < 1e-14      # entire signal discarded by `real`

    mu_all = KPM.kpm_1d_current(hn_ring, Jring, NCr, N, N; psi_in = psir,
        avg_output = false)
    mu_cplx = vec(sum(mu_all, dims = 1) ./ N)
    @test mu_cplx ≈ [tr(Jring' * Tr_[n]) / N for n = 1:NCr] rtol = 1e-12 atol = 1e-14
    @test mu_cplx ≈ -tr_J rtol = 1e-12 atol = 1e-14   # = Tr[J† T_n] = -Tr[J T_n]
    # review: kpm_1d_current contracts J† instead of the documented J.
    @test_broken mu_cplx ≈ tr_J rtol = 1e-12 atol = 1e-14
end

@testset "kpm_1d_current: independent bra/ket is documented as unimplemented" begin
    # src/moment.jl throws a bare String rather than an Exception subtype
    psil = KPM.random_phase_vectors(Xoshiro(1), D, 1)
    psir = KPM.random_phase_vectors(Xoshiro(2), D, 1)
    @test_throws String KPM.kpm_1d_current(
        H_norm,
        Jxx,
        NC,
        1,
        D;
        psi_in_l = psil,
        psi_in_r = psir,
    )
    @test_throws String KPM.kpm_1d_current!(
        H_norm,
        Jxx,
        NC,
        1,
        D,
        zeros(ComplexF64, 1, NC),
        psil,
        psir,
    )
end

@testset "Λn (Eq. 41) closed form" begin
    # With the documented default quadrature (quadgk over [-1+δ, 1-δ]),
    #   Λ_n = ∫ dϵ Δ_n(ϵ) f(ϵ),  Δ_n = 2 T_n(ϵ)/(π√(1-ϵ²)).
    # Substituting ϵ = cos θ turns this into (2/π)∫ cos(nθ) dθ over
    # θ ∈ (θ_hi, π - θ_lo), which integrates in closed form. At T = 0 the
    # Fermi function cuts the upper limit at ϵ = E_f.
    δ = 1e-5
    θ_lo = acos(1 - δ)                      # ϵ = 1 - δ  (upper cutoff)
    closed(n, E_f) = begin
        θ1 = acos(min(E_f, 1 - δ))          # occupied states are ϵ < E_f, i.e. θ > θ1
        θ2 = π - θ_lo                       # ϵ = -(1 - δ)
        n == 0 ? (2 / π) * (θ2 - θ1) : (2 / (n * π)) * (sin(n * θ2) - sin(n * θ1))
    end
    for n = 0:6
        # completely filled band: the integrand is smooth, so quadgk is exact
        # to roundoff. Λ_n = 2δ_{n0} up to the δ-cutoff deficit.
        @test KPM.Λn([n]; E_f = 1.0, beta = Inf) ≈ closed(n, 1.0) rtol = 1e-10 atol = 1e-12

        # partially filled: at T = 0 the Fermi factor is a jump discontinuity
        # at ϵ = E_f inside the integration interval, which the *default*
        # adaptive quadgk (rtol ≈ √eps ≈ 1.5e-8) cannot resolve better than
        # ~1e-7 absolute — that is the meaning of the loose atol here.
        @test KPM.Λn([n]; E_f = 0.3, beta = Inf) ≈ closed(n, 0.3) rtol = 1e-6 atol = 1e-6

        # ...and with the discontinuity handed to quadgk as a break point the
        # same call reproduces the closed form to quadrature roundoff
        sharp(f) = KPM.quadgk(f, -1 + δ, 0.3, 1 - δ; rtol = 1e-13, atol = 1e-15)
        @test KPM.Λn([n]; E_f = 0.3, beta = Inf, quad = sharp) ≈ closed(n, 0.3) rtol =
            1e-11 atol = 1e-13
    end
    # numerical value of the sum rule, spelled out: the n = 0 moment of a
    # filled band is 2 - (4/π)·arccos(1-δ) = 1.9943058949…
    @test KPM.Λn([0]; E_f = 1.0, beta = Inf) ≈ 1.994305894907872 rtol = 1e-12
end

@testset "Λnm (Eq. 42) assembles the retarded/advanced pair" begin
    # A fixed node set on a sub-interval where the integrand is smooth: only
    # the integrand assembly (which propagator carries which index, which
    # frequency offset, the Fermi weight) is under test, not the quadrature.
    xs = collect(range(-0.75, 0.75; length = 37))
    ws = fill(1.5 / 37, 37)
    myquad(f) = (sum(ws .* f.(xs)), nothing)
    ω = 0.23
    E_f, beta = -0.1, 9.0
    ff = KPM.fermiFunctions(E_f, beta)
    for nm in ([0, 0], [3, 1], [5, 8], [2, 2])
        n, m = nm
        integrand(ϵ) =
            (KPM.gn_R(ϵ + ω, n) * KPM.Δn(ϵ, m) + KPM.Δn(ϵ, n) * KPM.gn_A(ϵ - ω, m)) *
            ff(ϵ)
        ref = sum(ws .* integrand.(xs))
        @test KPM.Λnm(nm, ω; E_f = E_f, beta = beta, quad = myquad) ≈ ref rtol = 1e-12
    end
end

@testset "optical_cond1/2 reconstruction from exact moments" begin
    # NOTE on the quadrature: `optical_cond1`/`optical_cond2` build
    # `quad(f) = dot(weights, f.(nodes))` from an NC*8-node Gauss–Chebyshev
    # rule, which evaluates ∫ f(ϵ)/√(1-ϵ²) dϵ, whereas the documented default
    # `quad` of `Λn`/`Λnm` is `quadgk(f, -1+δ, 1-δ)` = ∫ f(ϵ) dϵ. Since Δn and
    # gn already carry a 1/√(1-ϵ²), the two rules are not interchangeable and
    # the Gauss–Chebyshev value of Λ_0 is not even node-count independent
    # (6.34 / 7.22 / 8.10 at N = 32 / 64 / 128, against the exact 1.9943).
    # See the report. The rule is reproduced verbatim here so that this test
    # pins the contraction, damping and prefactor and nothing else.
    ω = 0.35
    E_f, beta, δ = 0.0, Inf, 1e-5
    nodes, weights = KPM.gausschebyshevt(NC * 8)
    ff = KPM.fermiFunctions(Float64(E_f), Float64(beta))
    g = [KPM.JacksonKernel(n, NC) * KPM.hn(n) for n = 0:(NC-1)]

    # --- Λn contribution (1D diamagnetic moments)
    Γ1 = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi)
    Λ1 = [sum(weights .* (KPM.Δn.(nodes, n, δ) .* ff.(nodes))) for n = 0:(NC-1)]
    ref1 = -1im * sum(g .* Γ1 .* Λ1) / ω
    @test KPM.optical_cond1(Γ1, NC, ω; beta = beta, E_f = E_f, δ = δ) ≈ ref1 rtol = 1e-12
    corrected_weights = weights .* sqrt.(1 .- nodes .^2)
    Λ1_corrected = [
        sum(corrected_weights .* (KPM.Δn.(nodes, n, δ) .* ff.(nodes))) for n = 0:(NC-1)
    ]
    corrected_ref1 = -1im * sum(g .* Γ1 .* Λ1_corrected) / ω
    # review: quadrature double-counts 1/√(1−ε²); flip to @test when src is fixed
    @test_broken KPM.optical_cond1(Γ1, NC, ω; beta = beta, E_f = E_f, δ = δ) ≈
                 corrected_ref1 rtol = 1e-12
    @test abs(ref1) > 0

    # --- Λnm contribution (2D current-current moments)
    Γ2 = KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in = psi)
    Λ2 = zeros(ComplexF64, NC, NC)
    for m = 0:(NC-1), n = 0:(NC-1)
        integrand(ϵ) =
            (
                KPM.gn_R(ϵ + ω, n, 0.0, δ) * KPM.Δn(ϵ, m, δ) +
                KPM.Δn(ϵ, n, δ) * KPM.gn_A(ϵ - ω, m, 0.0, δ)
            ) * ff(ϵ)
        Λ2[n+1, m+1] = sum(weights .* integrand.(nodes))
    end
    ref2 = -1im * sum((g * g') .* Γ2 .* Λ2) / ω
    @test KPM.optical_cond2(Γ2, NC, ω; beta = beta, E_f = E_f, δ = δ) ≈ ref2 rtol = 1e-12
    Λ2_corrected = zeros(ComplexF64, NC, NC)
    for m = 0:(NC-1), n = 0:(NC-1)
        integrand(ϵ) =
            (
                KPM.gn_R(ϵ + ω, n, 0.0, δ) * KPM.Δn(ϵ, m, δ) +
                KPM.Δn(ϵ, n, δ) * KPM.gn_A(ϵ - ω, m, 0.0, δ)
            ) * ff(ϵ)
        Λ2_corrected[n+1, m+1] = sum(corrected_weights .* integrand.(nodes))
    end
    corrected_ref2 = -1im * sum((g * g') .* Γ2 .* Λ2_corrected) / ω
    # review: quadrature double-counts 1/√(1−ε²); flip to @test when src is fixed
    @test_broken KPM.optical_cond2(Γ2, NC, ω; beta = beta, E_f = E_f, δ = δ) ≈
                 corrected_ref2 rtol = 1e-12
    @test abs(ref2) > 0

    # the 2D moments themselves are exact (identity probes)
    Γ2_ref = [tr(Jx * Tmat[m] * Jy * Tmat[n]) / D for n = 1:NC, m = 1:NC]
    @test Γ2 ≈ Γ2_ref rtol = 1e-12
    Γ2_paper = [tr(Jx * Tmat[n] * Jy * Tmat[m]) / D for n = 1:NC, m = 1:NC]
    paper_ref2 = -1im * sum((g * g') .* Γ2_paper .* Λ2) / ω
    # review: optical_cond2 contracts the transposed table
    @test_broken KPM.optical_cond2(Γ2, NC, ω; beta = beta, E_f = E_f, δ = δ) ≈
                 paper_ref2 rtol = 1e-12
end

@testset "d_optical_cond1/2 integrate to optical_cond1/2 on the same nodes" begin
    # optical_cond* = -i/ω ∫ dϵ f(ϵ) [ϵ-resolved summand], with the summand
    # computed by the independent broadcast path `d_optical_cond*`. Evaluating
    # it on the same quadrature nodes must reproduce optical_cond* to
    # roundoff — a cross-check of two separate implementations.
    ω = 0.35
    nodes, weights = KPM.gausschebyshevt(NC * 8)
    ff = KPM.fermiFunctions(0.0, Inf)

    Γ1 = KPM.kpm_1d_current(H_norm, Jxx, NC, D, D; psi_in = psi)
    acc1 = sum(
        weights[k] * ff(nodes[k]) * KPM.d_optical_cond1(Γ1, NC, Float64(nodes[k])) for
        k in eachindex(nodes)
    )
    @test -1im * acc1 / ω ≈ KPM.optical_cond1(Γ1, NC, ω) rtol = 1e-12

    Γ2 = KPM.kpm_2d(H_norm, Jx, Jx, NC, D, D; psi_in = psi)
    acc2 = sum(
        weights[k] * ff(nodes[k]) * KPM.d_optical_cond2(Γ2, NC, ω, Float64(nodes[k]))
        for k in eachindex(nodes)
    )
    @test -1im * acc2 / ω ≈ KPM.optical_cond2(Γ2, NC, ω) rtol = 1e-12

    # grid entry points agree with the scalar ones
    ϵ_grid1, r1 = KPM.d_optical_cond1(Γ1, NC; N_int = 6, e_range = [-0.9, 0.9])
    for (k, ϵ) in enumerate(ϵ_grid1)
        @test r1[k] ≈ KPM.d_optical_cond1(Γ1, NC, Float64(ϵ)) rtol = 1e-12
    end
    ϵ_grid2, r2 = KPM.d_optical_cond2(Γ2, NC, ω; N_int = 6, e_range = [-0.9, 0.9])
    for (k, ϵ) in enumerate(ϵ_grid2)
        @test r2[k] ≈ KPM.d_optical_cond2(Γ2, NC, ω, Float64(ϵ)) rtol = 1e-12
    end
end
