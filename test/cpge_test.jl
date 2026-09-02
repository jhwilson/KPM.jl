using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Third-order (CPGE) moments and reconstruction.
#
# Two independent things are pinned here:
#
#  1. `kpm_3d!`/`kpm_3d` moments against dense Chebyshev matrices T_k(H_norm)
#     built by the three-term recurrence. The trace is evaluated exactly with
#     an identity probe matrix, so the comparison is a machine-precision
#     identity, not a stochastic estimate.
#
#  2. The `cpge` reconstruction — kernel/hₙ damping, the Λnmp contraction and
#     the overall prefactor — given *exact* moments. UNITS: the `cpge`
#     docstring quotes e³/(Ω ħ³) but states explicitly that, unlike
#     `kubo_bastin_cond`, this normalization has **not** been validated
#     against exact diagonalization. There is therefore no independent
#     absolute reference to test against; what is tested below is that the
#     reconstruction assembles the documented Eq.-43/45 expression from the
#     moments. The quadrature rule `cpge` uses internally is reproduced here
#     rather than replaced (see the comment on `_gauss_chebyshev` below).
#
# Model: a chiral 6-site helix. Positions are user data (CLAUDE.md: "models
# are user data") — the current operators are built from them by the package
# convention (J_α)_ij = H_ij (r_i - r_j)_α with the *unrescaled* H, and are
# never inferred from the matrix.

# --- helix cluster ----------------------------------------------------------

function helix_cluster(; t = 1.0, t2 = 0.45, φ = 0.7, w = 0.3)
    N = 6
    pos = [
        [cos(2π * (j - 1) / 3), sin(2π * (j - 1) / 3), 0.4 * (j - 1)] for j = 1:N
    ]
    Hd = zeros(ComplexF64, N, N)
    for j = 1:(N-1)
        Hd[j+1, j] += -t
        Hd[j, j+1] += -t
    end
    # chiral second-neighbour hopping along the helix: breaks inversion and
    # time reversal, so all three-current responses are generically nonzero
    for j = 1:(N-2)
        Hd[j+2, j] += t2 * cis(φ)
        Hd[j, j+2] += t2 * cis(-φ)
    end
    for j = 1:N
        Hd[j, j] = w * (-1)^j
    end
    @assert Hd ≈ Hd'
    J = [
        sparse([Hd[i, j] * (pos[i][α] - pos[j][α]) for i = 1:N, j = 1:N]) for α = 1:3
    ]
    return sparse(Hd), J, pos
end

# Dense Chebyshev matrices T_k(A), k = 0 … NC-1 (index k+1).
function chebyshev_matrices(A, NC)
    D = size(A, 1)
    T = Vector{Matrix{ComplexF64}}(undef, NC)
    T[1] = Matrix{ComplexF64}(I, D, D)
    NC >= 2 && (T[2] = Matrix{ComplexF64}(A))
    for k = 3:NC
        T[k] = 2 * Matrix(A) * T[k-1] - T[k-2]
    end
    return T
end

# Chebyshev polynomials of the first (T) and second (U) kind by their
# recurrences — used as an acos-free reference for the Eq.-35/36 kernels.
function cheb_TU(z, n)
    T0, T1 = one(z), z
    U0, U1 = one(z), 2z
    n == 0 && return (one(z), zero(z))          # (T_0, U_{-1})
    n == 1 && return (z, one(z))                # (T_1, U_0)
    Tm, Tc = T0, T1
    Um, Uc = U0, U1
    for _ = 2:n
        Tm, Tc = Tc, 2z * Tc - Tm
        Um, Uc = Uc, 2z * Uc - Um
    end
    return (Tc, Um)                              # (T_n, U_{n-1})
end

# The Gauss–Chebyshev rule `cpge`/`optical_cond*` build internally.
_gauss_chebyshev(N) = KPM.gausschebyshevt(N)

# Centered rescaling from the dense spectrum. `normalizeH(...; center=true)`
# goes through Arpack, whose default subspace size exceeds this 6×6 matrix;
# the arithmetic reproduced here is exactly normalizeH's (eps = 0.1 margin).
function dense_rescale(H; eps = 0.1)
    λ = eigvals(Hermitian(Matrix(H)))
    b = (maximum(λ) + minimum(λ)) / 2
    a = (maximum(λ) - minimum(λ)) / (2 - eps)
    return a, b, sparse((Matrix(H) - b * I) / a)
end

H, J, pos = helix_cluster()
D = size(H, 1)
a, b, H_norm = dense_rescale(H)
NC = 8
psi = Matrix{ComplexF64}(I, D, D)      # exact trace, no stochastic error
Tmat = chebyshev_matrices(H_norm, NC)

@testset "kpm_3d moments vs dense Chebyshev matrices" begin
    @test maximum(abs, eigvals(Hermitian(Matrix(H_norm)))) < 1

    Γ = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    @test size(Γ) == (NC, NC, NC)

    # IMPLEMENTED index convention (verified here, differs from the `kpm_3d!`
    # docstring — see the report accompanying this test file):
    #
    #   Γ[n, m, p] = ⟨ψ| T_m(H̃) Jγ T_n(H̃) Jβ T_p(H̃) Jα |ψ⟩ / D
    #
    # i.e. the *second* index carries the leftmost Chebyshev factor, exactly
    # as in `kpm_2d` where mu2D[n, m] = Tr[Jα T_m Jβ T_n]/D (CLAUDE.md,
    # "note the transposed-looking order"). The docstring of `kpm_3d!` names
    # the indices (n3, n2, n1) with n3 leftmost, which is the transpose of
    # this in the first two slots.
    Γ_ref = [
        tr(Tmat[m] * J[3] * Tmat[n] * J[2] * Tmat[p] * J[1]) / D for n = 1:NC, m = 1:NC,
        p = 1:NC
    ]
    # machine precision: identical arithmetic up to summation order
    @test Γ ≈ Γ_ref rtol = 1e-12

    # the transposed convention is *not* what is computed — guards against a
    # silent index swap being introduced to match the docstring
    Γ_swapped = permutedims(Γ_ref, (2, 1, 3))
    @test !isapprox(Γ, Γ_swapped; rtol = 1e-3)

    # in-place entry point writes the same thing
    Γ_ip = zeros(ComplexF64, NC, NC, NC)
    KPM.kpm_3d!(H_norm, J[1], J[2], J[3], NC, D, D, Γ_ip, psi, psi)
    @test Γ_ip ≈ Γ rtol = 1e-12

    # stochastic path: random-phase probes reproduce the exact trace with the
    # documented O(1/sqrt(NR·D)) accuracy. Measured max|ΔΓ|/max|Γ| at this
    # seed: 0.100 (NR=256), 0.049 (NR=1024), 0.028 (NR=4096) — the 1/sqrt(NR)
    # scaling. 0.15 is 3× the NR=1024 value.
    rng = Xoshiro(20260901)
    psi_r = KPM.random_phase_vectors(rng, D, 1024)
    Γ_r = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, 1024, D; psi_in = psi_r)
    @test maximum(abs, Γ_r .- Γ_ref) < 0.15 * maximum(abs, Γ_ref)
end

@testset "Eq. 35/36 kernels: Δn, gn_R, gn_A vs Chebyshev polynomials" begin
    # Δn(ϵ) = 2 T_n(ϵ) / (π √(1-ϵ²));
    # g_n^{R}(ϵ) = -2i e^{-i n arccos ϵ}/√(1-ϵ²) = -2i T_n/√(1-ϵ²) - 2 U_{n-1}
    # g_n^{A}(ϵ) = +2i e^{+i n arccos ϵ}/√(1-ϵ²) = +2i T_n/√(1-ϵ²) - 2 U_{n-1}
    # (e^{∓inθ} = T_n ∓ i sinθ U_{n-1}). These are acos-free identities, so a
    # sign or branch error in the source cannot be reproduced by the reference.
    # atol covers the cases where the exact value is 0 (T_n(0) = 0 for odd n);
    # all quantities here are O(1), so 1e-14 is roundoff-level
    for ϵ in (-0.87, -0.3, 0.0, 0.11, 0.62), n in (0, 1, 2, 5, 9)
        Tn, Unm1 = cheb_TU(ϵ, n)
        s = sqrt(1 - ϵ^2)
        @test KPM.Δn(ϵ, n) ≈ 2 * Tn / (π * s) rtol = 1e-12 atol = 1e-14
        @test KPM.gn_R(ϵ, n) ≈ -2im * Tn / s - 2 * Unm1 rtol = 1e-12 atol = 1e-14
        @test KPM.gn_A(ϵ, n) ≈ 2im * Tn / s - 2 * Unm1 rtol = 1e-12 atol = 1e-14
        # spectral identity g^R - g^A = -2πi Δn
        @test KPM.gn_R(ϵ, n) - KPM.gn_A(ϵ, n) ≈ -2im * π * KPM.Δn(ϵ, n) rtol = 1e-12 atol =
            1e-14
    end

    # hard cutoff |ϵ| < 1 - δ is applied by all three (δ = 1e-5 by default)
    for f in (KPM.Δn, KPM.gn_R, KPM.gn_A)
        @test f(1 - 1e-6, 3) == 0
        @test f(-(1 - 1e-6), 3) == 0
    end
    # gn_R/gn_A take a complex argument internally, so they also survive
    # |ϵ| > 1 (where Δn's real acos would not be defined)
    @test KPM.gn_R(1.5, 3) == 0
    @test KPM.gn_A(-1.5, 3) == 0
end

@testset "Λnmp assembles Eq. 43 from the Eq. 35/36 kernels" begin
    # Fix the quadrature rule (a deterministic Gauss–Legendre-free node set on
    # a sub-interval where the integrand is smooth) so that *only* the
    # integrand assembly — which propagator carries which index, which
    # frequency offset, and the Fermi weight — is under test.
    xs = collect(range(-0.8, 0.8; length = 41))
    ws = fill(1.6 / 41, 41)
    myquad(f) = (sum(ws .* f.(xs)), nothing)

    ω₁, ω₂ = 0.31, -0.29
    E_f, beta = 0.15, 12.0
    ff = KPM.fermiFunctions(E_f, beta)
    for nmp in ([0, 0, 0], [1, 3, 2], [4, 0, 5], [2, 2, 7])
        n, m, p = nmp
        integrand(ϵ) =
            (
                KPM.gn_R(ϵ + ω₁ + ω₂, n) * KPM.gn_R(ϵ + ω₂, m) * KPM.Δn(ϵ, p) +
                KPM.gn_R(ϵ + ω₁, n) * KPM.Δn(ϵ, m) * KPM.gn_A(ϵ - ω₂, p) +
                KPM.Δn(ϵ, n) * KPM.gn_A(ϵ - ω₁, m) * KPM.gn_A(ϵ - ω₁ - ω₂, p)
            ) * ff(ϵ)
        ref = sum(ws .* integrand.(xs))
        @test KPM.Λnmp(nmp, ω₁, ω₂; E_f = E_f, beta = beta, quad = myquad) ≈ ref rtol =
            1e-12
    end
end

@testset "cpge reconstruction from exact moments" begin
    Γ = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    ω = 0.3
    Ω = ω / 20
    E_f = 0.0
    beta = Inf
    δ = 1e-5

    val = KPM.cpge(Γ, NC, ω; beta = beta, E_f = E_f, δ = δ, Ω = Ω)

    # Independent reassembly of Eq. 45. NOTE on the quadrature: `cpge` builds
    # `quad(f) = dot(weights, f.(nodes))` from an NC*8-node Gauss–Chebyshev
    # rule, which evaluates ∫ f(ϵ)/√(1-ϵ²) dϵ, whereas the documented default
    # `quad` of `Λnmp` is `quadgk(f, -1+δ, 1-δ)` = ∫ f(ϵ) dϵ. The two differ by
    # the Chebyshev weight and are not interchangeable (see the report). The
    # rule is therefore reproduced verbatim here: this test pins the
    # contraction, kernel/hₙ damping and prefactor, not the quadrature.
    nodes, weights = _gauss_chebyshev(NC * 8)
    ff = KPM.fermiFunctions(Float64(E_f), Float64(beta))
    ω₁ = ω
    ω₂ = Ω - ω
    g = [KPM.JacksonKernel(n, NC) * KPM.hn(n) for n = 0:(NC-1)]

    Λ = zeros(ComplexF64, NC, NC, NC)
    for p = 0:(NC-1), m = 0:(NC-1), n = 0:(NC-1)
        integrand(ϵ) =
            (
                KPM.gn_R(ϵ + ω₁ + ω₂, n, 0.0, δ) * KPM.gn_R(ϵ + ω₂, m, 0.0, δ) *
                KPM.Δn(ϵ, p, δ) +
                KPM.gn_R(ϵ + ω₁, n, 0.0, δ) * KPM.Δn(ϵ, m, δ) *
                KPM.gn_A(ϵ - ω₂, p, 0.0, δ) +
                KPM.Δn(ϵ, n, δ) * KPM.gn_A(ϵ - ω₁, m, 0.0, δ) *
                KPM.gn_A(ϵ - ω₁ - ω₂, p, 0.0, δ)
            ) * ff(ϵ)
        Λ[n+1, m+1, p+1] = sum(weights .* integrand.(nodes))
    end

    ref = zero(ComplexF64)
    for p = 1:NC, m = 1:NC, n = 1:NC
        ref += g[n] * g[m] * g[p] * Γ[n, m, p] * Λ[n, m, p]
    end
    ref *= 1im / (ω₁ * ω₂) * Ω

    @test val ≈ ref rtol = 1e-12
    @test isfinite(val)
    @test abs(val) > 0            # chiral model: response is not accidentally zero

    # Ω must stay small compared with ω (documented precondition)
    @test_throws AssertionError KPM.cpge(Γ, NC, ω; Ω = 0.5 * ω)
end

@testset "d_cpge integrand integrates to cpge on the same nodes" begin
    # `cpge` = i Ω/(ω₁ω₂) ∫ dϵ f(ϵ) Σ_nmp g̃ Γ Λ-integrand(ϵ), and `d_cpge`
    # evaluates exactly that ϵ-resolved summand through an independent
    # (broadcast) code path. Contracting it on the same quadrature nodes must
    # reproduce `cpge` to roundoff — this cross-validates the two
    # implementations against each other.
    Γ = KPM.kpm_3d(H_norm, J[1], J[2], J[3], NC, D, D; psi_in = psi)
    ω = 0.3
    Ω = ω / 20
    ω₁, ω₂ = ω, Ω - ω
    nodes, weights = _gauss_chebyshev(NC * 8)
    ff = KPM.fermiFunctions(0.0, Inf)

    acc = zero(ComplexF64)
    for (k, x) in enumerate(nodes)
        acc += weights[k] * ff(x) * KPM.d_cpge(Γ, NC, ω₁, ω₂, Float64(x))
    end
    acc *= 1im / (ω₁ * ω₂) * Ω
    @test acc ≈ KPM.cpge(Γ, NC, ω; Ω = Ω) rtol = 1e-12

    # grid entry point returns the same values as the scalar one
    ϵ_grid, res = KPM.d_cpge(Γ, NC, ω₁, ω₂; N_int = 8, e_range = [-0.9, 0.9])
    @test length(res) == 8
    for (k, ϵ) in enumerate(ϵ_grid)
        @test res[k] ≈ KPM.d_cpge(Γ, NC, ω₁, ω₂, Float64(ϵ)) rtol = 1e-12
    end
end
