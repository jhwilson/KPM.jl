using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# End-to-end run of the main pipeline on a model built here, deterministically:
# rescale → 1D moments → DOS, and → 3D moments → CPGE. This file used to load
# an undocumented JLD2 fixture (test/test_data/ops_small.jld2) whose provenance
# was unknown; the model below replaces it with something whose spectrum is
# known analytically, so the assertions can be sharp instead of smoke-level.
#
# Model: dimerized (SSH) ring, alternating hoppings t1, t2. Bands
# ±sqrt(t1² + t2² + 2 t1 t2 cos k), i.e. a gap 2|t1 - t2| at E = 0 and a
# bandwidth 2(t1 + t2). Positions alternate along the chain, so the bond
# currents (J_α)_ij = H_ij (r_i - r_j)_α follow the package convention from
# user-supplied geometry.

t1 = 1.0
t2 = 0.4
Ncell = 128
NH = 2 * Ncell            # 256 sites
gap = 2 * abs(t1 - t2)    # 1.2
bandwidth = 2 * (t1 + t2) # 2.8

rows, cols, vals, dvals = Int[], Int[], ComplexF64[], ComplexF64[]
# intracell bond A(n)-B(n) with displacement 1/2, intercell B(n)-A(n+1) with 1/2
pos = zeros(NH)
for n = 1:Ncell
    A = 2n - 1
    B = 2n
    pos[A] = (n - 1) * 1.0
    pos[B] = (n - 1) * 1.0 + 0.5
end
function addbond!(i, j, amp, d)
    push!(rows, i); push!(cols, j); push!(vals, amp); push!(dvals, amp * d)
    push!(rows, j); push!(cols, i); push!(vals, conj(amp)); push!(dvals, -conj(amp) * d)
end
for n = 1:Ncell
    addbond!(2n, 2n - 1, -t1 + 0im, 0.5)                       # A(n) -> B(n)
    addbond!(mod1(2n + 1, NH), 2n, -t2 + 0im, 0.5)             # B(n) -> A(n+1)
end
H = sparse(rows, cols, vals, NH, NH)
Jx = sparse(rows, cols, dvals, NH, NH)

@testset "rescaling and 1D moments are exact against dense Chebyshev" begin
    @test ishermitian(H)
    λ = eigvals(Hermitian(Matrix(H)))
    @test maximum(λ) ≈ bandwidth / 2 rtol = 1e-10
    # k = π is on the 128-cell grid, gap edge exact
    @test minimum(abs, λ) ≈ gap / 2 rtol = 1e-10

    a, H_norm = KPM.normalizeH(H)
    # normalizeH's Arpack estimate is only tol=1e-3 (src/utils/external.jl)
    @test a ≈ 2 * (bandwidth / 2) / (2 - 0.1) rtol = 2e-3

    NC = 64
    psi = Matrix{ComplexF64}(I, NH, NH)              # exact trace
    mu = KPM.kpm_1d(H_norm, NC, NH; psi_in = psi)

    # μ_n = Tr[T_n(H_norm)]/D from the dense spectrum
    λn = eigvals(Hermitian(Matrix(H_norm)))
    mu_ed = [sum(cos.(n .* acos.(clamp.(λn, -1, 1)))) / NH for n = 0:(NC-1)]
    @test mu ≈ mu_ed rtol = 1e-10
    @test mu[1] ≈ 1.0 rtol = 1e-12
end

@testset "DOS vanishes in the gap and reproduces the ED counting function" begin
    a, H_norm = KPM.normalizeH(H)
    NC = 2048
    NR = 21
    rng = Xoshiro(2718)
    psi_in = KPM.random_phase_vectors(rng, NH, NR)
    mu = KPM.kpm_1d(H_norm, NC, NR; psi_in = psi_in)

    # Jackson resolution at the band centre is ~ π a / NC ≈ 2.3e-3, and the
    # nearest state sits gap/2 = 0.6 away — 260 resolution widths, where the
    # Jackson tail is far below the stochastic noise floor 1/sqrt(NR·NH) ~ 1e-2
    # times the local DOS (which is 0 here). Measured |ρ(0)| ~ 1e-7.
    rho_0 = KPM.dos0(mu, a)
    @test abs(rho_0) < 1e-4
    @test isfinite(KPM.dos0(mu, a; dE_order = 2))

    # counting function N(E) = ∫_{-∞}^{E} ρ: compare with the exact spectrum.
    # The KPM curve is the ED one convolved with the Jackson kernel, so they
    # agree to O(kernel width × ρ) plus the stochastic error; 0.02 (2% of the
    # states) is comfortable at NR = 21 on 256 sites.
    E, rho = KPM.dos(mu, a; N_tilde = 4096)
    dE = E[2] - E[1]
    N_kpm = cumsum(rho) .* dE
    λ = eigvals(Hermitian(Matrix(H)))
    N_ed = [count(<=(e), λ) / NH for e in E]
    inband = findall(e -> abs(abs(e) - gap / 2) > 0.15 && abs(e) < 0.9 * bandwidth / 2, E)
    @test maximum(abs.(N_kpm[inband] .- N_ed[inband])) < 0.02
    @test N_kpm[end] ≈ 1.0 atol = 0.02
end

@testset "third-order moments and cpge run end to end" begin
    a, H_norm = KPM.normalizeH(H)
    NC = 16
    NR = 2
    rng = Xoshiro(31415)
    psi_in = KPM.random_phase_vectors(rng, NH, NR)

    Gamma = zeros(ComplexF64, NC, NC, NC)
    KPM.kpm_3d!(H_norm, Jx, Jx, Jx, NC, NR, NH, Gamma, psi_in, psi_in)
    @test all(isfinite, Gamma)

    # the chain is inversion symmetric about a bond centre, so the xxx
    # third-order response is not expected to be large; what is asserted here
    # is that the whole chain returns a finite number (the sharp checks of the
    # kpm_3d convention and the cpge reconstruction live in cpge_test.jl)
    cpge_val = KPM.cpge(Gamma, NC, 0.1)
    @test isfinite(cpge_val)
end
