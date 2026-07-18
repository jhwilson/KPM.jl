using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Complex matrix-element moments ⟨ψl|T_n(H)|ψr⟩ with independent bra/ket, and
# the Green-function reconstruction built on them. Moment checks are exact
# (direct recurrence at 1e-12, ED spectral sums at 1e-8); reconstruction is
# anchored by the CPGF route at finite η, which is exact up to the series tail.

# Shared prescaled test Hamiltonian: dense complex Hermitian, spectrum inside
# (-1, 1), as in dos_test.jl.
function _prescaled_hermitian(rng, NH; margin = 0.95)
    A = randn(rng, ComplexF64, NH, NH)
    Hd = Matrix((A + A') / 2)
    a = maximum(abs, eigvals(Hermitian(Hd))) / margin
    Hd ./= a
    return Hd, sparse(Hd)
end

@testset "left/right moments vs direct recurrence and ED" begin
    rng = Xoshiro(7)
    NH = 64
    NC = 33   # odd on purpose: the non-doubling path must accept it
    NR = 2
    Hd, H = _prescaled_hermitian(rng, NH)

    ψl = randn(rng, ComplexF64, NH, NR)
    ψr = randn(rng, ComplexF64, NH, NR)
    foreach(j -> ψl[:, j] ./= norm(ψl[:, j]), 1:NR)
    foreach(j -> ψr[:, j] ./= norm(ψr[:, j]), 1:NR)

    mu_all = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_all, ψl, ψr)

    # direct three-term recurrence, complex dots
    mu_direct = zeros(ComplexF64, NR, NC)
    for j in 1:NR
        α_prev = ψr[:, j]
        α_curr = Hd * ψr[:, j]
        mu_direct[j, 1] = dot(ψl[:, j], α_prev)
        mu_direct[j, 2] = dot(ψl[:, j], α_curr)
        for n in 3:NC
            α_prev, α_curr = α_curr, 2 .* (Hd * α_curr) .- α_prev
            mu_direct[j, n] = dot(ψl[:, j], α_curr)
        end
    end
    @test mu_all ≈ mu_direct atol = 1e-12

    # ED: μ_n = Σ_m ⟨ψl|m⟩⟨m|ψr⟩ T_n(λ_m), complex
    λ, V = eigen(Hermitian(Hd))
    for j in 1:NR
        w = conj.(V' * ψl[:, j]) .* (V' * ψr[:, j])
        mu_ed = [sum(w .* cos.(n .* acos.(clamp.(λ, -1, 1)))) for n in 0:(NC - 1)]
        @test vec(mu_all[j, :]) ≈ mu_ed atol = 1e-8
    end

    # genuinely complex off-diagonal moments (not an accidental real case)
    @test maximum(abs.(imag.(mu_all))) > 1e-3
end

@testset "left == right: non-doubling path agrees with doubling path" begin
    rng = Xoshiro(11)
    NH = 48
    NC = 32   # even so the doubling path is legal
    NR = 3
    _, H = _prescaled_hermitian(rng, NH)

    ψ = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(ψ, NR)

    mu_doubling = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_doubling, ψ)

    mu_lr = zeros(ComplexF64, NR, NC)
    KPM.kpm_1d!(H, NC, NR, NH, mu_lr, copy(ψ), copy(ψ))

    @test mu_lr ≈ mu_doubling atol = 1e-12
    # diagonal moments of a Hermitian H are real to roundoff
    @test maximum(abs.(imag.(mu_lr))) < 1e-12
end

@testset "kpm_1d wrapper: complex average for left/right pairs" begin
    rng = Xoshiro(23)
    NH = 32
    NC = 17
    NR = 2
    _, H = _prescaled_hermitian(rng, NH)

    ψl = randn(rng, ComplexF64, NH, NR)
    ψr = randn(rng, ComplexF64, NH, NR)

    mu_avg = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr))
    @test eltype(mu_avg) <: Complex
    @test size(mu_avg) == (NC,)

    mu_all = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr),
                        avg_output = false)
    @test vec(sum(mu_all, dims = 1) ./ NR) ≈ mu_avg atol = 1e-12

    mu_ser = KPM.kpm_1d(H, NC, NR; psi_in_l = copy(ψl), psi_in_r = copy(ψr),
                        NR_parallel = false)
    @test mu_ser ≈ mu_avg atol = 1e-12

    # equal-vector path still returns real averages
    ψ = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(ψ, NR)
    mu_eq = KPM.kpm_1d(H, 16, NR; psi_in = ψ)
    @test eltype(mu_eq) <: Real
end

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference: ed_greens

# Physical-scale fixture: H = a·H_norm + b with known (a, b), plus unit-site
# probe pairs (i,i), (j,j), (i,j) so diagonal and off-diagonal reconstruction
# are both exercised.
function _greens_fixture(rng; NH = 64, NC = 512, a = 2.3, b = 0.4)
    Hn_d, Hn = _prescaled_hermitian(rng, NH)
    H_phys = a .* Hn_d .+ b .* Matrix(I, NH, NH)
    i, j = 3, NH - 5
    ψl = zeros(ComplexF64, NH, 3)
    ψr = zeros(ComplexF64, NH, 3)
    ψl[i, 1] = 1; ψr[i, 1] = 1      # (i, i)
    ψl[j, 2] = 1; ψr[j, 2] = 1      # (j, j)
    ψl[i, 3] = 1; ψr[j, 3] = 1      # (i, j)
    mu_all = zeros(ComplexF64, 3, NC)
    KPM.kpm_1d!(Hn, NC, 3, NH, mu_all, ψl, ψr)
    mu = permutedims(mu_all)         # (NC, 3) layout of the reconstruction
    u = ψl[:, 3]
    v = ψr[:, 3]
    return (; H_phys, mu, a, b, i, j, u, v, NH, NC)
end

@testset "CPGF route vs exact resolvent (finite η, tight)" begin
    rng = Xoshiro(101)
    fx = _greens_fixture(rng)
    η = 0.05 * fx.a                          # η̃ = 0.05; tail e^{-511·0.05} ≈ 8e-12
    E = collect(range(fx.b - 0.9 * fx.a, fx.b + 0.9 * fx.a; length = 41))

    G = KPM.greens(fx.mu, fx.a, E; b = fx.b, eta = η)
    e_i = zeros(ComplexF64, fx.NH); e_i[fx.i] = 1
    e_j = zeros(ComplexF64, fx.NH); e_j[fx.j] = 1
    @test G[:, 1] ≈ ed_greens(fx.H_phys, e_i, e_i, E; eta = η) atol = 1e-8
    @test G[:, 2] ≈ ed_greens(fx.H_phys, e_j, e_j, E; eta = η) atol = 1e-8
    @test G[:, 3] ≈ ed_greens(fx.H_phys, e_i, e_j, E; eta = η) atol = 1e-8

    # advanced branch against the ED advanced resolvent
    GA = KPM.greens(fx.mu, fx.a, E; b = fx.b, eta = η, branch = :advanced)
    @test GA[:, 3] ≈ ed_greens(fx.H_phys, e_i, e_j, E; eta = η, branch = :advanced) atol = 1e-8

    # scalar-E and vector-mu conveniences agree with the batched call
    @test KPM.greens(fx.mu[:, 3], fx.a, E[7]; b = fx.b, eta = η) ≈ G[7, 3] atol = 1e-12

    # route exclusivity and argument validation
    @test_throws ArgumentError KPM.greens(fx.mu, fx.a, E; b = fx.b)
    @test_throws ArgumentError KPM.greens(fx.mu, fx.a, E; b = fx.b, eta = η,
                                          kernel = KPM.JacksonKernel)
    @test_throws ArgumentError KPM.greens(fx.mu, fx.a, E; b = fx.b, eta = -1.0)
    @test_throws ArgumentError KPM.greens(fx.mu, fx.a, E; b = fx.b, eta = η,
                                          branch = :sideways)
end

@testset "Lorentz-kernel route vs exact resolvent at η(E) = (aλ/NC)·√(1-x̃²)" begin
    rng = Xoshiro(103)
    fx = _greens_fixture(rng)
    λ = 4.0
    E = collect(range(fx.b - 0.8 * fx.a, fx.b + 0.8 * fx.a; length = 31))

    G = KPM.greens(fx.mu, fx.a, E; b = fx.b, kernel = KPM.LorentzKernels(λ))
    e_i = zeros(ComplexF64, fx.NH); e_i[fx.i] = 1
    e_j = zeros(ComplexF64, fx.NH); e_j[fx.j] = 1
    # uniform damping of the θ-series is a Lorentzian of width λ/NC in θ, so
    # the equivalent physical broadening is position dependent:
    # η(E) = (aλ/NC)·√(1-x̃²), largest at band center.
    η_of = E -> (fx.a * λ / fx.NC) * sqrt(1 - ((E - fx.b) / fx.a)^2)
    G_ed = hcat([[ed_greens(fx.H_phys, l, r, e; eta = η_of(e)) for e in E]
                 for (l, r) in ((e_i, e_i), (e_j, e_j), (e_i, e_j))]...)
    # the sinh-kernel/Lorentzian correspondence is approximate; honest
    # scale-relative tolerance, calibrated once (not tuned to pass)
    scale = maximum(abs.(G_ed))
    @test maximum(abs.(G .- G_ed)) / scale < 0.05
end

@testset "branch identity G^A_uv = conj(G^R_vu)" begin
    rng = Xoshiro(107)
    fx = _greens_fixture(rng)
    NH, NC = fx.NH, fx.NC
    Hn = sparse((fx.H_phys .- fx.b .* Matrix(I, NH, NH)) ./ fx.a)

    # swapped moments ⟨v|T_n|u⟩
    mu_swap = zeros(ComplexF64, 1, NC)
    KPM.kpm_1d!(Hn, NC, 1, NH, mu_swap, reshape(fx.v, NH, 1), reshape(fx.u, NH, 1))
    mu_uv = fx.mu[:, 3]
    mu_vu = vec(permutedims(mu_swap))

    E = collect(range(fx.b - 0.7 * fx.a, fx.b + 0.7 * fx.a; length = 17))
    for kw in ((; eta = 0.05 * fx.a), (; kernel = KPM.LorentzKernels(3.0)))
        GA_uv = KPM.greens(mu_uv, fx.a, E; b = fx.b, branch = :advanced, kw...)
        GR_vu = KPM.greens(mu_vu, fx.a, E; b = fx.b, branch = :retarded, kw...)
        @test GA_uv ≈ conj.(GR_vu) atol = 1e-12
    end
end

@testset "Kramers–Kronig consistency of Re G and A on a grid" begin
    rng = Xoshiro(109)
    fx = _greens_fixture(rng)
    η = 0.08 * fx.a
    # wide fine grid; the PV Hilbert transform of the broadened spectral
    # function must recover Re G up to discretization and window truncation
    E = collect(range(fx.b - 3 * fx.a, fx.b + 3 * fx.a; length = 12001))
    G = KPM.greens(fx.mu[:, 1], fx.a, E; b = fx.b, eta = η)
    A = -imag.(G) ./ pi
    dE = E[2] - E[1]
    k = 101                                   # probe interior points only
    idx = k:200:(length(E) - k)
    for m in idx
        pv = sum(A[l] / (E[m] - E[l]) for l in eachindex(E) if l != m) * dE
        @test pv ≈ real(G[m]) atol = 0.02 * maximum(abs.(real.(G)))
    end
end

@testset "diagonal positivity and sum rules" begin
    rng = Xoshiro(113)
    fx = _greens_fixture(rng)

    # kernel routes: A vanishes outside the band, so the in-band Chebyshev
    # measure integrates it exactly: substitute x = cos φ and trapezoid in φ
    φ = range(1e-3, pi - 1e-3; length = 20001)
    E_φ = fx.b .+ fx.a .* cos.(φ)
    for kernel in (KPM.JacksonKernel, KPM.LorentzKernels(4.0))
        A = KPM.ldos(fx.mu[:, 1:2], fx.a, collect(E_φ); b = fx.b, kernel = kernel)
        @test minimum(A) > -1e-12             # positive diagonal spectral weight
        for p in 1:2
            integrand = A[:, p] .* (fx.a .* sin.(φ))   # |dE| = a sinφ dφ
            w = sum(integrand) * step(φ)
            @test w ≈ 1.0 atol = 1e-4
        end
    end

    # CPGF route: Lorentzian tails leak past any finite window; wide window,
    # honest tolerance
    η = 0.05 * fx.a
    E = collect(range(fx.b - 6 * fx.a, fx.b + 6 * fx.a; length = 24001))
    A = KPM.ldos(fx.mu[:, 1], fx.a, E; b = fx.b, eta = η)
    @test minimum(A) > -1e-12
    @test sum(A) * (E[2] - E[1]) ≈ 1.0 atol = 1e-2

    # spectral_function: diagonal matches ldos; off-diagonal is genuinely
    # complex and integrates to the overlap ⟨u|v⟩ = 0 for orthogonal probes
    Ashort = KPM.spectral_function(fx.mu, fx.a, E[1:200:end]; b = fx.b, eta = η)
    @test real.(Ashort[:, 1]) ≈ KPM.ldos(fx.mu[:, 1], fx.a, E[1:200:end]; b = fx.b, eta = η) atol = 1e-12
    @test maximum(abs.(imag.(Ashort[:, 1]))) < 1e-12
    Aoff = KPM.spectral_function(fx.mu[:, 3], fx.a, E; b = fx.b, eta = η)
    @test sum(Aoff) * (E[2] - E[1]) ≈ 0.0 atol = 1e-2
    @test maximum(abs.(imag.(Aoff))) > 1e-4   # off-diagonal A is complex
end

@testset "energies outside the band: real analytic continuation" begin
    rng = Xoshiro(127)
    fx = _greens_fixture(rng)
    E_out = [fx.b - 1.6 * fx.a, fx.b + 1.3 * fx.a, fx.b + 2.5 * fx.a]
    e_i = zeros(ComplexF64, fx.NH); e_i[fx.i] = 1

    # kernel route at η = 0: G is exactly real outside the band, and close to
    # the exact resolvent (kernel damping perturbs the small-n coefficients)
    G = KPM.greens(fx.mu[:, 1], fx.a, E_out; b = fx.b, kernel = KPM.JacksonKernel)
    @test maximum(abs.(imag.(G))) < 1e-12
    G_ed = ed_greens(fx.H_phys, e_i, e_i, E_out; eta = 1e-9)
    @test real.(G) ≈ real.(G_ed) rtol = 5e-3

    # CPGF route out of band is exact up to the series tail
    η = 0.05 * fx.a
    Gc = KPM.greens(fx.mu[:, 1], fx.a, E_out; b = fx.b, eta = η)
    @test Gc ≈ ed_greens(fx.H_phys, e_i, e_i, E_out; eta = η) atol = 1e-8
end
