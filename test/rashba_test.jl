using Test
using KPM
using LinearAlgebra
using SparseArrays

function rashba_channel_dense(h, mu, U, n, D)
    N = size(h, 1)
    xi = Matrix{ComplexF64}(h) - mu * I - Diagonal(U .* n ./ 2)
    return [
        xi Matrix{ComplexF64}(D);
        adjoint(Matrix{ComplexF64}(D)) -conj(xi)
    ]
end

function rashba_reconstruct_directed(mu_F, a, beta; Np = 2size(mu_F, 1))
    NC = size(mu_F, 1)
    gh = KPM.JacksonKernel.(0:(NC-1), NC) .* KPM.hn.(0:(NC-1))
    nodes, _ = KPM.gausschebyshevt(Np)
    C = cos.((0:(NC-1)) .* acos.(nodes'))
    wf = KPM.fermiFunctions(0.0, beta).(a .* nodes) ./ Np
    return transpose(gh .* mu_F) * (C * wf)
end

function rashba_channel_update_ed(H, channel, beta)
    N = size(H, 1) ÷ 2
    eig = eigen(Hermitian(H))
    occ = KPM.fermiFunctions(0.0, beta).(eig.values)
    values = ComplexF64[]
    sign = channel.parity === :even ? 1.0 : -1.0
    for (b, (i, j)) in pairs(channel.bonds)
        Fij = sum(
            eig.vectors[i, s] * conj(eig.vectors[j+N, s]) * occ[s] for
            s in eachindex(eig.values)
        )
        Fji = sum(
            eig.vectors[j, s] * conj(eig.vectors[i+N, s]) * occ[s] for
            s in eachindex(eig.values)
        )
        push!(values, -channel.V[b] * (Fij + sign * Fji) / 2)
    end
    return values
end

function rashba_h(t, lambda)
    h = zeros(ComplexF64, 4, 4)
    h[1, 3] = h[3, 1] = -t
    h[2, 4] = h[4, 2] = -t
    h[1, 4] = im * lambda
    h[4, 1] = -im * lambda
    h[2, 3] = im * lambda
    h[3, 2] = -im * lambda
    return sparse(h)
end

function ed_fixed_point(
    h,
    channel;
    mu,
    beta,
    seed = 0.2,
    mix = 0.5,
    tol = 1e-11,
    maxiter = 10_000,
)
    amplitudes = fill(ComplexF64(seed), length(channel.bonds))
    residual = Inf
    for _ = 1:maxiter
        D = KPM.pairing_matrix(size(h, 1), [channel]; amplitudes = [amplitudes])
        update = rashba_channel_update_ed(
            rashba_channel_dense(h, mu, zeros(size(h, 1)), zeros(size(h, 1)), D),
            channel,
            beta,
        )
        residual = norm(update .- amplitudes, Inf)
        @. amplitudes = (1 - mix) * amplitudes + mix * update
        residual < tol && break
    end
    return amplitudes, residual
end

const RASHBA_T = 1.0
const RASHBA_LAMBDA = 0.6
const RASHBA_MU = 0.2
const RASHBA_BETA = 8.0
const RASHBA_V = 2.5
const RASHBA_H = rashba_h(RASHBA_T, RASHBA_LAMBDA)
const RASHBA_CHANNEL = KPM.PairingChannel([(1, 2), (3, 4)], 1.0, RASHBA_V, :odd)

@testset "singlet is odd with explicit spin" begin
    @test ishermitian(RASHBA_H)
    D0 = KPM.pairing_matrix(4, [RASHBA_CHANNEL]; amplitude = 0.2)
    @test transpose(D0) == -D0
    H = rashba_channel_dense(RASHBA_H, RASHBA_MU, zeros(4), zeros(4), D0)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    taux = [zeros(ComplexF64, 4, 4) I4; I4 zeros(ComplexF64, 4, 4)]
    tauy = [
        zeros(ComplexF64, 4, 4) -im * I4;
        im * I4 zeros(ComplexF64, 4, 4)
    ]
    @test taux * conj(H) * taux ≈ -H atol=1e-12
    @test maximum(abs.(tauy * conj(H) * tauy + H)) > 1e-3
end

@testset "channel update vs ED" begin
    D0 = KPM.pairing_matrix(4, [RASHBA_CHANNEL]; amplitude = 0.2)
    H = rashba_channel_dense(RASHBA_H, RASHBA_MU, zeros(4), zeros(4), D0)
    delta_ed = rashba_channel_update_ed(H, RASHBA_CHANNEL, RASHBA_BETA)
    @test delta_ed[1] ≈ delta_ed[2] atol=1e-10

    op = KPM.BdGOperator(
        RASHBA_H;
        mu = RASHBA_MU,
        U = 0.0,
        n = zeros(4),
        D = D0,
        hole_convention = :conjugate,
    )
    rh = KPM.rescale(op)
    errors = Float64[]
    for NC in (128, 512)
        moments = KPM.bdg_channel_moments(rh, [RASHBA_CHANNEL]; NC = NC, g_rho = 1)
        _, amplitudes = KPM.bdg_update(moments; beta = RASHBA_BETA)
        push!(errors, maximum(abs.(only(amplitudes) .- delta_ed)))
    end
    println("Rashba channel-update errors: NC=128 $(errors[1]), NC=512 $(errors[2])")
    @test errors[2] < errors[1]
    @test errors[2] < 1e-3
end

@testset "SCF vs ED fixed point" begin
    delta_ed, residual_ed =
        ed_fixed_point(RASHBA_H, RASHBA_CHANNEL; mu = RASHBA_MU, beta = RASHBA_BETA)
    @test residual_ed < 1e-11

    D0 = KPM.pairing_matrix(4, [RASHBA_CHANNEL]; amplitude = 0.2)
    op = KPM.BdGOperator(
        RASHBA_H;
        mu = RASHBA_MU,
        U = 0.0,
        n = zeros(4),
        D = D0,
        hole_convention = :conjugate,
    )
    result = KPM.bdg_solve!(
        op,
        [RASHBA_CHANNEL];
        beta = RASHBA_BETA,
        NC = 512,
        g_rho = 1,
        mix = 0.3,
        tol_delta = 1e-8,
        update_density = false,
    )
    delta_kpm = ComplexF64[op.D[i, j] for (i, j) in RASHBA_CHANNEL.bonds]
    println("Rashba SCF: ED=$(abs.(delta_ed)), KPM=$(abs.(delta_kpm))")
    @test result.converged
    @test abs(delta_kpm[1]) ≈ abs(delta_kpm[2]) atol=1e-6
    @test abs(delta_kpm[1]) ≈ abs(delta_ed[1]) rtol=2e-3
end

@testset "lambda=0 reduces to the legacy reduced-singlet solver" begin
    h0 = rashba_h(RASHBA_T, 0.0)
    D0 = KPM.pairing_matrix(4, [RASHBA_CHANNEL]; amplitude = 0.2)
    op_explicit = KPM.BdGOperator(
        h0;
        mu = RASHBA_MU,
        U = 0.0,
        n = zeros(4),
        D = D0,
        hole_convention = :conjugate,
    )
    result_explicit = KPM.bdg_solve!(
        op_explicit,
        [RASHBA_CHANNEL];
        beta = RASHBA_BETA,
        NC = 512,
        g_rho = 1,
        mix = 0.3,
        tol_delta = 1e-8,
        update_density = false,
    )
    delta_explicit =
        abs.(ComplexF64[op_explicit.D[i, j] for (i, j) in RASHBA_CHANNEL.bonds])

    h2 = sparse([0.0 -RASHBA_T; -RASHBA_T 0.0])
    op_legacy = KPM.BdGOperator(
        h2;
        mu = RASHBA_MU,
        U = RASHBA_V,
        n = zeros(2),
        Delta = fill(0.2, 2),
        hole_convention = :conjugate,
    )
    result_legacy = KPM.bdg_solve!(
        op_legacy;
        beta = RASHBA_BETA,
        NC = 512,
        mix = 0.3,
        tol_delta = 1e-8,
        update_density = false,
    )
    delta_legacy = abs.(op_legacy.Δ)
    println("lambda=0 gaps: explicit=$(delta_explicit), legacy=$(delta_legacy)")
    @test result_explicit.converged
    @test result_legacy.converged
    @test maximum(abs.(delta_explicit .- delta_legacy)) < 1e-5
end
