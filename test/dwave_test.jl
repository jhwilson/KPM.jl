using Test
using KPM
using LinearAlgebra
using SparseArrays

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

function dwave_channel_dense(h, mu, U, n, D)
    N = size(h, 1)
    xi = Matrix{ComplexF64}(h) - mu * I - Diagonal(U .* n ./ 2)
    return [xi Matrix{ComplexF64}(D);
            adjoint(Matrix{ComplexF64}(D)) -conj(xi)]
end

function dwave_reconstruct_directed(mu_F, a, beta; Np=2size(mu_F, 1))
    NC = size(mu_F, 1)
    gh = KPM.JacksonKernel.(0:NC-1, NC) .* KPM.hn.(0:NC-1)
    nodes, _ = KPM.gausschebyshevt(Np)
    C = cos.((0:NC-1) .* acos.(nodes'))
    wf = KPM.fermiFunctions(0.0, beta).(a .* nodes) ./ Np
    return transpose(gh .* mu_F) * (C * wf)
end

function dwave_channel_update_ed(H, channel, beta)
    N = size(H, 1) ÷ 2
    eig = eigen(Hermitian(H))
    occ = KPM.fermiFunctions(0.0, beta).(eig.values)
    values = ComplexF64[]
    sign = channel.parity === :even ? 1.0 : -1.0
    for (b, (i, j)) in pairs(channel.bonds)
        Fij = sum(eig.vectors[i, s] * conj(eig.vectors[j + N, s]) * occ[s]
                  for s in eachindex(eig.values))
        Fji = sum(eig.vectors[j, s] * conj(eig.vectors[i + N, s]) * occ[s]
                  for s in eachindex(eig.values))
        push!(values, -channel.V[b] * (Fij + sign * Fji) / 2)
    end
    return values
end

const DWAVE_H, DWAVE_POS, DWAVE_DISP = square_model(3, 3)
const DWAVE_N = 9
const DWAVE_MU = -0.4
const DWAVE_BETA = 8.0

const DWAVE_BONDS, DWAVE_WEIGHTS, DWAVE_X = let
    rows, cols, _ = findnz(DWAVE_H)
    bonds = Tuple{Int, Int}[]
    weights = ComplexF64[]
    x_bonds = BitVector()
    for (i, j) in zip(rows, cols)
        i < j || continue
        d = DWAVE_DISP(i, j)
        isx = abs(d[1]) == 1
        isy = abs(d[2]) == 1
        @assert xor(isx, isy)
        push!(bonds, (i, j))
        push!(weights, isx ? 1.0 : -1.0)
        push!(x_bonds, isx)
    end
    bonds, weights, x_bonds
end
const DWAVE_CHANNEL = KPM.PairingChannel(DWAVE_BONDS, DWAVE_WEIGHTS, 1.6, :even)
const DWAVE_D0 = KPM.pairing_matrix(DWAVE_N, [DWAVE_CHANNEL]; amplitude=0.35)

@testset "d-wave PH branch" begin
    H = dwave_channel_dense(DWAVE_H, DWAVE_MU, zeros(DWAVE_N), zeros(DWAVE_N),
                            DWAVE_D0)
    I9 = Matrix{ComplexF64}(I, DWAVE_N, DWAVE_N)
    tauy = [zeros(ComplexF64, DWAVE_N, DWAVE_N) -im * I9;
            im * I9 zeros(ComplexF64, DWAVE_N, DWAVE_N)]
    @test tauy * conj(H) * tauy ≈ -H atol=1e-12
end

@testset "C4 covariance of the gap update (ED)" begin
    H = dwave_channel_dense(DWAVE_H, DWAVE_MU, zeros(DWAVE_N), zeros(DWAVE_N),
                            DWAVE_D0)
    delta_new = dwave_channel_update_ed(H, DWAVE_CHANNEL, DWAVE_BETA)
    xvalues = delta_new[DWAVE_X]
    yvalues = delta_new[.!DWAVE_X]
    @test maximum(abs.(xvalues .- first(xvalues))) < 1e-10
    @test maximum(abs.(yvalues .- first(yvalues))) < 1e-10
    @test first(yvalues) ≈ -first(xvalues) atol=1e-10
    c = KPM.channel_amplitude(DWAVE_CHANNEL, delta_new)
    @test c ≈ first(xvalues) atol=1e-10
    @test maximum(abs.(delta_new .- c .* DWAVE_CHANNEL.weights)) < 1e-10
end

@testset "KPM channel update vs ED" begin
    H = dwave_channel_dense(DWAVE_H, DWAVE_MU, zeros(DWAVE_N), zeros(DWAVE_N),
                            DWAVE_D0)
    delta_ed = dwave_channel_update_ed(H, DWAVE_CHANNEL, DWAVE_BETA)
    op = KPM.BdGOperator(DWAVE_H; mu=DWAVE_MU, U=0.0, n=zeros(DWAVE_N),
                         D=DWAVE_D0, hole_convention=:conjugate)
    rh = KPM.rescale(op)
    errors = Float64[]
    for NC in (128, 512)
        moments = KPM.bdg_channel_moments(rh, [DWAVE_CHANNEL]; NC=NC,
                                           g_rho=1)
        _, amplitudes = KPM.bdg_update(moments; beta=DWAVE_BETA)
        push!(errors, maximum(abs.(only(amplitudes) .- delta_ed)))
    end
    println("d-wave channel-update errors: NC=128 $(errors[1]), NC=512 $(errors[2])")
    @test errors[2] < errors[1]
    @test errors[2] < 1e-3
end
