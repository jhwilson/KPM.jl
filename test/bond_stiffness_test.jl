using Test
using KPM
using LinearAlgebra
using SparseArrays

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

const BOND_STIFFNESS_H, BOND_STIFFNESS_POS, BOND_STIFFNESS_DISP = square_model(3, 3)
const BOND_STIFFNESS_N = 9
const BOND_STIFFNESS_QY = [0.0, 2pi / 3]
const BOND_STIFFNESS_BETA = 10.0
const BOND_STIFFNESS_VOLUME = 9.0
const BOND_STIFFNESS_G_J = 1.0
const BOND_STIFFNESS_MU = -0.7
const BOND_STIFFNESS_DENSITY = fill(0.4, BOND_STIFFNESS_N)
const BOND_STIFFNESS_INTERACTION = fill(2.0, BOND_STIFFNESS_N)
const BOND_STIFFNESS_PSI = Matrix{ComplexF64}(I, 2BOND_STIFFNESS_N, 2BOND_STIFFNESS_N)

const BOND_STIFFNESS_CHANNEL = let
    rows, cols, _ = findnz(BOND_STIFFNESS_H)
    bonds = Tuple{Int,Int}[]
    weights = ComplexF64[]
    for (i, j) in zip(rows, cols)
        i < j || continue
        d = BOND_STIFFNESS_DISP(i, j)
        isx = abs(d[1]) == 1
        isy = abs(d[2]) == 1
        @assert xor(isx, isy)
        push!(bonds, (i, j))
        push!(weights, isx ? 1.0 : -1.0)
    end
    KPM.PairingChannel(bonds, weights, 1.0, :even)
end

function bond_stiffness_response(D)
    op = KPM.BdGOperator(
        BOND_STIFFNESS_H;
        mu = BOND_STIFFNESS_MU,
        U = BOND_STIFFNESS_INTERACTION,
        n = BOND_STIFFNESS_DENSITY,
        D = D,
        hole_convention = :conjugate,
    )
    return KPM.superfluid_stiffness(
        op,
        BOND_STIFFNESS_POS,
        BOND_STIFFNESS_QY;
        beta = BOND_STIFFNESS_BETA,
        eta = 0.0,
        NC = 256,
        psi_in = BOND_STIFFNESS_PSI,
        volume = BOND_STIFFNESS_VOLUME,
        g_J = BOND_STIFFNESS_G_J,
        disp = BOND_STIFFNESS_DISP,
        include_diamagnetic = true,
    )
end

function bond_stiffness_dense_response(H, Jq, Jmq)
    eta = 0.0
    omega = 0.0
    F = eigen(Hermitian(H))
    Jq_e = F.vectors' * Matrix(Jq) * F.vectors
    Jmq_e = F.vectors' * Matrix(Jmq) * F.vectors
    fermi(e) = 1 / (exp(BOND_STIFFNESS_BETA * e) + 1)
    occupations = fermi.(F.values)
    acc = zero(ComplexF64)
    for p in eachindex(F.values), q in eachindex(F.values)
        delta_E = F.values[p] - F.values[q]
        divided_difference = if eta == 0
            if abs(delta_E) < 1e-8
                fm = fermi((F.values[p] + F.values[q]) / 2)
                -BOND_STIFFNESS_BETA * fm * (1 - fm)
            else
                (occupations[p] - occupations[q]) / delta_E
            end
        else
            (occupations[p] - occupations[q]) / (omega + delta_E + im * eta)
        end
        acc += Jq_e[p, q] * Jmq_e[q, p] * divided_difference
    end
    return BOND_STIFFNESS_G_J * acc / BOND_STIFFNESS_VOLUME
end

@testset "zero-gap cancellation with structural bond zeros" begin
    D0 = KPM.pairing_matrix(BOND_STIFFNESS_N, [BOND_STIFFNESS_CHANNEL]; amplitude = 0.0)
    r0 = bond_stiffness_response(D0)
    @test abs(r0.Ds_over_pi_complete) < 1e-10
    @test abs(r0.Ds_over_pi) < 1e-10
end

@testset "rigid d-wave stiffness vs free-energy curvature" begin
    D = KPM.pairing_matrix(BOND_STIFFNESS_N, [BOND_STIFFNESS_CHANNEL]; amplitude = 0.45)
    D_SC = Matrix(D)
    D_zero = zeros(ComplexF64, BOND_STIFFNESS_N, BOND_STIFFNESS_N)
    r = bond_stiffness_response(D)

    free_energy(D_or_zero, A) = ed_bdg_free_energy(
        BOND_STIFFNESS_H,
        BOND_STIFFNESS_POS,
        BOND_STIFFNESS_DISP,
        BOND_STIFFNESS_MU,
        BOND_STIFFNESS_INTERACTION,
        BOND_STIFFNESS_DENSITY,
        Matrix(D_or_zero),
        A;
        q = BOND_STIFFNESS_QY,
        dir = 1,
        beta = BOND_STIFFNESS_BETA,
    )
    curvature(D_or_zero, delta) =
        (
            free_energy(D_or_zero, delta) - 2free_energy(D_or_zero, 0.0) +
            free_energy(D_or_zero, -delta)
        ) / delta^2

    Fpp_SC = curvature(D_SC, 1e-3)
    Fpp_N = curvature(D_zero, 1e-3)
    Fpp_SC_halfstep = curvature(D_SC, 5e-4)
    Fpp_N_halfstep = curvature(D_zero, 5e-4)
    anchor = (2BOND_STIFFNESS_G_J / BOND_STIFFNESS_VOLUME) * (Fpp_SC - Fpp_N)
    anchor_halfstep =
        (2BOND_STIFFNESS_G_J / BOND_STIFFNESS_VOLUME) * (Fpp_SC_halfstep - Fpp_N_halfstep)
    stiffness_relerr = abs(r.Ds_over_pi_complete - anchor) / abs(anchor)
    curvature_relerr = abs(anchor - anchor_halfstep) / abs(anchor)

    println(
        "bond free-energy curvatures delta=1e-3: SC=$(Fpp_SC), N=$(Fpp_N), anchor=$(anchor)",
    )
    println(
        "bond free-energy curvatures delta=5e-4: SC=$(Fpp_SC_halfstep), N=$(Fpp_N_halfstep), anchor=$(anchor_halfstep)",
    )
    println(
        "bond complete stiffness: curvature anchor=$(anchor), KPM=$(r.Ds_over_pi_complete), relerr=$(stiffness_relerr)",
    )
    println("bond curvature step relative error=$(curvature_relerr)")
    println(
        "bond paramagnetic-only stiffness=$(r.Ds_over_pi), diamagnetic correction=$(r.Dia_SC - r.Dia_N)",
    )

    @test isapprox(r.Ds_over_pi_complete, anchor; rtol = 2e-2)
    @test isapprox(anchor, anchor_halfstep; rtol = 1e-3)
    @test abs(r.Ds_over_pi_complete - r.Ds_over_pi) > 1e-4
end

@testset "paramagnetic response vs dense ED" begin
    D = KPM.pairing_matrix(BOND_STIFFNESS_N, [BOND_STIFFNESS_CHANNEL]; amplitude = 0.45)
    r = bond_stiffness_response(D)
    xi =
        Matrix{ComplexF64}(BOND_STIFFNESS_H) - BOND_STIFFNESS_MU * I -
        Diagonal(BOND_STIFFNESS_INTERACTION .* BOND_STIFFNESS_DENSITY ./ 2)
    H = [
        xi Matrix{ComplexF64}(D);
        adjoint(Matrix{ComplexF64}(D)) -conj(xi)
    ]
    Jq = KPM.nambu_current_q(
        BOND_STIFFNESS_H,
        BOND_STIFFNESS_POS,
        BOND_STIFFNESS_QY;
        dir = 1,
        disp = BOND_STIFFNESS_DISP,
        hole_convention = :conjugate,
    )
    Jmq = KPM.nambu_current_q(
        BOND_STIFFNESS_H,
        BOND_STIFFNESS_POS,
        -BOND_STIFFNESS_QY;
        dir = 1,
        disp = BOND_STIFFNESS_DISP,
        hole_convention = :conjugate,
    )
    Pi_SC_ed = bond_stiffness_dense_response(H, Jq, Jmq)
    relerr = abs(real(r.Pi_SC) - real(Pi_SC_ed)) / abs(real(Pi_SC_ed))

    println("bond paired stiffness Pi_SC: ED=$(Pi_SC_ed), KPM=$(r.Pi_SC), relerr=$(relerr)")
    @test isapprox(real(r.Pi_SC), real(Pi_SC_ed); rtol = 0.05)
end

@testset "U(1) rigidity sanity" begin
    D = KPM.pairing_matrix(BOND_STIFFNESS_N, [BOND_STIFFNESS_CHANNEL]; amplitude = 0.45)
    r = bond_stiffness_response(D)
    r_phase = bond_stiffness_response(exp(0.4im) * D)
    @test r_phase.Ds_over_pi ≈ r.Ds_over_pi atol=1e-10
    @test r_phase.Ds_over_pi_complete ≈ r.Ds_over_pi_complete atol=1e-10
end
