# Generic two-energy response reconstruction checks against dense ED.
using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

isdefined(@__MODULE__, :EDReference) || include("ed_reference.jl")
using .EDReference

const STIFFNESS_D = 6
const STIFFNESS_RNG = Xoshiro(42)
const STIFFNESS_A = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_H0 = Matrix(Hermitian((STIFFNESS_A + STIFFNESS_A') / 2))
const STIFFNESS_JL = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_JR = randn(STIFFNESS_RNG, ComplexF64, STIFFNESS_D, STIFFNESS_D)
const STIFFNESS_JL_H = Matrix(Hermitian((STIFFNESS_JL + STIFFNESS_JL') / 2))
const STIFFNESS_JR_H = Matrix(Hermitian((STIFFNESS_JR + STIFFNESS_JR') / 2))
const STIFFNESS_EV = eigvals(Hermitian(STIFFNESS_H0))
const STIFFNESS_B0 = 0.2
const STIFFNESS_A0 = maximum(abs, STIFFNESS_EV .- STIFFNESS_B0) / 0.9
const STIFFNESS_HN = (STIFFNESS_H0 - STIFFNESS_B0 * I) / STIFFNESS_A0
const STIFFNESS_PSI = Matrix{ComplexF64}(I, STIFFNESS_D, STIFFNESS_D)

stiffness_moments(NC) = KPM.kpm_2d(
    STIFFNESS_HN,
    STIFFNESS_JL,
    STIFFNESS_JR,
    NC,
    STIFFNESS_D,
    STIFFNESS_D;
    psi_in_l = STIFFNESS_PSI,
    psi_in_r = STIFFNESS_PSI,
)

mu2D_64 = stiffness_moments(64)
mu2D_256 = stiffness_moments(256)

@testset "two-point Lehmann oracle (omega≠0, complex vertices)" begin
    Pi_kpm_64 = KPM.two_energy_response(
        mu2D_64,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    Pi_kpm_256 = KPM.two_energy_response(
        mu2D_256,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0,
        STIFFNESS_JL,
        STIFFNESS_JR;
        beta = 8.0,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
    )
    err64 = abs(Pi_kpm_64 - Pi_ed)
    err256 = abs(Pi_kpm_256 - Pi_ed)
    println(
        "two-point oracle: ED=$(Pi_ed), KPM NC=64=$(Pi_kpm_64), KPM NC=256=$(Pi_kpm_256)",
    )
    println("two-point oracle errors: NC=64 $(err64), NC=256 $(err256)")

    @test err256 < err64
    @test abs(real(Pi_kpm_256 - Pi_ed)) < abs(real(Pi_kpm_64 - Pi_ed))
    @test abs(imag(Pi_kpm_256 - Pi_ed)) < abs(imag(Pi_kpm_64 - Pi_ed))
    TOL_DYNAMIC = 0.015
    @test isapprox(Pi_kpm_256, Pi_ed; rtol = TOL_DYNAMIC)
end

@testset "static eta=0 divided-difference kernel" begin
    # For the deliberately non-Hermitian oracle vertices the divided-
    # difference kernel is real, but its contraction need not be. Preserve
    # that generic comparison, then check the requested reality property on
    # Hermitian versions of the same fixed vertices.
    Pi_kpm_generic = KPM.two_energy_response(
        mu2D_256,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.0,
        omega = 0.0,
        Ef = 0.15,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    Pi_ed_generic = ed_two_energy_response(
        STIFFNESS_H0,
        STIFFNESS_JL,
        STIFFNESS_JR;
        beta = 8.0,
        eta = 0.0,
        omega = 0.0,
        Ef = 0.15,
    )
    println(
        "static divided difference (non-Hermitian): ED=$(Pi_ed_generic), KPM NC=256=$(Pi_kpm_generic), error=$(abs(Pi_kpm_generic - Pi_ed_generic))",
    )
    @test isapprox(Pi_kpm_generic, Pi_ed_generic; rtol = 0.012)

    mu2D_static = KPM.kpm_2d(
        STIFFNESS_HN,
        STIFFNESS_JL_H,
        STIFFNESS_JR_H,
        256,
        STIFFNESS_D,
        STIFFNESS_D;
        psi_in_l = STIFFNESS_PSI,
        psi_in_r = STIFFNESS_PSI,
    )
    Pi_kpm = KPM.two_energy_response(
        mu2D_static,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.0,
        omega = 0.0,
        Ef = 0.15,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0,
        STIFFNESS_JL_H,
        STIFFNESS_JR_H;
        beta = 8.0,
        eta = 0.0,
        omega = 0.0,
        Ef = 0.15,
    )
    println(
        "static divided difference (Hermitian): ED=$(Pi_ed), KPM NC=256=$(Pi_kpm), error=$(abs(Pi_kpm - Pi_ed))",
    )
    @test imag(Pi_kpm) ≈ 0 atol=1e-10
    @test imag(Pi_ed) ≈ 0 atol=1e-10
    TOL_STATIC = 0.05
    @test isapprox(Pi_kpm, Pi_ed; rtol = TOL_STATIC)
    @test_throws ArgumentError KPM.two_energy_response(
        mu2D_256,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.0,
        omega = 0.3,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    @test_throws ArgumentError KPM.two_energy_response(
        mu2D_256,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = Inf,
        eta = 0.0,
        omega = 0.0,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
end

@testset "typed wrapper parity and beta=Inf" begin
    m = KPM.ConductivityMoments(
        mu2D_256,
        STIFFNESS_A0,
        STIFFNESS_B0,
        STIFFNESS_D,
        STIFFNESS_D,
    )
    raw = KPM.two_energy_response(
        mu2D_256,
        STIFFNESS_A0;
        b = STIFFNESS_B0,
        beta = 8.0,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
        NH = STIFFNESS_D,
        volume = 1.0,
    )
    wrapped = KPM.two_energy_response(
        m;
        beta = 8.0,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
        volume = 1.0,
    )
    @test wrapped == raw
    @test_throws ArgumentError KPM.two_energy_response(
        m;
        beta = 8.0,
        eta = 0.25,
        volume = 1.0,
        b = STIFFNESS_B0,
    )
    @test_throws ArgumentError KPM.two_energy_response(
        m;
        beta = 8.0,
        eta = 0.25,
        volume = 1.0,
        NH = STIFFNESS_D,
    )

    Pi_kpm = KPM.two_energy_response(
        m;
        beta = Inf,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
        volume = 1.0,
    )
    Pi_ed = ed_two_energy_response(
        STIFFNESS_H0,
        STIFFNESS_JL,
        STIFFNESS_JR;
        beta = Inf,
        eta = 0.25,
        omega = 0.3,
        Ef = 0.15,
    )
    println(
        "beta=Inf oracle: ED=$(Pi_ed), KPM NC=256=$(Pi_kpm), error=$(abs(Pi_kpm - Pi_ed))",
    )
    @test isfinite(real(Pi_kpm)) && isfinite(imag(Pi_kpm))
    TOL_BETA_INF = 0.02
    @test isapprox(Pi_kpm, Pi_ed; rtol = TOL_BETA_INF)
end

(h4, pos4, disp4) = ring_model(4)
(hsq, possq, dispsq) = square_model(3, 3)
qy = [0.0, 2pi / 3]

@testset "vertex adjoint identity" begin
    # q must be commensurate with the periodic 3x3 torus (reciprocal-grid
    # vectors): wrapped-bond midpoints shift by a lattice vector L, so the
    # adjoint identity needs exp(im q . L) = 1. See the nambu_current_q
    # docstring; an incommensurate q like [0.7, 0.3] breaks it by design.
    for q in ([0.0, 0.0], qy, [2pi / 3, 4pi / 3])
        Jq = KPM.nambu_current_q(hsq, possq, q; dir = 1, disp = dispsq)
        Jmq = KPM.nambu_current_q(hsq, possq, -q; dir = 1, disp = dispsq)
        @test Matrix(Jq') ≈ -Matrix(Jmq) atol=1e-13
    end

    J0 = Matrix(KPM.nambu_current_q(hsq, possq, [0.0, 0.0]; dir = 1, disp = dispsq))
    @test J0[1:9, 1:9] ≈ J0[10:18, 10:18] atol=1e-13
    Jx_manual = zeros(ComplexF64, 9, 9)
    I, J, V = findnz(hsq)
    for k in eachindex(V)
        i, j = I[k], J[k]
        Jx_manual[i, j] = V[k] * dispsq(i, j)[1]
    end
    @test J0[1:9, 1:9] ≈ Jx_manual atol=1e-13
end

@testset "finite-difference vertex vs Peierls" begin
    delta = 1e-6
    FD_uniform =
        (
            bdg_peierls_matrix(hsq, possq, dispsq, +delta) -
            bdg_peierls_matrix(hsq, possq, dispsq, -delta)
        ) / (2delta * im)
    J0 = Matrix(KPM.nambu_current_q(hsq, possq, [0.0, 0.0]; dir = 1, disp = dispsq))
    @test FD_uniform ≈ J0 atol=1e-7

    FD_modulated =
        (
            bdg_peierls_matrix(hsq, possq, dispsq, +delta; q = qy) -
            bdg_peierls_matrix(hsq, possq, dispsq, -delta; q = qy)
        ) / (2delta * im)
    Jq = KPM.nambu_current_q(hsq, possq, qy; dir = 1, disp = dispsq)
    Jmq = KPM.nambu_current_q(hsq, possq, -qy; dir = 1, disp = dispsq)
    @test FD_modulated ≈ Matrix((Jq + Jmq) ./ 2) atol=1e-7

    H0 = bdg_peierls_matrix(hsq, possq, dispsq, 0.0)
    @test H0 ≈ bdg_matrix(hsq, 0.0, zeros(9), zeros(9), zeros(ComplexF64, 9)) atol=1e-13
end

@testset "zero-gap cancellation (paired subtraction)" begin
    op0 = KPM.BdGOperator(
        hsq;
        mu = -0.7,
        U = 2.0,
        n = fill(0.4, 9),
        Delta = zeros(ComplexF64, 9),
    )
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    r = KPM.superfluid_stiffness(
        op0,
        possq,
        qy;
        beta = 10.0,
        eta = 0.3,
        NC = 128,
        psi_in = psi_id,
        volume = 9.0,
        disp = dispsq,
    )
    @test abs(r.Ds_over_pi) < 1e-10
    @test r.a_SC == r.a_N

    r_stochastic = KPM.superfluid_stiffness(
        op0,
        possq,
        qy;
        beta = 10.0,
        eta = 0.3,
        NC = 128,
        NR = 4,
        volume = 9.0,
        disp = dispsq,
    )
    @test abs(r_stochastic.Ds_over_pi) < 1e-10
end

@testset "Delta != 0: KPM vs ED at identical (q, beta, eta)" begin
    delta_field = fill(0.45 + 0.2im, 9)
    opd = KPM.BdGOperator(hsq; mu = -0.7, U = 2.0, n = fill(0.4, 9), Delta = delta_field)
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    r = KPM.superfluid_stiffness(
        opd,
        possq,
        qy;
        beta = 10.0,
        eta = 0.3,
        NC = 256,
        psi_in = psi_id,
        volume = 1.0,
        g_J = 1.0,
        disp = dispsq,
    )

    Jq = KPM.nambu_current_q(hsq, possq, qy; dir = 1, disp = dispsq)
    Jmq = KPM.nambu_current_q(hsq, possq, -qy; dir = 1, disp = dispsq)
    Hd = bdg_matrix(hsq, -0.7, fill(2.0, 9), fill(0.4, 9), delta_field)
    Hn = bdg_matrix(hsq, -0.7, fill(2.0, 9), fill(0.4, 9), zeros(ComplexF64, 9))
    Pi_SC_ed = ed_two_energy_response(
        Hd,
        Matrix(Jq),
        Matrix(Jmq);
        beta = 10.0,
        eta = 0.3,
        omega = 0.0,
        Ef = 0.0,
    )
    Pi_N_ed = ed_two_energy_response(
        Hn,
        Matrix(Jq),
        Matrix(Jmq);
        beta = 10.0,
        eta = 0.3,
        omega = 0.0,
        Ef = 0.0,
    )
    Ds_ed = real(Pi_N_ed) - real(Pi_SC_ed)
    println(
        "paired stiffness Pi_SC: ED=$(Pi_SC_ed), KPM=$(r.Pi_SC), relerr=$(abs(r.Pi_SC - Pi_SC_ed) / abs(Pi_SC_ed))",
    )
    println(
        "paired stiffness Pi_N: ED=$(Pi_N_ed), KPM=$(r.Pi_N), relerr=$(abs(r.Pi_N - Pi_N_ed) / abs(Pi_N_ed))",
    )
    println(
        "paired stiffness Ds/pi: ED=$(Ds_ed), KPM=$(r.Ds_over_pi), relerr=$(abs(r.Ds_over_pi - Ds_ed) / abs(Ds_ed))",
    )

    @test isapprox(r.Pi_SC, Pi_SC_ed; rtol = 0.05)
    @test isapprox(r.Pi_N, Pi_N_ed; rtol = 0.05)
    @test isapprox(r.Ds_over_pi, Ds_ed; rtol = 0.05)
    if Ds_ed > 0
        @test r.Ds_over_pi > 0
    end
end

@testset "moment_parity :EVEN opt-in (real h, real Delta)" begin
    opr = KPM.BdGOperator(
        hsq;
        mu = -0.7,
        U = 2.0,
        n = fill(0.4, 9),
        Delta = fill(0.5 + 0.0im, 9),
    )
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    Jq = KPM.nambu_current_q(hsq, possq, qy; dir = 1, disp = dispsq)
    Jmq = KPM.nambu_current_q(hsq, possq, -qy; dir = 1, disp = dispsq)
    rh = KPM.rescale(opr; eps = 0.2)
    mu_none = KPM.kpm_2d(rh.H, Jq, Jmq, 64, 18, 18; psi_in = psi_id)
    odd_max = maximum(
        abs(mu_none[n, m]) for n in axes(mu_none, 1), m in axes(mu_none, 2) if isodd(m - n)
    )
    all_max = maximum(abs, mu_none)
    println(
        "moment parity premise: odd max=$(odd_max), all max=$(all_max), ratio=$(odd_max / all_max)",
    )
    @test odd_max < 1e-12 * all_max

    common =
        (; beta = 10.0, eta = 0.3, NC = 128, psi_in = psi_id, volume = 1.0, disp = dispsq)
    r_none = KPM.superfluid_stiffness(opr, possq, qy; common..., moment_parity = :NONE)
    r_even = KPM.superfluid_stiffness(opr, possq, qy; common..., moment_parity = :EVEN)
    @test isapprox(r_even.Pi_SC, r_none.Pi_SC; rtol = 1e-10)
    @test isapprox(r_even.Pi_N, r_none.Pi_N; rtol = 1e-10)
    @test isapprox(r_even.Ds_over_pi, r_none.Ds_over_pi; rtol = 1e-10)
end

@testset "volume and g_J prefactor routing" begin
    opd = KPM.BdGOperator(
        hsq;
        mu = -0.7,
        U = 2.0,
        n = fill(0.4, 9),
        Delta = fill(0.45 + 0.2im, 9),
    )
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    common = (; beta = 10.0, eta = 0.3, NC = 64, psi_in = psi_id, disp = dispsq)
    r1 = KPM.superfluid_stiffness(opd, possq, qy; common..., volume = 1.0)
    r4 = KPM.superfluid_stiffness(opd, possq, qy; common..., volume = 4.0)
    rg = KPM.superfluid_stiffness(opd, possq, qy; common..., volume = 1.0, g_J = 2.0)
    @test r4.Pi_SC ≈ r1.Pi_SC / 4 atol=1e-14
    @test r4.Pi_N ≈ r1.Pi_N / 4 atol=1e-14
    @test r4.Ds_over_pi ≈ r1.Ds_over_pi / 4 atol=1e-14
    @test rg.Pi_SC ≈ 2 * r1.Pi_SC atol=1e-14
    @test rg.Ds_over_pi ≈ 2 * r1.Ds_over_pi atol=1e-14
end

@testset "kpm_2d! parity zeroes skipped entries (overwrite contract)" begin
    opr = KPM.BdGOperator(
        hsq;
        mu = -0.7,
        U = 2.0,
        n = fill(0.4, 9),
        Delta = fill(0.5 + 0.0im, 9),
    )
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    Jq = KPM.nambu_current_q(hsq, possq, qy; dir = 1, disp = dispsq)
    Jmq = KPM.nambu_current_q(hsq, possq, -qy; dir = 1, disp = dispsq)
    rh = KPM.rescale(opr; eps = 0.2)
    NCp = 8
    mu_pre = fill(one(ComplexF64), NCp, NCp)
    KPM.kpm_2d!(rh.H, Jq, Jmq, NCp, 18, 18, mu_pre, psi_id; moment_parity = :EVEN)
    for n in axes(mu_pre, 1), m in axes(mu_pre, 2)
        if isodd(m - n)
            @test mu_pre[n, m] == 0
        end
    end
end

@testset "convention-aware vertices (complex flux ring)" begin
    N = 4
    hf, posf, dispf = flux_ring_model(N; phi = 0.35)
    q0 = [0.0, 0.0]
    delta = 1e-6

    FD_singlet =
        (
            bdg_peierls_matrix(hf, posf, dispf, +delta; hole_convention = :singlet) -
            bdg_peierls_matrix(hf, posf, dispf, -delta; hole_convention = :singlet)
        ) / (2delta * im)
    J_singlet =
        KPM.nambu_current_q(hf, posf, q0; dir = 1, disp = dispf, hole_convention = :singlet)
    @test FD_singlet ≈ Matrix(J_singlet) atol=1e-7

    FD_intervalley =
        (
            bdg_peierls_matrix(hf, posf, dispf, +delta; hole_convention = :intervalley) -
            bdg_peierls_matrix(hf, posf, dispf, -delta; hole_convention = :intervalley)
        ) / (2delta * im)
    J_intervalley = KPM.nambu_current_q(
        hf,
        posf,
        q0;
        dir = 1,
        disp = dispf,
        hole_convention = :intervalley,
    )
    @test FD_intervalley ≈ Matrix(J_intervalley) atol=1e-7
    cross_convention_mismatch = maximum(abs.(FD_singlet .- Matrix(J_intervalley)))
    println("flux-ring vertex cross-convention mismatch=$(cross_convention_mismatch)")
    @test cross_convention_mismatch > 1e-3

    qx = [2pi / 4, 0.0]
    for convention in (:singlet, :intervalley)
        Jq = KPM.nambu_current_q(
            hf,
            posf,
            qx;
            dir = 1,
            disp = dispf,
            hole_convention = convention,
        )
        Jmq = KPM.nambu_current_q(
            hf,
            posf,
            -qx;
            dir = 1,
            disp = dispf,
            hole_convention = convention,
        )
        @test Matrix(Jq') ≈ -Matrix(Jmq) atol=1e-13
    end

    mu = -0.4
    U = fill(2.0, N)
    n = fill(0.9, N)
    Delta = fill(0.45 + 0.2im, N)
    op_s = KPM.BdGOperator(
        hf;
        mu = mu,
        U = U,
        n = n,
        Delta = Delta,
        hole_convention = :singlet,
    )
    psi_id = Matrix{ComplexF64}(I, 2N, 2N)
    # This longitudinal ring calculation checks only vertex/convention
    # consistency; it is not a physical transverse-stiffness claim.
    response =
        @test_logs (:warn, r"q has a longitudinal component") KPM.superfluid_stiffness(
            op_s,
            posf,
            qx;
            beta = 10.0,
            eta = 0.3,
            dir = 1,
            disp = dispf,
            NC = 256,
            psi_in = psi_id,
            volume = 1.0,
        )
    Jq_s =
        KPM.nambu_current_q(hf, posf, qx; dir = 1, disp = dispf, hole_convention = :singlet)
    Jmq_s = KPM.nambu_current_q(
        hf,
        posf,
        -qx;
        dir = 1,
        disp = dispf,
        hole_convention = :singlet,
    )
    Hd_s = bdg_matrix_singlet(hf, mu, U, n, Delta)
    Pi_SC_ed = ed_two_energy_response(
        Hd_s,
        Matrix(Jq_s),
        Matrix(Jmq_s);
        beta = 10.0,
        eta = 0.3,
        omega = 0.0,
        Ef = 0.0,
    )
    stiffness_relerr = abs(response.Pi_SC - Pi_SC_ed) / abs(Pi_SC_ed)
    println(
        "flux-ring singlet Pi_SC: ED=$(Pi_SC_ed), KPM=$(response.Pi_SC), relerr=$(stiffness_relerr)",
    )
    @test isapprox(response.Pi_SC, Pi_SC_ed; rtol = 0.05)
end

@testset "diamagnetic term vs ED" begin
    delta_field = fill(0.45 + 0.2im, 9)
    opd = KPM.BdGOperator(hsq; mu = -0.7, U = 2.0, n = fill(0.4, 9), Delta = delta_field)
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    Dia_kpm = KPM.diamagnetic_term(
        opd,
        possq;
        disp = dispsq,
        beta = 10.0,
        NC = 256,
        psi_in = psi_id,
        volume = 1.0,
    )
    Dhat = KPM.nambu_diamagnetic(hsq, possq; disp = dispsq)
    Hd = bdg_matrix(hsq, -0.7, fill(2.0, 9), fill(0.4, 9), delta_field)
    Dia_ed = ed_diamagnetic(Hd, Matrix(Dhat); beta = 10.0)
    Dia_relerr = abs(Dia_kpm - Dia_ed) / abs(Dia_ed)
    println("diamagnetic term: ED=$(Dia_ed), KPM=$(Dia_kpm), relerr=$(Dia_relerr)")
    TOL_DIAMAGNETIC = 5e-3
    @test isapprox(Dia_kpm, Dia_ed; rtol = TOL_DIAMAGNETIC)

    @test Matrix(Dhat) ≈ Matrix(Dhat)' atol=1e-13
    Dhat_intervalley =
        KPM.nambu_diamagnetic(hsq, possq; disp = dispsq, hole_convention = :intervalley)
    @test Dhat == Dhat_intervalley

    hf, posf, dispf = flux_ring_model(4; phi = 0.35)
    Dhat_singlet = KPM.nambu_diamagnetic(hf, posf; disp = dispf, hole_convention = :singlet)
    Dhat_intervalley_flux =
        KPM.nambu_diamagnetic(hf, posf; disp = dispf, hole_convention = :intervalley)
    diamagnetic_convention_mismatch =
        maximum(abs, Matrix(Dhat_singlet - Dhat_intervalley_flux))
    println(
        "flux-ring diamagnetic cross-convention mismatch=$(diamagnetic_convention_mismatch)",
    )
    @test diamagnetic_convention_mismatch > 1e-3
end

@testset "normal-state Meissner consistency and complete stiffness vs free-energy curvature" begin
    beta = 10.0
    volume = 9.0
    g_J = 1.0
    delta_field = fill(0.45 + 0.2im, 9)
    zero_delta = zeros(ComplexF64, 9)
    density = fill(0.4, 9)
    interaction = fill(2.0, 9)
    opd = KPM.BdGOperator(hsq; mu = -0.7, U = interaction, n = density, Delta = delta_field)
    op0 = KPM.BdGOperator(hsq; mu = -0.7, U = interaction, n = density, Delta = zero_delta)
    psi_id = Matrix{ComplexF64}(I, 18, 18)
    r = KPM.superfluid_stiffness(
        opd,
        possq,
        qy;
        beta = beta,
        eta = 0.0,
        NC = 256,
        psi_in = psi_id,
        volume = volume,
        g_J = g_J,
        disp = dispsq,
        include_diamagnetic = true,
    )

    free_energy(Delta, A) = ed_bdg_free_energy(
        hsq,
        possq,
        dispsq,
        -0.7,
        interaction,
        density,
        Delta,
        A;
        q = qy,
        dir = 1,
        beta = beta,
    )
    function curvature(Delta, delta)
        return (
            free_energy(Delta, delta) - 2free_energy(Delta, 0.0) +
            free_energy(Delta, -delta)
        ) / delta^2
    end
    Fpp_SC = curvature(delta_field, 1e-3)
    Fpp_N = curvature(zero_delta, 1e-3)
    Fpp_SC_halfstep = curvature(delta_field, 5e-4)
    Fpp_N_halfstep = curvature(zero_delta, 5e-4)
    anchor = (2g_J / volume) * (Fpp_SC - Fpp_N)
    anchor_halfstep = (2g_J / volume) * (Fpp_SC_halfstep - Fpp_N_halfstep)
    stiffness_relerr = abs(r.Ds_over_pi_complete - anchor) / abs(anchor)

    println("free-energy curvatures delta=1e-3: SC=$(Fpp_SC), N=$(Fpp_N), anchor=$(anchor)")
    println(
        "free-energy curvatures delta=5e-4: SC=$(Fpp_SC_halfstep), N=$(Fpp_N_halfstep), anchor=$(anchor_halfstep)",
    )
    println(
        "complete stiffness: curvature anchor=$(anchor), KPM=$(r.Ds_over_pi_complete), relerr=$(stiffness_relerr)",
    )
    println(
        "paramagnetic-only stiffness=$(r.Ds_over_pi), diamagnetic correction=$(r.Dia_SC - r.Dia_N)",
    )
    TOL_CURVATURE = 2e-2
    @test isapprox(r.Ds_over_pi_complete, anchor; rtol = TOL_CURVATURE)
    @test abs(r.Ds_over_pi_complete - r.Ds_over_pi) > 1e-4

    normal_kernel = r.Dia_N - real(r.Pi_N)
    println(
        "normal-state Meissner check: Dia_N=$(r.Dia_N), Re(Pi_N)=$(real(r.Pi_N)), K_N=$(normal_kernel)",
    )

    r0 = KPM.superfluid_stiffness(
        op0,
        possq,
        qy;
        beta = beta,
        eta = 0.0,
        NC = 256,
        psi_in = psi_id,
        volume = volume,
        g_J = g_J,
        disp = dispsq,
        include_diamagnetic = true,
    )
    println(
        "zero-gap complete stiffness=$(r0.Ds_over_pi_complete), paramagnetic=$(r0.Ds_over_pi), diamagnetic correction=$(r0.Dia_SC - r0.Dia_N)",
    )
    @test abs(r0.Ds_over_pi_complete) < 1e-10
end
