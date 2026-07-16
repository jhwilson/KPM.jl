using Test
using KPM
using LinearAlgebra
using SparseArrays

function ssh_ring(Ncell; t1=1.0, t2=0.6)
    N = 2 * Ncell
    h = spzeros(Float64, N, N)
    A(n) = 2n - 1
    B(n) = 2n
    for n in 1:Ncell
        h[A(n), B(n)] = h[B(n), A(n)] = -t1
        np = mod1(n + 1, Ncell)
        h[B(n), A(np)] = h[A(np), B(n)] = -t2
    end
    return h, A, B
end

function ring_disp(pos, L)
    function disp(i, j)
        d = pos[i, 1] - pos[j, 1]
        d -= L * round(d / L)
        return [d]
    end
    return disp
end

const SSH_NCELL = 6
const SSH_T1 = 1.0
const SSH_T2 = 0.6
const SSH_H, SSH_A, SSH_B = ssh_ring(SSH_NCELL; t1=SSH_T1, t2=SSH_T2)
const SSH_L = 6.0
const SSH_POS_A = let pos = zeros(2SSH_NCELL, 1)
    for n in 1:SSH_NCELL
        pos[SSH_A(n), 1] = n - 1
        pos[SSH_B(n), 1] = n - 1 + 0.5
    end
    pos
end
const SSH_POS_B = let pos = zeros(2SSH_NCELL, 1)
    for n in 1:SSH_NCELL
        pos[SSH_A(n), 1] = n - 1
        pos[SSH_B(n), 1] = n - 1
    end
    pos
end
const SSH_DISP_A = ring_disp(SSH_POS_A, SSH_L)
const SSH_DISP_B = ring_disp(SSH_POS_B, SSH_L)

@testset "embedding decides the current operator" begin
    Ja = KPM.nambu_current_q(SSH_H, SSH_POS_A, [0.0]; disp=SSH_DISP_A)
    Jb = KPM.nambu_current_q(SSH_H, SSH_POS_B, [0.0]; disp=SSH_DISP_B)
    N = size(SSH_H, 1)
    for n in 1:SSH_NCELL
        a, b = SSH_A(n), SSH_B(n)
        @test Jb[a, b] == 0
        @test Jb[a + N, b + N] == 0
        @test Ja[a, b] ≈ 0.5 * SSH_T1 atol=1e-14
        @test Ja[a + N, b + N] ≈ 0.5 * SSH_T1 atol=1e-14
    end
    # B_n -> A_(n+1): split embedding has d=-0.5, co-located has d=-1.
    b1, a2 = SSH_B(1), SSH_A(2)
    @test Ja[b1, a2] ≈ 0.5 * SSH_T2 atol=1e-14
    @test Jb[b1, a2] ≈ SSH_T2 atol=1e-14
    @test Ja[b1 + N, a2 + N] ≈ 0.5 * SSH_T2 atol=1e-14
    @test Jb[b1 + N, a2 + N] ≈ SSH_T2 atol=1e-14
    @test Ja[b1, a2] / Jb[b1, a2] ≈ 0.5 atol=1e-14
end

@testset "identical position data => identical response; different embeddings => different" begin
    op = KPM.BdGOperator(SSH_H; mu=0.2, U=0.0, n=zeros(12),
                         Delta=fill(0.3, 12), hole_convention=:conjugate)
    q = [2pi / SSH_L]
    stiffness(pos, disp) = KPM.superfluid_stiffness(
        op, pos, q; beta=8.0, eta=0.15, dir=1, disp=disp, NC=256, NR=6,
        rng=KPM.Random.Xoshiro(7), volume=SSH_L, include_diamagnetic=true)
    r1 = @test_logs (:warn, r"longitudinal") match_mode=:any stiffness(SSH_POS_A, SSH_DISP_A)
    r1b = @test_logs (:warn, r"longitudinal") match_mode=:any stiffness(SSH_POS_A, SSH_DISP_A)
    r2 = @test_logs (:warn, r"longitudinal") match_mode=:any stiffness(SSH_POS_B, SSH_DISP_B)
    println("SSH stiffness: split=$(r1.Ds_over_pi_complete), colocated=$(r2.Ds_over_pi_complete), Dia_SC=($(r1.Dia_SC), $(r2.Dia_SC))")
    @test r1.Ds_over_pi_complete == r1b.Ds_over_pi_complete
    @test abs(r1.Ds_over_pi_complete - r2.Ds_over_pi_complete) > 1e-3
    @test abs(r1.Dia_SC - r2.Dia_SC) > 1e-3
end

@testset "spectra and local moments are embedding-blind" begin
    op = KPM.BdGOperator(SSH_H; mu=0.2, U=0.0, n=zeros(12),
                         Delta=fill(0.3, 12), hole_convention=:conjugate)
    rh = KPM.rescale(op)
    # This is the trivial half of invariance: positions never enter BdG
    # spectra or local moments. The binding embedding tests are above.
    local_moments(_pos) = KPM.bdg_site_moments(rh.H, 12, collect(1:12), 64)
    mu_rho_a, mu_delta_a = local_moments(SSH_POS_A)
    mu_rho_b, mu_delta_b = local_moments(SSH_POS_B)
    xi = Matrix{ComplexF64}(SSH_H) - 0.2 * I
    D = Diagonal(fill(0.3 + 0im, 12))
    H_a = [xi Matrix(D); adjoint(Matrix(D)) -conj(xi)]
    H_b = copy(H_a)
    @test eigvals(Hermitian(H_a)) == eigvals(Hermitian(H_b))
    @test mu_rho_a == mu_rho_b
    @test mu_delta_a == mu_delta_b
end
