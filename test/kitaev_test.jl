using Test
using KPM
using LinearAlgebra
using SparseArrays

function bdg_dense(h, mu, U, n, D; hole_convention)
    N = size(h, 1)
    ξ = Matrix{ComplexF64}(h) - mu * I - Diagonal(U .* n ./ 2)
    hole = hole_convention === :conjugate ? -conj(ξ) : -ξ
    return Matrix{ComplexF64}(
        [
            ξ Matrix{ComplexF64}(D);
            adjoint(Matrix{ComplexF64}(D)) hole
        ],
    )
end

function kitaev_ring(N; t = 1.0, Delta_p = 0.3)
    h = zeros(ComplexF64, N, N)
    D = zeros(ComplexF64, N, N)
    for i = 1:N
        j = mod1(i + 1, N)
        h[i, j] = -t
        h[j, i] = -t
        D[i, j] = Delta_p
        D[j, i] = -Delta_p
    end
    return sparse(h), sparse(D)
end

@testset "Kitaev pairing operator action" begin
    N = 6
    h, D = kitaev_ring(N; t = 1.0, Delta_p = 0.3)
    mu = 0.4
    U = zeros(N)
    n = zeros(N)
    op = KPM.BdGOperator(h; mu = mu, U = U, n = n, D = D, hole_convention = :conjugate)
    Hd = bdg_dense(h, mu, U, n, D; hole_convention = :conjugate)

    x = ComplexF64[sin(i) + im * cos(2i) for i = 1:2N]
    y = ComplexF64[cos(i / 2) - im * sin(3i) for i = 1:2N]
    α = 1.3 - 0.2im
    β = 0.7 + 0.1im
    @test mul!(copy(y), op, x, α, β) ≈ α * Hd * x + β * y atol=1e-12
    @test mul!(copy(y), op, x) ≈ Hd * x atol=1e-12

    A = sparse(
        ComplexF64[
            0 0.2 + 0.4im 0 0 0 0;
            0 0 0.3 - 0.2im 0 0 0;
            0 0 0 0 0.1 + 0.5im 0;
            0 0 0 0 0 0.6 - 0.1im;
            0.2im 0 0 0 0 0;
            0 0 0.4 - 0.3im 0 0 0
        ],
    )
    h_complex = sparse((A + A') / 2)
    D_complex = sparse(
        ComplexF64[
            0 0.1 + 0.2im 0 0 0 0.3im;
            -0.4im 0 0.5 0 0 0;
            0 0.2im 0 0.1 0 0;
            0 0 0.3im 0 0.4 - 0.2im 0;
            0.2 0 0 0 0 0.1im;
            0 0.3 0 0.2im 0 0
        ],
    )
    U_complex = [0.2, 0.4, 0.1, 0.3, 0.6, 0.5]
    n_complex = [0.7, 0.2, 0.9, 0.4, 0.1, 0.6]
    op_complex = KPM.BdGOperator(
        h_complex;
        mu = 0.17,
        U = U_complex,
        n = n_complex,
        D = D_complex,
        hole_convention = :conjugate,
    )
    Hd_complex = bdg_dense(
        h_complex,
        0.17,
        U_complex,
        n_complex,
        D_complex;
        hole_convention = :conjugate,
    )
    @test mul!(copy(y), op_complex, x, α, β) ≈ α * Hd_complex * x + β * y atol=1e-12
    @test mul!(copy(y), op_complex, x) ≈ Hd_complex * x atol=1e-12
end

@testset "Kitaev parity-resolved particle-hole symmetry" begin
    N = 6
    h, D_odd = kitaev_ring(N; t = 1.0, Delta_p = 0.3)
    U = zeros(N)
    n = zeros(N)
    Hd = bdg_dense(h, 0.4, U, n, D_odd; hole_convention = :conjugate)
    τx = [zeros(N, N) I; I zeros(N, N)]
    τy = [zeros(N, N) -im * I; im * I zeros(N, N)]
    @test τx * conj(Hd) * τx ≈ -Hd atol=1e-12
    ev_odd = sort(eigvals(Hermitian(Hd)))
    @test ev_odd ≈ -reverse(ev_odd) atol=1e-10

    D_even = Matrix{ComplexF64}(D_odd)
    D_even .= abs.(D_even)
    Hd_even = bdg_dense(h, 0.4, U, n, D_even; hole_convention = :conjugate)
    @test τy * conj(Hd_even) * τy ≈ -Hd_even atol=1e-12

    # A uniform real onsite component has an accidental momentum-block ±E
    # pairing on this translation-invariant ring. Break that special case so
    # this gate exercises generic mixed parity without either PH symmetry.
    D_mixed = Diagonal(ComplexF64[0.25, 0.0, 0.5, 0.1, 0.4, 0.05]) + D_odd
    Hd_mixed = bdg_dense(h, 0.4, U, n, D_mixed; hole_convention = :conjugate)
    ev_mixed = sort(eigvals(Hermitian(Hd_mixed)))
    @test maximum(abs.(ev_mixed .+ reverse(ev_mixed))) > 1e-3

    op_mixed = KPM.BdGOperator(
        h;
        mu = 0.4,
        U = U,
        n = n,
        D = D_mixed,
        hole_convention = :conjugate,
    )
    rad = KPM.spectral_radius(op_mixed)[1]
    @test rad > 0
    Hs = KPM.ScaledOperator(op_mixed, 2 * rad / (2 - 0.2), 0.0)
    @test KPM.chebyshev_stability_probe(Hs, 2N, 256) <= 1.5
end

@testset "Kitaev ring dispersion" begin
    N = 64
    t = 1.0
    mu = 0.5
    Delta_p = 0.3
    h, D = kitaev_ring(N; t = t, Delta_p = Delta_p)
    Hd = bdg_dense(h, mu, zeros(N), zeros(N), D; hole_convention = :conjugate)
    k = 2π .* (0:(N-1)) ./ N
    E = sqrt.((2t .* cos.(k) .+ mu) .^ 2 .+ 4Delta_p^2 .* sin.(k) .^ 2)
    @test eigvals(Hermitian(Hd)) ≈ sort(vcat(E, -E)) atol=1e-10
end

@testset "BdG pairing operator API contract" begin
    N = 4
    h, D = kitaev_ring(N; Delta_p = 0.2)
    Delta = ComplexF64[0.1, 0.2im, -0.3, 0.4-0.1im]
    op_onsite = KPM.BdGOperator(h; mu = 0.0, U = 0.0, Delta = Delta)
    @test op_onsite.D isa Diagonal
    @test op_onsite.Δ == Delta
    @test_throws ArgumentError KPM.BdGOperator(h; mu = 0.0, U = 0.0, Delta = Delta, D = D)
    h_bound = sparse(1.0I, 2, 2)
    h_hole_bound = sparse(100.0I, 2, 2)
    @test_throws ArgumentError KPM.BdGOperator(
        h_bound;
        mu = 0.0,
        U = 0.0,
        h_hole = h_hole_bound,
        hole_convention = :conjugate,
    )

    op_bond = KPM.BdGOperator(h; mu = 0.0, U = 0.0, D = D, hole_convention = :conjugate)
    @test_throws ArgumentError op_bond.Δ
    @test_throws ArgumentError KPM.bdg_solve!(op_bond; beta = 1.0)
    op_alias =
        KPM.BdGOperator(h; mu = 0.0, U = 0.0, Delta = Delta, hole_convention = :singlet)
    @test op_alias.hole_convention === :conjugate

    pos = reshape(collect(0.0:3.0), :, 1)
    q = [0.0]
    @test KPM.nambu_current_q(h, pos, q; hole_convention = :conjugate) ==
          KPM.nambu_current_q(h, pos, q; hole_convention = :singlet)
end
