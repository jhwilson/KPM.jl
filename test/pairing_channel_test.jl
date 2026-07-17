using Test
using KPM
using LinearAlgebra
using SparseArrays

function channel_dense(h, mu, U, n, D)
    N = size(h, 1)
    xi = Matrix{ComplexF64}(h) - mu * I - Diagonal(U .* n ./ 2)
    return [xi Matrix{ComplexF64}(D);
            adjoint(Matrix{ComplexF64}(D)) -conj(xi)]
end

function channel_chain(N; t=1.0, periodic=true)
    h = zeros(ComplexF64, N, N)
    bonds = [(i, i + 1) for i in 1:N-1]
    periodic && push!(bonds, (N, 1))
    for (i, j) in bonds
        h[i, j] = -t
        h[j, i] = -t
    end
    return sparse(h), bonds
end

function reconstruct_directed(mu_F, a, beta; Np=2size(mu_F, 1))
    NC = size(mu_F, 1)
    gh = KPM.JacksonKernel.(0:NC-1, NC) .* KPM.hn.(0:NC-1)
    nodes, _ = KPM.gausschebyshevt(Np)
    C = cos.((0:NC-1) .* acos.(nodes'))
    wf = KPM.fermiFunctions(0.0, beta).(a .* nodes) ./ Np
    return transpose(gh .* mu_F) * (C * wf)
end

@testset "two-site analytic pairing-channel fixed point" begin
    t = 1.0
    mu = 0.3
    beta = 8.0
    V = 3.0
    equation(R) = V / (4R) *
        (tanh(beta * (R + t) / 2) + tanh(beta * (R - t) / 2)) - 1
    # The larger root is locally stable but thermodynamically METASTABLE:
    # Omega(Delta*) = -0.7826 > Omega(0) = -1.0005 for
    # Omega(Delta) = Delta^2/V - (1/beta)[ln 2cosh(beta(R+t)/2) +
    # ln 2cosh(beta(R-t)/2)]. This gate pins the fixed-point algebra against
    # a closed form, not the phase diagram. Bracket that paired branch.
    grid = range(abs(mu) + 1e-9, V; length=10_001)
    last_positive = findlast(R -> equation(R) > 0, grid)
    @test last_positive !== nothing
    lo = grid[last_positive]
    hi = grid[last_positive + 1]
    for _ in 1:200
        mid = (lo + hi) / 2
        if equation(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    R = (lo + hi) / 2
    delta_exact = sqrt(R^2 - mu^2)

    h = sparse(ComplexF64[0 -t; -t 0])
    channel = KPM.PairingChannel([(1, 2)], 1.0, V, :odd)
    D = KPM.pairing_matrix(2, [channel]; amplitude=delta_exact)
    H = channel_dense(h, mu, zeros(2), zeros(2), D)
    eig = eigen(Hermitian(H))
    occ = KPM.fermiFunctions(0.0, beta).(eig.values)
    F12 = sum(eig.vectors[1, s] * conj(eig.vectors[4, s]) * occ[s]
              for s in eachindex(eig.values))
    F_closed = -delta_exact / (4R) *
        (tanh(beta * (R + t) / 2) + tanh(beta * (R - t) / 2))
    @test F12 ≈ F_closed atol=1e-12
    @test -V * F12 ≈ delta_exact atol=1e-10

    op = KPM.BdGOperator(h; mu=mu, U=0.0, n=zeros(2), D=D,
                         hole_convention=:conjugate)
    rh = KPM.rescale(op)
    errors = Float64[]
    for NC in (128, 512)
        moments = KPM.bdg_channel_moments(rh, [channel]; NC=NC, g_rho=1)
        _, amplitudes = KPM.bdg_update(moments; beta=beta)
        push!(errors, abs(only(only(amplitudes)) - delta_exact))
    end
    println("two-site channel errors: NC=128 $(errors[1]), NC=512 $(errors[2])")
    @test errors[2] < errors[1]
    # The mandated Jackson path at the default eps=0.2 radial scale gives
    # 2.9e-4 for this four-level spectrum at NC=512.
    @test errors[2] < 3e-4
end

@testset "grand-potential stationarity (independent variational gate)" begin
    function Omega(h, mu, beta, amps, channel)
        D = KPM.pairing_matrix(size(h, 1), [channel]; amplitudes=[amps])
        H = channel_dense(h, mu, zeros(size(h, 1)), zeros(size(h, 1)), D)
        interaction = sum(abs2(amps[b]) / channel.V[b]
                          for b in eachindex(amps))
        quasiparticles = sum(log(2cosh(beta * E / 2))
                             for E in eigvals(Hermitian(H)))
        return interaction - quasiparticles / (2beta)
    end

    function finite_gradient(omega, amps; step=1e-5)
        gradient = Float64[]
        for b in eachindex(amps), direction in (1.0, 1.0im)
            plus = copy(amps)
            minus = copy(amps)
            plus[b] += step * direction
            minus[b] -= step * direction
            push!(gradient, (omega(plus) - omega(minus)) / (2step))
        end
        return gradient
    end

    t = 1.0
    mu = 0.3
    beta = 8.0
    V = 3.0
    equation(R) = V / (4R) *
        (tanh(beta * (R + t) / 2) + tanh(beta * (R - t) / 2)) - 1
    grid = range(abs(mu) + 1e-9, V; length=10_001)
    last_positive = findlast(R -> equation(R) > 0, grid)
    lo = grid[last_positive]
    hi = grid[last_positive + 1]
    for _ in 1:200
        mid = (lo + hi) / 2
        if equation(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    delta_exact = sqrt(((lo + hi) / 2)^2 - mu^2)
    h_two = sparse(ComplexF64[0 -t; -t 0])
    channel_two = KPM.PairingChannel([(1, 2)], 1.0, V, :odd)
    gradient_two = finite_gradient(
        amps -> Omega(h_two, mu, beta, amps, channel_two),
        ComplexF64[delta_exact])
    @test maximum(abs, gradient_two) < 1e-6

    N = 6
    mu_ring = 0.2
    h_ring, bonds = channel_chain(N)
    channel_ring = KPM.PairingChannel(bonds, 1.0, V, :odd)
    amplitudes_ed = fill(0.3 + 0im, length(bonds))
    residual_ed = Inf
    for _ in 1:10_000
        D = KPM.pairing_matrix(N, [channel_ring]; amplitudes=[amplitudes_ed])
        eig = eigen(Hermitian(
            channel_dense(h_ring, mu_ring, zeros(N), zeros(N), D)))
        occ = KPM.fermiFunctions(0.0, beta).(eig.values)
        amplitudes_new = similar(amplitudes_ed)
        for (b, (i, j)) in pairs(bonds)
            Fij = sum(eig.vectors[i, s] * conj(eig.vectors[j + N, s]) * occ[s]
                      for s in eachindex(eig.values))
            Fji = sum(eig.vectors[j, s] * conj(eig.vectors[i + N, s]) * occ[s]
                      for s in eachindex(eig.values))
            amplitudes_new[b] = -V * (Fij - Fji) / 2
        end
        residual_ed = norm(amplitudes_new .- amplitudes_ed, Inf)
        @. amplitudes_ed = 0.7 * amplitudes_ed + 0.3 * amplitudes_new
        residual_ed < 1e-11 && break
    end
    @test residual_ed < 1e-11

    # This functional never touches the implementation's F extraction or
    # symmetrization, so a sign, parity, or factor-of-two gap-equation error
    # would break stationarity.
    omega_ring = amps -> Omega(h_ring, mu_ring, beta, amps, channel_ring)
    gradient_ring = finite_gradient(omega_ring, amplitudes_ed)
    @test maximum(abs, gradient_ring) < 1e-6
    displaced_gradient = finite_gradient(omega_ring, 1.3 .* amplitudes_ed)
    @test norm(displaced_gradient) > 1e-3
end

@testset "bond moments vs exact eigenvector sums" begin
    N = 4
    mu = -0.3
    beta = 10.0
    delta_p = 0.25 + 0.35im
    h, bonds = channel_chain(N)
    channel = KPM.PairingChannel(bonds, 1.0, 1.0, :odd)
    D = KPM.pairing_matrix(N, [channel];
                           amplitudes=[fill(delta_p, length(bonds))])
    H = channel_dense(h, mu, zeros(N), zeros(N), D)
    eig = eigen(Hermitian(H))
    occ = KPM.fermiFunctions(0.0, beta).(eig.values)
    directed = Tuple{Int, Int}[]
    for (i, j) in bonds
        push!(directed, (i, j), (j, i))
    end
    exact = ComplexF64[
        sum(eig.vectors[i, s] * conj(eig.vectors[j + N, s]) * occ[s]
            for s in eachindex(eig.values)) for (i, j) in directed]

    op = KPM.BdGOperator(h; mu=mu, U=0.0, n=zeros(N), D=D,
                         hole_convention=:conjugate)
    rh = KPM.rescale(op)
    errors = Float64[]
    reconstructed = nothing
    for NC in (64, 256)
        _, mu_F = KPM.bdg_channel_moments(
            rh.H, N, collect(1:N), directed, NC; batch_size=3)
        F = reconstruct_directed(mu_F, rh.a, beta)
        push!(errors, maximum(abs.(F .- exact)))
        NC == 256 && (reconstructed = F)
    end
    println("bond moment errors: NC=64 $(errors[1]), NC=256 $(errors[2])")
    @test errors[2] < 1e-3
    @test errors[1] > errors[2]
    for b in 1:2:length(directed)
        @test reconstructed[b + 1] ≈ -reconstructed[b] atol=1e-3
    end
end

@testset "Kitaev channel SCF vs ED and U(1) covariance" begin
    N = 6
    mu = 0.2
    beta = 8.0
    V = 3.0
    h, bonds = channel_chain(N)
    channel = KPM.PairingChannel(bonds, 1.0, V, :odd)

    amplitudes_ed = fill(0.3 + 0im, length(bonds))
    residual_ed = Inf
    for _ in 1:10_000
        D = KPM.pairing_matrix(N, [channel]; amplitudes=[amplitudes_ed])
        eig = eigen(Hermitian(channel_dense(h, mu, zeros(N), zeros(N), D)))
        occ = KPM.fermiFunctions(0.0, beta).(eig.values)
        amplitudes_new = similar(amplitudes_ed)
        for (b, (i, j)) in pairs(bonds)
            Fij = sum(eig.vectors[i, s] * conj(eig.vectors[j + N, s]) * occ[s]
                      for s in eachindex(eig.values))
            Fji = sum(eig.vectors[j, s] * conj(eig.vectors[i + N, s]) * occ[s]
                      for s in eachindex(eig.values))
            amplitudes_new[b] = -V * (Fij - Fji) / 2
        end
        residual_ed = norm(amplitudes_new .- amplitudes_ed, Inf)
        @. amplitudes_ed = 0.7 * amplitudes_ed + 0.3 * amplitudes_new
        residual_ed < 1e-11 && break
    end
    @test residual_ed < 1e-11

    function solve_seed(seed)
        D = KPM.pairing_matrix(N, [channel]; amplitude=seed)
        op = KPM.BdGOperator(h; mu=mu, U=0.0, n=zeros(N), D=D,
                             hole_convention=:conjugate)
        result = KPM.bdg_solve!(op, [channel]; beta=beta, NC=512,
                                g_rho=1, mix=0.3, update_density=false,
                                tol_delta=1e-8, maxiter=800)
        return op, result, ComplexF64[op.D[i, j] for (i, j) in bonds]
    end

    op, result, amplitudes = solve_seed(0.3)
    @test result.converged
    @test maximum(abs.(abs.(amplitudes) .- sum(abs, amplitudes) / N)) < 1e-6
    @test abs(amplitudes[1]) ≈ abs(amplitudes_ed[1]) rtol=2e-3

    _, phase_result, phase_amplitudes = solve_seed(0.3exp(0.7im))
    @test phase_result.converged
    @test abs.(phase_amplitudes) ≈ abs.(amplitudes) atol=1e-6 rtol=0
    phase_errors = angle.(exp.(im .* (angle.(phase_amplitudes) .-
                                      angle.(amplitudes) .- 0.7)))
    @test maximum(abs, phase_errors) < 1e-3
end

@testset "open-chain Majorana check" begin
    N = 40
    h, bonds = channel_chain(N; periodic=false)
    channel = KPM.PairingChannel(bonds, 1.0, 1.0, :odd)
    D = KPM.pairing_matrix(N, [channel]; amplitude=0.4)
    for mu in (0.5, 3.0)
        values = sort(abs.(eigvals(Hermitian(
            channel_dense(h, mu, zeros(N), zeros(N), D)))))
        if mu == 0.5
            @test values[1] < 1e-3
            @test values[2] < 1e-3
            @test values[3] > 0.1
        else
            @test values[1] > 0.3
        end
    end
end

@testset "legacy onsite equivalence" begin
    N = 4
    h, _ = channel_chain(N)
    U = fill(2.0, N)
    n0 = fill(0.4, N)
    delta0 = fill(0.25 + 0.35im, N)
    channel = KPM.PairingChannel([(i, i) for i in 1:N], 1.0, U, :even)
    op_legacy = KPM.BdGOperator(h; mu=-0.3, U=U, n=n0, Delta=delta0)
    D = KPM.pairing_matrix(N, [channel]; amplitudes=[delta0])
    op_channel = KPM.BdGOperator(h; mu=-0.3, U=U, n=n0, D=D,
                                 hole_convention=:conjugate)
    result_legacy = KPM.bdg_solve!(op_legacy; beta=10.0, NC=256,
                                   mix=0.5, tol_delta=1e-8,
                                   tol_n=1e-8, maxiter=600)
    result_channel = KPM.bdg_solve!(op_channel, [channel]; beta=10.0,
                                    NC=256, mix=0.5, tol_delta=1e-8,
                                    tol_n=1e-8, maxiter=600)
    @test result_legacy.converged
    @test result_channel.converged
    channel_delta = ComplexF64[op_channel.D[i, i] for i in 1:N]
    @test abs.(channel_delta) ≈ abs.(op_legacy.Δ) atol=1e-8 rtol=0
    @test op_channel.n ≈ op_legacy.n atol=1e-8 rtol=0

    rh = KPM.rescale(op_channel)
    sites = collect(1:N)
    onsite = [(i, i) for i in sites]
    local_rho, local_F = KPM.bdg_site_moments(rh.H, N, sites, 64;
                                               batch_size=3)
    channel_rho, channel_F = KPM.bdg_channel_moments(
        rh.H, N, sites, onsite, 64; batch_size=3)
    @test local_rho == channel_rho
    @test local_F == channel_F
end

@testset "pairing-channel checkpoint v3" begin
    N = 4
    h, bonds = channel_chain(N)
    channel = KPM.PairingChannel(bonds, 1.0, 3.0, :odd)
    make_op() = KPM.BdGOperator(
        h; mu=0.2, U=0.0, n=zeros(N),
        D=KPM.pairing_matrix(N, [channel]; amplitude=0.3),
        hole_convention=:conjugate)
    mktempdir() do dir
        checkpoint = joinpath(dir, "channel-checkpoint.bin")
        op_a = make_op()
        KPM.bdg_solve!(op_a, [channel]; beta=8.0, NC=256, g_rho=1,
                       mix=0.3, update_density=false, tol_delta=1e-14,
                       maxiter=5, checkpoint_path=checkpoint,
                       checkpoint_every=5)
        state = open(KPM.deserialize, checkpoint)
        @test state.version == 3
        @test state.delta === nothing
        @test length(state.amplitudes) == 1
        @test state.hole_convention === :conjugate
        @test length(state.D_triplets.I) == nnz(op_a.D)

        op_restart = make_op()
        KPM.bdg_solve!(op_restart, [channel]; beta=8.0, NC=256,
                       g_rho=1, mix=0.3, update_density=false,
                       tol_delta=1e-14, maxiter=5, restart=checkpoint)
        op_full = make_op()
        KPM.bdg_solve!(op_full, [channel]; beta=8.0, NC=256, g_rho=1,
                       mix=0.3, update_density=false, tol_delta=1e-14,
                       maxiter=10)
        if KPM.whichcore()
            # GPU kernels are not guaranteed bitwise-deterministic; the
            # bitwise checkpoint/restart contract holds on the CPU path.
            @test nonzeros(op_restart.D) ≈ nonzeros(op_full.D) atol=1e-10 rtol=0
            @test op_restart.n ≈ op_full.n atol=1e-10 rtol=0
        else
            @test nonzeros(op_restart.D) == nonzeros(op_full.D)
            @test op_restart.n == op_full.n
        end

        mismatch = KPM.PairingChannel(bonds, 1.0, 2.5, :odd)
        @test_throws ArgumentError KPM.bdg_restore!(make_op(), [mismatch], checkpoint)
        weight_mismatch = KPM.PairingChannel(bonds, 2.0, 3.0, :odd)
        @test_throws ArgumentError KPM.bdg_restore!(make_op(),
                                                    [weight_mismatch], checkpoint)

        h_fixed, _ = channel_chain(3; periodic=false)
        fixed_channel = KPM.PairingChannel([(1, 2)], 1.0, 3.0, :odd)
        D_fixed = KPM.pairing_matrix(3, [fixed_channel]; amplitude=0.2)
        D_fixed[1, 3] = 0.7
        D_fixed[3, 1] = -0.7
        fixed_op = KPM.BdGOperator(
            h_fixed; mu=0.2, U=0.0, n=zeros(3), D=D_fixed,
            hole_convention=:conjugate)
        fixed_checkpoint = joinpath(dir, "fixed-D-checkpoint.bin")
        KPM.bdg_checkpoint(fixed_checkpoint, fixed_op, [fixed_channel],
                           [ComplexF64[0.2]], NamedTuple[], nothing,
                           (beta=8.0,))
        D_wrong = KPM.pairing_matrix(3, [fixed_channel]; amplitude=0.2)
        D_wrong[1, 3] = -4.0
        D_wrong[3, 1] = 4.0
        wrong_op = KPM.BdGOperator(
            h_fixed; mu=0.2, U=0.0, n=zeros(3), D=D_wrong,
            hole_convention=:conjugate)
        @test_throws ArgumentError KPM.bdg_restore!(
            wrong_op, [fixed_channel], fixed_checkpoint)

        legacy = KPM.BdGOperator(h; mu=0.2, U=0.0, n=zeros(N),
                                 Delta=fill(0.3 + 0im, N))
        @test_throws ArgumentError KPM.bdg_restore!(legacy, checkpoint)
        legacy_checkpoint = joinpath(dir, "legacy-checkpoint.bin")
        KPM.bdg_checkpoint(legacy_checkpoint, legacy, NamedTuple[], nothing,
                           (beta=8.0,))
        @test_throws ArgumentError KPM.bdg_restore!(make_op(), [channel],
                                                    legacy_checkpoint)
    end
end

@testset "pairing-channel API validation" begin
    @test_throws ArgumentError KPM.PairingChannel([(1, 2), (2, 1)], 1, 1, :even)
    @test_throws ArgumentError KPM.PairingChannel([(1, 1)], 1, 1, :odd)
    scalar = KPM.PairingChannel([(1, 2), (2, 3)], 2im, 3, :odd)
    @test scalar.weights == fill(2im, 2)
    @test scalar.V == fill(3.0, 2)
    amplitudes = (0.4 - 0.2im) .* scalar.weights
    @test KPM.channel_amplitude(scalar, amplitudes) ≈ 0.4 - 0.2im

    c1 = KPM.PairingChannel([(1, 2)], 1, 1, :odd)
    c2 = KPM.PairingChannel([(2, 1)], 1, 1, :odd)
    @test nnz(KPM.pairing_matrix(2, [c1]; amplitude=0.0)) == 2
    @test_throws ArgumentError KPM.pairing_matrix(2, [c1, c2]; amplitude=0.2)

    oriented_bonds = [(1, 2), (2, 3), (3, 1)]
    reversed_bonds = [(2, 1), (2, 3), (3, 1)]
    oriented = KPM.PairingChannel(oriented_bonds, ones(3), 3.0, :odd)
    reversed = KPM.PairingChannel(reversed_bonds, [-1.0, 1.0, 1.0],
                                  3.0, :odd)
    oriented_amplitudes = ComplexF64[0.3, 0.4im, -0.2 + 0.1im]
    reversed_amplitudes = copy(oriented_amplitudes)
    reversed_amplitudes[1] *= -1
    D_oriented = KPM.pairing_matrix(
        3, [oriented]; amplitudes=[oriented_amplitudes])
    D_reversed = KPM.pairing_matrix(
        3, [reversed]; amplitudes=[reversed_amplitudes])
    @test Matrix(D_reversed) == Matrix(D_oriented)
    @test Matrix(KPM.pairing_matrix(3, [reversed]; amplitude=0.3)) ==
          Matrix(KPM.pairing_matrix(3, [oriented]; amplitude=0.3))

    h = sparse(ComplexF64[0 -1; -1 0])
    overlap_D = sparse(ComplexF64[0 0.2; -0.2 0])
    overlap_op = KPM.BdGOperator(h; mu=0.0, U=0.0, D=overlap_D,
                                 hole_convention=:conjugate)
    @test_throws ArgumentError KPM.bdg_solve!(overlap_op, [c1, c2]; beta=1.0)
    intervalley_op = KPM.BdGOperator(h; mu=0.0, U=0.0, D=overlap_D,
                                     hole_convention=:intervalley)
    @test_throws ArgumentError KPM.bdg_solve!(intervalley_op, [c1]; beta=1.0)

    missing_D = sparse([1], [2], ComplexF64[0.2], 2, 2)
    missing_op = KPM.BdGOperator(h; mu=0.0, U=0.0, D=missing_D,
                                 hole_convention=:conjugate)
    @test_throws ArgumentError KPM.bdg_solve!(missing_op, [c1]; beta=1.0)

    N = 6
    h_ring, ring_bonds = channel_chain(N)
    reversed_ring_bonds = copy(ring_bonds)
    reversed_ring_bonds[1] = reverse(reversed_ring_bonds[1])
    ring_channel = KPM.PairingChannel(ring_bonds, ones(N), 3.0, :odd)
    reversed_ring_weights = ones(N)
    reversed_ring_weights[1] = -1
    reversed_ring_channel = KPM.PairingChannel(
        reversed_ring_bonds, reversed_ring_weights, 3.0, :odd)
    ring_seed = fill(0.3 + 0im, N)
    reversed_ring_seed = copy(ring_seed)
    reversed_ring_seed[1] *= -1
    ring_op = KPM.BdGOperator(
        h_ring; mu=0.2, U=0.0, n=zeros(N),
        D=KPM.pairing_matrix(N, [ring_channel]; amplitudes=[ring_seed]),
        hole_convention=:conjugate)
    reversed_ring_op = KPM.BdGOperator(
        h_ring; mu=0.2, U=0.0, n=zeros(N),
        D=KPM.pairing_matrix(N, [reversed_ring_channel];
                             amplitudes=[reversed_ring_seed]),
        hole_convention=:conjugate)
    ring_result = KPM.bdg_solve!(
        ring_op, [ring_channel]; beta=8.0, NC=512, g_rho=1, mix=0.3,
        update_density=false, tol_delta=1e-8, maxiter=800)
    reversed_ring_result = KPM.bdg_solve!(
        reversed_ring_op, [reversed_ring_channel]; beta=8.0, NC=512,
        g_rho=1, mix=0.3, update_density=false, tol_delta=1e-8,
        maxiter=800)
    @test ring_result.converged
    @test reversed_ring_result.converged
    @test Matrix(reversed_ring_op.D) ≈ Matrix(ring_op.D) atol=1e-8 rtol=0

    filling_h = sparse(reshape(ComplexF64[0.5], 1, 1))
    filling_op = KPM.BdGOperator(
        filling_h; mu=0.0, U=0.0, n=zeros(1), Delta=zeros(ComplexF64, 1))
    @test_throws ArgumentError KPM.bdg_solve!(
        filling_op; beta=1.0, NC=32, update_density=false,
        target_filling=0.7, mu_bracket=(-4.0, 4.0))
    exhausted_op = KPM.BdGOperator(
        filling_h; mu=0.0, U=0.0, n=zeros(1), Delta=zeros(ComplexF64, 1))
    exhaustion_error = try
        KPM.bdg_solve!(
            exhausted_op; beta=1.0, NC=64, mix=1.0, tol_delta=1e-12,
            tol_n=1e-12, maxiter=10, target_filling=0.7,
            mu_bracket=(-4.0, 4.0), mu_tol=1e-8, mu_maxiter=1)
        nothing
    catch err
        err
    end
    @test exhaustion_error isa ErrorException
    @test occursin("target-filling bisection did not converge",
                   sprint(showerror, exhaustion_error))
end
