"""
    ScaledOperator(op, a, b)

Lazy matrix-free representation of `(op - b I) / a` for duck-typed
operators. This is used where `normalizeH`, which requires `ishermitian` and
Arpack `eigs`, cannot be applied.
"""
struct ScaledOperator{TO}
    op::TO
    a::Float64
    b::Float64
end

Base.size(S::ScaledOperator) = size(S.op)
Base.size(S::ScaledOperator, d::Int) = size(S.op, d)
Base.eltype(S::ScaledOperator) = promote_type(eltype(S.op), Float64)

function LinearAlgebra.mul!(Y::AbstractVecOrMat, S::ScaledOperator,
                            X::AbstractVecOrMat, α::Number, β::Number)
    mul!(Y, S.op, X, α / S.a, β)
    iszero(S.b) || (@. Y -= (α * S.b / S.a) * X)
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, S::ScaledOperator, X::AbstractVecOrMat) =
    mul!(Y, S, X, true, false)

# Workspaces follow the residence of the wrapped operator.
to_device_of(S::ScaledOperator, x) = to_device_of(S.op, x)
device_zeros_of(S::ScaledOperator, T::Type, dims...) = device_zeros_of(S.op, T, dims...)

"""
    spectral_radius(op; tol=1e-4, maxiter=300, miniter=32, restarts=2,
                    rng=Xoshiro(0), v0=nothing) -> (radius, v)

Estimate the spectral radius of a Hermitian matrix-free operator with norm-ratio
power iteration. The norm ratio converges to the spectral radius even when the
extremal eigenvalues occur as a `±Emax` pair, as in BdG systems. The default
deterministic RNG makes restarts reproducible, and `v0` enables a warm-started
primary run. No run exits for convergence before `miniter`, and `restarts`
additional runs use fresh random vectors. The maximum estimate and its vector
are returned. This remains a lower estimate; see [`rescale(::BdGOperator)`](@ref)
for certified bounds and runtime guards.

`op` may be device-resident (e.g. the assembled BdG matrix moved to GPU); the
iteration then runs on the device. `v0` must always be a host vector, and the
returned vector lives on the operator's device (`maybe_to_host` it before
serializing).
"""
function spectral_radius(op; tol=1e-4, maxiter=300, miniter::Integer=32,
                         restarts::Integer=2, rng=Xoshiro(0), v0=nothing)
    N = size(op, 1)
    maxiter > 0 || throw(ArgumentError("spectral_radius: maxiter must be positive (got $maxiter)"))
    1 <= miniter <= maxiter ||
        throw(ArgumentError("spectral_radius: miniter must satisfy 1 <= miniter <= maxiter (got miniter=$miniter, maxiter=$maxiter)"))
    restarts >= 0 || throw(ArgumentError("spectral_radius: restarts must be nonnegative (got $restarts)"))

    function power_run(v_start)
        # `to_device_of` keeps the iteration vector on the operator's device;
        # `v_start` itself is always a host vector (driver and RNG contract).
        v = to_device_of(op, collect(ComplexF64, v_start))
        length(v) == N || throw(ArgumentError("spectral_radius: v0 has length $(length(v)); expected $N"))
        nv = norm(v)
        iszero(nv) && throw(ArgumentError("spectral_radius: v0 must be nonzero"))
        v ./= nv
        w = similar(v)
        λ_prev = Inf

        for iter in 1:maxiter
            mul!(w, op, v)
            λ = norm(w)
            iszero(λ) && return 0.0, v
            v .= w ./ λ
            if iter >= miniter && abs(λ - λ_prev) <= tol * max(λ, eps())
                return Float64(λ), v
            end
            λ_prev = λ
        end

        @debug "spectral_radius run reached maxiter without convergence" maxiter miniter tol λ=λ_prev
        return Float64(λ_prev), v
    end

    primary = v0 === nothing ? randn(rng, ComplexF64, N) : v0
    radius_best, v_best = power_run(primary)
    for _ in 1:restarts
        radius, v = power_run(randn(rng, ComplexF64, N))
        if radius > radius_best
            radius_best, v_best = radius, v
        end
    end
    return radius_best, v_best
end

"""
    BdGOperator(h; mu, U, n=zeros(size(h, 1)),
                Delta=nothing, D=nothing,
                hole_convention=:intervalley, h_hole=nothing,
                assume_intervalley=false)

Matrix-free Nambu BdG operator with particle-hole layout `[particle; hole]`
and hole index `i + N`:

    H_BdG = [ ξ          D       ]
            [ adjoint(D) hole    ]

with `ξ = h - μ I - Diagonal(U n / 2)`.

Here `U > 0` is attractive, with `H_int = -U Σ n↑n↓`, Hartree shift
`-(U/2)n`. A matrix-free `D` must support `size` and five-argument `mul!` for
both `D` and `adjoint(D)`; the `Delta` keyword remains as an onsite
compatibility shorthand and constructs `Diagonal(Delta)`. The matrix-free
action itself is host-only; with a CUDA GPU active, the BdG compute paths
instead run on the assembled sparse matrix ([`bdg_assemble`](@ref)) moved to
the device, which requires assembled matrix blocks — operators with
matrix-free blocks stay on the host. This reduced convention supports two
hole-block conventions:

  * `hole_convention=:intervalley` (the default) uses `hole = -ξ`, hence the
    same `h` in both blocks, and presumes `h_{-K}^* = h_K`. For matrix inputs a
    complex Hermitian `h` is rejected unless `assume_intervalley=true`
    explicitly asserts that it is the intervalley-identified operator. For
    complex `h` with a nonuniform gap phase its spectrum need not be
    particle-hole symmetric; the package's `b=0` rescaling remains a safe
    radial bound, not a symmetry statement.
  * `hole_convention=:conjugate` is the standard same-valley
    `(c_up, c_down^dagger)` convention and uses `hole = -conj(ξ)`. It obeys the
    exact particle-hole symmetry `tau_y * conj(H) * tau_y = -H` when
    `transpose(D) == D`, or `tau_x * conj(H) * tau_x = -H` when
    `transpose(D) == -D`; mixed-parity `D` has neither exact symmetry.
    `:singlet` remains an accepted alias. For an assembled matrix the
    conjugated hole operator is built automatically. Matrix-free callers must
    supply it as `h_hole`.

The conventions coincide identically for real-symmetric `h`.
"""
mutable struct BdGOperator{TH, THH, TD}
    const h::TH
    const h_hole::THH
    μ::Float64
    const U::Vector{Float64}
    const n::Vector{Float64}
    const D::TD
    const N::Int
    const hole_convention::Symbol
end

function Base.getproperty(B::BdGOperator, s::Symbol)
    if s === :Δ
        D = getfield(B, :D)
        D isa Diagonal ||
            throw(ArgumentError("BdGOperator: Δ is only defined for onsite (Diagonal) pairing; access op.D"))
        return D.diag
    end
    return getfield(B, s)
end

Base.propertynames(B::BdGOperator, private::Bool=false) =
    getfield(B, :D) isa Diagonal ? (fieldnames(typeof(B))..., :Δ) : fieldnames(typeof(B))

function _canonical_hole_convention(s::Symbol; caller::AbstractString="BdGOperator")
    s === :singlet && return :conjugate
    s in (:conjugate, :intervalley) && return s
    throw(ArgumentError("$caller: hole_convention must be :intervalley or :conjugate (:singlet is an alias) (got $s)"))
end

function BdGOperator(h; mu::Real, U, n=zeros(size(h, 1)),
                     Delta=nothing, D=nothing,
                     hole_convention::Symbol=:intervalley, h_hole=nothing,
                     assume_intervalley::Bool=false)
    N = size(h, 1)
    size(h, 2) == N || throw(ArgumentError("BdGOperator: h must be square (got $(size(h)))"))
    hole_convention = _canonical_hole_convention(hole_convention)
    U_vec = U isa Number ? fill(Float64(U), N) : collect(Float64, U)
    n_vec = collect(Float64, n)
    length(U_vec) == N || throw(ArgumentError("BdGOperator: U has length $(length(U_vec)); expected $N"))
    length(n_vec) == N || throw(ArgumentError("BdGOperator: n has length $(length(n_vec)); expected $N"))
    Delta !== nothing && D !== nothing &&
        throw(ArgumentError("BdGOperator: supply only one of Delta or D"))
    D_stored = if Delta !== nothing
        Delta isa AbstractVector ||
            throw(ArgumentError("BdGOperator: Delta must be a vector"))
        Δ_vec = collect(ComplexF64, Delta)
        length(Δ_vec) == N ||
            throw(ArgumentError("BdGOperator: Delta has length $(length(Δ_vec)); expected $N"))
        Diagonal(Δ_vec)
    elseif D !== nothing
        size(D) == (N, N) ||
            throw(ArgumentError("BdGOperator: D has size $(size(D)); expected ($N, $N)"))
        D
    else
        Diagonal(zeros(ComplexF64, N))
    end

    h_is_matrix = h isa AbstractMatrix
    if h_is_matrix
        ishermitian(h) || throw(ArgumentError("BdGOperator: matrix h must be Hermitian"))
    end

    h_hole_stored = if hole_convention === :intervalley
        h_hole === nothing ||
            throw(ArgumentError("BdGOperator: h_hole must be nothing for hole_convention=:intervalley"))
        if h_is_matrix && eltype(h) <: Complex
            values = h isa SparseMatrixCSC ? nonzeros(h) : h
            max_imag = maximum(x -> abs(imag(x)), values; init=0.0)
            max_abs = maximum(abs, values; init=0.0)
            symmetric = max_imag <= sqrt(eps()) * max(max_abs, 1e-300)
            if !symmetric && !assume_intervalley
                throw(ArgumentError("BdGOperator: for complex Hermitian h the reduced same-h hole block is only the correct physics if h is the intervalley-identified operator (h_{-K}^* = h_K); pass assume_intervalley=true to assert that, or use a real-symmetric h."))
            end
        end
        h
    else
        assume_intervalley &&
            throw(ArgumentError("BdGOperator: assume_intervalley is meaningless for hole_convention=:conjugate; leave it false"))
        if h_is_matrix && h_hole !== nothing &&
                (!(h_hole isa AbstractMatrix) || size(h_hole) != size(h) ||
                 !iszero(h_hole - conj(h)))
            throw(ArgumentError("BdGOperator: for an assembled h the conjugate hole block is conj(h); a different h_hole would silently invalidate gershgorin_bound and the Hermitian BdG action"))
        end
        if h_hole === nothing
            h_is_matrix ||
                throw(ArgumentError("BdGOperator: matrix-free h with hole_convention=:conjugate requires h_hole, the conjugated normal-state operator"))
            conj(h)
        else
            size(h_hole) == size(h) ||
                throw(ArgumentError("BdGOperator: h_hole has size $(size(h_hole)); expected $(size(h))"))
            h_hole
        end
    end
    return BdGOperator(h, h_hole_stored, Float64(mu), U_vec, n_vec, D_stored, N,
                       hole_convention)
end

Base.size(B::BdGOperator) = (2B.N, 2B.N)
Base.size(B::BdGOperator, d::Int) = d <= 2 ? 2B.N : 1
Base.eltype(::BdGOperator) = ComplexF64

_nambu_block(x::AbstractVector, N, blk) = view(x, (blk - 1) * N + 1:blk * N)
_nambu_block(x::AbstractMatrix, N, blk) = view(x, (blk - 1) * N + 1:blk * N, :)

function LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator,
                            X::AbstractVecOrMat, α::Number, β::Number)
    N = B.N
    Yp = _nambu_block(Y, N, 1); Yh = _nambu_block(Y, N, 2)
    Xp = _nambu_block(X, N, 1); Xh = _nambu_block(X, N, 2)
    mul!(Yp, B.h, Xp, α, β)
    mul!(Yh, B.h_hole, Xh, -α, β)
    if B.D isa Diagonal
        @. Yp += α * ((-B.μ - (B.U / 2) * B.n) * Xp + B.D.diag * Xh)
        @. Yh += α * (conj(B.D.diag) * Xp - (-B.μ - (B.U / 2) * B.n) * Xh)
    else
        @. Yp += α * ((-B.μ - (B.U / 2) * B.n) * Xp)
        @. Yh += α * (-(-B.μ - (B.U / 2) * B.n) * Xh)
        mul!(Yp, B.D, Xh, α, true)
        mul!(Yh, adjoint(B.D), Xp, α, true)
    end
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator, X::AbstractVecOrMat) =
    mul!(Y, B, X, true, false)

# An operator is assemblable when every block is an actual matrix.
_bdg_assemblable(op::BdGOperator) =
    getfield(op, :h) isa AbstractMatrix &&
    getfield(op, :h_hole) isa AbstractMatrix &&
    getfield(op, :D) isa AbstractMatrix

_as_sparse_cplx(x::SparseMatrixCSC{ComplexF64, Int}) = x
_as_sparse_cplx(x) = SparseMatrixCSC{ComplexF64, Int}(sparse(x))

"""
    bdg_assemble(op::BdGOperator) -> SparseMatrixCSC{ComplexF64, Int}

Assemble the full `2N x 2N` sparse BdG matrix realized by the matrix-free
action of `op`:

    H_BdG = [ h - (mu + U n / 2) I    D                            ]
            [ adjoint(D)              -(h_hole - (mu + U n / 2) I) ]

Requires assembled matrix blocks (`h`, `h_hole`, `D`); matrix-free blocks
throw. This is the operator the CUDA extension moves to the device: the GPU
BdG paths run standard sparse `mul!` on the assembled matrix rather than the
blockwise matrix-free action.
"""
function bdg_assemble(op::BdGOperator)
    _bdg_assemblable(op) ||
        throw(ArgumentError("bdg_assemble: requires assembled matrix blocks h, h_hole, and D (matrix-free blocks cannot be assembled)"))
    onsite = spdiagm(0 => ComplexF64.(op.μ .+ (op.U ./ 2) .* op.n))
    xi = _as_sparse_cplx(op.h) - onsite
    hole = -(_as_sparse_cplx(op.h_hole) - onsite)
    D = _as_sparse_cplx(op.D)
    return [xi D; _as_sparse_cplx(adjoint(D)) hole]
end

"""
    PairingChannel(bonds, weights, V, parity)

Plain-data description of a pairing channel. `bonds` contains oriented
representatives of undirected bonds: the stored `(i, j)` order defines the
positive orientation, so reversing a representative changes the sign of odd
channel weights and amplitudes while leaving even-channel values unchanged.
`weights` is the seed pattern and projection form factor, `V` is the per-bond
attractive coupling, and `parity` is `:even` or `:odd`. Scalar `weights` and
`V` values are broadcast over all bonds.
"""
struct PairingChannel
    bonds::Vector{Tuple{Int, Int}}
    weights::Vector{ComplexF64}
    V::Vector{Float64}
    parity::Symbol

    function PairingChannel(bonds::Vector{Tuple{Int, Int}}, weights, V,
                            parity::Symbol)
        parity in (:even, :odd) ||
            throw(ArgumentError("PairingChannel: parity must be :even or :odd (got $parity)"))
        nb = length(bonds)
        weights_vec = weights isa Number ? fill(ComplexF64(weights), nb) :
                      collect(ComplexF64, weights)
        V_vec = V isa Number ? fill(Float64(V), nb) : collect(Float64, V)
        length(weights_vec) == nb ||
            throw(ArgumentError("PairingChannel: weights has length $(length(weights_vec)); expected $nb"))
        length(V_vec) == nb ||
            throw(ArgumentError("PairingChannel: V has length $(length(V_vec)); expected $nb"))

        seen = Set{Tuple{Int, Int}}()
        for (i, j) in bonds
            i >= 1 && j >= 1 ||
                throw(ArgumentError("PairingChannel: bond indices must be at least 1 (got ($i, $j))"))
            i == j && parity === :odd &&
                throw(ArgumentError("PairingChannel: onsite bond ($i, $j) is not allowed in an odd channel"))
            key = minmax(i, j)
            key in seen &&
                throw(ArgumentError("PairingChannel: duplicate undirected bond ($i, $j)"))
            push!(seen, key)
        end
        new(copy(bonds), weights_vec, V_vec, parity)
    end
end

function _validate_channel_ownership(channels::AbstractVector{<:PairingChannel},
                                     caller::AbstractString)
    owners = Dict{Tuple{Int, Int}, Int}()
    for (channel_index, channel) in pairs(channels)
        for (i, j) in channel.bonds
            key = minmax(i, j)
            if haskey(owners, key)
                throw(ArgumentError("$caller: undirected bond $key is shared by channels $(owners[key]) and $channel_index"))
            end
            owners[key] = channel_index
        end
    end
    return owners
end

"""
    pairing_matrix(N, channels; amplitudes=nothing, amplitude=nothing)

Assemble the sparse pairing block declared by `channels`. Supply either
per-channel, per-bond `amplitudes`, or a scalar `amplitude` that seeds each
bond as `amplitude * channel.weights[b]`. Explicit zero structural entries
are retained. Bonds are oriented representatives of undirected bonds: under
reversal, odd-channel weights and amplitudes change sign while even-channel
values are unchanged. Each undirected bond may belong to only one channel.
"""
function pairing_matrix(N::Integer, channels::Vector{PairingChannel};
                        amplitudes=nothing, amplitude=nothing)
    N >= 0 || throw(ArgumentError("pairing_matrix: N must be nonnegative (got $N)"))
    (amplitudes === nothing) != (amplitude === nothing) ||
        throw(ArgumentError("pairing_matrix: supply exactly one of amplitudes or amplitude"))
    _validate_channel_ownership(channels, "pairing_matrix")

    amplitudes_vec = if amplitudes === nothing
        amplitude isa Number ||
            throw(ArgumentError("pairing_matrix: amplitude must be a number"))
        [ComplexF64(amplitude) .* channel.weights for channel in channels]
    else
        length(amplitudes) == length(channels) ||
            throw(ArgumentError("pairing_matrix: amplitudes has length $(length(amplitudes)); expected $(length(channels))"))
        [begin
             values = collect(ComplexF64, amplitudes[c])
             length(values) == length(channels[c].bonds) ||
                 throw(ArgumentError("pairing_matrix: amplitudes[$c] has length $(length(values)); expected $(length(channels[c].bonds))"))
             values
         end for c in eachindex(channels)]
    end

    I = Int[]
    J = Int[]
    values = ComplexF64[]
    for c in eachindex(channels)
        channel = channels[c]
        for b in eachindex(channel.bonds)
            i, j = channel.bonds[b]
            i <= N && j <= N ||
                throw(ArgumentError("pairing_matrix: bond ($i, $j) exceeds N=$N"))
            delta = amplitudes_vec[c][b]
            push!(I, i); push!(J, j); push!(values, delta)
            if i != j
                push!(I, j); push!(J, i)
                push!(values, channel.parity === :even ? delta : -delta)
            end
        end
    end
    return sparse(I, J, values, Int(N), Int(N))
end

"""
    channel_amplitude(channel, amplitudes)

Return the form-factor-projected scalar channel amplitude. Channel weights
act as a seed pattern and projection diagnostic; the self-consistency
unknowns are the individual per-bond amplitudes.
"""
function channel_amplitude(channel::PairingChannel,
                           amplitudes::Vector{<:Number})
    length(amplitudes) == length(channel.weights) ||
        throw(ArgumentError("channel_amplitude: amplitudes has length $(length(amplitudes)); expected $(length(channel.weights))"))
    denominator = sum(abs2, channel.weights)
    iszero(denominator) &&
        throw(ArgumentError("channel_amplitude: channel weights must not all be zero"))
    return sum(conj.(channel.weights) .* amplitudes) / denominator
end

"""
    gershgorin_bound(op::BdGOperator) -> Float64

Certified Gershgorin upper bound on the spectral radius of an assembled BdG
operator. Off-diagonal hopping row sums are accumulated from the stored matrix
entries, while the diagonal is replaced explicitly by
`abs(h[i,i] - mu - U[i]*n[i]/2)`. For onsite pairing the contribution is
`abs(D.diag[i])`; for an assembled general pairing matrix it is the larger of
the absolute row and column sums of `D` at site `i`. Matrix-free normal
operators and non-matrix pairing operators are not supported.
"""
function gershgorin_bound(op::BdGOperator)
    op.h isa AbstractMatrix ||
        throw(ArgumentError("gershgorin_bound: BdGOperator must have an assembled matrix h"))
    # The conjugate hole block has identical row sums because abs(conj(h)) == abs(h).
    N = op.N
    iszero(N) && return 0.0
    rowsums = zeros(Float64, N)
    diagonal = zeros(ComplexF64, N)

    if op.h isa SparseMatrixCSC
        rows = rowvals(op.h)
        values = nonzeros(op.h)
        for j in axes(op.h, 2)
            for k in nzrange(op.h, j)
                i = rows[k]
                if i == j
                    diagonal[i] = values[k]
                else
                    rowsums[i] += abs(values[k])
                end
            end
        end
    else
        for j in axes(op.h, 2), i in axes(op.h, 1)
            if i == j
                diagonal[i] = op.h[i, j]
            else
                rowsums[i] += abs(op.h[i, j])
            end
        end
    end

    pairing_rowsums = if op.D isa Diagonal
        abs.(op.D.diag)
    elseif op.D isa AbstractMatrix
        row_sums = zeros(Float64, N)
        col_sums = zeros(Float64, N)
        if op.D isa SparseMatrixCSC
            rows_D = rowvals(op.D)
            values_D = nonzeros(op.D)
            for j in axes(op.D, 2)
                for k in nzrange(op.D, j)
                    value = abs(values_D[k])
                    row_sums[rows_D[k]] += value
                    col_sums[j] += value
                end
            end
        else
            for j in axes(op.D, 2), i in axes(op.D, 1)
                value = abs(op.D[i, j])
                row_sums[i] += value
                col_sums[j] += value
            end
        end
        max.(row_sums, col_sums)
    else
        throw(ArgumentError("gershgorin_bound: D must be an AbstractMatrix"))
    end

    @inbounds for i in 1:N
        rowsums[i] += abs(diagonal[i] - op.μ - op.U[i] * op.n[i] / 2) + pairing_rowsums[i]
    end
    return maximum(rowsums)
end

function _check_chebyshev_columns(slot::AbstractMatrix, iteration::Integer)
    # Columnwise norms as a reduction so the check runs on device arrays too.
    max_norm = sqrt(maximum(sum(abs2, slot; dims=1)))
    if !(max_norm <= 1.5)
        error("Chebyshev recurrence is unstable at iteration $iteration (maximum column norm $max_norm > 1.5); use rescale(...; bound=:gershgorin) or a larger eps.")
    end
    return nothing
end

"""
    bdg_channel_moments(Hs, N, sites, directed_bonds, NC;
                        batch_size=64, verbose=0)

Compute particle-density moments for `sites` and anomalous moments for every
directed bond `(i, j)`. Each bond is extracted from the recurrence seeded by
the particle unit vector at `i` as
`mu_F[m, b] = conj(T_{m-1}(Hs)[j + N, i])`; consequently every bond source
must occur in `sites`.

Workspaces and the gather-based moment extraction follow the residence of
`Hs`: pass a host operator for CPU execution or a device-resident assembled
operator (`ScaledOperator(maybe_to_device(op), a, 0.0)`) to run the recurrence
on the GPU. The returned moment matrices are always host arrays.
"""
function bdg_channel_moments(Hs, N::Integer,
                             sites::AbstractVector{<:Integer},
                             directed_bonds::Vector{Tuple{Int, Int}},
                             NC::Integer; batch_size::Integer=64,
                             verbose::Integer=0)
    NC >= 2 || throw(ArgumentError("bdg_channel_moments: NC must be at least 2 (got $NC)"))
    batch_size > 0 || throw(ArgumentError("bdg_channel_moments: batch_size must be positive (got $batch_size)"))
    all(i -> 1 <= i <= N, sites) ||
        throw(ArgumentError("bdg_channel_moments: all sites must satisfy 1 <= site <= N=$N"))
    all(b -> 1 <= b[1] <= N && 1 <= b[2] <= N, directed_bonds) ||
        throw(ArgumentError("bdg_channel_moments: all directed bond indices must satisfy 1 <= index <= N=$N"))

    site_columns = Dict{Int, Int}()
    for (c, i) in pairs(sites)
        haskey(site_columns, i) &&
            throw(ArgumentError("bdg_channel_moments: sites contains duplicate site $i"))
        site_columns[Int(i)] = c
    end
    extraction = [Tuple{Int, Int}[] for _ in eachindex(sites)]
    for (bond_index, (i, j)) in pairs(directed_bonds)
        haskey(site_columns, i) ||
            throw(ArgumentError("bdg_channel_moments: directed bond ($i, $j) requires source site $i to occur in sites"))
        push!(extraction[site_columns[i]], (j, bond_index))
    end

    ns = length(sites)
    mu_rho = zeros(dt_real, NC, ns)
    mu_F = zeros(dt_cplx, NC, length(directed_bonds))
    iszero(ns) && return mu_rho, mu_F

    twoN = 2Int(N)
    batch_capacity = min(Int(batch_size), ns)
    # All batch workspaces live where the operator lives (host, or device for
    # an assembled GPU operator); extraction is gather/scatter-based so the
    # same code path serves both. Plain 2D slot buffers keep every device
    # operation on unwrapped arrays; they are reallocated only when the batch
    # width changes (at most once, for a final partial batch).
    psi_slots = (device_zeros_of(Hs, dt_cplx, twoN, batch_capacity),
                 device_zeros_of(Hs, dt_cplx, twoN, batch_capacity))
    batch_starts = 1:batch_capacity:ns
    verbose >= 1 && println("NC = $(NC), sites = $(ns), batch_size = $(batch_capacity)")

    for first_site in batch_starts
        last_site = min(first_site + batch_capacity - 1, ns)
        B = last_site - first_site + 1
        if B != size(psi_slots[1], 2)
            psi_slots = (device_zeros_of(Hs, dt_cplx, twoN, B),
                         device_zeros_of(Hs, dt_cplx, twoN, B))
        else
            foreach(slot -> fill!(slot, zero(dt_cplx)), psi_slots)
        end

        # Linear indices into a (2N, B) slot: seeds double as density gathers.
        rho_idx_host = [(c - 1) * twoN + Int(sites[cg])
                        for (c, cg) in enumerate(first_site:last_site)]
        bond_idx_host = Int[]
        bond_cols = Int[]
        for (c, cg) in enumerate(first_site:last_site)
            for (j, bond_index) in extraction[cg]
                push!(bond_idx_host, (c - 1) * twoN + j + Int(N))
                push!(bond_cols, bond_index)
            end
        end
        nb = length(bond_idx_host)
        rho_idx = to_device_of(Hs, rho_idx_host)
        bond_idx = to_device_of(Hs, bond_idx_host)
        rho_work = device_zeros_of(Hs, dt_cplx, B, NC)
        F_work = device_zeros_of(Hs, dt_cplx, nb, NC)

        psi_slots[1][rho_idx] = to_device_of(Hs, fill(one(dt_cplx), B))

        function extract_moment!(m, slot)
            rho_work[:, m] = slot[rho_idx]
            nb == 0 || (F_work[:, m] = slot[bond_idx])
            return nothing
        end

        extract_moment!(1, psi_slots[1])
        mul!(psi_slots[2], Hs, psi_slots[1])
        extract_moment!(2, psi_slots[2])
        NC == 2 && _check_chebyshev_columns(psi_slots[2], 1)

        ip = 2
        ipp = 1
        for m in 3:NC
            chebyshev_iter_single(Hs, psi_slots[ipp], psi_slots[ip])
            extract_moment!(m, psi_slots[ipp])
            iteration = m - 1
            (iteration % 16 == 0 || m == NC) &&
                _check_chebyshev_columns(psi_slots[ipp], iteration)
            ip = 3 - ip
            ipp = 3 - ipp
        end

        mu_rho[:, first_site:last_site] .= transpose(real.(maybe_to_host(rho_work)))
        nb == 0 || (mu_F[:, bond_cols] .= conj.(transpose(maybe_to_host(F_work))))
    end

    return mu_rho, mu_F
end

"""
    bdg_site_moments(Hs, N, sites, NC; batch_size=64, verbose=0)

Compute raw local density and onsite anomalous moments. This compatibility
wrapper uses one onsite directed bond per seed and preserves the recurrence
and extraction order of the original local implementation.
"""
function bdg_site_moments(Hs, N::Integer, sites::AbstractVector{<:Integer},
                          NC::Integer; batch_size::Integer=64,
                          verbose::Integer=0)
    directed_bonds = [(Int(i), Int(i)) for i in sites]
    return bdg_channel_moments(Hs, N, sites, directed_bonds, NC;
                               batch_size=batch_size, verbose=verbose)
end

"""
    chebyshev_stability_probe(Hs, NH::Integer, NC::Integer; rng=Xoshiro(1)) -> Float64

Run the plain two-slot Chebyshev recurrence from one random unit vector for
`NC` moments and return the maximum vector norm encountered. Values above
`1.5` indicate that the rescaled spectrum has escaped the stable interval.
"""
function chebyshev_stability_probe(Hs, NH::Integer, NC::Integer; rng=Xoshiro(1))
    NH > 0 || throw(ArgumentError("chebyshev_stability_probe: NH must be positive (got $NH)"))
    size(Hs) == (NH, NH) ||
        throw(ArgumentError("chebyshev_stability_probe: Hs has size $(size(Hs)); expected ($NH, $NH)"))
    NC > 0 || throw(ArgumentError("chebyshev_stability_probe: NC must be positive (got $NC)"))

    previous = to_device_of(Hs, randn(rng, ComplexF64, NH))
    previous ./= norm(previous)
    max_norm = 1.0
    NC == 1 && return max_norm

    current = similar(previous)
    mul!(current, Hs, previous)
    max_norm = max(max_norm, norm(current))
    T = ComplexF64
    for _ in 3:NC
        # two-address Chebyshev step: previous <- 2 Hs current - previous
        mul!(previous, Hs, current, T(2), T(-1))
        max_norm = max(max_norm, norm(previous))
        previous, current = current, previous
    end
    return Float64(max_norm)
end

"""
    bdg_update(mu_rho, mu_delta, a; U, beta, g_rho=2.0,
               kernel=JacksonKernel, Np=2size(mu_rho, 1))

Update local BdG fields at Fermi level zero from the Jackson-dressed moments:

    n_c = g_rho / Np * sum_k gamma_rho_c(x_k) f(a x_k)
    Delta_c = -U_c / Np * sum_k gamma_delta_c(x_k) f(a x_k)

where `gamma_c(x) = sum_m kernel(m-1, NC) hn(m-1) mu[m,c]
T_{m-1}(x)` and `x_k` are Gauss-Chebyshev nodes. There is no extra
`a`-Jacobian: the spectral `1/a` cancels the `dE = a dx` measure exactly.
The reduced block integrates one spin species, so the default `g_rho=2`
reconstructs the full spin-singlet site density. Set `g_rho=1` for per-spin
density; the caller then owns the corresponding Hartree interpretation.
"""
function bdg_update(mu_rho::AbstractMatrix{<:Real},
                    mu_delta::AbstractMatrix{<:Complex}, a::Real;
                    U::AbstractVector{<:Real}, beta::Real, g_rho::Real=2.0,
                    kernel=JacksonKernel, Np::Integer=2 * size(mu_rho, 1))
    size(mu_rho) == size(mu_delta) ||
        throw(ArgumentError("bdg_update: mu_rho and mu_delta must have the same size (got $(size(mu_rho)) and $(size(mu_delta)))"))
    NC, ns = size(mu_rho)
    length(U) == ns || throw(ArgumentError("bdg_update: U has length $(length(U)); expected $ns"))
    beta > 0 || throw(ArgumentError("bdg_update: beta must be positive (got $beta)"))
    a > 0 || throw(ArgumentError("bdg_update: a must be positive (got $a)"))
    Np > 0 || throw(ArgumentError("bdg_update: Np must be positive (got $Np)"))

    gh = kernel.(0:NC-1, NC) .* hn.(0:NC-1)
    nodes, _ = gausschebyshevt(Np)
    C = cos.((0:NC-1) .* acos.(nodes'))
    wf = fermiFunctions(0.0, Float64(beta)).(a .* nodes) ./ Np
    integrated = C * wf
    n_new = g_rho .* real.(transpose(gh .* mu_rho) * integrated)
    Delta_new = -collect(Float64, U) .* (transpose(gh .* mu_delta) * integrated)
    return collect(Float64, n_new), collect(ComplexF64, Delta_new)
end

"""
    bdg_channel_update(mu_rho, mu_F, a; channels, directed_bonds, beta,
                       g_rho=2.0, kernel=JacksonKernel,
                       Np=2size(mu_rho, 1)) -> (n_new, amplitudes_new)

Reconstruct density and directed anomalous averages using the same
Jackson/Gauss-Chebyshev integration as [`bdg_update`](@ref), then apply the
parity-symmetrized per-bond gap equation. The returned amplitudes are nested
in channel and bond order.
"""
function bdg_channel_update(mu_rho::AbstractMatrix{<:Real},
                            mu_F::AbstractMatrix{<:Complex}, a::Real;
                            channels::Vector{PairingChannel},
                            directed_bonds::Vector{Tuple{Int, Int}},
                            beta::Real, g_rho::Real=2.0,
                            kernel=JacksonKernel,
                            Np::Integer=2 * size(mu_rho, 1))
    size(mu_rho, 1) == size(mu_F, 1) ||
        throw(ArgumentError("bdg_channel_update: mu_rho and mu_F must have the same number of rows (got $(size(mu_rho, 1)) and $(size(mu_F, 1)))"))
    size(mu_F, 2) == length(directed_bonds) ||
        throw(ArgumentError("bdg_channel_update: mu_F has $(size(mu_F, 2)) columns; expected $(length(directed_bonds)) directed bonds"))
    beta > 0 || throw(ArgumentError("bdg_channel_update: beta must be positive (got $beta)"))
    a > 0 || throw(ArgumentError("bdg_channel_update: a must be positive (got $a)"))
    Np > 0 || throw(ArgumentError("bdg_channel_update: Np must be positive (got $Np)"))
    _validate_channel_ownership(channels, "bdg_channel_update")

    directed_index = Dict{Tuple{Int, Int}, Int}()
    for (index, bond) in pairs(directed_bonds)
        haskey(directed_index, bond) &&
            throw(ArgumentError("bdg_channel_update: duplicate directed bond $bond"))
        directed_index[bond] = index
    end
    for channel in channels, (i, j) in channel.bonds
        haskey(directed_index, (i, j)) ||
            throw(ArgumentError("bdg_channel_update: missing directed bond ($i, $j)"))
        i == j || haskey(directed_index, (j, i)) ||
            throw(ArgumentError("bdg_channel_update: missing directed bond ($j, $i)"))
    end

    NC = size(mu_rho, 1)
    gh = kernel.(0:NC-1, NC) .* hn.(0:NC-1)
    nodes, _ = gausschebyshevt(Np)
    C = cos.((0:NC-1) .* acos.(nodes'))
    wf = fermiFunctions(0.0, Float64(beta)).(a .* nodes) ./ Np
    integrated = C * wf
    n_new = g_rho .* real.(transpose(gh .* mu_rho) * integrated)
    F = transpose(gh .* mu_F) * integrated

    amplitudes_new = Vector{Vector{ComplexF64}}(undef, length(channels))
    for c in eachindex(channels)
        channel = channels[c]
        values = Vector{ComplexF64}(undef, length(channel.bonds))
        parity_sign = channel.parity === :even ? 1.0 : -1.0
        for b in eachindex(channel.bonds)
            i, j = channel.bonds[b]
            Fsym = if i == j
                F[directed_index[(i, j)]]
            else
                (F[directed_index[(i, j)]] +
                 parity_sign * F[directed_index[(j, i)]]) / 2
            end
            values[b] = -channel.V[b] * Fsym
        end
        amplitudes_new[c] = values
    end
    return collect(Float64, n_new), amplitudes_new
end

"""
    BdGSCFResult

Summary of a BdG self-consistency solve. `history` stores one entry per
fixed-point iteration, including entries restored from a checkpoint.
"""
struct BdGSCFResult
    converged::Bool
    iterations::Int
    residual_delta::Float64
    residual_n::Float64
    a::Float64
    history::Vector{NamedTuple}
end

Base.show(io::IO, r::BdGSCFResult) = print(io,
    "BdGSCFResult(converged=$(r.converged), iterations=$(r.iterations), " *
    "residual_delta=$(r.residual_delta), residual_n=$(r.residual_n), a=$(r.a))")

"""
    bdg_checkpoint(path, op, history, v_power, params) -> nothing

Atomically write the plain-data state needed to restart a BdG
self-consistency solve. The operator and callback are never serialized.
"""
function bdg_checkpoint(path::AbstractString, op::BdGOperator, history, v_power, params)
    state = (version=2,
             delta=copy(op.Δ),
             n=copy(op.n),
             mu=op.μ,
             U=copy(op.U),
             N=op.N,
             history=copy(history),
             v_power=v_power === nothing ? nothing : copy(v_power),
             params=params)
    tmp_path = "$(path).tmp"
    open(tmp_path, "w") do io
        serialize(io, state)
    end
    mv(tmp_path, path; force=true)
    return nothing
end

function _sparse_nz_indices(D::SparseMatrixCSC)
    indices = Dict{Tuple{Int, Int}, Int}()
    I, J, _ = findnz(D)
    for k in eachindex(I)
        indices[(I[k], J[k])] = k
    end
    return indices
end

function _channel_structure(op::BdGOperator,
                            channels::Vector{PairingChannel},
                            caller::AbstractString)
    op.D isa SparseMatrixCSC ||
        throw(ArgumentError("$caller: channel self-consistency requires op.D isa SparseMatrixCSC"))
    _validate_channel_ownership(channels, caller)
    nz_indices = _sparse_nz_indices(op.D)
    structure = Vector{Vector{Tuple{Int, Int}}}(undef, length(channels))
    for c in eachindex(channels)
        channel = channels[c]
        channel_structure = Vector{Tuple{Int, Int}}(undef, length(channel.bonds))
        for b in eachindex(channel.bonds)
            i, j = channel.bonds[b]
            i <= op.N && j <= op.N ||
                throw(ArgumentError("$caller: bond ($i, $j) exceeds N=$(op.N)"))
            haskey(nz_indices, (i, j)) ||
                throw(ArgumentError("$caller: op.D is missing structural entry ($i, $j)"))
            if i == j
                channel_structure[b] = (nz_indices[(i, j)], nz_indices[(i, j)])
            else
                haskey(nz_indices, (j, i)) ||
                    throw(ArgumentError("$caller: op.D is missing structural entry ($j, $i)"))
                channel_structure[b] = (nz_indices[(i, j)], nz_indices[(j, i)])
            end
        end
        structure[c] = channel_structure
    end
    return structure
end

function _read_channel_amplitudes(op::BdGOperator,
                                  structure::Vector{Vector{Tuple{Int, Int}}})
    values = nonzeros(op.D)
    return [[ComplexF64(values[structure[c][b][1]])
             for b in eachindex(structure[c])] for c in eachindex(structure)]
end

function _write_channel_amplitudes!(op::BdGOperator,
                                    channels::Vector{PairingChannel},
                                    structure::Vector{Vector{Tuple{Int, Int}}},
                                    amplitudes)
    length(amplitudes) == length(channels) ||
        throw(ArgumentError("bdg_restore!: checkpoint amplitudes has invalid channel count"))
    values = nonzeros(op.D)
    for c in eachindex(channels)
        length(amplitudes[c]) == length(channels[c].bonds) ||
            throw(ArgumentError("bdg_restore!: checkpoint amplitudes[$c] has invalid length"))
        parity_sign = channels[c].parity === :even ? 1.0 : -1.0
        for b in eachindex(channels[c].bonds)
            delta = ComplexF64(amplitudes[c][b])
            forward, reverse = structure[c][b]
            values[forward] = delta
            forward == reverse || (values[reverse] = parity_sign * delta)
        end
    end
    return nothing
end

function _checkpoint_channel_data(channels::Vector{PairingChannel})
    return [(bonds=copy(channel.bonds), weights=copy(channel.weights),
             V=copy(channel.V), parity=channel.parity) for channel in channels]
end

function bdg_checkpoint(path::AbstractString, op::BdGOperator,
                        channels::Vector{PairingChannel}, amplitudes,
                        history, v_power, params)
    I_D, J_D, V_D = findnz(op.D)
    state = (version=3,
             delta=nothing,
             channels=_checkpoint_channel_data(channels),
             amplitudes=[copy(values) for values in amplitudes],
             D_triplets=(I=copy(I_D), J=copy(J_D), V=copy(V_D)),
             hole_convention=op.hole_convention,
             n=copy(op.n),
             mu=op.μ,
             U=copy(op.U),
             N=op.N,
             history=copy(history),
             v_power=v_power === nothing ? nothing : copy(v_power),
             params=params)
    tmp_path = "$(path).tmp"
    open(tmp_path, "w") do io
        serialize(io, state)
    end
    mv(tmp_path, path; force=true)
    return nothing
end

"""
    bdg_restore!(op, path) -> (history, v_power, params)

Restore a checkpointed BdG field state into `op`. The checkpoint must have
the same number of sites and interaction field as `op`. Version-1 checkpoints
are accepted and return `nothing` for their missing solver parameters.
"""
function bdg_restore!(op::BdGOperator, path::AbstractString)
    state = open(deserialize, path)
    state.version == 3 &&
        throw(ArgumentError("bdg_restore!: version-3 checkpoint contains pairing channels; call bdg_restore!(op, channels, path)"))
    state.version in (1, 2) || throw(ArgumentError("bdg_restore!: unsupported checkpoint version $(state.version)"))
    state.N == op.N || throw(ArgumentError("bdg_restore!: checkpoint has N=$(state.N); expected N=$(op.N)"))
    state.U ≈ op.U || throw(ArgumentError("bdg_restore!: checkpoint interaction U does not match the operator"))
    length(state.delta) == op.N || throw(ArgumentError("bdg_restore!: checkpoint delta has invalid length"))
    length(state.n) == op.N || throw(ArgumentError("bdg_restore!: checkpoint n has invalid length"))
    op.Δ .= state.delta
    op.n .= state.n
    op.μ = state.mu
    params = state.version == 1 ? nothing : state.params
    return copy(state.history), state.v_power === nothing ? nothing : copy(state.v_power), params
end

"""
    bdg_restore!(op, channels, path) -> (history, v_power, params)

Restore a version-3 channel checkpoint, validating its site count, Hartree
interaction, hole convention, channels, and fixed pairing entries.
"""
function bdg_restore!(op::BdGOperator, channels::Vector{PairingChannel},
                      path::AbstractString)
    state = open(deserialize, path)
    state.version in (1, 2) &&
        throw(ArgumentError("bdg_restore!: channel restore requires a version-3 checkpoint; got legacy version $(state.version)"))
    state.version == 3 ||
        throw(ArgumentError("bdg_restore!: unsupported checkpoint version $(state.version)"))
    state.N == op.N || throw(ArgumentError("bdg_restore!: checkpoint has N=$(state.N); expected N=$(op.N)"))
    state.U ≈ op.U || throw(ArgumentError("bdg_restore!: checkpoint interaction U does not match the operator"))
    state.hole_convention === op.hole_convention ||
        throw(ArgumentError("bdg_restore!: checkpoint hole convention does not match the operator"))
    length(state.n) == op.N || throw(ArgumentError("bdg_restore!: checkpoint n has invalid length"))
    length(state.channels) == length(channels) ||
        throw(ArgumentError("bdg_restore!: checkpoint channel count does not match"))
    for c in eachindex(channels)
        saved = state.channels[c]
        current = channels[c]
        saved.bonds == current.bonds ||
            throw(ArgumentError("bdg_restore!: checkpoint channel $c bonds do not match"))
        saved.weights ≈ current.weights ||
            throw(ArgumentError("bdg_restore!: checkpoint channel $c weights do not match"))
        saved.V ≈ current.V ||
            throw(ArgumentError("bdg_restore!: checkpoint channel $c coupling V does not match"))
        saved.parity === current.parity ||
            throw(ArgumentError("bdg_restore!: checkpoint channel $c parity does not match"))
    end
    structure = _channel_structure(op, channels, "bdg_restore!")
    _write_channel_amplitudes!(op, channels, structure, state.amplitudes)
    triplets = state.D_triplets
    saved_D = sparse(triplets.I, triplets.J, triplets.V, op.N, op.N)
    channel_slots = Set{Tuple{Int, Int}}()
    for channel in channels, (i, j) in channel.bonds
        push!(channel_slots, (i, j))
        i == j || push!(channel_slots, (j, i))
    end
    current_I, current_J, _ = findnz(op.D)
    fixed_slots = Set(zip(triplets.I, triplets.J))
    union!(fixed_slots, zip(current_I, current_J))
    for (i, j) in sort!(collect(fixed_slots))
        if !((i, j) in channel_slots) && !isequal(op.D[i, j], saved_D[i, j])
            throw(ArgumentError("bdg_restore!: pairing entry outside declared channels at ($i, $j) does not match the checkpoint"))
        end
    end
    op.n .= state.n
    op.μ = state.mu
    return copy(state.history), state.v_power === nothing ? nothing : copy(state.v_power), state.params
end

function _bdg_scf_driver!(op::BdGOperator;
                          compute_update, pack_state, pack_out, set_state!,
                          linear_update!, residuals, max_delta, checkpoint!,
                          history, v_power, params,
                          mix::Real, tol_delta::Real, tol_n::Real,
                          maxiter::Integer, update_density::Bool,
                          target_filling, mu_bracket, mu_tol::Real,
                          mu_maxiter::Integer, rescale_eps::Real,
                          callback, checkpoint_path, checkpoint_every::Integer,
                          mixing::Symbol, anderson_history::Integer,
                          anderson_delay::Integer, anderson_max_step::Real,
                          verbose::Integer)
    target_filling !== nothing && !update_density &&
        throw(ArgumentError("bdg_solve!: target_filling cannot be combined with update_density=false: the filling estimator requires density updates"))

    function inner_solve!()
        consecutive = false
        converged = false
        res_d_abs = Inf
        res_n_abs = update_density ? Inf : 0.0
        a = NaN
        iter_offset = isempty(history) ? 0 : last(history).iter
        # Anderson memory is per inner solve (and empty after a restart or
        # between target-filling evaluations): the first accelerated step
        # only happens once two (X, F) pairs exist.
        X_hist = Vector{Vector{Float64}}()
        F_hist = Vector{Vector{Float64}}()

        for local_iter in 1:Int(maxiter)
            iter = iter_offset + local_iter
            # In translation-invariant problems an exact prior power vector
            # can remain a nonextremal eigenvector as the fields change.
            # A deterministic perturbation retains the warm start while
            # restoring overlap with every eigenspace; using `iter` preserves
            # checkpoint/restart bitwise reproducibility.
            if v_power !== nothing
                v_power .+= 0.1 .* randn(Xoshiro(iter), ComplexF64, length(v_power))
            end
            # With a GPU active, the fields updated last iteration are baked
            # into a fresh assembled device operator; power iteration and the
            # moment recurrence both run on it. On CPU this is `op` itself.
            op_run = maybe_to_device(op)
            rad, v_power_new = spectral_radius(op_run; v0=v_power)
            rad > 0 || throw(ArgumentError("bdg_solve!: BdG operator has zero spectral radius"))
            v_power = maybe_to_host(v_power_new)
            a = 2rad / (2 - rescale_eps)
            Hs = ScaledOperator(op_run, a, 0.0)
            update = compute_update(Hs, a)

            res_d_abs, res_d_rel, res_n_abs, res_n_rel = residuals(update)
            delta_pass = res_d_abs <= tol_delta || res_d_rel <= tol_delta
            n_pass = update_density ?
                (res_n_abs <= tol_n || res_n_rel <= tol_n) : true
            passes = delta_pass && n_pass
            converged = consecutive && passes
            consecutive = passes

            if mixing === :anderson
                X_k = pack_state()
                F_k = pack_out(update) .- X_k
                push!(X_hist, X_k)
                push!(F_hist, F_k)
                while length(X_hist) > anderson_history + 1
                    popfirst!(X_hist)
                    popfirst!(F_hist)
                end
                p = length(X_hist)
                X_next = nothing
                # Warm-up delay: the accelerated (Newton-like) step converges
                # to ANY root of F, including the unstable normal-state
                # solution Delta = 0; a few damped steps first move the
                # iterate into the physical basin. The step cap rejects wild
                # extrapolations for the same reason.
                if p >= 2 && local_iter > anderson_delay
                    dX = reduce(hcat, (X_hist[j+1] .- X_hist[j] for j in 1:p-1))
                    dF = reduce(hcat, (F_hist[j+1] .- F_hist[j] for j in 1:p-1))
                    gamma = try
                        dF \ F_k
                    catch
                        nothing
                    end
                    if gamma !== nothing && all(isfinite, gamma)
                        # Type-II Anderson step with damping `mix`
                        candidate = X_k .+ mix .* F_k .- (dX .+ mix .* dF) * gamma
                        damped_step = mix * norm(F_k)
                        if norm(candidate .- X_k) <= anderson_max_step * max(damped_step, eps())
                            X_next = candidate
                        end
                    end
                end
                X_next === nothing && (X_next = X_k .+ mix .* F_k)
                set_state!(X_next)
            else
                linear_update!(update)
            end

            entry = (iter=iter, res_delta=Float64(res_d_abs), res_n=Float64(res_n_abs),
                     max_delta=Float64(max_delta()),
                     mean_n=Float64(sum(op.n) / op.N), mu=op.μ, a=Float64(a))
            push!(history, entry)
            callback === nothing || callback(op, iter, entry)
            if checkpoint_path !== nothing &&
                    (iter % checkpoint_every == 0 || converged || local_iter == maxiter)
                checkpoint!(history, v_power, params)
            end
            verbose >= 1 && println("BdG iter $(iter): res_delta=$(res_d_abs), res_n=$(res_n_abs), mu=$(op.μ), a=$(a)")
            converged && break
        end

        iterations = isempty(history) ? 0 : last(history).iter
        return BdGSCFResult(converged, iterations, Float64(res_d_abs),
                            Float64(res_n_abs), Float64(a), history)
    end

    target_filling === nothing && return inner_solve!()

    mu_lo, mu_hi = mu_bracket
    isfinite(mu_lo) && isfinite(mu_hi) ||
        throw(ArgumentError("bdg_solve!: target_filling requires finite mu_bracket endpoints"))
    mu_lo < mu_hi || throw(ArgumentError("bdg_solve!: mu_bracket must satisfy lo < hi"))

    function filling_error!(mu)
        op.μ = Float64(mu)
        result = inner_solve!()
        result.converged ||
            error("bdg_solve!: target-filling inner solve did not converge at mu=$(op.μ); raise maxiter before continuing the bisection")
        return sum(op.n) / op.N - target_filling, result
    end

    err_lo, result = filling_error!(mu_lo)
    abs(err_lo) <= mu_tol && return result
    err_hi, result = filling_error!(mu_hi)
    abs(err_hi) <= mu_tol && return result
    signbit(err_lo) == signbit(err_hi) &&
        throw(ArgumentError("bdg_solve!: target_filling is not bracketed by mu_bracket"))

    for _ in 1:Int(mu_maxiter)
        mu_mid = (mu_lo + mu_hi) / 2
        err_mid, result = filling_error!(mu_mid)
        abs(err_mid) <= mu_tol && return result
        if signbit(err_mid) == signbit(err_lo)
            mu_lo, err_lo = mu_mid, err_mid
        else
            mu_hi, err_hi = mu_mid, err_mid
        end
    end
    error("bdg_solve!: target-filling bisection did not converge after $(mu_maxiter) iterations; raise mu_maxiter or mu_tol")
end

"""
    bdg_solve!(op; beta, NC=512, g_rho=2.0, mix=0.1, tol_delta=1e-6,
               tol_n=1e-6, maxiter=500, kernel=JacksonKernel, Np=2NC,
               batch_size=64, update_density=true, target_filling=nothing,
               mu_bracket=(-Inf, Inf), mu_tol=1e-4, mu_maxiter=60,
               rescale_eps=0.2, callback=nothing, checkpoint_path=nothing,
               checkpoint_every=10, restart=nothing, verbose=0) -> BdGSCFResult

Solve the local reduced BdG fixed-point equations by damped fixed-point
iteration. `mixing=:linear` (default) is plain linear mixing with weight
`mix`; `mixing=:anderson` is Type-II Anderson acceleration over the packed
state `(Re Δ, Im Δ[, n])` with window `anderson_history` and damping `mix`.
Because the accelerated step is Newton-like it can converge to ANY root of
the gap equation — including the unstable normal state `Δ = 0` — so it is
safeguarded: the first `anderson_delay` iterations use plain damped steps
(entering the physical basin), extrapolations larger than
`anderson_max_step` times the damped step are rejected, and an unusable
least-squares system falls back to the damped step. Anderson memory is not
checkpointed: a restart (or each target-filling evaluation) begins with an
empty acceleration window. A field
update must meet both channel tolerances on two consecutive iterations before
the solve is reported converged. With `target_filling`, a bisection over
`op.μ` performs a full inner solve at every chemical-potential evaluation;
the fields carry over between evaluations as a warm start.
The reduced block integrates one spin species; default `g_rho=2` reconstructs
the full spin-singlet site density used by the Hartree term. On restart,
solver-parameter differences are warned about but allowed. Restart cannot be
combined with `target_filling` because the outer bisection state is not
checkpointed.

With a CUDA GPU active and assembled matrix blocks, each iteration bakes the
current fields into a fresh assembled device operator ([`bdg_assemble`](@ref))
and runs the power iteration and moment recurrence on the GPU; fields,
mixing, and checkpoints stay on the host. Operators with matrix-free blocks
run entirely on the host.
"""
function bdg_solve!(op::BdGOperator; beta::Real, NC::Integer=512,
                    g_rho::Real=2.0, mix::Real=0.1,
                    tol_delta::Real=1e-6, tol_n::Real=1e-6,
                    maxiter::Integer=500, kernel=JacksonKernel,
                    Np::Integer=2 * NC, batch_size::Integer=64,
                    update_density::Bool=true,
                    target_filling::Union{Nothing, Real}=nothing,
                    mu_bracket::Tuple{Real, Real}=(-Inf, Inf), mu_tol::Real=1e-4,
                    mu_maxiter::Integer=60, rescale_eps::Real=0.2,
                    callback=nothing,
                    checkpoint_path::Union{Nothing, AbstractString}=nothing,
                    checkpoint_every::Integer=10,
                    restart::Union{Nothing, AbstractString}=nothing,
                    mixing::Symbol=:linear, anderson_history::Integer=6,
                    anderson_delay::Integer=5, anderson_max_step::Real=50.0,
                    verbose::Integer=0)
    op.D isa Diagonal ||
        throw(ArgumentError("bdg_solve!: the onsite gap equation requires Diagonal (onsite) pairing; bond-channel self-consistency is not yet implemented"))
    beta > 0 || throw(ArgumentError("bdg_solve!: beta must be positive (got $beta)"))
    NC >= 2 || throw(ArgumentError("bdg_solve!: NC must be at least 2 (got $NC)"))
    Np > 0 || throw(ArgumentError("bdg_solve!: Np must be positive (got $Np)"))
    0 < mix <= 1 || throw(ArgumentError("bdg_solve!: mix must satisfy 0 < mix <= 1 (got $mix)"))
    tol_delta >= 0 || throw(ArgumentError("bdg_solve!: tol_delta must be nonnegative (got $tol_delta)"))
    tol_n >= 0 || throw(ArgumentError("bdg_solve!: tol_n must be nonnegative (got $tol_n)"))
    maxiter > 0 || throw(ArgumentError("bdg_solve!: maxiter must be positive (got $maxiter)"))
    batch_size > 0 || throw(ArgumentError("bdg_solve!: batch_size must be positive (got $batch_size)"))
    0 < rescale_eps < 2 || throw(ArgumentError("bdg_solve!: rescale_eps must satisfy 0 < rescale_eps < 2 (got $rescale_eps)"))
    mu_tol >= 0 || throw(ArgumentError("bdg_solve!: mu_tol must be nonnegative (got $mu_tol)"))
    mu_maxiter > 0 || throw(ArgumentError("bdg_solve!: mu_maxiter must be positive (got $mu_maxiter)"))
    checkpoint_every > 0 || throw(ArgumentError("bdg_solve!: checkpoint_every must be positive (got $checkpoint_every)"))
    restart !== nothing && target_filling !== nothing &&
        throw(ArgumentError("bdg_solve!: restart cannot be combined with target_filling because the bisection state is not checkpointed"))
    mixing in (:linear, :anderson) ||
        throw(ArgumentError("bdg_solve!: mixing must be :linear or :anderson (got $mixing)"))
    anderson_history >= 1 ||
        throw(ArgumentError("bdg_solve!: anderson_history must be positive (got $anderson_history)"))
    anderson_delay >= 0 ||
        throw(ArgumentError("bdg_solve!: anderson_delay must be nonnegative (got $anderson_delay)"))
    anderson_max_step > 0 ||
        throw(ArgumentError("bdg_solve!: anderson_max_step must be positive (got $anderson_max_step)"))

    kernel_name = try
        string(nameof(kernel))
    catch
        string(nameof(typeof(kernel)))
    end
    params = (beta=beta, NC=NC, Np=Np, g_rho=g_rho, mix=mix,
              tol_delta=tol_delta, tol_n=tol_n, kernel=kernel_name,
              update_density=update_density, rescale_eps=rescale_eps,
              mixing=string(mixing), anderson_history=anderson_history,
              anderson_delay=anderson_delay,
              anderson_max_step=anderson_max_step)
    history, v_power, saved_params = restart === nothing ?
        (NamedTuple[], nothing, nothing) : bdg_restore!(op, restart)
    if saved_params !== nothing
        differences = String[]
        for name in propertynames(params)
            # older checkpoints may predate newly added parameters
            saved = hasproperty(saved_params, name) ?
                getproperty(saved_params, name) : missing
            current = getproperty(params, name)
            isequal(saved, current) || push!(differences, "$(name): saved=$(repr(saved)), current=$(repr(current))")
        end
        if !isempty(differences)
            diff_text = join(differences, "; ")
            @warn "bdg_solve!: restart parameters differ from checkpoint: $diff_text"
        end
    end

    Nsites = op.N
    pack_state() = update_density ?
        vcat(real.(op.Δ), imag.(op.Δ), copy(op.n)) :
        vcat(real.(op.Δ), imag.(op.Δ))
    pack_out(update) = update_density ?
        vcat(real.(update[2]), imag.(update[2]), update[1]) :
        vcat(real.(update[2]), imag.(update[2]))
    function set_state!(x)
        @views op.Δ .= complex.(x[1:Nsites], x[Nsites+1:2Nsites])
        update_density && (@views op.n .= x[2Nsites+1:3Nsites])
        return nothing
    end

    function compute_update(Hs, a)
        mu_rho, mu_delta = bdg_site_moments(Hs, op.N, 1:op.N, Int(NC);
                                             batch_size=Int(batch_size))
        return bdg_update(mu_rho, mu_delta, a; U=op.U, beta=beta,
                          g_rho=g_rho, kernel=kernel, Np=Int(Np))
    end
    function residuals(update)
        n_new, Delta_new = update
        res_d_abs = norm(Delta_new .- op.Δ, Inf)
        res_d_rel = res_d_abs / max(norm(Delta_new, Inf), eps())
        if update_density
            res_n_abs = norm(n_new .- op.n, Inf)
            res_n_rel = res_n_abs / max(norm(n_new, Inf), eps())
        else
            res_n_abs = 0.0
            res_n_rel = 0.0
        end
        return res_d_abs, res_d_rel, res_n_abs, res_n_rel
    end
    function linear_update!(update)
        n_new, Delta_new = update
        @. op.Δ = (1 - mix) * op.Δ + mix * Delta_new
        if update_density
            @. op.n = (1 - mix) * op.n + mix * n_new
        end
        return nothing
    end
    checkpoint!(history, v_power, params) =
        bdg_checkpoint(checkpoint_path, op, history, v_power, params)

    return _bdg_scf_driver!(op; compute_update=compute_update,
                            pack_state=pack_state, pack_out=pack_out,
                            set_state! = set_state!, linear_update! = linear_update!,
                            residuals=residuals,
                            max_delta=(() -> maximum(abs, op.Δ)),
                            checkpoint! = checkpoint!, history=history,
                            v_power=v_power, params=params, mix=mix,
                            tol_delta=tol_delta, tol_n=tol_n, maxiter=maxiter,
                            update_density=update_density,
                            target_filling=target_filling,
                            mu_bracket=mu_bracket, mu_tol=mu_tol,
                            mu_maxiter=mu_maxiter, rescale_eps=rescale_eps,
                            callback=callback, checkpoint_path=checkpoint_path,
                            checkpoint_every=checkpoint_every, mixing=mixing,
                            anderson_history=anderson_history,
                            anderson_delay=anderson_delay,
                            anderson_max_step=anderson_max_step,
                            verbose=verbose)
end

function _channel_directed_bonds(channels::Vector{PairingChannel})
    directed = Tuple{Int, Int}[]
    for channel in channels, (i, j) in channel.bonds
        push!(directed, (i, j))
        i == j || push!(directed, (j, i))
    end
    return directed
end

_flatten_amplitudes(amplitudes) = reduce(vcat, amplitudes; init=ComplexF64[])

"""
    bdg_solve!(op, channels; kwargs...) -> BdGSCFResult

Solve the per-bond channel gap equations. The packed SCF unknowns are the
real and imaginary parts of every channel amplitude, followed by density
when `update_density=true`. `op.U` retains only its Hartree role; pairing is
generated exclusively by `channels`. Pairing entries in `op.D` outside the
declared channels are held fixed and are never written by the solver.
"""
function bdg_solve!(op::BdGOperator, channels::Vector{PairingChannel};
                    beta::Real, NC::Integer=512, g_rho::Real=2.0,
                    mix::Real=0.1, tol_delta::Real=1e-6,
                    tol_n::Real=1e-6, maxiter::Integer=500,
                    kernel=JacksonKernel, Np::Integer=2 * NC,
                    batch_size::Integer=64, update_density::Bool=true,
                    target_filling::Union{Nothing, Real}=nothing,
                    mu_bracket::Tuple{Real, Real}=(-Inf, Inf),
                    mu_tol::Real=1e-4, mu_maxiter::Integer=60,
                    rescale_eps::Real=0.2, callback=nothing,
                    checkpoint_path::Union{Nothing, AbstractString}=nothing,
                    checkpoint_every::Integer=10,
                    restart::Union{Nothing, AbstractString}=nothing,
                    mixing::Symbol=:linear, anderson_history::Integer=6,
                    anderson_delay::Integer=5,
                    anderson_max_step::Real=50.0,
                    verbose::Integer=0)
    op.hole_convention === :conjugate ||
        throw(ArgumentError("bdg_solve!: pairing-channel self-consistency requires hole_convention=:conjugate because transpose-parity symmetrization presumes same-block fermionic exchange; an intervalley partner-valley pairing map is not implemented"))
    beta > 0 || throw(ArgumentError("bdg_solve!: beta must be positive (got $beta)"))
    NC >= 2 || throw(ArgumentError("bdg_solve!: NC must be at least 2 (got $NC)"))
    Np > 0 || throw(ArgumentError("bdg_solve!: Np must be positive (got $Np)"))
    0 < mix <= 1 || throw(ArgumentError("bdg_solve!: mix must satisfy 0 < mix <= 1 (got $mix)"))
    tol_delta >= 0 || throw(ArgumentError("bdg_solve!: tol_delta must be nonnegative (got $tol_delta)"))
    tol_n >= 0 || throw(ArgumentError("bdg_solve!: tol_n must be nonnegative (got $tol_n)"))
    maxiter > 0 || throw(ArgumentError("bdg_solve!: maxiter must be positive (got $maxiter)"))
    batch_size > 0 || throw(ArgumentError("bdg_solve!: batch_size must be positive (got $batch_size)"))
    0 < rescale_eps < 2 || throw(ArgumentError("bdg_solve!: rescale_eps must satisfy 0 < rescale_eps < 2 (got $rescale_eps)"))
    mu_tol >= 0 || throw(ArgumentError("bdg_solve!: mu_tol must be nonnegative (got $mu_tol)"))
    mu_maxiter > 0 || throw(ArgumentError("bdg_solve!: mu_maxiter must be positive (got $mu_maxiter)"))
    checkpoint_every > 0 || throw(ArgumentError("bdg_solve!: checkpoint_every must be positive (got $checkpoint_every)"))
    restart !== nothing && target_filling !== nothing &&
        throw(ArgumentError("bdg_solve!: restart cannot be combined with target_filling because the bisection state is not checkpointed"))
    mixing in (:linear, :anderson) ||
        throw(ArgumentError("bdg_solve!: mixing must be :linear or :anderson (got $mixing)"))
    anderson_history >= 1 ||
        throw(ArgumentError("bdg_solve!: anderson_history must be positive (got $anderson_history)"))
    anderson_delay >= 0 ||
        throw(ArgumentError("bdg_solve!: anderson_delay must be nonnegative (got $anderson_delay)"))
    anderson_max_step > 0 ||
        throw(ArgumentError("bdg_solve!: anderson_max_step must be positive (got $anderson_max_step)"))

    structure = _channel_structure(op, channels, "bdg_solve!")
    directed_bonds = _channel_directed_bonds(channels)
    kernel_name = try
        string(nameof(kernel))
    catch
        string(nameof(typeof(kernel)))
    end
    params = (beta=beta, NC=NC, Np=Np, g_rho=g_rho, mix=mix,
              tol_delta=tol_delta, tol_n=tol_n, kernel=kernel_name,
              update_density=update_density, rescale_eps=rescale_eps,
              mixing=string(mixing), anderson_history=anderson_history,
              anderson_delay=anderson_delay,
              anderson_max_step=anderson_max_step)
    history, v_power, saved_params = restart === nothing ?
        (NamedTuple[], nothing, nothing) : bdg_restore!(op, channels, restart)
    if saved_params !== nothing
        differences = String[]
        for name in propertynames(params)
            saved = hasproperty(saved_params, name) ?
                getproperty(saved_params, name) : missing
            current = getproperty(params, name)
            isequal(saved, current) ||
                push!(differences, "$(name): saved=$(repr(saved)), current=$(repr(current))")
        end
        if !isempty(differences)
            diff_text = join(differences, "; ")
            @warn "bdg_solve!: restart parameters differ from checkpoint: $diff_text"
        end
    end

    amplitudes = _read_channel_amplitudes(op, structure)
    namplitudes = sum(length, amplitudes)
    function pack_state()
        flat = _flatten_amplitudes(amplitudes)
        return update_density ?
            vcat(real.(flat), imag.(flat), copy(op.n)) :
            vcat(real.(flat), imag.(flat))
    end
    function pack_out(update)
        flat = _flatten_amplitudes(update[2])
        return update_density ?
            vcat(real.(flat), imag.(flat), update[1]) :
            vcat(real.(flat), imag.(flat))
    end
    function set_state!(x)
        flat = @views complex.(x[1:namplitudes],
                               x[namplitudes+1:2namplitudes])
        offset = 0
        for c in eachindex(amplitudes)
            nb = length(amplitudes[c])
            amplitudes[c] .= view(flat, offset+1:offset+nb)
            offset += nb
        end
        _write_channel_amplitudes!(op, channels, structure, amplitudes)
        update_density &&
            (@views op.n .= x[2namplitudes+1:2namplitudes+op.N])
        return nothing
    end
    function compute_update(Hs, a)
        mu_rho, mu_F = bdg_channel_moments(
            Hs, op.N, 1:op.N, directed_bonds, Int(NC);
            batch_size=Int(batch_size))
        return bdg_channel_update(mu_rho, mu_F, a; channels=channels,
                                  directed_bonds=directed_bonds, beta=beta,
                                  g_rho=g_rho, kernel=kernel, Np=Int(Np))
    end
    function residuals(update)
        n_new, amplitudes_new = update
        old_flat = _flatten_amplitudes(amplitudes)
        new_flat = _flatten_amplitudes(amplitudes_new)
        res_d_abs = norm(new_flat .- old_flat, Inf)
        res_d_rel = res_d_abs / max(norm(new_flat, Inf), eps())
        if update_density
            res_n_abs = norm(n_new .- op.n, Inf)
            res_n_rel = res_n_abs / max(norm(n_new, Inf), eps())
        else
            res_n_abs = 0.0
            res_n_rel = 0.0
        end
        return res_d_abs, res_d_rel, res_n_abs, res_n_rel
    end
    function linear_update!(update)
        n_new, amplitudes_new = update
        for c in eachindex(amplitudes), b in eachindex(amplitudes[c])
            amplitudes[c][b] = (1 - mix) * amplitudes[c][b] +
                               mix * amplitudes_new[c][b]
        end
        _write_channel_amplitudes!(op, channels, structure, amplitudes)
        if update_density
            @. op.n = (1 - mix) * op.n + mix * n_new
        end
        return nothing
    end
    checkpoint!(history, v_power, params) = bdg_checkpoint(
        checkpoint_path, op, channels, amplitudes, history, v_power, params)

    return _bdg_scf_driver!(op; compute_update=compute_update,
                            pack_state=pack_state, pack_out=pack_out,
                            set_state! = set_state!, linear_update! = linear_update!,
                            residuals=residuals,
                            max_delta=(() -> maximum(abs, _flatten_amplitudes(amplitudes); init=0.0)),
                            checkpoint! = checkpoint!, history=history,
                            v_power=v_power, params=params, mix=mix,
                            tol_delta=tol_delta, tol_n=tol_n, maxiter=maxiter,
                            update_density=update_density,
                            target_filling=target_filling,
                            mu_bracket=mu_bracket, mu_tol=mu_tol,
                            mu_maxiter=mu_maxiter, rescale_eps=rescale_eps,
                            callback=callback, checkpoint_path=checkpoint_path,
                            checkpoint_every=checkpoint_every, mixing=mixing,
                            anderson_history=anderson_history,
                            anderson_delay=anderson_delay,
                            anderson_max_step=anderson_max_step,
                            verbose=verbose)
end
