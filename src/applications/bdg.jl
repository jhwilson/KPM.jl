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

"""
    spectral_radius(op; tol=1e-4, maxiter=300, rng=Xoshiro(0), v0=nothing) -> (radius, v)

Estimate the spectral radius of a Hermitian matrix-free operator with norm-ratio
power iteration. The norm ratio converges to the spectral radius even when the
extremal eigenvalues occur as a `±Emax` pair, as in BdG systems. The default
deterministic RNG makes restarts reproducible, and `v0` enables warm starts.
"""
function spectral_radius(op; tol=1e-4, maxiter=300, rng=Xoshiro(0), v0=nothing)
    N = size(op, 1)
    v = v0 === nothing ? randn(rng, ComplexF64, N) : collect(ComplexF64, v0)
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
        if abs(λ - λ_prev) <= tol * max(λ, eps())
            v .= w ./ λ
            return Float64(λ), v
        end
        v .= w ./ λ
        λ_prev = λ
    end

    @debug "spectral_radius reached maxiter without convergence" maxiter tol λ=λ_prev
    return Float64(λ_prev), v
end

"""
    BdGOperator(h; mu, U, n=zeros(size(h, 1)), Delta=zeros(ComplexF64, size(h, 1)))

Matrix-free reduced spin-singlet Nambu BdG operator with particle-hole layout
`[particle; hole]` and hole index `i + N`:

    H_BdG = [ ξ                         Diagonal(Δ)       ]
            [ Diagonal(conj(Δ))         -ξ                ]

with `ξ = h - μ I - Diagonal(U n / 2)`.

Here `U > 0` is attractive, with `H_int = -U Σ n↑n↓`, Hartree shift
`-(U/2)n`, and `Δ_i = -U_i⟨c_{i↓}c_{i↑}⟩`. This CPU-only reduced convention
uses the same `h` in the hole block and presumes `h_{-K}^* = h_K`; for matrix
inputs this is exact when `h` is real-symmetric. Matrix-free `h` is the
caller's responsibility.
"""
mutable struct BdGOperator{TH}
    const h::TH
    μ::Float64
    const U::Vector{Float64}
    const n::Vector{Float64}
    const Δ::Vector{ComplexF64}
    const N::Int
end

function BdGOperator(h; mu::Real, U, n=zeros(size(h, 1)), Delta=zeros(ComplexF64, size(h, 1)))
    N = size(h, 1)
    size(h, 2) == N || throw(ArgumentError("BdGOperator: h must be square (got $(size(h)))"))
    U_vec = U isa Number ? fill(Float64(U), N) : collect(Float64, U)
    n_vec = collect(Float64, n)
    Δ_vec = collect(ComplexF64, Delta)
    length(U_vec) == N || throw(ArgumentError("BdGOperator: U has length $(length(U_vec)); expected $N"))
    length(n_vec) == N || throw(ArgumentError("BdGOperator: n has length $(length(n_vec)); expected $N"))
    length(Δ_vec) == N || throw(ArgumentError("BdGOperator: Delta has length $(length(Δ_vec)); expected $N"))

    if h isa AbstractMatrix
        ishermitian(h) || throw(ArgumentError("BdGOperator: matrix h must be Hermitian"))
        h ≈ transpose(h) || @warn "BdGOperator reduced convention presumes h_{-K}^* = h_K (i.e. symmetric h); the spectrum need not be particle-hole symmetric."
    end
    return BdGOperator(h, Float64(mu), U_vec, n_vec, Δ_vec, N)
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
    mul!(Yh, B.h, Xh, -α, β)
    @. Yp += α * ((-B.μ - (B.U / 2) * B.n) * Xp + B.Δ * Xh)
    @. Yh += α * (conj(B.Δ) * Xp - (-B.μ - (B.U / 2) * B.n) * Xh)
    return Y
end

LinearAlgebra.mul!(Y::AbstractVecOrMat, B::BdGOperator, X::AbstractVecOrMat) =
    mul!(Y, B, X, true, false)
