using SpecialFunctions: besselj

## Time-independent unitary evolution by Chebyshev propagation
## (Tal-Ezer & Kosloff):
##   e^{-iHt}|ψ⟩ = e^{-ibt} [ J_0(at) T_0(H_norm)
##                            + 2 Σ_{n≥1} (-i)^n J_n(at) T_n(H_norm) ] |ψ⟩
## with H = a H_norm + b I. The converged series is unitary: no Jackson or
## Lorentz kernel is ever applied here — kernel damping would lose norm.

"""
    evolution_order(a, t; tol=1e-12, NC_min=8, NC_cap=10_000_000)

Smallest Chebyshev order `NC` whose dropped time-evolution tail is below
`tol`: past ``n \\approx |a t|`` the coefficients ``J_n(a t)`` decay
superexponentially, and the scan stops once two consecutive orders fall
below `tol/2` (two, so an isolated zero of ``J_n`` cannot stop it early).
Throws when `NC_cap` is reached — pass `NC` explicitly or split the
propagation into shorter times.
"""
function evolution_order(a::Real, t::Real; tol::Real=1e-12, NC_min::Integer=8,
                         NC_cap::Integer=10_000_000)
    isfinite(a) && a > 0 || throw(ArgumentError("a must be finite and positive, got $a"))
    isfinite(t) || throw(ArgumentError("t must be finite, got $t"))
    0 < tol < 1 || throw(ArgumentError("tol must be in (0, 1), got $tol"))
    NC_min >= 2 || throw(ArgumentError("NC_min must be >= 2, got $NC_min"))
    z = abs(a * t)
    n = max(Int(NC_min), ceil(Int, z))
    while !(abs(besselj(n, z)) < tol / 2 && abs(besselj(n + 1, z)) < tol / 2)
        n += 1
        n > NC_cap &&
            throw(ArgumentError("evolution_order exceeded NC_cap = $NC_cap at |a t| = $z; pass NC explicitly or split the propagation into shorter times"))
    end
    return n + 1
end

"""
    evolution_coefficients(a, b, ts; NC=0, tol=1e-12) -> (C, NC, tail)

Chebyshev coefficient table for ``e^{-iHt}`` with ``H = a H_{\\rm norm} + bI``:

    C[n, k] = e^{-i b t_k} (2 - δ_{n1}) (-i)^{n-1} J_{n-1}(a t_k)

ready for [`chebyshev_action!`](@ref) (rows multiply ``T_0 … T_{NC-1}``; the
``(2 - δ_{n0})`` weight and the center-shift phase are already folded in).
The Bessel argument keeps the sign of `t`: ``J_n(-z) = (-1)^n J_n(z)``
combines with ``(-i)^n`` to the complex conjugate, so reverse propagation
needs no special-casing. `NC=0` (default) selects the order adaptively via
[`evolution_order`](@ref); a caller-fixed `NC` whose tail estimate
`tail[k] = 2(|J_NC(a t_k)| + |J_{NC+1}(a t_k)|)` exceeds `tol` triggers a
warning.
"""
function evolution_coefficients(a::Real, b::Real, ts::AbstractVector{<:Real};
                                NC::Integer=0, tol::Real=1e-12)
    isfinite(a) && a > 0 || throw(ArgumentError("a must be finite and positive, got $a"))
    isfinite(b) || throw(ArgumentError("b must be finite, got $b"))
    isempty(ts) && throw(ArgumentError("ts must not be empty"))
    all(isfinite, ts) || throw(ArgumentError("all times must be finite"))
    0 < tol < 1 || throw(ArgumentError("tol must be in (0, 1), got $tol"))
    adaptive = NC == 0
    if adaptive
        NC = maximum(evolution_order(a, t; tol=tol) for t in ts)
    else
        NC >= 2 || throw(ArgumentError("NC must be >= 2 (or 0 for adaptive selection), got $NC"))
    end
    NT = length(ts)
    C = Matrix{dt_cplx}(undef, NC, NT)
    tail = Vector{dt_real}(undef, NT)
    for (k, t) in enumerate(ts)
        z = a * t   # signed on purpose; see the docstring
        phase = cis(-b * t)
        C[1, k] = phase * besselj(0, z)
        minus_i_pow = one(dt_cplx)
        for n in 2:NC
            minus_i_pow *= -im
            C[n, k] = phase * 2 * minus_i_pow * besselj(n - 1, z)
        end
        tail[k] = 2 * (abs(besselj(NC, z)) + abs(besselj(NC + 1, z)))
    end
    # the adaptive path already stopped at tol; only a caller-fixed NC can
    # leave a fat tail
    if !adaptive && maximum(tail) > tol
        @warn "evolution series tail estimate $(maximum(tail)) exceeds tol = $tol; increase NC or tol"
    end
    return C, NC, tail
end

"""
    evolve!(out, H_norm, a, b, psi0, ts; NC=0, tol=1e-12, check_every=16, verbose=0)

Raw propagation path: fill `out[:, :, k]` with
``e^{-i(a H_{\\rm norm} + bI) t_k}`` `psi0` for every `t_k` in `ts`, sharing
one Chebyshev recurrence across all times ([`chebyshev_action!`](@ref) with
the [`evolution_coefficients`](@ref) table). `H_norm` must already be
rescaled and resident where the caller wants the recurrence to run; `out`
must follow its residence. Prefer the typed [`evolve`](@ref) front end.
"""
function evolve!(out::AbstractArray{<:Complex, 3}, H_norm, a::Real, b::Real,
                 psi0::AbstractMatrix, ts::AbstractVector{<:Real};
                 NC::Integer=0, tol::Real=1e-12, check_every::Integer=16,
                 verbose::Integer=0)
    C, NC_used, tail = evolution_coefficients(a, b, ts; NC=NC, tol=tol)
    verbose >= 1 &&
        println("evolve: NC = $NC_used, max tail estimate = $(maximum(tail))")
    chebyshev_action!(out, H_norm, psi0, C; check_every=check_every, verbose=verbose)
    return out
end
