using SpecialFunctions: besselj

## Time-independent unitary evolution by Chebyshev propagation
## (Tal-Ezer & Kosloff):
##   e^{-iHt}|ψ⟩ = e^{-ibt} [ J_0(at) T_0(H_norm)
##                            + 2 Σ_{n≥1} (-i)^n J_n(at) T_n(H_norm) ] |ψ⟩
## with H = a H_norm + b I. The converged series is unitary: no Jackson or
## Lorentz kernel is ever applied here — kernel damping would lose norm.

"""
    evolution_tail(z, NC)

Sum of the dropped time-evolution coefficient magnitudes,
``2\\sum_{m \\ge NC} |J_m(z)|``. Since ``|T_n(x)| \\le 1`` on ``[-1, 1]``,
this is a rigorous truncation-error bound for the propagated state (up to
recurrence roundoff). Past the turning point ``m \\approx |z|`` the terms
decay superexponentially, so the sum is evaluated exactly to machine
precision in ``O(|z|^{1/3})`` terms.
"""
function evolution_tail(z::Real, NC::Integer)
    z = abs(z)
    isfinite(z) || throw(ArgumentError("z must be finite, got $z"))
    NC >= 1 || throw(ArgumentError("NC must be >= 1, got $NC"))
    s = 0.0
    m = Int(NC)
    negligible = 0
    while negligible < 3
        term = 2 * abs(besselj(m, z))
        s += term
        # three consecutive terms below the running sum's ulp end the sum
        # (three, so isolated Bessel zeros cannot end it early)
        negligible = term <= eps(dt_real) * max(s, floatmin(dt_real)) ? negligible + 1 : 0
        m += 1
    end
    return s
end

"""
    evolution_order(a, t; tol=1e-12, NC_min=8, NC_cap=10_000_000)

Smallest Chebyshev order `NC >= NC_min` whose dropped time-evolution tail
``2\\sum_{m \\ge NC}|J_m(a t)|`` ([`evolution_tail`](@ref), a rigorous
truncation bound) is below `tol`. Past ``n \\approx |a t|`` the terms decay
superexponentially, so the search costs ``O(|a t|^{1/3})`` Bessel
evaluations beyond the turning point. Throws when the required order would
exceed `NC_cap` (including `|a t| >= NC_cap` up front) — pass `NC`
explicitly or split the propagation into shorter times.
"""
function evolution_order(a::Real, t::Real; tol::Real=1e-12, NC_min::Integer=8,
                         NC_cap::Integer=10_000_000)
    isfinite(a) && a > 0 || throw(ArgumentError("a must be finite and positive, got $a"))
    isfinite(t) || throw(ArgumentError("t must be finite, got $t"))
    0 < tol < 1 || throw(ArgumentError("tol must be in (0, 1), got $tol"))
    NC_min >= 2 || throw(ArgumentError("NC_min must be >= 2, got $NC_min"))
    NC_min <= NC_cap || throw(ArgumentError("NC_min = $NC_min must not exceed NC_cap = $NC_cap"))
    z = abs(a * t)
    _evolution_order_capcheck(z, NC_cap)
    # coarse scan from the turning point: two consecutive orders below tol/2
    # (an isolated zero of J_n cannot stop it), then certify with the full
    # tail sum, which bounds the actual truncation error
    n = ceil(Int, z)
    while !(abs(besselj(n, z)) < tol / 2 && abs(besselj(n + 1, z)) < tol / 2)
        n += 1
        _evolution_order_capcheck(n, NC_cap)
    end
    NC = n + 1
    while evolution_tail(z, NC) >= tol
        NC += 1
        _evolution_order_capcheck(NC, NC_cap)
    end
    return max(NC, Int(NC_min))
end

_evolution_order_capcheck(x::Real, NC_cap::Integer) =
    (isfinite(x) && x < NC_cap) ||
    throw(ArgumentError("evolution order would exceed NC_cap = $NC_cap (reached $x); pass NC explicitly or split the propagation into shorter times"))

"""
    evolution_coefficients(a, b, ts; NC=0, tol=1e-12) -> (C, NC, tail)

Chebyshev coefficient table for ``e^{-iHt}`` with ``H = a H_{\\rm norm} + bI``:

    C[n, k] = e^{-i b t_k} (2 - δ_{n1}) (-i)^{n-1} J_{n-1}(a t_k)

ready for [`chebyshev_action!`](@ref) (rows multiply ``T_0 … T_{NC-1}``; the
``(2 - δ_{n0})`` weight and the center-shift phase are already folded in).
The Bessel argument keeps the sign of `t`: ``J_n(-z) = (-1)^n J_n(z)``
combines with ``(-i)^n`` to the complex conjugate, so reverse propagation
needs no special-casing. `NC=0` (default) selects the order adaptively via
[`evolution_order`](@ref); a caller-fixed `NC` whose dropped tail
`tail[k] = 2 Σ_{m≥NC} |J_m(a t_k)|` ([`evolution_tail`](@ref), a rigorous
truncation bound) exceeds `tol` triggers a warning.
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
        tail[k] = evolution_tail(z, NC)
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
