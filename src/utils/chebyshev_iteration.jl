using Logging
using LoopVectorization


const ASA = Array{T} where {T <: SubArray}

"""
Advance the Chebyshev recurrence, filling slots 3 to `n` of the workspace:
ψ[i] = 2 H ψ[i-1] - ψ[i-2].
"""
function chebyshev_iter(H, ψall::AbstractArray{T, 2} where T, n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψall, i-2, i-1, i)
    end
end

function chebyshev_iter(H, ψviews::Array{T} where {T <: SubArray}, n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψviews[i-2], ψviews[i-1], ψviews[i])
    end
end

"""
Continue the recurrence past the end of an `n`-slot ring buffer: slots 1 and 2
are overwritten with the next two Chebyshev vectors.
"""
function chebyshev_iter_wrap(H, ψall::AbstractArray{T, 2} where T, n::Int64)
    chebyshev_iter_single(H, ψall, n - 1, n, 1)
    chebyshev_iter_single(H, ψall, n, 1, 2)
end

function chebyshev_iter_wrap(H, ψviews::Array{T} where {T <: SubArray}, n::Int64)
    chebyshev_iter_single(H, ψviews[n-1], ψviews[n], ψviews[1])
    chebyshev_iter_single(H, ψviews[n], ψviews[1], ψviews[2])
end

chebyshev_iter_wrap(H, ψall) = chebyshev_iter_wrap(H, ψall, size(ψall)[1])

# Index-based variant on a 3D workspace [NH, NR, slot]: (i_pp, i_p) -> i_pp.
function chebyshev_iter_single(H, V_all::Union{Array, SubArray}, i_pp_in::Int64, i_p_in::Int64)
    V_p_in = @view V_all[:, :, i_p_in]
    V_out = @view V_all[:, :, i_pp_in]
    chebyshev_iter_single(H, V_out, V_p_in)
    return nothing
end

# Below this many elements per block, per-step task spawning costs more than
# the matvecs themselves (tiny test systems, local BdG seeds), so run serially.
const _CHEB_SERIAL_CUTOFF = 1 << 14

# The core step: V_pp <- 2 H V_p - V_pp, one fused 5-arg mul! per column,
# multithreaded over the NR random-vector columns (serial for tiny blocks).
# Accepts plain Matrix workspaces too (BdG batch buffers); CuArray workspaces
# dispatch to the extension's fused method instead.
function chebyshev_iter_single(H,
                               V_pp_in::Union{Matrix, SubArray},
                               V_p_in::Union{Matrix, SubArray})
    T = eltype(V_pp_in)
    if Threads.nthreads() == 1 || length(V_pp_in) <= _CHEB_SERIAL_CUTOFF
        for i = 1:size(V_pp_in, 2)
            mul!(view(V_pp_in, :, i), H, view(V_p_in, :, i), T(2), T(-1))
        end
    else
        Threads.@threads for i = 1:size(V_pp_in, 2)
            mul!(view(V_pp_in, :, i), H, view(V_p_in, :, i), T(2), T(-1))
        end
    end
    return nothing
end

# Same, over precomputed column views (avoids constructing views in the loop).
function chebyshev_iter_single(H,
                               V_pp_in::ASA,
                               V_p_in::ASA)
    T = eltype(first(V_pp_in))
    if Threads.nthreads() == 1 || length(V_pp_in) * length(first(V_pp_in)) <= _CHEB_SERIAL_CUTOFF
        for i in eachindex(V_pp_in)
            mul!(V_pp_in[i], H, V_p_in[i], T(2), T(-1))
        end
    else
        Threads.@threads for i in eachindex(V_pp_in)
            mul!(V_pp_in[i], H, V_p_in[i], T(2), T(-1))
        end
    end
    return nothing
end

"""
    _check_chebyshev_columns(slot, iteration)

Throw when the recurrence has left the Chebyshev stability region (a column
norm exceeding 1.5 means the rescaled spectrum escaped (−1, 1) and the
three-term recurrence is growing exponentially).
"""
function _check_chebyshev_columns(slot::AbstractMatrix, iteration::Integer)
    # Columnwise norms as a reduction so the check runs on device arrays too.
    max_norm = sqrt(maximum(sum(abs2, slot; dims=1)))
    if !(max_norm <= 1.5)
        error("Chebyshev recurrence is unstable at iteration $iteration (maximum column norm $max_norm > 1.5); use rescale(...; bound=:gershgorin) or a larger eps.")
    end
    return nothing
end

# Relative variant for arbitrary-norm probe columns (the two-arg form assumes
# unit-norm seeds): |T_n(x)| ≤ 1 on [−1, 1], so for a valid rescaling every
# column norm is bounded by its seed norm. `ref_sq` is the seed's columnwise
# abs2-sum, `sum(abs2, seed; dims=1)`. The check is scale-invariant for every
# nonzero seed column; an exactly zero seed column stays exactly zero under
# the linear recurrence, so its ratio is taken against 1 (always passes).
# It detects growth only in the propagated probe subspace — an unstable
# eigenvector orthogonal to every seed column is invisible to it.
function _check_chebyshev_columns(slot::AbstractMatrix, iteration::Integer,
                                  ref_sq::AbstractMatrix)
    denom = ifelse.(iszero.(ref_sq), one(dt_real), ref_sq)
    growth_sq = maximum(sum(abs2, slot; dims=1) ./ denom)
    if !(growth_sq <= 1.5^2)
        error("Chebyshev recurrence is unstable at iteration $iteration (maximum column growth $(sqrt(growth_sq)) > 1.5); use a larger eps (rescaling safety margin) in rescale/normalizeH, or bound=:gershgorin for BdG operators.")
    end
    return nothing
end

# Three-address variants: (i_pp, i_p) -> i_out, leaving slot i_pp intact.
function chebyshev_iter_single(H, V_all::Union{Array, SubArray}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    (@view V_all[:, :, i_out]) .= (@view V_all[:, :, i_pp_in])
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function chebyshev_iter_single(H, V_all::Array{T} where {T <: ASA}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    V_all[i_out] .= V_all[i_pp_in]
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function chebyshev_iter_single(H, V_pp_in::Union{SubArray, ASA}, V_p_in::Union{SubArray, ASA}, V_out::Union{SubArray, ASA})
    V_out .= V_pp_in
    chebyshev_iter_single(H, V_out, V_p_in)
end
