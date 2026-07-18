# Coefficient-accumulating matrix-function action and the shared fused-axpy
# accumulation kernel. `broadcast_assign!` and its helpers moved here from
# dc_cond_long.jl: the accumulator is the shared primitive, the DC path is one
# consumer. `dc_long` itself keeps its specialized driver (shrinking per-n
# `idx_max` cutoffs, fused hn·kernel·T_n(E_f) weights) — a future unification
# would have to reproduce that contract on top of `chebyshev_action!`.

"""
    chebyshev_action!(out, Hn, V, C; slots=nothing, check_every=16, verbose=0)

Accumulate the matrix-function action

    out[:, :, k] = Σ_{n=1}^{NC} C[n, k] · T_{n-1}(Hn) V

over the two-slot Chebyshev recurrence, overwriting `out`. `Hn` is the
*rescaled* Hamiltonian (spectrum inside (−1, 1)); `V` is a host `NH × NR`
probe block (`SubArray`s are materialized before transfer) and `C` is an
`NC × K` coefficient table, real or complex, whose rows multiply
`T_0 … T_{NC-1}`.

Internal primitive: coefficients are used verbatim — no `hn` factor, no
kernel damping, no moment doubling — so callers supply fully formed series
(e.g. `(2 - δ_{n0})` factors already included). All `K` coefficient columns
share one recurrence at an `O(NH · NR · K)` per-step accumulation cost;
columns of different effective order are zero-padded (exact). Workspaces and
`out` follow the residence of `Hn` (`to_device_of`/`device_zeros_of`); `out`
must be a plain dense array resident with `Hn` and is always complex, even
for real `Hn` and `V`.

`slots` optionally supplies the two preallocated complex `NH × NR` recurrence
workspaces. Every `check_every` iterations (and at the last) the freshly
written slot is checked by [`_check_chebyshev_columns`](@ref) against the
seed column norms (`|T_n(x)| ≤ 1` on `[−1, 1]` makes every column
non-expanding for a valid rescaling), throwing when the recurrence has left
the stability region — the rescaling margin was too tight; `check_every=0`
disables the guard.
"""
function chebyshev_action!(out::AbstractArray{<:Complex, 3}, Hn, V::AbstractMatrix,
                           C::AbstractMatrix;
                           slots=nothing, check_every::Integer=16, verbose::Integer=0)
    NH, NR = size(V)
    NC, K = size(C)
    NC >= 1 || throw(ArgumentError("C must have at least one coefficient row"))
    size(Hn, 2) == NH ||
        throw(ArgumentError("size(Hn, 2) = $(size(Hn, 2)) must equal size(V, 1) = $NH"))
    size(out, 1) == NH && size(out, 2) == NR ||
        throw(ArgumentError("out must be $NH × $NR × K, got $(size(out))"))
    size(out, 3) == K ||
        throw(ArgumentError("size(out, 3) = $(size(out, 3)) must equal size(C, 2) = $K"))
    check_every >= 0 || throw(ArgumentError("check_every must be >= 0"))
    Base.mightalias(out, V) && throw(ArgumentError("out must not alias V"))

    if slots === nothing
        slots = (device_zeros_of(Hn, dt_cplx, NH, NR),
                 device_zeros_of(Hn, dt_cplx, NH, NR))
    else
        length(slots) == 2 || throw(ArgumentError("slots must hold two workspaces"))
        all(s -> size(s) == (NH, NR) && eltype(s) <: Complex, slots) ||
            throw(ArgumentError("slots must be two complex $NH × $NR workspaces"))
        (Base.mightalias(slots[1], slots[2]) ||
         any(s -> Base.mightalias(s, out) || Base.mightalias(s, V), slots)) &&
            throw(ArgumentError("slots must not alias each other, out, or V"))
    end

    # K × NC transposed layout: each step's per-column coefficients are one
    # contiguous column, and column getindex yields the plain-array copy that
    # broadcast_assign! dispatches on for both devices.
    Ct = to_device_of(Hn, permutedims(Matrix{dt_cplx}(C)))

    # Seed T_0 |V⟩ by indexed assignment (the validated GPU idiom); the
    # Matrix call widens and materializes SubArrays, which to_device_of
    # deliberately does not move.
    slots[1][:, :] = to_device_of(Hn, Matrix{dt_cplx}(V))
    # Seed column norms: the stability guard measures growth relative to these.
    ref_sq = sum(abs2, slots[1]; dims=1)

    fill!(out, zero(eltype(out)))
    out_views = map(k -> view(out, :, :, k), 1:K)

    broadcast_assign!(out, out_views, slots[1], Ct[:, 1], K)
    NC == 1 && return nothing

    mul!(slots[2], Hn, slots[1])
    broadcast_assign!(out, out_views, slots[2], Ct[:, 2], K)
    NC == 2 && check_every > 0 && _check_chebyshev_columns(slots[2], 1, ref_sq)

    ip = 2
    ipp = 1
    n_enum = 3:NC
    if verbose >= 1
        println("chebyshev_action: NC = $NC, K = $K")
        n_enum = ProgressBar(n_enum)
    end
    for n in n_enum
        chebyshev_iter_single(Hn, slots[ipp], slots[ip])
        broadcast_assign!(out, out_views, slots[ipp], Ct[:, n], K)
        iteration = n - 1
        check_every > 0 && (iteration % check_every == 0 || n == NC) &&
            _check_chebyshev_columns(slots[ipp], iteration, ref_sq)
        ip = 3 - ip
        ipp = 3 - ipp
    end
    return nothing
end

function chebyshev_action!(out::AbstractMatrix{<:Complex}, Hn, V::AbstractMatrix,
                           C::AbstractVector; kwargs...)
    chebyshev_action!(reshape(out, size(out, 1), size(out, 2), 1), Hn, V,
                      reshape(C, :, 1); kwargs...)
end

"""
    chebyshev_action(Hn, V, C; kwargs...)

Allocating form of [`chebyshev_action!`](@ref). The output follows the
residence of `Hn` and drops the singleton dimensions of vector inputs:
matrix `V` with matrix `C` gives `NH × NR × K`, vector `C` drops the last
axis, vector `V` drops the probe axis.
"""
function chebyshev_action(Hn, V::AbstractMatrix, C::AbstractMatrix; kwargs...)
    out = device_zeros_of(Hn, dt_cplx, size(V, 1), size(V, 2), size(C, 2))
    chebyshev_action!(out, Hn, V, C; kwargs...)
    return out
end

chebyshev_action(Hn, V::AbstractMatrix, C::AbstractVector; kwargs...) =
    dropdims(chebyshev_action(Hn, V, reshape(C, :, 1); kwargs...); dims=3)
chebyshev_action(Hn, V::AbstractVector, C::AbstractMatrix; kwargs...) =
    dropdims(chebyshev_action(Hn, reshape(V, :, 1), C; kwargs...); dims=2)
chebyshev_action(Hn, V::AbstractVector, C::AbstractVector; kwargs...) =
    vec(chebyshev_action(Hn, reshape(V, :, 1), reshape(C, :, 1); kwargs...))

# --- fused-axpy accumulation (moved from dc_cond_long.jl) -------------------
# y_all[j] .+= x .* c_all[j] for j = 1:idx_max; the CUDA extension adds the
# CuArray method (one kernel launch over all output slots).

function broadcast_assign!(y_all::Array, y_all_views::Array{T, 1} where {T<:Union{Array, SubArray}}, x::Union{Array, SubArray}, c_all::Array, idx_max::Int)
    if idx_max > Threads.nthreads()
        mt_broadcast_assign!(y_all_views, x, c_all, 1:idx_max)
    else
        finer_mt_broadcast_assign!(y_all_views, x, c_all, 1:idx_max)
    end
end


function mt_broadcast_assign!(y_all, x, c_all, idx)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    Threads.@threads for j in idx
        @inbounds y_all[j] .+= x .* c_all[j]
    end
    return nothing
end

function finer_mt_broadcast_assign!(y_all, x, c_all, idx)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    x_cols = size(x, 2)
    split_by_col = x_cols >= Threads.nthreads()
    if split_by_col
        N_splits = x_cols
        xv_all = map(i->view(x, :, i), 1:x_cols)
    else
        N_splits = Threads.nthreads()
        xv_all = _split_vector(x, N_splits)
    end

    for j in idx
        if split_by_col
            yjv_all = map(i->view(y_all[j], :, i), 1:x_cols)
        else
            yjv_all = _split_vector(y_all[j], N_splits)
        end
        cj = c_all[j]
        Threads.@threads for i in 1:length(xv_all)
            @inbounds yjv_all[i] .+= xv_all[i] .* cj
        end
    end

    return nothing
end

function _split_vector(x, N)
    pieces = _partition_l(length(x), N)

    ub = cumsum(pieces)
    lb = ub - pieces
    lb .+= 1

    return map((l,u)->view(x, l:u), lb, ub)

end

function _partition_l(l, N)
    each = cld(l, N) - 1
    res = fill(each, N)
    excess = l - each*N
    res[1:excess].+=1;
    return res
end
