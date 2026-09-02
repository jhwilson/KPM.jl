using DocStringExtensions
using ProgressBars
using SparseArrays
using Logging

"""
$(METHODLIST)

The in-place version of 1D KPM.
Calculate the moments μ defined in KPM. Output is saved in `mu`.

  - `H`           -- Hamiltonian. A matrix or sparse matrix.

  - `NC`          -- Integer. the cut off dimension.

  - `NR`          -- Integer. number of random vectors used for KPM evaluation.

  - `NH`          -- Integer. the size of hamiltonian.

  - `mu_all`          -- Array. Output for each random vector. Size (NR, NC).

  - `psi_in`      -- Array (optional). Input array on the right side. A ket.

  - `psi_in_l`, `psi_in_r` -- Arrays (optional, together). Independent bra/ket
    probe blocks, size (NH, NR) each; column i of `psi_in_l` pairs with column i
    of `psi_in_r` and yields complex matrix-element moments
    μ_n[i] = ⟨ψl_i|T_n(H)|ψr_i⟩ (`dot` conjugates the bra). This path cannot use
    moment doubling: it runs the full NC-step recurrence (2x the matvecs of the
    equal-vector path) and accepts odd NC. The equal-vector method requires even
    NC.
"""
function kpm_1d! end

"""
$(METHODLIST)

The simple version of 1D KPM that returns the moment.

  - `H`           -- Hamiltonian. A matrix or sparse matrix

  - `NC`          -- Integer. the cut off dimension

  - `NR`          -- Integer. number of random vectors used for KPM evaluation

  - `NH`          -- Integer. the size of hamiltonian

  - `psi_in`      -- Optional. Allow setting random vector manually.

  - `force_norm`  -- Boolean, Optional. Apply normalization.

  - `verbose`     -- Integer. Default is 0. Enables progress bar if set `verbose=1`.

  - `avg_output`  -- Boolean. Default is true. Whether to output averaged μ (hence size NC) or separate μs (size NR x NC).
    The averaged output is real for the equal-vector path (`psi_in` or random
    probes) and complex when `psi_in_l`/`psi_in_r` are given, since
    ⟨ψl|T_n|ψr⟩ between independent bra/ket has no reality constraint.
"""
function kpm_1d end

"""
$(METHODLIST)

The in-place version of 1D KPM with current operator.
Calculate the moments μ defined in KPM: Γ_n^α = Tr[J_α T_n(H)].
Output is saved in `mu`.

  - `H`           -- Hamiltonian. A matrix or sparse matrix.

  - `Jα`          -- Current operator. A matrix or sparse matrix.

  - `NC`          -- Integer. the cut off dimension.

  - `NR`          -- Integer. number of random vectors used for KPM evaluation.

  - `NH`          -- Integer. the size of hamiltonian.

  - `mu_all`      -- Array. Output for each random vector. Size (NR, NC).

  - `psi_in`      -- Array (optional). Input array on the right side. A ket.
"""
function kpm_1d_current! end

"""
$(METHODLIST)

The simple version of 1D KPM with current operator that returns the moment.
Calculate moments Γ_n^α = Tr[J_α T_n(H)].

  - `H`           -- Hamiltonian. A matrix or sparse matrix

  - `Jα`          -- Current operator. A matrix or sparse matrix

  - `NC`          -- Integer. the cut off dimension

  - `NR`          -- Integer. number of random vectors used for KPM evaluation

  - `NH`          -- Integer. the size of hamiltonian

  - `psi_in`      -- Optional. Allow setting random vector manually.

  - `force_norm`  -- Boolean, Optional. Apply normalization.

  - `verbose`     -- Integer. Default is 0. Enables progress bar if set `verbose=1`.

  - `avg_output`  -- Boolean. Default is true. Whether to output averaged μ (hence size NC) or separate μs (size NR x NC).
"""
function kpm_1d_current end

"""
$(METHODLIST)

In place KPM2D. This is also the main building block for KPM_2D.

Calculates `ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r`.  When `ψ0r` and `ψ0l` are
chosen to be random and identical, the output approximates `tr(Tm(H) Jβ Tn(H) Jα)`.
The accuracy is ``\\sim O(1/sqrt(NR * NH))`` with NR repetitions. NC controls
the energy resolution of the result.

Output: nothing. Result is saved on μ.

**ARGS**

  - `H` : Hamiltonian. A sparse 2D array.

  - `Jα` : Current operator. A sparse 2D array.

  - `Jβ` : Current operator. A sparse 2D array.

  - `NC` : Integer. KPM cutoff order.

  - `NR` : Integer. Number of random vectors.

  - `NH` : Integer. Dimension of H, Jα and Jβ

  - `μ` : 2D Array of dimension (NC, NC). Results will be updated here. Any data
    will be overwritten.

  - `psi_in` : Set `psi_in_l` and `psi_in_r`. Size is (NH, NR). The array is not updated.
    Whether the input is normalized or not, it is assumed to be intended.
    Usually `psi_in` should be normalized.

  - `psi_in_l` : Passes value to ψ0l. Size is (NH, NR). The array is not updated.
    Whether the input is normalized or not, it is assumed to be intended.
    Usually `psi_in_l` should be normalized. `psi_in_l` is given as column vector
    of ket ``|ψl> = <ψl|^\\dagger``

  - `psi_in_r` : Passes value to ψ0r. Size is (NH, NR). The array is not updated.
    Whether the input is normalized or not, it is assumed to be intended.
    Usually `psi_in_r` should be normalized. `psi_in_r` is given as column vector
    of ket ``|ψr>``.

**KWARGS**

  - `arr_size=:auto` : Number `K_left` of consecutive left Chebyshev states
    retained for each recurrence pass (minimum 2). Larger blocks reduce sparse
    matvecs from approximately `NC^2 / 3` to `NC^2 / K_left`.

  - `right_block=:auto` : Number `K_right` of consecutive transformed right
    states contracted at once. The default is `min(NC, 16)`.

  - `workspace_bytes=:auto` : Memory budget used by [`kpm_2d_blocking`](@ref).
    The automatic budget is 25% of total host memory or 50% of available device
    memory. Explicit block sizes take precedence over the budget.

The `NC^2` inner products are evaluated as blocked dense GEMMs. Space is
`O(NH * NR * (K_left + K_right))`, plus the small dense result block.

  - `moment_parity` : The condition enforced on μmn. Choose from `:NONE`, `:ODD` and `:EVEN`.
    `:NONE` keeps all μmn; `:ODD` keeps only μmn with `mod(m+n, 2)==1` and
    `:EVEN` only `mod(m+n, 2)==0`, zeroing the rest. The blocked GEMM computes
    every entry regardless, so parity costs nothing and saves nothing in time;
    `moment_parity=:EVEN` on a particle-hole symmetric model removes the
    stochastic noise in the entries that vanish by symmetry.

**working spaces KWARGS**: The following keyword args are simply providing working
place arrays to avoid repetitive allocation and GC. They are automatically
created if not set. However, when using `KPM_2D!` for many times, it
is beneficial to reuse those arrays.  CONVENTION: args with `ψ` are all
working space arr.

  - `ψ0r=maybe_on_device_zeros(NH, NR)`
  - `Jψ0r=maybe_on_device_zeros(NH, NR)`
  - `JTnHJψr=maybe_on_device_zeros(NH, NR, right_block)`
  - `ψall_r=maybe_on_device_zeros(NH, NR, 3)`
  - `ψ0l=maybe_on_device_zeros(NH, NR)`
  - `ψall_l=maybe_on_device_zeros(NH, NR, arr_size)`
  - `ψw=maybe_on_device_zeros(NH, NR)`
  - `μblock=maybe_on_device_zeros(right_block, arr_size)`
"""
function kpm_2d! end

"""
$(METHODLIST)

The simple version of 2D KPM that returns the moment.
Calculate moments for 2D KPM.

Calculates `ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r`.
When ψ0r and ψ0l are chosen to be random and identical, the output approximates
tr(Tm(H) Jβ Tn(H) Jα). The accuracy is ~ O(1/sqrt(NR * NH)). NC controls the
energy resolution of the result.

Output: μ, a 2D array in ComplexF64. μ[n, m] is the momentum for 2D KPM.

**ARGS**

  - `H`
    Hamiltonian. A sparse 2D array.

  - `Jα`
    Current operator. A sparse 2D array.

  - `Jβ`
    Current operator. A sparse 2D array.

  - `NC`
    Integer. KPM cutoff order.

  - `NR`
    Integer. Number of random vectors to choose from. When skipped, understood as NR=1.

  - `NH`
    Integer. Dimension of H, Jα and Jβ

**KWARGS**

  - `psi_in_l`

Passes value to ψ0l. The array is not updated. Size must be (NH, NR).

  - `psi_in_r`

Passes value to ψ0r. The array is not updated. Size must be (NH, NR).

  - `psi_in`

Cannot be used together with psi_in_l and psi_in_r. Sets psi_in_l=psi_in_r=psi_in if set.

  - `arr_size=:auto`, `right_block=:auto`, `workspace_bytes=:auto`

Choose the left recurrence block, right GEMM block, and workspace budget as
documented for [`kpm_2d!`](@ref). Automatic blocking reduces sparse matvecs to
approximately `NC^2 / arr_size`, while all `NC^2` inner products run in dense
BLAS blocks.

  - `kwargs`

other kwargs in KPM_2D!
"""
function kpm_2d end

"""
$(METHODLIST)

The simple version of tripple KPM that returns the moment.
Calculate moments for tripple KPM.

Calculates `ψ0l * Tn3(H) * Jγ * Tn2(H) * Jβ * Tn1(H) * Jα * ψ0r`.
When ψ0r and ψ0l are chosen to be random and identical, the output approximates
tr(Tn3(H) Jγ Tn2(H) Jβ Tn1(H) Jα). The accuracy is ~ O(1/sqrt(NR * NH)). NC controls the
energy resolution of the result.

Output: μ, a 3D array in ComplexF64. μ[n3, n2, n1] is the momentum for 2D KPM.

**ARGS**

  - `H`
    Hamiltonian. A sparse 2D array.

  - `Jα`
    Current operator. A sparse 2D array.

  - `Jβ`
    Current operator. A sparse 2D array.

  - `Jγ`
    Current operator. A sparse 2D array.

  - `NC`
    Integer. KPM cutoff order.

  - `NR`
    Integer. Number of random vectors to choose from. When skipped, understood as NR=1.

  - `NH`
    Integer. Dimension of H, Jα, Jβ and Jγ

**KWARGS**

  - `psi_in_l`

Passes value to ψ0l. The array is not updated. Size must be (NH, NR).

  - `psi_in_r`

Passes value to ψ0r. The array is not updated. Size must be (NH, NR).

  - `psi_in`

Cannot be used together with psi_in_l and psi_in_r. Sets psi_in_l=psi_in_r=psi_in if set.

  - `kwargs`

other kwargs in KPM_2D!
"""
function kpm_3d! end

"""
$(METHODLIST)

TODO: add doc.
"""
function kpm_3d end

kpm_1d(H, NC::Int64, NR::Int64; kwargs...) = kpm_1d(H, NC, NR, size(H)[1]; kwargs...)
function kpm_1d(
    H,
    NC::Int64,
    NR::Int64,
    NH::Int64;
    psi_in = nothing,
    psi_in_l = nothing,
    psi_in_r = nothing,
    force_norm = false,
    verbose = 0,
    avg_output = true,
    NR_parallel = true,
)
    mu_all = on_host_zeros(dt_cplx, NR, NC) # this mu is never large enough to be worth putting on GPU
    lr_mode = false
    if isnothing(psi_in)
        if (!isnothing(psi_in_l) | !isnothing(psi_in_r))
            @assert (!isnothing(psi_in_l) & !isnothing(psi_in_r)) "must set both `psi_in_l` and `psi_in_r` or neither."
            lr_mode = true
            if force_norm
                normalize_by_col(psi_in_l, NR)
                normalize_by_col(psi_in_r, NR)
            end
            if NR_parallel
                kpm_1d!(H, NC, NR, NH, mu_all, psi_in_l, psi_in_r; verbose = verbose)
            else
                for NRi = 1:NR
                    kpm_1d!(
                        H,
                        NC,
                        1,
                        NH,
                        view(mu_all, NRi:NRi, :),
                        view(psi_in_l, :, NRi:NRi),
                        view(psi_in_r, :, NRi:NRi);
                        verbose = verbose,
                    )
                end
            end
        else
            if NR_parallel
                kpm_1d!(H, NC, NR, NH, mu_all; verbose = verbose)
            else
                for NRi = 1:NR
                    kpm_1d!(H, NC, 1, NH, view(mu_all, NRi:NRi, :); verbose = verbose)
                end
            end
        end
    else
        @assert (isnothing(psi_in_l) & isnothing(psi_in_r)) "must either set `psi_in` or set `psi_in_l` and `psi_in_r`, but not both."
        if force_norm
            normalize_by_col(psi_in, NR)
        end
        if NR_parallel
            kpm_1d!(H, NC, NR, NH, mu_all, psi_in; verbose = verbose)
        else
            for NRi = 1:NR
                kpm_1d!(
                    H,
                    NC,
                    1,
                    NH,
                    view(mu_all, NRi:NRi, :),
                    view(psi_in, :, NRi:NRi);
                    verbose = verbose,
                )
            end
        end
    end

    if avg_output
        avg = dropdims(sum(mu_all, dims = 1), dims = 1) ./ NR
        # ⟨ψl|T_n|ψr⟩ between independent bra/ket is complex; only the
        # equal-vector trace estimate is real.
        return maybe_to_host(lr_mode ? avg : real.(avg))
    end

    return mu_all
end

function kpm_1d!(
    H,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    mu_all,
    psi_in;
    verbose = 0,
    # working arrays
    α_all = maybe_on_device_zeros(dt_cplx, NH, NR, 2),
)
    @assert size(mu_all) == (NR, NC)
    H = maybe_to_device(H, eltype(α_all))

    @assert (mod(NC, 2) == 0) "Invalid NC: NC should be even."
    NChalf = div(NC, 2)

    psi_in_size = size(psi_in)
    @assert (psi_in_size == (NH, NR)) "Invalid `psi_in` with size $(psi_in_size). Expecting $(NH), $(NR)"

    α_all[:, :, 1] = maybe_to_device(psi_in)

    mul!((@view α_all[:, :, 2]), H, (@view α_all[:, :, 1]))

    # μ0 and μ1 feed the doubling identities below. The accumulator follows
    # α_all, so GPU reductions remain asynchronous until the final host copy.
    mu_acc = moment_accumulator(α_all, mu_all)
    α_views = [view(α_all, :, :, 1), view(α_all, :, :, 2)]
    columnwise_dot!(mu_acc, 1, α_views[1], α_views[1])
    columnwise_dot!(mu_acc, 2, α_views[1], α_views[2])

    ip = 2
    ipp = 1

    n_enum = 2:NChalf
    if verbose >= 1
        println("NC/2 = $(NC/2)")
        n_enum = ProgressBar(n_enum)
    end

    for n in n_enum
        chebyshev_iter_single(H, α_views[ipp], α_views[ip])

        # doubling trick: μ_{2n-2} = 2⟨α_{n-1}|α_{n-1}⟩ - μ0,
        #                 μ_{2n-1} = 2⟨α_n|α_{n-1}⟩ - μ1
        columnwise_dot!(
            mu_acc,
            2n - 1,
            α_views[ip],
            α_views[ip];
            alpha = 2.0,
            beta_col = 1,
            beta_scale = -1,
        )

        columnwise_dot!(
            mu_acc,
            2n,
            α_views[ip],
            α_views[ipp];
            alpha = 2.0,
            beta_col = 2,
            beta_scale = -1,
        )

        ip = 3-ip
        ipp = 3-ipp
    end

    copy_moment_accumulator!(mu_all, mu_acc)

    return nothing
end
function kpm_1d!(H, NC::Int64, NR::Int64, NH::Int64, mu; kwargs...)
    psi_in = exp.(maybe_on_device_rand(dt_real, NH, NR) * (2.0im * pi))
    normalize_by_col(psi_in, NR)
    kpm_1d!(H, NC, NR, NH, mu, psi_in; kwargs...)
end
function kpm_1d!(
    H,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    mu_all,
    psi_in_l,
    psi_in_r;
    verbose = 0,
    # working arrays
    α_all = maybe_on_device_zeros(dt_cplx, NH, NR, 2),
    ψl = maybe_on_device_zeros(dt_cplx, NH, NR),
)
    @assert size(mu_all) == (NR, NC)
    @assert NC >= 2 "Invalid NC: NC should be at least 2."
    H = maybe_to_device(H, eltype(α_all))

    @assert (size(psi_in_l) == (NH, NR)) "Invalid `psi_in_l` with size $(size(psi_in_l)). Expecting ($(NH), $(NR))"
    @assert (size(psi_in_r) == (NH, NR)) "Invalid `psi_in_r` with size $(size(psi_in_r)). Expecting ($(NH), $(NR))"
    # the bra is loaded into ψl before the ket seeds α_all; an aliased ψl
    # would be overwritten and corrupt every moment silently
    Base.mightalias(ψl, α_all) && throw(ArgumentError("ψl workspace must not alias α_all"))

    # Moment doubling folds T_m T_n products of one and the same ket; with an
    # independent bra it does not apply, so this path runs the full NC-step
    # recurrence (2x the matvecs of the equal-vector path) and the moments
    # μ_n[i] = ⟨ψl_i|T_n(H)|ψr_i⟩ stay complex. NC need not be even here.
    # (indexed assignment, not broadcast: the equal-vector path's validated
    # GPU idiom for copying host probes into a device workspace)
    ψl[:, :] = maybe_to_device(psi_in_l)
    α_all[:, :, 1] = maybe_to_device(psi_in_r)
    mul!((@view α_all[:, :, 2]), H, (@view α_all[:, :, 1]))

    α_views = [view(α_all, :, :, 1), view(α_all, :, :, 2)]
    mu_acc = moment_accumulator(α_all, mu_all)

    columnwise_dot!(mu_acc, 1, ψl, α_views[1])
    columnwise_dot!(mu_acc, 2, ψl, α_views[2])

    ip = 2
    ipp = 1

    n_enum = 3:NC
    if verbose >= 1
        println("NC = $(NC)")
        n_enum = ProgressBar(n_enum)
    end

    for n in n_enum
        chebyshev_iter_single(H, α_views[ipp], α_views[ip])
        columnwise_dot!(mu_acc, n, ψl, α_views[ipp])
        ip = 3 - ip
        ipp = 3 - ipp
    end

    copy_moment_accumulator!(mu_all, mu_acc)

    return nothing
end

kpm_1d_current(H, Jα, NC::Int64, NR::Int64; kwargs...) =
    kpm_1d_current(H, Jα, NC, NR, size(H)[1]; kwargs...)
function kpm_1d_current(
    H,
    Jα,
    NC::Int64,
    NR::Int64,
    NH::Int64;
    psi_in = nothing,
    psi_in_l = nothing,
    psi_in_r = nothing,
    force_norm = false,
    verbose = 0,
    avg_output = true,
    NR_parallel = true,
)
    mu_all = on_host_zeros(dt_cplx, NR, NC) # this mu is never large enough to be worth putting on GPU
    if isnothing(psi_in)
        if (!isnothing(psi_in_l) | !isnothing(psi_in_r))
            @assert (!isnothing(psi_in_l) & !isnothing(psi_in_r)) "must set both `psi_in_l` and `psi_in_r` or neither."
            if force_norm
                normalize_by_col(psi_in_l, NR)
                normalize_by_col(psi_in_r, NR)
            end
            if NR_parallel
                kpm_1d_current!(
                    H,
                    Jα,
                    NC,
                    NR,
                    NH,
                    mu_all,
                    psi_in_l,
                    psi_in_r;
                    verbose = verbose,
                )
            else
                for NRi = 1:NR
                    kpm_1d_current!(
                        H,
                        Jα,
                        NC,
                        1,
                        NH,
                        view(mu_all, NRi:NRi, :),
                        view(psi_in_l, :, NRi:NRi),
                        view(psi_in_r, :, NRi:NRi);
                        verbose = verbose,
                    )
                end
            end
        else
            if NR_parallel
                kpm_1d_current!(H, Jα, NC, NR, NH, mu_all; verbose = verbose)
            else
                for NRi = 1:NR
                    kpm_1d_current!(
                        H,
                        Jα,
                        NC,
                        1,
                        NH,
                        view(mu_all, NRi:NRi, :);
                        verbose = verbose,
                    )
                end
            end
        end
    else
        @assert (isnothing(psi_in_l) & isnothing(psi_in_r)) "must either set `psi_in` or set `psi_in_l` and `psi_in_r`, but not both."
        if force_norm
            normalize_by_col(psi_in, NR)
        end
        if NR_parallel
            kpm_1d_current!(H, Jα, NC, NR, NH, mu_all, psi_in; verbose = verbose)
        else
            for NRi = 1:NR
                kpm_1d_current!(
                    H,
                    Jα,
                    NC,
                    1,
                    NH,
                    view(mu_all, NRi:NRi, :),
                    view(psi_in, :, NRi:NRi);
                    verbose = verbose,
                )
            end
        end
    end

    if avg_output
        return maybe_to_host(real.(dropdims(sum(mu_all, dims = 1), dims = 1) ./ NR))
    end

    return mu_all
end

function kpm_1d_current!(
    H,
    Jα,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    mu_all,
    psi_in;
    verbose = 0,
    # working arrays
    α_all = maybe_on_device_zeros(dt_cplx, NH, NR, 2),
    Jα_psi = maybe_on_device_zeros(dt_cplx, NH, NR),
)
    @assert size(mu_all) == (NR, NC)
    H = maybe_to_device(H, dt_cplx)
    Jα = maybe_to_device(Jα, dt_cplx)

    psi_in_size = size(psi_in)
    @assert (psi_in_size == (NH, NR)) "Invalid `psi_in` with size $(psi_in_size). Expecting $(NH), $(NR)"

    # Initialize right vector for Chebyshev iteration: T_0(H)|ψ> = |ψ>
    α_all[:, :, 1] = maybe_to_device(psi_in)

    # Apply current operator to left vector once: <ψ|J_α
    mul!(Jα_psi, Jα, α_all[:, :, 1])

    mu_acc = moment_accumulator(α_all, mu_all)
    # Compute first moment: <ψ|J_α T_0(H)|ψ> = <ψ|J_α|ψ>
    columnwise_dot!(mu_acc, 1, Jα_psi, view(α_all, :, :, 1))
    # Compute T_1(H)|ψ> = H|ψ>
    mul!((@view α_all[:, :, 2]), H, (@view α_all[:, :, 1]))
    # Compute second moment: <ψ|J_α T_1(H)|ψ>
    columnwise_dot!(mu_acc, 2, Jα_psi, view(α_all, :, :, 2))

    n_enum = 3:NC
    if verbose >= 1
        println("NC = $(NC)")
        n_enum = ProgressBar(n_enum)
    end

    α_views = [view(α_all, :, :, 1), view(α_all, :, :, 2)]

    ip = 2
    ipp = 1
    for n in n_enum
        # Chebyshev iteration: T_{n+1}(H) = 2H T_n(H) - T_{n-1}(H)
        chebyshev_iter_single(H, α_views[ipp], α_views[ip])
        # cannot use the moment doubling trick here.
        # Compute moment <ψ|J_α T_{n-1}(H)|ψ>.
        columnwise_dot!(mu_acc, n, Jα_psi, α_views[ipp])

        ip = 3-ip
        ipp = 3-ipp
    end

    copy_moment_accumulator!(mu_all, mu_acc)

    return nothing
end

function kpm_1d_current!(H, Jα, NC::Int64, NR::Int64, NH::Int64, mu; kwargs...)
    psi_in = exp.(maybe_on_device_rand(dt_real, NH, NR) * (2.0im * pi))
    normalize_by_col(psi_in, NR)
    kpm_1d_current!(H, Jα, NC, NR, NH, mu, psi_in; kwargs...)
end

function kpm_1d_current!(
    H,
    Jα,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    mu,
    psi_in_l,
    psi_in_r;
    kwargs...,
)
    # with different left and right.
    throw("unimplemented.")
end

# Host budget is a fraction of *total* (cgroup-constrained) memory, not free
# memory: macOS and Linux both report only truly idle pages as free (a few
# hundred MB on a busy laptop), which would collapse the left block to 2.
_kpm2d_auto_budget(::CPUDevice) = div(Sys.total_memory(), 4)
_kpm2d_auto_budget(device::AbstractDevice) = div(device_free_memory(device), 2)

"""
    kpm_2d_blocking(NH, NR, NC; workspace_bytes=:auto,
                    arr_size=:auto, right_block=:auto)

Choose the dense blocking for [`kpm_2d!`](@ref). One recurrence block costs
`NH * NR * sizeof(ComplexF64)` bytes. The automatic workspace budget is 25%
of total host memory (cgroup-limited where one applies; on a shared node
without a cgroup set `KPM_WORKSPACE_BYTES` in the environment or pass
`workspace_bytes`) or 50% of currently available device memory. The right
block defaults to `min(NC, 16)`; the left block is the largest that fits
`(arr_size + right_block + 7)` recurrence blocks plus the
`right_block × arr_size` result buffer in the budget, capped at `NC`. Explicit
integer block sizes always take precedence and are not checked against the
budget. The returned `bytes` is the footprint of that choice (the host
`NC × NC` moment matrix is extra).
"""
function kpm_2d_blocking(
    NH::Integer,
    NR::Integer,
    NC::Integer;
    workspace_bytes = :auto,
    arr_size = :auto,
    right_block = :auto,
)
    NH > 0 || throw(ArgumentError("NH must be positive (got $NH)"))
    NR > 0 || throw(ArgumentError("NR must be positive (got $NR)"))
    NC >= 2 || throw(ArgumentError("NC must be at least 2 (got $NC)"))
    block_bytes = Int(NH) * Int(NR) * sizeof(dt_cplx)

    budget = if workspace_bytes === :auto
        env = get(ENV, "KPM_WORKSPACE_BYTES", "")
        isempty(env) ? Int(_kpm2d_auto_budget(ACTIVE_DEVICE[])) : parse(Int, env)
    elseif workspace_bytes isa Integer
        Int(workspace_bytes)
    else
        throw(ArgumentError("workspace_bytes must be :auto or an integer"))
    end
    budget >= 0 || throw(ArgumentError("workspace_bytes must be nonnegative"))

    K_right = if right_block === :auto
        min(Int(NC), 16)
    elseif right_block isa Integer && right_block >= 1
        Int(right_block)
    else
        throw(ArgumentError("right_block must be :auto or a positive integer"))
    end
    # Footprint: (K_left + K_right + 7) recurrence blocks plus the
    # K_right × K_left GEMM result buffer, so each left slot costs
    # block_bytes + K_right * sizeof(dt_cplx).
    base = (K_right + 7) * block_bytes
    per_left = block_bytes + K_right * sizeof(dt_cplx)
    K_left = if arr_size === :auto
        K_left_max = fld(budget - base, per_left)
        K_left_max >= 2 || throw(
            ArgumentError(
                "workspace_bytes=$budget cannot fit the minimum kpm_2d workspace ($(base + 2 * per_left) bytes at right_block=$K_right)",
            ),
        )
        min(K_left_max, Int(NC))
    elseif arr_size isa Integer && arr_size >= 2
        Int(arr_size)
    else
        throw(ArgumentError("arr_size must be :auto or an integer at least 2"))
    end

    bytes = base + K_left * per_left
    return (arr_size = K_left, right_block = K_right, bytes = bytes)
end

function kpm_2d(
    H,
    Jα,
    Jβ,
    NC::Int64,
    NR::Int64,
    NH::Int64;
    psi_in = nothing,
    psi_in_l = nothing,
    psi_in_r = nothing,
    arr_size = :auto,
    right_block = :auto,
    workspace_bytes = :auto,
    moment_parity = :NONE,
    verbose = 0,
    kwargs...,
)
    mu = on_host_zeros(dt_cplx, NC, NC)
    if isnothing(psi_in) & isnothing(psi_in_l) & isnothing(psi_in_r)
        kpm_2d!(
            H,
            Jα,
            Jβ,
            NC,
            NR,
            NH,
            mu;
            arr_size = arr_size,
            right_block = right_block,
            workspace_bytes = workspace_bytes,
            verbose = verbose,
            moment_parity = moment_parity,
            kwargs...,
        )
    elseif !isnothing(psi_in) & isnothing(psi_in_l) & isnothing(psi_in_r)
        kpm_2d!(
            H,
            Jα,
            Jβ,
            NC,
            NR,
            NH,
            mu,
            psi_in;
            arr_size = arr_size,
            right_block = right_block,
            workspace_bytes = workspace_bytes,
            verbose = verbose,
            moment_parity = moment_parity,
            kwargs...,
        )
    elseif isnothing(psi_in) & !isnothing(psi_in_l) & !isnothing(psi_in_r)
        kpm_2d!(
            H,
            Jα,
            Jβ,
            NC,
            NR,
            NH,
            mu,
            psi_in_l,
            psi_in_r;
            arr_size = arr_size,
            right_block = right_block,
            workspace_bytes = workspace_bytes,
            verbose = verbose,
            moment_parity = moment_parity,
            kwargs...,
        )
    else
        throw("unimplemented")
    end
    return mu
end

function kpm_2d!(
    H,
    Jα,
    Jβ,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    μ,
    psi_in_l,
    psi_in_r;
    arr_size = :auto,
    right_block = :auto,
    workspace_bytes = :auto,
    verbose = 0,
    mn_sym = false,
    moment_parity = :NONE,
    # workspace kwargs
    ψ0r = nothing,
    Jψ0r = nothing,
    JTnHJψr = nothing,
    ψall_r = nothing,
    ψ0l = nothing,
    ψall_l = nothing,
    ψw = nothing,
    μblock = nothing,
)
    if !(moment_parity in (:NONE, :ODD, :EVEN))
        throw(ArgumentError("moment_parity=$(moment_parity) not understood."))
    end
    fill!(μ, 0)

    block_workspaces_supplied =
        ψall_l !== nothing || JTnHJψr !== nothing || μblock !== nothing
    if block_workspaces_supplied
        K_left = if ψall_l !== nothing
            ndims(ψall_l) == 3 || throw(ArgumentError("ψall_l must be three-dimensional"))
            size(ψall_l, 3)
        elseif μblock !== nothing
            size(μblock, 2)
        elseif arr_size isa Integer
            Int(arr_size)
        else
            throw(ArgumentError("supplied block workspaces must determine arr_size"))
        end
        K_right = if JTnHJψr !== nothing
            ndims(JTnHJψr) == 3 ||
                throw(ArgumentError("JTnHJψr must be three-dimensional"))
            size(JTnHJψr, 3)
        elseif μblock !== nothing
            size(μblock, 1)
        elseif right_block isa Integer
            Int(right_block)
        else
            throw(ArgumentError("supplied block workspaces must determine right_block"))
        end
    else
        blocking = kpm_2d_blocking(
            NH,
            NR,
            NC;
            workspace_bytes = workspace_bytes,
            arr_size = arr_size,
            right_block = right_block,
        )
        K_left = blocking.arr_size
        K_right = blocking.right_block
    end
    K_left >= 2 || throw(ArgumentError("ψall_l must have at least two slots"))
    K_right >= 1 || throw(ArgumentError("JTnHJψr must have at least one slot"))

    ψ0r === nothing && (ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR))
    Jψ0r === nothing && (Jψ0r = maybe_on_device_zeros(dt_cplx, NH, NR))
    JTnHJψr === nothing &&
        (JTnHJψr = maybe_on_device_zeros(dt_cplx, NH, NR, K_right))
    ψall_r === nothing && (ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR, 3))
    ψ0l === nothing && (ψ0l = maybe_on_device_zeros(dt_cplx, NH, NR))
    ψall_l === nothing &&
        (ψall_l = maybe_on_device_zeros(dt_cplx, NH, NR, K_left))
    ψw === nothing && (ψw = maybe_on_device_zeros(dt_cplx, NH, NR))
    μblock === nothing && (μblock = maybe_on_device_zeros(dt_cplx, K_right, K_left))

    size(JTnHJψr) == (NH, NR, K_right) ||
        throw(ArgumentError("JTnHJψr workspace has incompatible size"))
    size(ψall_l) == (NH, NR, K_left) ||
        throw(ArgumentError("ψall_l workspace has incompatible size"))
    size(μblock, 1) >= K_right && size(μblock, 2) >= K_left ||
        throw(ArgumentError("μblock workspace must be at least ($K_right, $K_left)"))

    footprint = (K_left + K_right + 7) * NH * NR * sizeof(dt_cplx) +
                K_left * K_right * sizeof(dt_cplx)
    verbose >= 1 && println(
        "kpm_2d blocking: arr_size=$K_left, right_block=$K_right, workspace=$footprint bytes",
    )

    # Function barrier: the workspace kwargs default to `nothing` and are
    # reassigned above, so their types are unions here; the blocked loop
    # must see concrete array types or every moment write boxes.
    return _kpm_2d_blocked!(H, Jα, Jβ, NC, NR, NH, μ, psi_in_l, psi_in_r, K_left, K_right, mn_sym, moment_parity, verbose, ψ0r, Jψ0r, JTnHJψr, ψall_r, ψ0l, ψall_l, ψw, μblock)
end

function _kpm_2d_blocked!(
    H, Jα, Jβ, NC, NR, NH, μ, psi_in_l, psi_in_r, K_left, K_right, mn_sym, moment_parity, verbose, ψ0r, Jψ0r, JTnHJψr, ψall_r, ψ0l, ψall_l, ψw, μblock,
)
    # do not enforce normalization
    @assert (size(psi_in_r) == (NH, NR)) "`psi_in_r` has size $(size(psi_in_r)) but expecting $(NH), $(NR)"
    @assert (size(psi_in_l) == (NH, NR)) "`psi_in_l` has size $(size(psi_in_l)) but expecting $(NH), $(NR)"
    ψ0r .= maybe_to_device(psi_in_r)
    ψ0l .= maybe_to_device(psi_in_l)

    #mn_sym = false
    #if Jα ≡ Jβ
    #    mn_sym = true
    #    println("Jα and Jβ are identical. using m <-> n symmetry.")
    #end

    H = maybe_to_device(H, dt_cplx)
    Jα = maybe_to_device(Jα, dt_cplx)
    Jβ = maybe_to_device(Jβ, dt_cplx)

    # generate all views
    ψall_l_views = map(x -> view(ψall_l, :, :, x), 1:K_left)
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)
    L = reshape(ψall_l, NH * NR, K_left)
    R = reshape(JTnHJψr, NH * NR, K_right)

    # left starter
    ψall_l_views[1] .= ψ0l
    if verbose >= 1
        println("$(typeof(ψw)), $(typeof(H)), $(typeof(ψ0l))")
    end
    mul!(ψw, H, ψ0l)
    ψall_l_views[2] .= ψw

    # right starter
    mul!(Jψ0r, Jα, ψ0r)

    reps = cld(NC, K_left)
    for rep = 1:reps
        m1 = (rep - 1) * K_left + 1
        m2 = min(rep * K_left, NC)
        if verbose >= 1
            println("step $(rep)/$reps: $(m1) to $(m2)")
        end
        rep_size = m2 - m1 + 1
        chebyshev_iter(H, ψall_l_views, rep_size)

        nmax = mn_sym ? m2 : NC
        for n1 = 1:K_right:nmax
            n2 = min(n1 + K_right - 1, nmax)
            n_count = n2 - n1 + 1
            for j = 1:n_count
                n = n1 + j - 1
                if n == 1
                    ψall_r_views[1] .= Jψ0r
                elseif n == 2
                    mul!(ψall_r_views[2], H, Jψ0r)
                else
                    chebyshev_iter_single(
                        H,
                        ψall_r_views[r_ipp(n)],
                        ψall_r_views[r_ip(n)],
                        ψall_r_views[r_i(n)],
                    )
                end
                mul!(view(JTnHJψr, :, :, j), Jβ, ψall_r_views[r_i(n)])
            end

            μb = view(μblock, 1:n_count, 1:rep_size)
            mul!(
                μb,
                adjoint(view(R, :, 1:n_count)),
                view(L, :, 1:rep_size),
                inv(dt_real(NR)),
                zero(dt_cplx),
            )
            host_block = maybe_to_host(μb)
            for jm = 1:rep_size
                m = m1 + jm - 1
                for jn = 1:n_count
                    n = n1 + jn - 1
                    keep = moment_parity == :NONE ||
                           (moment_parity == :ODD && isodd(m + n)) ||
                           (moment_parity == :EVEN && iseven(m + n))
                    keep && (μ[n, m] += conj(host_block[jn, jm]))
                end
            end
        end

        rep < reps && chebyshev_iter_wrap(H, ψall_l_views, K_left)
    end

    if mn_sym
        # apply symmetry
        for m = 1:NC
            for n = (m+1):NC
                μ[m, n] = real(μ[m, n])
                μ[n, m] = μ[m, n]
            end
        end
    end
    return nothing
end

#aliases
function kpm_2d!(H, Jα, Jβ, NC::Int64, NR::Int64, NH::Int64, μ, psi_in; kwargs...)
    kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in, psi_in; kwargs...)
    return nothing
end
function kpm_2d!(H, Jα, Jβ, NC::Int64, NR::Int64, NH::Int64, μ; kwargs...)

    # random vector
    psi_in = exp.(2im*pi*maybe_on_device_rand(dt_real, NH, NR))
    normalize_by_col(psi_in, NR)

    kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in; kwargs...)
    return nothing
end
### END OF ALIASES

function kpm_3d!(
    H,
    Jα,
    Jβ,
    Jγ,
    NC::Int64,
    NR::Int64,
    NH::Int64,
    μ,
    psi_in_l,
    psi_in_r;
    arr_size::Int64 = 3,
    right_block::Int64 = min(NC, 16),
    verbose = 0,
    # workspace kwargs
    ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
    JTn1HJψr = maybe_on_device_zeros(dt_cplx, NH, NR),
    ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR, 3),
    ψ0l = maybe_on_device_zeros(dt_cplx, NH, NR),
    # workspace for sub problem (kpm_2d)
    sub_ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
    sub_Jψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
    sub_JTnHJψr = maybe_on_device_zeros(dt_cplx, NH, NR, right_block),
    sub_ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR, 3),
    sub_ψ0l = maybe_on_device_zeros(dt_cplx, NH, NR),
    sub_ψall_l = maybe_on_device_zeros(dt_cplx, NH, NR, arr_size),
    sub_ψw = maybe_on_device_zeros(dt_cplx, NH, NR),
    sub_μblock = maybe_on_device_zeros(dt_cplx, right_block, arr_size),
)
    #println("Developing")

    # do not enforce normalization
    @assert (size(psi_in_r) == (NH, NR)) "`psi_in_r` has size $(size(psi_in_r)) but expecting $(NH), $(NR)"
    @assert (size(psi_in_l) == (NH, NR)) "`psi_in_l` has size $(size(psi_in_l)) but expecting $(NH), $(NR)"
    ψ0r .= maybe_to_device(psi_in_r)
    ψ0l .= maybe_to_device(psi_in_l)

    H = maybe_to_device(H, dt_cplx)
    Jα = maybe_to_device(Jα, dt_cplx)
    Jβ = maybe_to_device(Jβ, dt_cplx)
    Jγ = maybe_to_device(Jγ, dt_cplx)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)
    μ_views = map(x -> view(μ, :, :, x), 1:NC) # Jα, n1, last index

    n1 = 1
    mul!(ψall_r_views[n1], Jα, ψ0r)
    kpm_2d!(
        H,
        Jβ,
        Jγ,
        NC,
        NR,
        NH,
        μ_views[n1],
        ψ0l, # psi_in_l
        ψall_r_views[r_i(n1)]; # psi_in_r
        arr_size = arr_size,
        right_block = right_block,
        ψ0r = sub_ψ0r,
        Jψ0r = sub_Jψ0r,
        JTnHJψr = sub_JTnHJψr,
        ψall_r = sub_ψall_r,
        ψ0l = sub_ψ0l,
        ψall_l = sub_ψall_l,
        ψw = sub_ψw,
        μblock = sub_μblock,
    )

    n1 = 2
    mul!(ψall_r_views[n1], H, ψall_r_views[r_ip(n1)])
    kpm_2d!(
        H,
        Jβ,
        Jγ,
        NC,
        NR,
        NH,
        μ_views[n1],
        ψ0l, # psi_in_l
        ψall_r_views[r_i(n1)]; # psi_in_r
        arr_size = arr_size,
        right_block = right_block,
        ψ0r = sub_ψ0r,
        Jψ0r = sub_Jψ0r,
        JTnHJψr = sub_JTnHJψr,
        ψall_r = sub_ψall_r,
        ψ0l = sub_ψ0l,
        ψall_l = sub_ψall_l,
        ψw = sub_ψw,
        μblock = sub_μblock,
    )

    for n1 = 3:NC
        if verbose >= 1
            println("n1=$(n1) out of $(NC)")
        end
        chebyshev_iter_single(H, ψall_r, r_ipp(n1), r_ip(n1), r_i(n1))
        kpm_2d!(
            H,
            Jβ,
            Jγ,
            NC,
            NR,
            NH,
            μ_views[n1],
            ψ0l, # psi_in_l
            ψall_r_views[r_i(n1)]; # psi_in_r
            arr_size = arr_size,
            right_block = right_block,
            ψ0r = sub_ψ0r,
            Jψ0r = sub_Jψ0r,
            JTnHJψr = sub_JTnHJψr,
            ψall_r = sub_ψall_r,
            ψ0l = sub_ψ0l,
            ψall_l = sub_ψall_l,
            ψw = sub_ψw,
            μblock = sub_μblock,
        )
    end
end

function kpm_3d(
    H,
    Jα,
    Jβ,
    Jγ,
    NC::Int64,
    NR::Int64,
    NH::Int64;
    arr_size::Int64 = 3,
    right_block::Int64 = min(NC, 16),
    verbose = 0,
    psi_in_l = nothing,
    psi_in_r = nothing,
    psi_in = nothing,
)
    μ = zeros(dt_cplx, NC, NC, NC)
    if !isnothing(psi_in)
        if (!isnothing(psi_in_l) || !isnothing(psi_in_r))
            @warn "`psi_in_l`, `psi_in_r` and `psi_in` are simoutaneously set. Taking `psi_in` and discarding the others"
        end
        psi_in_l = psi_in
        psi_in_r = psi_in
    elseif !isnothing(psi_in_l) || !isnothing(psi_in_r)
        if isnothing(psi_in_l) || isnothing(psi_in_r)
            @warn "only one of `psi_in_l` and `psi_in_r` is set. Setting them as the same."
            psi_in_l = something(psi_in_l, psi_in_r)
            psi_in_r = psi_in_l
        end
    else
        @info "Using random phase as random vector"
        psi_in_l = exp.(2pi * 1im * rand(NH, NR));
        KPM.normalize_by_col(psi_in_l, NR)
        psi_in_r = psi_in_l
    end

    kpm_3d!(
        H,
        Jα,
        Jβ,
        Jγ,
        NC,
        NR,
        NH,
        μ,
        psi_in_l,
        psi_in_r;
        arr_size = arr_size,
        right_block = right_block,
        verbose = verbose,
    )
    return μ
end
