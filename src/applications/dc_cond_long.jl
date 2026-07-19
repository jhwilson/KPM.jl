using Logging
using LinearAlgebra
using SparseArrays
## Special algorithm for longitudinal DC conductivity

function dc_long(
    H,
    Jα,
    H_rescale_factor,
    NC_all::Vector{Int64},
    NR::Int64,
    NH::Int64;
    verbose = 0,
    psi_in = nothing,
    kernel = KPM.JacksonKernel,
    Ef = 0.0,
    # workspace kwargs
    ψr = maybe_on_device_zeros(dt_cplx, NH, NR * 2, length(NC_all)),
    ψ0 = maybe_on_device_zeros(dt_cplx, NH, NR * 2),
    ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR * 2, 2),
    avg_NR = true,
    debug_mode = false,
)
    Ef = KPM.dt_real(Ef)
    H_rescale_factor = KPM.dt_real(H_rescale_factor)
    NC_orig = NC_all
    NC_sort_i = sortperm(NC_orig, rev = true)
    NC_all = NC_orig[NC_sort_i]
    @assert issorted(NC_all, rev = true) "NC_all should be descend sorted"

    if !(typeof(kernel) <: Array)
        kernel = [kernel, kernel]
    end
    NC_max = maximum(NC_all)
    Ef_tilde = Ef / H_rescale_factor

    if Ef_tilde == 0
        Tn_e = chebyshevT_0.((0:(NC_max-1))')
    else
        Tn_e = chebyshevT_accurate.((0:(NC_max-1))', Ef_tilde)
    end

    kernel1_Tn = kernel[1].((0:(NC_max-1))', NC_all) .* hn.((0:(NC_max-1))') .* Tn_e
    kernel2_Tn = kernel[2].((0:(NC_max-1))', NC_all) .* hn.((0:(NC_max-1))') .* Tn_e

    kernel_Tn = maybe_to_device([kernel1_Tn kernel2_Tn])

    if isnothing(psi_in)
        psi_in = exp.(maybe_on_device_rand(dt_real, size(H, 1), NR) * 2im * pi);
        normalize_by_col(psi_in, NR)
    end
    psi_in = maybe_to_device(psi_in)

    # Hermitian warning
    if debug_mode
        if !ishermitian(H)
            @warn "Hamiltonian is not Hermitian. Please make sure it is upper triangular."
        end
        if !ishermitian(Jα)
            @warn "Current operator is not Hermitian. Please make sure it is upper triangular."
        end
    end

    # materialize the Hermitian operators on the host (keeps the documented
    # upper-triangular-input semantics), then move plain matrices to the
    # device: mul! through a Hermitian-wrapped CuSparseMatrixCSR multiplies
    # by the stored triangle only and silently diverges (A100-verified)
    H = maybe_to_device(sparse(Hermitian(H, :U)), dt_cplx)
    Jα = maybe_to_device(sparse(Hermitian(Jα, :U)), dt_cplx)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:2)
    ψr_views = map(x -> view(ψr, :, :, x), 1:length(NC_all))

    # right start
    view(ψ0, :, 1:NR) .= psi_in
    @debug "$(size(psi_in)), $(size(Jα)), $(size(ψ0))"
    mul!(view(ψ0, :, (NR+1):(2*NR)), Jα, psi_in)

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    ψall_r_views[r2_i(n)] .= ψ0
    NC_idx_max = findlast(i -> i >= n, NC_all)
    broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)

    n = 2
    mul!(ψall_r_views[r2_i(n)], H, ψall_r_views[r2_ip(n)])
    NC_idx_max = findlast(i -> i >= n, NC_all)
    broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)

    n_enum = 3:NC_max
    if verbose >= 1
        println("loop over n=3:$(NC_max)")
        n_enum = ProgressBar(n_enum)
    end
    for n in n_enum
        chebyshev_iter_single(H, ψall_r_views[r2_i(n)], ψall_r_views[r2_ip(n)])
        # output is stored at r2_i(n) === r2_ipp(n)

        NC_idx_max = findlast(i -> i >= n, NC_all)
        broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)
    end

    ψr_views_1 = map(x -> view(ψr, :, 1:NR, x), 1:length(NC_all))
    ψr_views_2 = map(x -> view(ψr, :, (NR+1):(2*NR), x), 1:length(NC_all))

    if avg_NR
        cond = on_host_zeros(dt_cplx, length(NC_all))
        for (NCi, NC_orig_i) in enumerate(NC_sort_i)
            # mul-then-dot (the generic 3-arg dot scalar-indexes on the GPU)
            cond[NC_orig_i] = dot(ψr_views_1[NCi], Jα * ψr_views_2[NCi])
        end
        cond ./= NR
    else
        cond = on_host_zeros(dt_cplx, length(NC_all), NR)
        for (NCi, NC_orig_i) in enumerate(NC_sort_i)
            for NRi = 1:NR
                cond[NC_orig_i, NRi] =
                    dot(view(ψr_views_1[NCi], :, NRi), Jα * view(ψr_views_2[NCi], :, NRi))
            end
        end
    end

    return cond / H_rescale_factor
end

# broadcast_assign! and its helpers moved to src/utils/chebyshev_action.jl
