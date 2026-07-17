using KPM
using LinearAlgebra
using SparseArrays
using Random
using Printf

# A small disordered simple-cubic sample. D = L^3 sites and the lattice
# constant is one. These defaults are intended to finish in about a minute.
const L = 8                    # Linear size; D = L^3
const NC = 256                 # Chebyshev order
const NR = 16                  # Random-phase probes
const W = 3.0                  # Uniform on-site disorder strength
const beta = 6.0               # Inverse temperature (energy^-1)
const t = 1.0                  # Nearest-neighbor hopping

site_index(x, y, z, L) = x + 1 + L * (y + L * z)

# For even L, put a displacement into [-L/2, L/2), so boundary-crossing NN bonds retain
# their physical length rather than their coordinate difference on the torus.
minimum_image(delta, L) = mod(delta + fld(L, 2), L) - fld(L, 2)

function cubic_hamiltonian_and_currents(L; t=1.0, W=0.0, rng=Xoshiro(42))
    D = L^3
    rows = Int[]
    cols = Int[]
    hvals = ComplexF64[]
    jxvals = ComplexF64[]
    jyvals = ComplexF64[]
    jzvals = ComplexF64[]
    positions = Vector{NTuple{3, Int}}(undef, D)

    for z in 0:(L - 1), y in 0:(L - 1), x in 0:(L - 1)
        positions[site_index(x, y, z, L)] = (x, y, z)
    end

    function add_entry!(i, j, value)
        ri = positions[i]
        rj = positions[j]
        dx = minimum_image(ri[1] - rj[1], L)
        dy = minimum_image(ri[2] - rj[2], L)
        dz = minimum_image(ri[3] - rj[3], L)
        push!(rows, i); push!(cols, j); push!(hvals, value)
        push!(jxvals, value * dx)
        push!(jyvals, value * dy)
        push!(jzvals, value * dz)
    end

    disorder = W .* (rand(rng, D) .- 0.5)
    for i in 1:D
        add_entry!(i, i, disorder[i])
    end

    # Add each undirected bond once, followed by its Hermitian partner.
    for z in 0:(L - 1), y in 0:(L - 1), x in 0:(L - 1)
        i = site_index(x, y, z, L)
        for (xn, yn, zn) in ((mod(x + 1, L), y, z),
                              (x, mod(y + 1, L), z),
                              (x, y, mod(z + 1, L)))
            j = site_index(xn, yn, zn, L)
            add_entry!(i, j, -t + 0im)
            add_entry!(j, i, -t + 0im)
        end
    end

    H = sparse(rows, cols, hvals, D, D)
    Jx = sparse(rows, cols, jxvals, D, D)
    Jy = sparse(rows, cols, jyvals, D, D)
    Jz = sparse(rows, cols, jzvals, D, D)
    return H, Jx, Jy, Jz
end

function seebeck_value(m, mu_chem; kwargs...)
    return KPM.seebeck_uVK(KPM.thermoelectric(m, mu_chem; kwargs...))
end

function report_drift(label, values)
    finite_values = filter(isfinite, values)
    if length(finite_values) != length(values)
        println("  $label: non-convergence indicated (one or more Seebeck values are NaN).")
        return
    end
    drift = maximum(finite_values) - minimum(finite_values)
    final = values[end]
    relative = iszero(final) ? Inf : abs(drift / final)
    @printf("  %s drift = % .4e uV/K, relative to final = %.2f%%\n",
            label, drift, 100 * relative)
    if relative > 0.05
        println("  $label: drift is large (>5%); this calculation is not converged at this level.")
    else
        println("  $label: drift is below 5% in this sweep, not a convergence proof.")
    end
end

elapsed = @elapsed begin
    H, Jx, Jy, Jz = cubic_hamiltonian_and_currents(L; t=t, W=W, rng=Xoshiro(42))
    volume = float(L^3)  # Cubic lattice constants.
    h = KPM.rescale(H; center=true)

    # For these parameters, 1/beta exceeds pi*a/NC (a is the rescaling
    # half-width), avoiding the thermal-resolution warning in the main sweep.
    @printf("D = %d, rescaling a = %.4f, b = %.4f, 1/beta = %.4f, pi*a/NC = %.4f\n",
            L^3, h.a, h.b, inv(beta), pi * h.a / NC)
    mxx = KPM.cond_moments(h, Jx, Jx; NC=NC, NR=NR, rng=Xoshiro(42))

    usable_halfwidth = h.a * (1 - 1e-3)
    mu_values = collect(range(h.b - 0.80 * usable_halfwidth,
                              h.b + 0.80 * usable_halfwidth; length=19))
    # This central-difference step is above the KPM energy resolution while
    # remaining small on the scale of the disorder-broadened band structure.
    dE = max(0.10, 2 * pi * h.a / NC)

    println("\nCaveat: at these demo defaults the table is noise-dominated (see convergence appendix);")
    println("increase L, NC, NR, and average disorder realizations before quoting numbers.")
    println("\n mu_chem          L0 [(e^2/h)/length]  S [uV/K]       S_Mott [uV/K]")
    println("----------------------------------------------------------------------------")
    for mu_chem in mu_values
        r = KPM.thermoelectric(mxx, mu_chem; beta=beta, volume=volume)
        sigma_minus, sigma_plus = KPM.transport_distribution(
            mxx, [mu_chem - dE, mu_chem + dE]; volume=volume)
        S_mott = NaN
        if isfinite(sigma_minus) && isfinite(sigma_plus) &&
           sigma_minus > 0 && sigma_plus > 0
            dlog_sigma = (log(sigma_plus) - log(sigma_minus)) / (2 * dE)
            S_mott_bar = -(pi^2 / (3 * beta)) * dlog_sigma
            S_mott = 86.17333262 * S_mott_bar
        end
        if isfinite(r.S_over_kB_over_e)
            @printf("% .6f  % .8e  % .6f  % .6f\n",
                    mu_chem, r.L0, KPM.seebeck_uVK(r), S_mott)
        else
            @printf("% .6f  % .8e  %12s  % .6f\n",
                    mu_chem, r.L0, "NaN (insulating)", S_mott)
        end
    end

    # ------------------------------------------------------------------------
    # Convergence appendix: use these sweeps before treating a regularized
    # finite sample as a bulk result. No sweep below is massaged to look good.
    # ------------------------------------------------------------------------
    mu_ref = clamp(-2.5, h.b - 0.70 * usable_halfwidth,
                   h.b + 0.70 * usable_halfwidth)
    @printf("\nConvergence appendix at mu_chem = %.4f\n", mu_ref)

    println("  NC sweep (the stored moments are truncated; no recomputation):")
    NC_values = (128, 192, 256)
    S_NC = Float64[]
    for nc_trial in NC_values
        S = seebeck_value(mxx, mu_ref; beta=beta, volume=volume, NC=nc_trial)
        push!(S_NC, S)
        @printf("    NC = %3d: % .6f uV/K\n", nc_trial, S)
    end
    report_drift("NC sweep", S_NC)

    println("  NR sweep (moments recomputed with the same random seed):")
    NR_values = (2, 4, 8)
    S_NR = Float64[]
    for nr_trial in NR_values
        m_trial = KPM.cond_moments(h, Jx, Jx; NC=NC, NR=nr_trial, rng=Xoshiro(42))
        S = seebeck_value(m_trial, mu_ref; beta=beta, volume=volume)
        push!(S_NR, S)
        @printf("    NR = %d: % .6f uV/K\n", nr_trial, S)
    end
    report_drift("NR sweep", S_NR)

    println("  Kernel sweep (Jackson default versus Lorentz broadening):")
    kernel_labels = ["Jackson", "Lorentz(lambda=3.0)",
                     "Lorentz(lambda=4.0)", "Lorentz(lambda=5.0)"]
    kernels = Any[KPM.JacksonKernel, KPM.LorentzKernels(3.0),
                  KPM.LorentzKernels(4.0), KPM.LorentzKernels(5.0)]
    S_kernel = Float64[]
    for (label, kernel) in zip(kernel_labels, kernels)
        S = seebeck_value(mxx, mu_ref; beta=beta, volume=volume, kernel=kernel)
        push!(S_kernel, S)
        @printf("    %-20s % .6f uV/K\n", label * ":", S)
    end
    report_drift("kernel sweep", S_kernel)
    println("  A full study must also sweep L and disorder realizations; this appendix covers NC/NR/kernel only.")
end

@printf("\nTotal elapsed time: %.2f s\n", elapsed)
