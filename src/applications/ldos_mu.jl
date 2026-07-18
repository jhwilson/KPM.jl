"""
    ldos_mu(H, NC, site; kwargs...)

Legacy raw path: real site-diagonal moments `⟨site|T_n(H)|site⟩` of an
already-rescaled `H`, with no `(a, b)` provenance and no reconstruction.
Prefer [`ldos_moments`](@ref) on a [`RescaledHamiltonian`](@ref), which
batches sites, carries the rescaling metadata, and reconstructs via
[`ldos`](@ref).
"""
function ldos_mu(
                 H, NC::Int64, site::Int64;
                 kwargs...
                )
    NH = size(H)[1]
    mu = zeros(ComplexF64,1, NC) #on_host_zeros(dt_cplx, NR, NC)
    sitevector = zeros(NH,1)
    sitevector[site] = 1
    psi_in = sitevector #KPM.maybe_on_device(sitevector) 
    KPM.kpm_1d!(H, NC, 1, NH, mu, psi_in; kwargs...)
    mu = dropdims(mu, dims=1)
    mu = real.(mu)
    return mu
end
