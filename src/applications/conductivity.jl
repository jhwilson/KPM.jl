using DocStringExtensions
using ProgressBars
using Zygote
using Logging
include("dc_cond_util.jl")
include("dc_cond_long.jl")

"""
$(METHODLIST)

Calculate the integrand for conductivity of an energy grid spanning `E_range`
with `N_tilde` total points.  If `E_range` is not set,  automatically set it to
be sightly smaller than full size.  Otherwise an explicit array of `E_grid` can
be passed in. Don't do both.

Either a) pass in a 2d array as moment and a normalization factor;
or b) pass in a Hamiltonian that is rescaled with an optional keyword
`rescale_factor` that default to 1, as well as two current operators Jα and Jβ

- `H_rescale_factor` is the normalization of H. Needed when μ is passed.

- `NR` random vectors. Needed when H is passed

"""
function d_dc_cond end

"""
$(METHODLIST)

Bare Chebyshev Fermi-surface sum at the band center (Garcia et al. Supp.
Eq. 25 structure): Σ_{nm} μ̃nm Tn(0) Tm(0) / a. Proportional to the
longitudinal Kubo–Greenwood conductivity at Ef = b, but **not** in physical
units — the D/A, (1-x_F²) and e²/h factors are not included. For an
absolutely normalized, ED-validated conductivity use [`kubo_bastin_cond`](@ref).
"""
function dc_cond0 end

"""
$(METHODLIST)

Bare Chebyshev Fermi-surface sum at Fermi energy `Ef` (physical units):
Σ_{nm} μ̃nm Tn(x_F) Tm(x_F) / a with x_F = Ef/a. Proportional to the
longitudinal Kubo–Greenwood conductivity, but **not** in physical units —
the D/A, (1-x_F²) and e²/h factors are not included. For an absolutely
normalized, ED-validated conductivity use [`kubo_bastin_cond`](@ref).
"""
function dc_cond_single end




"""
    kubo_bastin_cond(mu2D, a, Ef; b=0.0, NH, area, beta=Inf, kernel=JacksonKernel,
                     NC=size(mu2D, 1), quad_N=8*NC, edge_cutoff=1e-3)

DC conductivity σ_αβ(Ef) from the Kubo–Bastin formula, **in units of e²/h**.

`mu2D` are the 2D moments from `kpm_2d(H_norm, Jα, Jβ, NC, NR, NH)`, where
`H_norm = (H - b I)/a` (see `normalizeH`) and the current operators follow the
bond convention `(J_α)_ij = H_ij (r_i - r_j)_α`, i.e. J_α = iħ v_α (as built in
`examples/GrapheneModel.jl`). With unit-norm probe vectors the moments estimate
`mu2D[n, m] ≈ Tr[Jα T_m(H̃) Jβ T_n(H̃)] / D`, `D = NH`.

Starting from the Kubo–Bastin formula [Bastin et al. 1971; García, Covaci &
Rappoport, PRL 114, 116602 (2015)]

    σ_αβ(μ, T) = (i e² ħ / A) ∫ dε f(ε) Tr[ v_α δ(ε-H) v_β ∂_ε G⁺(ε)
                                           - v_α ∂_ε G⁻(ε) v_β δ(ε-H) ]

and inserting the Chebyshev expansions of δ(x - H̃) and G±(x) on the rescaled
Hamiltonian (x = (ε-b)/a), the trace collapses to the Γnm contraction of
`Γnmμnmαβ`, and the physical conductivity becomes

    σ_αβ(Ef) = -(2 e²/h) · (D / (A a²)) · ∫_{-1}^{1} dx f(a x + b)
               Re[Σ_nm Γnm(x) μ̃nm] / (1-x²)² ,

with μ̃ the kernel- and hn-improved moments. This function evaluates that
integral with a `quad_N`-node Gauss–Chebyshev quadrature, dropping nodes
within `edge_cutoff` of the band edges ±1.

Arguments: `a`, `b` — rescaling of H; `Ef` — Fermi energy (physical units);
`NH` — Hilbert-space dimension D; `area` — sample area A (or volume), in the
same length units as the bond displacements in J; `beta` — inverse
temperature (Inf for T=0).

Index convention: α (the `Jα` passed to `kpm_2d`) is the response direction,
β the field direction; σ_xy > 0 corresponds to Chern number C = +1 with this
sign convention (validated against exact diagonalization in
`test/kubo_bastin_test.jl`).
"""
function kubo_bastin_cond(mu2D, a::Real, Ef::Real; b::Real=0.0,
                          NH::Integer, area::Real,
                          beta::Real=Inf, kernel=JacksonKernel,
                          NC::Int64=size(mu2D, 1), quad_N::Int64=8*NC,
                          edge_cutoff::Real=1e-3)
    NC = min(NC, size(mu2D, 1), size(mu2D, 2))
    μtilde = maybe_to_host(mu2D_apply_kernel_and_h(mu2D[1:NC, 1:NC], NC, kernel))
    ff = fermiFunctions(dt_real(Ef), dt_real(beta))

    nodes, weights = gausschebyshevt(quad_N)
    acc = zero(dt_cplx)
    for (x, w) in zip(nodes, weights)
        abs(x) < 1 - edge_cutoff || continue
        fx = ff(a * x + b)
        fx == 0 && continue
        # Gauss–Chebyshev absorbs 1/√(1-x²); remaining weight is (1-x²)^{-3/2}
        acc += w * fx * Γnmμnmαβ(μtilde, x, NC) / (1 - x^2)^(3/2)
    end
    return -2 * NH / (area * a^2) * real(acc)
end


function d_dc_cond_old(μ, a::Float64; E_grid=nothing, NC::Integer=0, kernel=JacksonKernel)
    μ=complex(μ)
    μ=maybe_to_device(μ) # temporary

    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else
        if NC > size(μ)[1]
            @warn "NC=$(NC) exceeds the maximum size of μ; decreased to $(size(μ)[1])."
        end
        NC = min(size(μ)[1],NC)
    end

    if isnothing(E_grid)
        Erange = [-a+0.01,a-0.01]
        Ntilde = 2 * NC
        E_grid = collect(((0:(Ntilde)).*(Erange[2]-Erange[1]))./Ntilde .+ Erange[1])
    end

    dσE_full = zeros(ComplexF64, size(E_grid))
    idx = findall(abs.(E_grid) .< abs(a))

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)


    for idx_ in idx
        ϵ = E_grid[idx_] / a
        dσE_full[idx_] = Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2)
    end

    return E_grid, dσE_full
end


d_dc_cond(μ, a::Float64, E::Float64; kwargs...) = d_dc_cond(μ, a, [E]; kwargs...)

function d_dc_cond(μ, a::Float64, E::Array{Float64, 1}; NC::Integer=0, kernel=JacksonKernel, dE_order=0)
    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else
        if NC > size(μ)[1]
            @warn "NC=$(NC) exceeds the maximum size of μ; decreased to $(size(μ)[1])."
        end
        NC = min(size(μ)[1],NC)
    end

    dσE = zeros(Float64, size(E))

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)

    f(x) = _d_dc_cond_single(μtilde, a, x, NC)
    g(x) = real(Zygote.forwarddiff(f, x))
    for dE_order_i = 1:dE_order
        g = real ∘ g'
    end

    idx = abs.(E) .< abs(a)
    dσE[idx] .= real(g.(E[idx]))
    return dσE

end


function _d_dc_cond_single(μtilde, H_rescale_factor::Float64, E, NC::Int64)
    a = H_rescale_factor

    ϵ = E / a
    dσE = real(Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2))

    return dσE
end

function dc_cond0(mu, H_rescale_factor::Number; kernel=JacksonKernel, NC::Int64=size(mu, 1))
    mu_tilde = mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel)
    # Σ_{nm even} μ̃nm Tn(0) Tm(0): Tn(0) = +1 for n ≡ 0 (mod 4), -1 for n ≡ 2 (mod 4)
    oneone = sum(mu_tilde[1:4:NC, 1:4:NC]);
    onethree = sum(mu_tilde[1:4:NC, 3:4:NC]);
    threeone = sum(mu_tilde[3:4:NC, 1:4:NC]);
    threethree = sum(mu_tilde[3:4:NC, 3:4:NC]);
    # 1/a is one factor of the full physical normalization; see the docstring
    # and kubo_bastin_cond for absolute units
    return (oneone + threethree - onethree - threeone) / H_rescale_factor
end

function dc_cond_single(mu, H_rescale_factor::Number, Ef::Number; kernel=JacksonKernel, NC::Int64=size(mu, 1))
    Ef_tilde = Ef / H_rescale_factor
    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel))

    chebyT_poly_all = chebyshevT_accurate.((1:NC) .- 1, Ef_tilde)
    @debug "$(size(chebyT_poly_all)), $(size(mu_tilde))"

    mu_tilde .*= chebyT_poly_all
    mu_tilde .*= chebyT_poly_all'

    # 1/a is one factor of the full physical normalization; see the docstring
    # and kubo_bastin_cond for absolute units
    return sum(mu_tilde) / H_rescale_factor

end
