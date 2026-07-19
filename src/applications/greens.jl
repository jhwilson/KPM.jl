using DocStringExtensions

"""
$(METHODLIST)

Reconstruct the retarded or advanced Green function

    G^{R/A}_uv(E) = ⟨u|(E - H ± iη)^{-1}|v⟩

from matrix-element moments `mu[n, p] = ⟨u_p|T_{n-1}(H_norm)|v_p⟩` (size
NC × M; a vector is treated as a single pair) with the rescaling
`H = a H_norm + b`. Returns the full complex value on the energy grid `E`
(physical units), size `length(E) × M`; scalar `E` returns a length-M vector,
and a vector `mu` with scalar `E` returns a scalar.

Both the real and imaginary parts come from the same moments through the
closed-form Chebyshev expansion of the resolvent [Pixley et al., PRB 95,
235101 (2017), Eq. (14); Weiße et al., RMP 78, 275 (2006)]:

    G^{R/A}(E) = -(s i)/(a sin θ̃) [μ₀ g₀ + 2 Σ_{n≥1} μₙ gₙ e^{-i s n θ̃}],
    θ̃ = acos(z̃),  z̃ = (E - b + i s η)/a,  s = +1 (retarded) / -1 (advanced),

i.e. the Kramers–Kronig transform of the spectral part is analytic in
Chebyshev space — no numerical Hilbert transform is involved. The principal
branch of `acos` puts Im θ̃ on the decaying side for either sign of s, and
`sin θ̃` is computed from θ̃ itself so both factors always sit on the same
branch. Energies outside the band analytically continue to the correct real
G; only z̃ = ±1 exactly is singular.

Exactly one of the two broadening routes must be chosen:

  - `kernel`: finite-order damping at real energies (η → 0⁺ implicit). Pass
    `LorentzKernels(λ)` when the truncation should act as a Lorentzian lifetime
    broadening; the damping is uniform in θ̃, so the equivalent half-width is
    position dependent, η(E) ≈ (a λ / NC) √(1-x̃²) (≈ a λ / NC at band center).
    That correspondence holds where the kernel's reflected `e^{-2λ}` term is
    negligible — use λ ≳ 2, and expect few-percent accuracy only by λ ≈ 4.
    `JacksonKernel` gives near-Gaussian resolution (positive spectral weight,
    but *not* a constant imaginary self-energy).

  - `eta` (> 0, physical units): direct Chebyshev-polynomial Green function
    (CPGF) at the complex energy E ± iη with undamped coefficients (gₙ = 1).
    Strictly in band with η̃ = η/a ≪ 1-x̃², the series tail decays like
    exp(-n η̃ / √(1-x̃²)); at the band edges the decay softens to
    exp(-n √(2η̃)). The emitted warning uses the exact complex `acos`, so it
    covers all regimes: heed it by increasing `NC` or `eta`.

  - `b` : center shift of the rescaling (default 0). `NC` : truncate to the
    first NC moments (default: all).
"""
function greens end

function greens(
    mu::AbstractVecOrMat,
    H_rescale_factor::Real,
    E;
    b::Real = 0.0,
    kernel = nothing,
    eta = nothing,
    branch::Symbol = :retarded,
    NC::Int = 0,
)
    a = H_rescale_factor
    @assert a > 0

    if (kernel === nothing) == (eta === nothing)
        throw(
            ArgumentError(
                "choose exactly one broadening route: `kernel=` (finite-order " *
                "damping at real energies) or `eta=` (> 0; direct CPGF at " *
                "complex energy).",
            ),
        )
    end
    if eta !== nothing && !(eta > 0)
        throw(ArgumentError("`eta` must be positive (physical units); got $(eta)."))
    end
    s = if branch === :retarded
        1
    elseif branch === :advanced
        -1
    else
        throw(ArgumentError("`branch` must be :retarded or :advanced; got $(branch)."))
    end

    mu_mat = mu isa AbstractVector ? reshape(mu, :, 1) : mu
    NC >= 0 ||
        throw(ArgumentError("NC must be nonnegative (0 = use all moments); got $(NC)."))
    size(mu_mat, 1) >= 1 || throw(ArgumentError("mu must contain at least one moment."))
    if NC == 0
        NC = size(mu_mat, 1)
    else
        if NC > size(mu_mat, 1)
            @warn "NC=$(NC) exceeds the number of moments; decreased to $(size(mu_mat, 1))."
        end
        NC = min(NC, size(mu_mat, 1))
    end

    g = kernel === nothing ? ones(dt_real, NC) : kernel.(0:(NC-1), NC)
    μtilde = mu_mat[1:NC, :] .* (hn.(0:(NC-1)) .* g)

    E_vec = E isa Number ? [E] : E
    # signed zero matters on the real axis: s*0.0 = ±0.0 selects the side of
    # the branch cut of acos, so the η = 0 kernel route continues correctly
    # to real energies outside the band.
    η = eta === nothing ? 0.0 : float(eta)
    z̃ = complex.((E_vec .- b) ./ a, s * η / a)
    θ̃ = acos.(z̃)

    if eta !== nothing
        tail = exp((NC - 1) * maximum(s .* imag.(θ̃)))
        if tail > 1e-3
            @warn "CPGF series tail |e^{-isNCθ̃}| ≈ $(round(tail, sigdigits=2)) " *
                  "is not negligible; increase NC or eta." maxlog = 1
        end
    end

    G = Matrix{dt_cplx}(undef, length(E_vec), size(μtilde, 2))
    # chunk the e^{-isnθ̃} phase matrix so memory stays bounded at ~16 MB
    chunk = max(1, cld(1_000_000, NC))
    n_row = collect(0:(NC-1))'
    for lo = 1:chunk:length(E_vec)
        hi = min(lo + chunk - 1, length(E_vec))
        θc = view(θ̃, lo:hi)
        W = exp.((-im * s) .* θc .* n_row)
        G[lo:hi, :] .= (W * μtilde) .* ((-im * s) ./ (a .* sin.(θc)))
    end

    if E isa Number
        return mu isa AbstractVector ? G[1, 1] : vec(G[1, :])
    end
    return mu isa AbstractVector ? vec(G) : G
end

"""
$(METHODLIST)

Local density of states from *diagonal* matrix-element moments
`mu[n, p] = ⟨u_p|T_{n-1}(H_norm)|u_p⟩`:

    ρ_u(E) = -Im G^R_uu(E) / π  ≥ 0.

Same arguments and broadening routes as [`greens`](@ref); returns a real
array. Only meaningful for equal left/right probes — for independent bra/ket
pairs use [`spectral_function`](@ref), whose off-diagonal entries are
complex. With unit site probes and the per-state DOS convention, summing
over all sites gives NH times the [`dos`](@ref) reconstruction.
"""
function ldos(mu::AbstractVecOrMat, H_rescale_factor::Real, E; kwargs...)
    :branch in keys(kwargs) &&
        throw(ArgumentError("ldos is -Im G^R/π by definition; `branch` cannot be chosen"))
    G = greens(mu, H_rescale_factor, E; kwargs..., branch = :retarded)
    return -imag.(G) ./ pi
end

"""
$(METHODLIST)

Spectral function from matrix-element moments (same layout and broadening
routes as [`greens`](@ref)):

    A_uv(E) = (i/2π) (G^R - G^A)_uv(E).

`A_uv` is the (u, v) matrix element of δ(E - H) broadened by the chosen
route: complex in general, real and ≥ 0 on the diagonal, where it reduces to
-Im G^R_uu/π (= [`ldos`](@ref)). Integrated over all E it recovers the
zeroth moment ⟨u|v⟩. Both branches are reconstructed from the same moments —
no extra Chebyshev recurrences.
"""
function spectral_function(mu::AbstractVecOrMat, H_rescale_factor::Real, E; kwargs...)
    :branch in keys(kwargs) && throw(
        ArgumentError("spectral_function uses both branches; `branch` cannot be chosen"),
    )
    GR = greens(mu, H_rescale_factor, E; kwargs..., branch = :retarded)
    GA = greens(mu, H_rescale_factor, E; kwargs..., branch = :advanced)
    return (im / (2 * pi)) .* (GR .- GA)
end
