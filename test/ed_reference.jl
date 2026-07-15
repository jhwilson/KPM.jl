# Exact-diagonalization Kubo–Bastin reference (test-only).
#
# Conventions match the package: current operators are J_α = iħ v_α with
# (J_α)_ij = H_ij (r_i - r_j)_α built from the true bond displacement, and
# ħ = e = 1 so that conductivities are quoted in units of e²/h (= 1/2π in
# natural units; the 2π conversion is applied before returning).

module EDReference

using LinearAlgebra
using SparseArrays

export haldane_model, ed_kubo_bastin, ed_kubo_bastin_broadened, ed_hall_conductivity_T0

"""
    haldane_model(Lx, Ly; t=1.0, t2=0.2, ϕ=π/2, m=0.0)

Haldane model on an Lx×Ly honeycomb torus: NN hopping -t, complex NNN hopping
t2·e^{±iϕ} with the standard chirality pattern, staggered mass ±m. Returns
`(H, Jx, Jy, area)` with sparse H (Hermitian), bond-current operators
(J_α)_ij = H_ij (r_i - r_j)_α, and the total sample area.
"""
function haldane_model(Lx::Int, Ly::Int; t::Real=1.0, t2::Real=0.2, ϕ::Real=π/2, m::Real=0.0)
    a1 = [sqrt(3), 0.0]
    a2 = [sqrt(3) / 2, 1.5]
    δAB = [sqrt(3) / 2, 0.5]           # B-site offset within the cell

    cell(x, y) = mod(y - 1, Ly) + 1 + Ly * mod(x - 1, Lx)   # 1-based cell index
    siteA(x, y) = 2 * cell(x, y) - 1
    siteB(x, y) = 2 * cell(x, y)

    N = 2 * Lx * Ly
    I_, J_, V_ = Int[], Int[], ComplexF64[]
    Vx_, Vy_ = ComplexF64[], ComplexF64[]

    # add hop j -> i with amplitude amp and true displacement d = r_i - r_j,
    # plus its Hermitian conjugate
    function addhop!(i, j, amp, d)
        push!(I_, i); push!(J_, j); push!(V_, amp)
        push!(Vx_, amp * d[1]); push!(Vy_, amp * d[2])
        push!(I_, j); push!(J_, i); push!(V_, conj(amp))
        push!(Vx_, -conj(amp) * d[1]); push!(Vy_, -conj(amp) * d[2])
        return nothing
    end

    for x in 1:Lx, y in 1:Ly
        # NN: B(x,y) <- A from the three neighboring cells; displacements are
        # the three A->B bond vectors
        addhop!(siteB(x, y), siteA(x, y), -t + 0im, δAB)                    # δ1
        addhop!(siteB(x, y), siteA(x + 1, y), -t + 0im, δAB .- a1)          # δ2
        addhop!(siteB(x, y), siteA(x, y + 1), -t + 0im, δAB .- a2)          # δ3

        # NNN with Haldane chirality: on sublattice A the hops along
        # +a1, a2-a1, -a2 carry phase e^{+iϕ}; on B the same displacements
        # carry e^{-iϕ}. (Any consistent chirality convention is fine — the
        # ED reference below fixes the resulting Chern number.)
        for d in (a1, a2 .- a1, .-a2)
            xoff, yoff = _cell_offset(d, a1, a2)
            addhop!(siteA(x + xoff, y + yoff), siteA(x, y), t2 * cis(ϕ), d)
            addhop!(siteB(x + xoff, y + yoff), siteB(x, y), t2 * cis(-ϕ), d)
        end

        # staggered mass (no contribution to J: d = 0)
        push!(I_, siteA(x, y)); push!(J_, siteA(x, y)); push!(V_, m + 0im)
        push!(Vx_, 0); push!(Vy_, 0)
        push!(I_, siteB(x, y)); push!(J_, siteB(x, y)); push!(V_, -m + 0im)
        push!(Vx_, 0); push!(Vy_, 0)
    end

    H = sparse(I_, J_, V_, N, N)
    Jx = sparse(I_, J_, Vx_, N, N)
    Jy = sparse(I_, J_, Vy_, N, N)
    area = Lx * Ly * abs(a1[1] * a2[2] - a1[2] * a2[1])
    return H, Jx, Jy, area
end

# integer cell offset of a lattice displacement d = xoff*a1 + yoff*a2
function _cell_offset(d, a1, a2)
    M = [a1 a2]
    c = M \ collect(d)
    return round(Int, c[1]), round(Int, c[2])
end

"""
    ed_kubo_bastin(H, Jα, Jβ, area; Ef, eta, beta=Inf)

σ_αβ(Ef) in units of e²/h, by direct eigenbasis evaluation of the
Kubo–Bastin formula with Lorentzian broadening `eta`:

σ_αβ = (i e² ħ/A) Σ_{mn} f(ε_m) [ vα_{mn} vβ_{nm}/(ε_m-ε_n-iη)²
                                 - vβ_{mn} vα_{nm}/(ε_m-ε_n+iη)² ]

with v_α = J_α/(iħ). This is the ∫dε f(ε) Tr[...] form integrated exactly in
the eigenbasis; its η→0 limit at T=0 for α≠β is `ed_hall_conductivity_T0`.
"""
function ed_kubo_bastin(H, Jα, Jβ, area; Ef::Real, eta::Real, beta::Real=Inf)
    ev, U = eigen(Hermitian(Matrix(H)))
    Va = U' * (Matrix(Jα) ./ im) * U
    Vb = U' * (Matrix(Jβ) ./ im) * U
    D = length(ev)
    f(e) = isinf(beta) ? (e <= Ef ? 1.0 : 0.0) : 1 / (exp(beta * (e - Ef)) + 1)
    acc = zero(ComplexF64)
    for m in 1:D, n in 1:D
        Δ = ev[m] - ev[n]
        acc += f(ev[m]) * (Va[m, n] * Vb[n, m] / (Δ - im * eta)^2 -
                           Vb[m, n] * Va[n, m] / (Δ + im * eta)^2)
    end
    σ = real(im * acc / area)
    return 2π * σ
end

"""
    ed_kubo_bastin_broadened(H, Jα, Jβ, area; Ef, eta, beta=Inf, grid_N=4000)

Like `ed_kubo_bastin`, but with the spectral function δ(ε-H) broadened into a
Lorentzian of the same width `eta` as the Green's function, and the ε-integral
done numerically. This mirrors the KPM regularization, where the damping
kernel broadens both factors — on a discrete spectrum the dissipative
(longitudinal) response only exists with a broadened δ, so this is the right
reference for σ_xx. Returns σ_αβ in units of e²/h.
"""
function ed_kubo_bastin_broadened(H, Jα, Jβ, area; Ef::Real, eta::Real,
                                  beta::Real=Inf, grid_N::Int=4000)
    ev, U = eigen(Hermitian(Matrix(H)))
    Va = U' * (Matrix(Jα) ./ im) * U
    Vb = U' * (Matrix(Jβ) ./ im) * U
    W = Va .* transpose(Vb)            # W_{mn} = vα_{mn} vβ_{nm}
    f(e) = isinf(beta) ? (e <= Ef ? 1.0 : 0.0) : 1 / (exp(beta * (e - Ef)) + 1)

    # integration window: whole occupied region, padded by the broadening tails
    lo = minimum(ev) - 20 * eta
    hi = isinf(beta) ? Ef : maximum(ev) + 20 * eta
    εs = range(lo, hi; length=grid_N)
    dε = step(εs)

    L(x) = (eta / π) / (x^2 + eta^2)
    acc = zero(ComplexF64)
    for ε in εs
        Lv = L.(ε .- ev)
        Gp = @. 1 / (ε - ev + im * eta)^2   # -dG⁺/dε eigenvalues
        Gm = @. 1 / (ε - ev - im * eta)^2
        # Tr[vα δ vβ G⁺′] - Tr[vα G⁻′ vβ δ] with G±′ = -1/(ε-ε_m ± iη)²
        t1 = -transpose(Gp) * (W * Lv)
        t2 = -transpose(Lv) * (W * Gm)
        acc += f(ε) * (t1 - t2) * dε
    end
    return 2π * real(im * acc / area)
end

"""
    ed_hall_conductivity_T0(H, Jα, Jβ, area; Ef)

T=0, η→0 Hall conductivity in units of e²/h (the TKNN form of the
Kubo–Bastin formula): for Ef in a spectral gap this is the Chern number.

σ_αβ = -(2 e² ħ/A) Σ_{m occ, n unocc} Im[vα_{mn} vβ_{nm}]/(ε_m-ε_n)²
"""
function ed_hall_conductivity_T0(H, Jα, Jβ, area; Ef::Real)
    ev, U = eigen(Hermitian(Matrix(H)))
    Va = U' * (Matrix(Jα) ./ im) * U
    Vb = U' * (Matrix(Jβ) ./ im) * U
    occ = findall(<=(Ef), ev)
    unocc = findall(>(Ef), ev)
    acc = 0.0
    for m in occ, n in unocc
        acc += imag(Va[m, n] * Vb[n, m]) / (ev[m] - ev[n])^2
    end
    return 2π * (-2 / area) * acc
end

end # module
