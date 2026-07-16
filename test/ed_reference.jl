# Exact-diagonalization Kubo–Bastin reference (test-only).
#
# Conventions match the package: current operators are J_α = iħ v_α with
# (J_α)_ij = H_ij (r_i - r_j)_α built from the true bond displacement, and
# ħ = e = 1 so that conductivities are quoted in units of e²/h (= 1/2π in
# natural units; the 2π conversion is applied before returning).

module EDReference

using LinearAlgebra
using SparseArrays

export haldane_model, haldane_bloch, chern_number_fhs,
       ed_kubo_bastin, ed_kubo_bastin_broadened, ed_hall_conductivity_T0,
       ring_model, flux_ring_model, bdg_matrix, bdg_matrix_singlet,
       ed_two_energy_response

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
    haldane_bloch(; t=1.0, t2=0.2, ϕ=π/2, m=0.0)

Bloch Hamiltonian h(k) of the same Haldane model as `haldane_model`
(periodic gauge, k in reciprocal coordinates): returns a function
`(k1, k2) -> 2×2 Hermitian matrix` with k = k1·b1 + k2·b2, k1, k2 ∈ [0, 1).
"""
function haldane_bloch(; t::Real=1.0, t2::Real=0.2, ϕ::Real=π/2, m::Real=0.0)
    # cell offsets of the hops, matching haldane_model exactly:
    # NN B <- A from cells (0,0), (1,0), (0,1); NNN along a1, a2-a1, -a2
    nnn = ((1, 0), (-1, 1), (0, -1))
    function hk(k1, k2)
        phase(off) = cis(2π * (k1 * off[1] + k2 * off[2]))
        hBA = -t * (1 + phase((1, 0)) + phase((0, 1)))
        hAA = m + 0im
        hBB = -m + 0im
        # real-space hop <A_{R+v}|H|A_R> = t2 e^{iϕ} contributes
        # e^{iϕ} e^{-ik·v} + h.c. => 2 t2 cos(k·v - ϕ)
        for off in nnn
            hAA += t2 * cis(ϕ) * conj(phase(off)) + t2 * cis(-ϕ) * phase(off)
            hBB += t2 * cis(-ϕ) * conj(phase(off)) + t2 * cis(ϕ) * phase(off)
        end
        return [hAA conj(hBA); hBA hBB]
    end
    return hk
end

"""
    chern_number_fhs(hk; Nk=24, band=1)

Chern number of band `band` of a 2×2 Bloch Hamiltonian `hk(k1, k2)` (reciprocal
coordinates) via the Fukui–Hatsugai–Suzuki lattice-gauge method [JPSJ 74, 1674
(2005)]. Built purely from wavefunction overlaps — no velocity operators or
Kubo formula — so it independently anchors the absolute sign convention
σ_xy = C e²/h of the ED and KPM Hall calculations.
"""
function chern_number_fhs(hk; Nk::Int=24, band::Int=1)
    u = Matrix{Vector{ComplexF64}}(undef, Nk, Nk)
    for i in 1:Nk, j in 1:Nk
        F = eigen(Hermitian(hk((i - 1) / Nk, (j - 1) / Nk)))
        u[i, j] = F.vectors[:, band]
    end
    wrap(i) = mod(i - 1, Nk) + 1
    total = 0.0
    for i in 1:Nk, j in 1:Nk
        u00 = u[i, j]
        u10 = u[wrap(i + 1), j]
        u11 = u[wrap(i + 1), wrap(j + 1)]
        u01 = u[i, wrap(j + 1)]
        total += angle(dot(u00, u10) * dot(u10, u11) * dot(u11, u01) * dot(u01, u00))
    end
    # ⟨u_k|u_{k+δ}⟩ ≈ e^{-i A·δ} for the standard Berry connection
    # A = i⟨u|∂u⟩, so the counterclockwise plaquette angle accumulates
    # -Ω dk² with Ω = i(⟨∂₁u|∂₂u⟩ - ⟨∂₂u|∂₁u⟩) — the curvature for which
    # TKNN reads σ_xy = +C e²/h. Hence the sign flip.
    return -total / (2π)
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

    # integration window: whole occupied region, padded by the broadening
    # tails; the grid must resolve the Lorentzians (≥ ~20 points per eta)
    lo = minimum(ev) - 20 * eta
    hi = isinf(beta) ? Ef : maximum(ev) + 20 * eta
    grid_N = max(grid_N, ceil(Int, 20 * (hi - lo) / eta))
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

"""
    ring_model(N; t=1.0) -> (h, pos, disp)

N-site periodic one-dimensional nearest-neighbor ring reference model.
"""
function ring_model(N::Int; t::Real=1.0)
    N > 0 || throw(ArgumentError("ring_model: N must be positive"))
    h = spzeros(Float64, N, N)
    for i in 1:N
        h[i, mod1(i + 1, N)] = -Float64(t)
        h[i, mod1(i - 1, N)] = -Float64(t)
    end
    pos = hcat(collect(0.0:N-1), zeros(N))
    function disp(i, j)
        dx = pos[i, 1] - pos[j, 1]
        dx > N / 2 && (dx -= N)
        dx < -N / 2 && (dx += N)
        return [dx, 0.0]
    end
    return h, pos, disp
end

"""
    flux_ring_model(N; t=1.0, phi=0.35) -> (h, pos, disp)

N-site periodic ring with directed nearest-neighbor hopping
`h[i, i+1] = -t * exp(im * phi)` and total flux `N * phi`.
"""
function flux_ring_model(N::Int; t::Real=1.0, phi::Real=0.35)
    N > 0 || throw(ArgumentError("flux_ring_model: N must be positive"))
    h = spzeros(ComplexF64, N, N)
    hopping = -Float64(t) * cis(Float64(phi))
    for i in 1:N
        j = mod1(i + 1, N)
        h[i, j] = hopping
        h[j, i] = conj(hopping)
    end
    _, pos, disp = ring_model(N; t=t)
    return h, pos, disp
end

"""
    bdg_matrix(h, mu, U, n, Δ) -> Matrix{ComplexF64}

Assemble the dense reduced spin-singlet Nambu BdG reference matrix.
"""
function bdg_matrix(h, mu, U, n, Δ)
    ξ = Matrix(h) - mu * I - Diagonal(U .* n ./ 2)
    D = Diagonal(Δ)
    return ComplexF64[ξ D; Diagonal(conj.(Δ)) -ξ]
end

"""
    bdg_matrix_singlet(h, mu, U, n, Δ) -> Matrix{ComplexF64}

Assemble the standard same-valley spin-singlet Nambu matrix with hole block
`-conj(xi)`.
"""
function bdg_matrix_singlet(h, mu, U, n, Δ)
    ξ = Matrix(h) - mu * I - Diagonal(U .* n ./ 2)
    D = Diagonal(Δ)
    return ComplexF64[ξ D; Diagonal(conj.(Δ)) -conj(ξ)]
end

"""
    ed_two_energy_response(H, Jl, Jr; beta, eta, omega=0.0, Ef=0.0)
        -> ComplexF64

Dense Lehmann-sum reference for the generic two-energy response.
"""
function ed_two_energy_response(H, Jl, Jr; beta, eta, omega=0.0, Ef=0.0)
    eta == 0 && !(omega == 0 && isfinite(beta)) &&
        throw(ArgumentError("eta=0 requires omega=0 and finite beta"))

    F = eigen(Hermitian(Matrix(H)))
    Jl_e = F.vectors' * Matrix(Jl) * F.vectors
    Jr_e = F.vectors' * Matrix(Jr) * F.vectors
    fermi(e) = isinf(beta) ? ((e < Ef) + (e <= Ef)) / 2 :
                             1 / (exp(beta * (e - Ef)) + 1)
    occupations = fermi.(F.values)
    acc = zero(ComplexF64)
    for p in eachindex(F.values), q in eachindex(F.values)
        delta_E = F.values[p] - F.values[q]
        divided_difference = if eta == 0
            if abs(delta_E) < 1e-8
                fm = fermi((F.values[p] + F.values[q]) / 2)
                -beta * fm * (1 - fm)
            else
                (occupations[p] - occupations[q]) / delta_E
            end
        else
            (occupations[p] - occupations[q]) /
                (omega + delta_E + im * eta)
        end
        acc += Jl_e[p, q] * Jr_e[q, p] * divided_difference
    end
    return ComplexF64(acc)
end

export square_model, bdg_peierls_matrix

"""
    square_model(Lx, Ly; t=1.0) -> (h, pos, disp)

Periodic square-lattice nearest-neighbor model with hopping `-t`, coordinates
`(ix - 1, iy - 1)`, and column-major site index
`i = ix + (iy - 1) * Lx`. The returned closure supplies minimum-image
displacements in both periodic directions.
"""
function square_model(Lx::Int, Ly::Int; t::Real=1.0)
    Lx > 0 || throw(ArgumentError("square_model: Lx must be positive"))
    Ly > 0 || throw(ArgumentError("square_model: Ly must be positive"))
    N = Lx * Ly
    site(ix, iy) = mod1(ix, Lx) + (mod1(iy, Ly) - 1) * Lx
    h = spzeros(Float64, N, N)
    pos = zeros(Float64, N, 2)

    for iy in 1:Ly, ix in 1:Lx
        i = site(ix, iy)
        pos[i, :] .= (ix - 1, iy - 1)
        for (jx, jy) in ((ix + 1, iy), (ix - 1, iy),
                         (ix, iy + 1), (ix, iy - 1))
            h[i, site(jx, jy)] = -Float64(t)
        end
    end

    function disp(i, j)
        dx = pos[i, 1] - pos[j, 1]
        dy = pos[i, 2] - pos[j, 2]
        dx > Lx / 2 && (dx -= Lx)
        dx < -Lx / 2 && (dx += Lx)
        dy > Ly / 2 && (dy -= Ly)
        dy < -Ly / 2 && (dy += Ly)
        return [dx, dy]
    end
    return h, pos, disp
end

"""
    bdg_peierls_matrix(h, pos, disp, A::Real; q=nothing, dir::Integer=1,
                        hole_convention::Symbol=:singlet)
        -> Matrix{ComplexF64}

Build only the kinetic Nambu part of the Peierls-coupled BdG matrix. Local
chemical-potential, Hartree, and pairing terms do not couple to the vector
potential. A real modulation `u(m) = cos(q ⋅ m)` is used when `q` is given,
so the result remains Hermitian. The singlet hole block is `-conj(h(A))`;
the intervalley hole block uses the same `h` with the opposite Peierls phase.
"""
function bdg_peierls_matrix(h, pos, disp, A::Real; q=nothing, dir::Integer=1,
                            hole_convention::Symbol=:singlet)
    N = size(h, 1)
    size(h, 2) == N || throw(ArgumentError("bdg_peierls_matrix: h must be square"))
    size(pos, 1) == N || throw(ArgumentError("bdg_peierls_matrix: incompatible pos"))
    ndim = size(pos, 2)
    1 <= dir <= ndim || throw(ArgumentError("bdg_peierls_matrix: invalid dir=$dir"))
    q === nothing || length(q) == ndim ||
        throw(ArgumentError("bdg_peierls_matrix: q has length $(length(q)); expected $ndim"))
    hole_convention in (:intervalley, :singlet) ||
        throw(ArgumentError("bdg_peierls_matrix: invalid hole_convention=$hole_convention"))

    hp = zeros(ComplexF64, N, N)
    hh = zeros(ComplexF64, N, N)
    I, J, V = findnz(sparse(h))
    for k in eachindex(V)
        i = I[k]
        j = J[k]
        d = disp(i, j)
        m = [pos[j, ν] + d[ν] / 2 for ν in 1:ndim]
        u = q === nothing ? 1.0 : cos(dot(q, m))
        hp[i, j] = V[k] * exp(im * A * d[dir] * u)
        hole_hopping = hole_convention === :singlet ? conj(V[k]) : V[k]
        hh[i, j] = -hole_hopping * exp(-im * A * d[dir] * u)
    end
    H = [hp zeros(ComplexF64, N, N);
         zeros(ComplexF64, N, N) hh]
    @assert ishermitian(H)
    return Matrix{ComplexF64}(H)
end

end # module
