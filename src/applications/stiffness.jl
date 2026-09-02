# Generic two-point response layer; stiffness vertices arrive in a later stage.

"""
    two_energy_response(mu2D::AbstractMatrix, a::Real; b::Real=0.0,
                        beta::Real, eta::Real, omega::Real=0.0,
                        Ef::Real=0.0, NH::Integer, volume::Real,
                        g_J::Real=1.0, kernel=JacksonKernel,
                        NC::Integer=size(mu2D, 1), Np::Integer=2*NC)
        -> ComplexF64

Reconstruct the generic two-energy response

    Pi(omega) = (g_J / volume) sum_{p,q} (Jalpha)_{pq} (Jbeta)_{qp}
                (f_p - f_q) / (omega + E_p - E_q + i eta)

from moments produced by `kpm_2d`. The moment convention is

    mu2D[n,m] = Tr[Jalpha T_{m-1}(H_norm) Jbeta T_{n-1}(H_norm)] / D,

where `H_norm = (H - b I) / a` and `D = NH`. Consequently the reconstruction
row/node axis pairs with eigenstate `q`, the column/node axis pairs with
eigenstate `p`, and the Fermi kernel has `+f(E_P)` on the row-node axis.

Gauss--Chebyshev quadrature absorbs both spectral
`1 / (pi sqrt(1-x^2))` weights, so there is no edge cutoff and no residual
power of `1-x^2`. There is also no `a` or `a^2` Jacobian: the two spectral
`1/a` factors cancel the two physical-energy measures `dE = a dx` exactly.

When `eta == 0`, only the static finite-temperature case (`omega == 0` and
finite `beta`) is defined. It uses the continuous Fermi divided difference,
including `-beta*f*(1-f)` for coincident energies.

The result carries `[J]^2 / volume` in the caller's units. The primary use is
BdG response, where the Fermi level is normally `Ef = 0` because the chemical
potential is already included in `H_BdG`.
"""
function two_energy_response(
    mu2D::AbstractMatrix,
    a::Real;
    b::Real = 0.0,
    beta::Real,
    eta::Real,
    omega::Real = 0.0,
    Ef::Real = 0.0,
    NH::Integer,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    NC::Integer = size(mu2D, 1),
    Np::Integer = 2 * NC,
)
    size(mu2D, 1) == size(mu2D, 2) ||
        throw(ArgumentError("two_energy_response: mu2D must be square (got $(size(mu2D)))"))
    NC >= 2 || throw(ArgumentError("two_energy_response: NC must be at least 2 (got $NC)"))
    size(mu2D, 1) >= NC || throw(
        ArgumentError(
            "two_energy_response: mu2D size $(size(mu2D)) is smaller than NC=$NC",
        ),
    )
    a > 0 || throw(ArgumentError("two_energy_response: a must be positive (got $a)"))
    eta >= 0 ||
        throw(ArgumentError("two_energy_response: eta must be nonnegative (got $eta)"))
    beta > 0 ||
        throw(ArgumentError("two_energy_response: beta must be positive (got $beta)"))
    Np > 0 || throw(ArgumentError("two_energy_response: Np must be positive (got $Np)"))
    NH > 0 || throw(ArgumentError("two_energy_response: NH must be positive (got $NH)"))
    volume > 0 ||
        throw(ArgumentError("two_energy_response: volume must be positive (got $volume)"))
    eta == 0 &&
        !(omega == 0 && isfinite(beta)) &&
        throw(ArgumentError("eta=0 requires omega=0 and finite beta"))

    NC_int = Int(NC)
    Np_int = Int(Np)
    mu_tilde = mu2D_apply_kernel_and_h(mu2D[1:NC_int, 1:NC_int], NC_int, kernel)
    nodes, _ = gausschebyshevt(Np_int)
    C = cos.((0:(NC_int-1)) .* acos.(nodes'))
    Gamma = transpose(C) * maybe_to_host(mu_tilde) * C
    E = a .* nodes .+ b
    fermi = fermiFunctions(Float64(Ef), Float64(beta))
    f = fermi.(E)

    if eta == 0
        threshold = 1e-8 * max(a, one(a))
        F = [
            begin
                delta_E = E[P] - E[Q]
                if abs(delta_E) < threshold
                    fm = fermi((E[P] + E[Q]) / 2)
                    -beta * fm * (1 - fm)
                else
                    (f[P] - f[Q]) / delta_E
                end
            end for P in eachindex(E), Q in eachindex(E)
        ]
    else
        F = [
            (f[P] - f[Q]) / (omega + E[P] - E[Q] + im * eta) for
            P in eachindex(E), Q in eachindex(E)
        ]
    end

    return ComplexF64(g_J * NH / volume / Np_int^2 * sum(Gamma .* F))
end

"""
    nambu_current_q(h::SparseMatrixCSC, pos::AbstractMatrix{<:Real},
                    q::AbstractVector{<:Real}; dir::Integer=1, disp=nothing,
                    hole_convention::Symbol=:conjugate)
        -> SparseMatrixCSC{ComplexF64, Int}

Build the finite-wavevector current vertex in the reduced Nambu convention.
For a Peierls field `A_x(r) = A u_q(r)`, with
`u_q(r) = exp(-im * dot(q, r))`, the hopping is

    h(A)_ij = h_ij exp(+im A d_dir u_q(m_ij)),

where `d = r_i - r_j` is the bond displacement and the unwrapped midpoint is
`m_ij = r_j + d / 2`. Thus `J_BdG(q) = (1/im) dH_BdG/dA` has entries

    J(q)_ij         = h_ij d_dir exp(-im q ⋅ m_ij),
    J(q)_(i+N,j+N) = j_hole exp(-im q ⋅ m_ij),

and zero off-diagonal Nambu blocks. For `hole_convention=:conjugate`
(`:singlet` is an accepted alias),
differentiating `-conj(h(A))` gives `j_hole = conj(h_ij d_dir)`. For
`hole_convention=:intervalley`, differentiating the same-`h`, opposite-Peierls-
phase hole block gives `j_hole = h_ij d_dir`. The conjugation in the conjugate
branch acts on the bond factor, not on the phase. Both branches coincide for
real `h`. Zero-displacement entries are skipped.

The input `h` must be the unrescaled hopping matrix; using a rescaled hopping
would spuriously divide a response by the square of its energy scale. With the
package's anti-Hermitian bond-current convention both vertices obey
`J(q)' == -J(-q)`; at `q = 0` and real `h`, both Nambu blocks equal the bond
current `(J_dir)_ij = h_ij d_dir`.

`disp(i, j)` should return the minimum-image displacement `r_i - r_j`.
When `disp=nothing`, raw coordinate differences are used, which is correct
only for open boundaries.

On periodic geometries `q` must be commensurate with the simulation torus
(a reciprocal-grid vector, e.g. `2π/L` multiples): the directed unwrapped
midpoints of a wrapped bond differ by a lattice vector `L`, so
`J(q)' == -J(-q)` — and the single-valuedness of `A(r) = exp(-im q ⋅ r)`
itself — require `exp(im q ⋅ L) = 1`. Incommensurate `q` silently breaks
the adjoint identity.

This operator is kinetic-only by construction: it is built from `h`, and
pairing never enters, realizing the rigid-Δ convention.
"""
function nambu_current_q(
    h::SparseMatrixCSC,
    pos::AbstractMatrix{<:Real},
    q::AbstractVector{<:Real};
    dir::Integer = 1,
    disp = nothing,
    hole_convention::Symbol = :conjugate,
)
    N = size(h, 1)
    size(h, 2) == N ||
        throw(ArgumentError("nambu_current_q: h must be square (got $(size(h)))"))
    size(pos, 1) == N ||
        throw(ArgumentError("nambu_current_q: pos has $(size(pos, 1)) rows; expected $N"))
    ndim = size(pos, 2)
    length(q) == ndim ||
        throw(ArgumentError("nambu_current_q: q has length $(length(q)); expected $ndim"))
    1 <= dir <= ndim || throw(
        ArgumentError("nambu_current_q: dir must satisfy 1 <= dir <= $ndim (got $dir)"),
    )
    hole_convention =
        _canonical_hole_convention(hole_convention; caller = "nambu_current_q")

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    I, J, V = findnz(h)
    sizehint!(rows, 2length(V))
    sizehint!(cols, 2length(V))
    sizehint!(vals, 2length(V))

    for k in eachindex(V)
        i = I[k]
        j = J[k]
        d = disp === nothing ? collect(view(pos, i, :) .- view(pos, j, :)) : disp(i, j)
        length(d) == ndim || throw(
            ArgumentError(
                "nambu_current_q: disp($i, $j) has length $(length(d)); expected $ndim",
            ),
        )
        all(iszero, d) && continue
        phase = exp(-im * sum(q[ν] * (pos[j, ν] + d[ν] / 2) for ν = 1:ndim))
        bond = V[k] * d[dir]
        hole_bond = hole_convention === :conjugate ? conj(bond) : bond
        push!(rows, i);
        push!(cols, j);
        push!(vals, ComplexF64(bond * phase))
        push!(rows, i + N);
        push!(cols, j + N);
        push!(vals, ComplexF64(hole_bond * phase))
    end

    return sparse(rows, cols, vals, 2N, 2N)
end

"""
    nambu_diamagnetic(h::SparseMatrixCSC, pos::AbstractMatrix{<:Real};
                      dir::Integer=1, disp=nothing,
                      hole_convention::Symbol=:conjugate)
        -> SparseMatrixCSC{ComplexF64, Int}

Build the zero-wavevector second Peierls derivative
`Dhat = d^2 H_BdG / dA_dir^2`. The finite-`q` phases from `u_q u_-q`
cancel, so this diamagnetic operator is independent of `q`. For bond
displacement `d = r_i - r_j`, its nonzero entries are

    Dhat_ij             = -h_ij d_dir^2
    Dhat_(i+N,j+N)      = +conj(h_ij) d_dir^2  (:conjugate; :singlet alias)
                         +h_ij d_dir^2        (:intervalley).

The off-diagonal Nambu blocks vanish and zero-displacement entries are
skipped. For Hermitian `h` and consistent bond displacements the construction
is Hermitian in both hole conventions. `disp(i, j)` should return the
minimum-image displacement on periodic geometries; raw coordinate differences
are used when `disp=nothing`.

This operator is kinetic-only by construction: it is built from `h`, and
pairing never enters, realizing the rigid-Δ convention.
"""
function nambu_diamagnetic(
    h::SparseMatrixCSC,
    pos::AbstractMatrix{<:Real};
    dir::Integer = 1,
    disp = nothing,
    hole_convention::Symbol = :conjugate,
)
    N = size(h, 1)
    size(h, 2) == N ||
        throw(ArgumentError("nambu_diamagnetic: h must be square (got $(size(h)))"))
    size(pos, 1) == N ||
        throw(ArgumentError("nambu_diamagnetic: pos has $(size(pos, 1)) rows; expected $N"))
    ndim = size(pos, 2)
    1 <= dir <= ndim || throw(
        ArgumentError("nambu_diamagnetic: dir must satisfy 1 <= dir <= $ndim (got $dir)"),
    )
    hole_convention =
        _canonical_hole_convention(hole_convention; caller = "nambu_diamagnetic")

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    I, J, V = findnz(h)
    sizehint!(rows, 2length(V))
    sizehint!(cols, 2length(V))
    sizehint!(vals, 2length(V))

    for k in eachindex(V)
        i = I[k]
        j = J[k]
        d = disp === nothing ? collect(view(pos, i, :) .- view(pos, j, :)) : disp(i, j)
        length(d) == ndim || throw(
            ArgumentError(
                "nambu_diamagnetic: disp($i, $j) has length $(length(d)); expected $ndim",
            ),
        )
        all(iszero, d) && continue
        d2 = d[dir]^2
        particle_bond = -V[k] * d2
        hole_bond = (hole_convention === :conjugate ? conj(V[k]) : V[k]) * d2
        push!(rows, i);
        push!(cols, j);
        push!(vals, ComplexF64(particle_bond))
        push!(rows, i + N);
        push!(cols, j + N);
        push!(vals, ComplexF64(hole_bond))
    end

    # Hermiticity follows from Hermitian h and d_ij^2 == d_ji^2; avoid an
    # O(nnz) runtime assertion here because this vertex is built at scale.
    return sparse(rows, cols, vals, 2N, 2N)
end

"""
    diamagnetic_term(op::BdGOperator, pos::AbstractMatrix{<:Real};
                     dir::Integer=1, disp=nothing, beta::Real,
                     NC::Integer=256, NR::Integer=8, rng=Xoshiro(0),
                     psi_in=nothing, volume::Real, g_J::Real=1.0,
                     kernel=JacksonKernel, Np::Integer=2*NC,
                     rescale_eps::Real=0.2,
                     a::Union{Nothing,Real}=nothing) -> Float64

Compute the lattice diamagnetic contribution

    Dia = (g_J / volume) sum_n f(E_n) <n|Dhat|n>,

where `Dhat` is `nambu_diamagnetic` and the BdG Fermi level is zero. It enters
the transverse electromagnetic kernel as `K_xx(q) = Dia - Pi(q)`. The
Chebyshev moments are probe-averaged estimates of
`Tr[Dhat T_m(H_scaled)] / (2N)`; the final `2N` factor restores the trace.
There is no `a`-Jacobian because the spectral `1/a` cancels `dE = a dx`, as
in `bdg_update`.

The primary entry point is `superfluid_stiffness(...; include_diamagnetic=true)`, which evaluates superconducting and normal terms
with common probes and a common scale (and passes its already device-resident
scaled operator via the internal `Hs` keyword; when supplied, `Hs.a` must
equal the requested scale). With a CUDA GPU active, an operator that cannot
be assembled to the device (matrix-free or dense blocks) throws rather than
mixing host and device residences.
"""
function diamagnetic_term(
    op::BdGOperator,
    pos::AbstractMatrix{<:Real};
    dir::Integer = 1,
    disp = nothing,
    beta::Real,
    NC::Integer = 256,
    NR::Integer = 8,
    rng = Xoshiro(0),
    psi_in = nothing,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    Np::Integer = 2 * NC,
    rescale_eps::Real = 0.2,
    a::Union{Nothing,Real} = nothing,
    Hs = nothing,
)
    op.h isa AbstractMatrix || throw(
        ArgumentError("diamagnetic_term needs an assembled sparse h to build the vertex"),
    )
    size(pos, 1) == op.N || throw(
        ArgumentError("diamagnetic_term: pos has $(size(pos, 1)) rows; expected $(op.N)"),
    )
    beta > 0 || throw(ArgumentError("diamagnetic_term: beta must be positive (got $beta)"))
    volume > 0 ||
        throw(ArgumentError("diamagnetic_term: volume must be positive (got $volume)"))
    NC >= 2 || throw(ArgumentError("diamagnetic_term: NC must be at least 2 (got $NC)"))
    Np > 0 || throw(ArgumentError("diamagnetic_term: Np must be positive (got $Np)"))
    NR > 0 || throw(ArgumentError("diamagnetic_term: NR must be positive (got $NR)"))
    0 < rescale_eps < 2 || throw(
        ArgumentError(
            "diamagnetic_term: rescale_eps must satisfy 0 < rescale_eps < 2 (got $rescale_eps)",
        ),
    )

    h_sparse = op.h isa SparseMatrixCSC ? op.h : sparse(op.h)
    Dhat = nambu_diamagnetic(
        h_sparse,
        pos;
        dir = dir,
        disp = disp,
        hole_convention = op.hole_convention,
    )
    scale = a === nothing ? rescale(op; eps = Float64(rescale_eps)).a : Float64(a)
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("diamagnetic_term: a must be finite and positive (got $scale)"))
    # On GPU, the assembled BdG matrix moves to the device once here; the
    # kpm_1d_current workspaces and vertex transfer follow the active device
    # (a non-device-assemblable operator throws rather than mix residences).
    # Internal callers pass an already-resident `Hs` to avoid a re-upload.
    if Hs === nothing
        Hs = ScaledOperator(_device_operator(op, "diamagnetic_term"), scale, 0.0)
    else
        Hs.a == scale || throw(
            ArgumentError(
                "diamagnetic_term: supplied Hs has a=$(Hs.a) but the requested scale is $scale",
            ),
        )
    end

    NH = 2 * op.N
    NC_int = Int(NC)
    stability = chebyshev_stability_probe(Hs, NH, NC_int)
    if !(stability <= 1.5)
        error(
            "Chebyshev recurrence is unstable (maximum probe norm $stability > 1.5); use rescale(...; bound=:gershgorin) or a larger eps.",
        )
    end

    NR_int = Int(NR)
    if psi_in === nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    else
        NR_int = size(psi_in, 2)
    end
    size(psi_in) == (NH, NR_int) || throw(
        ArgumentError(
            "diamagnetic_term: psi_in has size $(size(psi_in)); expected ($NH, $NR_int)",
        ),
    )

    # The diamagnetic vertex is Hermitian, so its trace is real.
    Gamma = real.(kpm_1d_current(Hs, Dhat, NC_int, NR_int, NH; psi_in = psi_in))
    Np_int = Int(Np)
    gh = kernel.(0:(NC_int-1), NC_int) .* hn.(0:(NC_int-1))
    nodes, _ = gausschebyshevt(Np_int)
    wf = fermiFunctions(0.0, Float64(beta)).(scale .* nodes) ./ Np_int
    C = cos.((0:(NC_int-1)) .* acos.(nodes'))
    Dia_raw = dot(gh .* Gamma, C * wf)
    return Float64(g_J * NH / volume * Dia_raw)
end

"""
    _kpm2d_workspace(NH::Integer, NR::Integer;
                     arr_size::Integer=3, right_block::Integer=16) -> NamedTuple

Allocate the complete workspace keyword set consumed by `kpm_2d!`.
"""
function _kpm2d_workspace(
    NH::Integer,
    NR::Integer;
    arr_size::Integer = 3,
    right_block::Integer = 16,
)
    NH > 0 || throw(ArgumentError("_kpm2d_workspace: NH must be positive (got $NH)"))
    NR > 0 || throw(ArgumentError("_kpm2d_workspace: NR must be positive (got $NR)"))
    arr_size >= 2 || throw(
        ArgumentError("_kpm2d_workspace: arr_size must be at least 2 (got $arr_size)"),
    )
    right_block >= 1 || throw(
        ArgumentError(
            "_kpm2d_workspace: right_block must be at least 1 (got $right_block)",
        ),
    )
    return (
        ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
        Jψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
        JTnHJψr = maybe_on_device_zeros(dt_cplx, NH, NR, right_block),
        ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR, 3),
        ψ0l = maybe_on_device_zeros(dt_cplx, NH, NR),
        ψall_l = maybe_on_device_zeros(dt_cplx, NH, NR, arr_size),
        ψw = maybe_on_device_zeros(dt_cplx, NH, NR),
        μblock = maybe_on_device_zeros(dt_cplx, right_block, arr_size),
    )
end

"""
    superfluid_stiffness(op::BdGOperator, pos::AbstractMatrix{<:Real},
                         q::AbstractVector{<:Real}; beta, eta, ...)
        -> NamedTuple

Compute the finite-`q` paired paramagnetic response

    D_s / pi = Re Pi_N - Re Pi_SC.

By default this is the paired paramagnetic subtraction of Eq. 24, with no
separate diamagnetic term. It is exact for strictly linear
(continuum-Dirac-like) dispersions, where the diamagnetic operator vanishes
and the subtraction removes the ultraviolet cutoff dependence. On a lattice,
pass `include_diamagnetic=true` to also compute

    Ds_over_pi_complete = (Re Pi_N - Re Pi_SC) + (Dia_SC - Dia_N).

The added diamagnetic difference vanishes as `Delta -> 0` (so the zero-gap
cancellation is exact) but is generically `O(Delta^2)`. The normal reference
retains the superconducting state's assembled hopping, chemical potential,
interaction, and converged Hartree density, changing only `Delta` to zero.

This is the package's fixed-gauge, rigid-`D` functional: Peierls phases enter
kinetic bonds only, and the pairing block contributes no current or
diamagnetic vertex. It reduces to the standard mean-field Kubo
(Scalapino–White–Zhang) treatment for onsite pairing. The normal reference
zeroes the entire pairing block while retaining hopping, chemical potential,
interaction, and Hartree density, and the free-energy-curvature anchor tests
exactly this functional. Local gauge covariance of a nonlocal pair field and
the charge-2e pairing response are out of scope.

Both responses use `Jalpha = J(q)` and `Jbeta = J(-q)`, the same probes,
`NC`, kernel, `Np`, `eta`, chemical potential, and Hartree field. Candidate
SC and normal scales are estimated separately, then both calculations use
`a_common = max(a_SC, a_N)`. This gives the subtraction equal physical
regularization because finite-`NC` Jackson broadening scales as `a/NC`.

The default returned quantity is the finite-`q` paired paramagnetic response,
and its NamedTuple fields are unchanged. With `include_diamagnetic=true`, the
return additionally contains `Dia_SC`, `Dia_N`, and
`Ds_over_pi_complete`. `Ds_over_pi` is its transverse `q -> 0` limit only
under the documented continuum/paramagnetic-only definition. Choose `q`
perpendicular to `dir`, typically `q_y = 2pi/L_y` for an `x` current, and
scan several commensurate wavevectors for extrapolation. Choose `eta` between
the finite-size level spacing and the gap, and as a resolution rule use
`eta >= 5*a_common*pi/NC`. Vertices are always built from the unrescaled
assembled hopping.

With a CUDA GPU active, both moment computations (and the optional
diamagnetic traces) run on the device: the assembled BdG matrices, vertices,
probes, and Chebyshev workspaces are moved automatically, while scale
estimation and the spectral reconstruction stay on the host. Operators that
cannot be assembled to the device (matrix-free or dense blocks) throw an
`ArgumentError` here rather than mixing host and device residences —
sparsify/assemble the blocks or run without the GPU device.
"""
function superfluid_stiffness(
    op::BdGOperator,
    pos::AbstractMatrix{<:Real},
    q::AbstractVector{<:Real};
    beta::Real,
    eta::Real,
    omega::Real = 0.0,
    dir::Integer = 1,
    disp = nothing,
    NC::Integer = 256,
    NR::Integer = 8,
    rng = Xoshiro(0),
    psi_in = nothing,
    volume::Real,
    g_J::Real = 1.0,
    kernel = JacksonKernel,
    Np::Integer = 2 * NC,
    moment_parity::Symbol = :NONE,
    arr_size::Integer = 3,
    right_block::Integer = 16,
    rescale_eps::Real = 0.2,
    verbose::Integer = 0,
    include_diamagnetic::Bool = false,
)
    op.h isa AbstractMatrix || throw(
        ArgumentError(
            "superfluid_stiffness needs an assembled sparse h to build vertices; matrix-free users can call nambu_current_q-equivalent vertices + kpm_2d + two_energy_response directly",
        ),
    )
    size(pos, 1) == op.N || throw(
        ArgumentError(
            "superfluid_stiffness: pos has $(size(pos, 1)) rows; expected $(op.N)",
        ),
    )
    0 < rescale_eps < 2 || throw(
        ArgumentError(
            "superfluid_stiffness: rescale_eps must satisfy 0 < rescale_eps < 2 (got $rescale_eps)",
        ),
    )

    op_n = BdGOperator(
        op.h;
        mu = op.μ,
        U = copy(op.U),
        n = copy(op.n),
        Delta = zeros(ComplexF64, op.N),
        hole_convention = op.hole_convention,
        h_hole = op.hole_convention === :conjugate ? op.h_hole : nothing,
        assume_intervalley = op.hole_convention === :intervalley,
    )
    h_sparse = op.h isa SparseMatrixCSC ? op.h : sparse(op.h)
    Jq = nambu_current_q(
        h_sparse,
        pos,
        q;
        dir = dir,
        disp = disp,
        hole_convention = op.hole_convention,
    )
    Jmq = nambu_current_q(
        h_sparse,
        pos,
        -q;
        dir = dir,
        disp = disp,
        hole_convention = op.hole_convention,
    )
    q_norm = norm(q)
    if q_norm > 0 && abs(q[dir]) > 1e-12 * q_norm
        @warn "superfluid_stiffness: q has a longitudinal component; transverse stiffness requires q perpendicular to dir"
    end

    rh_sc = rescale(op; eps = Float64(rescale_eps))
    rh_n = rescale(op_n; eps = Float64(rescale_eps))
    a_common = max(rh_sc.a, rh_n.a)
    # Scale estimation runs on the host (deterministic, one-time); the moment
    # recurrences run wherever the active device puts the assembled operator.
    # A non-device-assemblable operator throws rather than mix residences.
    Hs_sc = ScaledOperator(_device_operator(op, "superfluid_stiffness"), a_common, 0.0)
    Hs_n = ScaledOperator(_device_operator(op_n, "superfluid_stiffness"), a_common, 0.0)

    NH = 2 * op.N
    NR_int = Int(NR)
    if psi_in === nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    else
        NR_int = size(psi_in, 2)
    end
    size(psi_in) == (NH, NR_int) || throw(
        ArgumentError(
            "superfluid_stiffness: psi_in has size $(size(psi_in)); expected ($NH, $NR_int)",
        ),
    )

    NC_int = Int(NC)
    stability_sc = chebyshev_stability_probe(Hs_sc, NH, NC_int)
    stability_n = chebyshev_stability_probe(Hs_n, NH, NC_int)
    max_stability = max(stability_sc, stability_n)
    if !(max_stability <= 1.5)
        error(
            "Chebyshev recurrence is unstable (maximum probe norm $max_stability > 1.5); use rescale(...; bound=:gershgorin) or a larger eps.",
        )
    end
    arr_size_int = Int(arr_size)
    right_block_int = Int(right_block)
    ws = _kpm2d_workspace(
        NH,
        NR_int;
        arr_size = arr_size_int,
        right_block = right_block_int,
    )
    mu_sc = zeros(dt_cplx, NC_int, NC_int)
    mu_n = zeros(dt_cplx, NC_int, NC_int)
    kpm_2d!(
        Hs_sc,
        Jq,
        Jmq,
        NC_int,
        NR_int,
        NH,
        mu_sc,
        psi_in;
        ws...,
        moment_parity = moment_parity,
        arr_size = arr_size_int,
        right_block = right_block_int,
        verbose = verbose,
    )
    kpm_2d!(
        Hs_n,
        Jq,
        Jmq,
        NC_int,
        NR_int,
        NH,
        mu_n,
        psi_in;
        ws...,
        moment_parity = moment_parity,
        arr_size = arr_size_int,
        right_block = right_block_int,
        verbose = verbose,
    )

    Np_int = Int(Np)
    Pi_SC = two_energy_response(
        mu_sc,
        a_common;
        b = 0.0,
        beta = beta,
        eta = eta,
        omega = omega,
        Ef = 0.0,
        NH = NH,
        volume = volume,
        g_J = g_J,
        kernel = kernel,
        Np = Np_int,
    )
    Pi_N = two_energy_response(
        mu_n,
        a_common;
        b = 0.0,
        beta = beta,
        eta = eta,
        omega = omega,
        Ef = 0.0,
        NH = NH,
        volume = volume,
        g_J = g_J,
        kernel = kernel,
        Np = Np_int,
    )

    result = (
        Ds_over_pi = real(Pi_N) - real(Pi_SC),
        Pi_SC = Pi_SC,
        Pi_N = Pi_N,
        a_SC = rh_sc.a,
        a_N = rh_n.a,
        a_common = a_common,
        q = collect(Float64, q),
        dir = Int(dir),
        eta = Float64(eta),
        beta = Float64(beta),
        omega = Float64(omega),
        NC = NC_int,
        NR = NR_int,
        Np = Np_int,
    )
    if include_diamagnetic
        Dia_SC = diamagnetic_term(
            op,
            pos;
            dir = dir,
            disp = disp,
            beta = beta,
            NC = NC_int,
            NR = NR_int,
            psi_in = psi_in,
            volume = volume,
            g_J = g_J,
            kernel = kernel,
            Np = Np_int,
            rescale_eps = rescale_eps,
            a = a_common,
            Hs = Hs_sc,
        )
        Dia_N = diamagnetic_term(
            op_n,
            pos;
            dir = dir,
            disp = disp,
            beta = beta,
            NC = NC_int,
            NR = NR_int,
            psi_in = psi_in,
            volume = volume,
            g_J = g_J,
            kernel = kernel,
            Np = Np_int,
            rescale_eps = rescale_eps,
            a = a_common,
            Hs = Hs_n,
        )
        return merge(
            result,
            (
                Dia_SC = Dia_SC,
                Dia_N = Dia_N,
                Ds_over_pi_complete = result.Ds_over_pi + Dia_SC - Dia_N,
            ),
        )
    end
    return result
end
