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
function two_energy_response(mu2D::AbstractMatrix, a::Real;
                             b::Real=0.0, beta::Real, eta::Real,
                             omega::Real=0.0, Ef::Real=0.0,
                             NH::Integer, volume::Real, g_J::Real=1.0,
                             kernel=JacksonKernel,
                             NC::Integer=size(mu2D, 1), Np::Integer=2 * NC)
    size(mu2D, 1) == size(mu2D, 2) ||
        throw(ArgumentError("two_energy_response: mu2D must be square (got $(size(mu2D)))"))
    NC >= 2 || throw(ArgumentError("two_energy_response: NC must be at least 2 (got $NC)"))
    size(mu2D, 1) >= NC ||
        throw(ArgumentError("two_energy_response: mu2D size $(size(mu2D)) is smaller than NC=$NC"))
    a > 0 || throw(ArgumentError("two_energy_response: a must be positive (got $a)"))
    eta >= 0 || throw(ArgumentError("two_energy_response: eta must be nonnegative (got $eta)"))
    beta > 0 || throw(ArgumentError("two_energy_response: beta must be positive (got $beta)"))
    Np > 0 || throw(ArgumentError("two_energy_response: Np must be positive (got $Np)"))
    NH > 0 || throw(ArgumentError("two_energy_response: NH must be positive (got $NH)"))
    volume > 0 || throw(ArgumentError("two_energy_response: volume must be positive (got $volume)"))
    eta == 0 && !(omega == 0 && isfinite(beta)) &&
        throw(ArgumentError("eta=0 requires omega=0 and finite beta"))

    NC_int = Int(NC)
    Np_int = Int(Np)
    mu_tilde = mu2D_apply_kernel_and_h(mu2D[1:NC_int, 1:NC_int], NC_int, kernel)
    nodes, _ = gausschebyshevt(Np_int)
    C = cos.((0:NC_int-1) .* acos.(nodes'))
    Gamma = transpose(C) * maybe_to_host(mu_tilde) * C
    E = a .* nodes .+ b
    fermi = fermiFunctions(Float64(Ef), Float64(beta))
    f = fermi.(E)

    if eta == 0
        threshold = 1e-8 * max(a, one(a))
        F = [begin
                 delta_E = E[P] - E[Q]
                 if abs(delta_E) < threshold
                     fm = fermi((E[P] + E[Q]) / 2)
                     -beta * fm * (1 - fm)
                 else
                     (f[P] - f[Q]) / delta_E
                 end
             end for P in eachindex(E), Q in eachindex(E)]
    else
        F = [(f[P] - f[Q]) / (omega + E[P] - E[Q] + im * eta)
             for P in eachindex(E), Q in eachindex(E)]
    end

    return ComplexF64(g_J * NH / volume / Np_int^2 * sum(Gamma .* F))
end
