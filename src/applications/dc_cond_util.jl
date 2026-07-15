using FastGaussQuadrature

# utility functions for conductivity

"""
The Kubo–Bastin kernel Γnm(ε) of Garcia et al., PRL 114, 116602 (2015).
Complex-valued for ε in (-1, 1).
"""
Γnm(n::Int64,m::Int64,ε) = ((ε - 1.0im * m * sqrt(1 - ε^2)) * exp(1.0im * m * acos(ε)) * chebyshevT(n, ε) +
                                     (ε + 1.0im * n * sqrt(1 - ε^2)) * exp(-1.0im * n * acos(ε)) * chebyshevT(m, ε))

"""
Contract the (complex) Γnm(ε) matrix with the kernel-improved moments:
Σ_{nm} Γnm(ε) μ̃nm. `μtilde` must already have the kernel and hn factors applied.
"""
function Γnmμnmαβ(μtilde::Array, ε, NC)
    Γnm_matrix = Γnm.(0:NC-1, (0:NC-1)', ε)
    @assert size(Γnm_matrix) == size(μtilde)
    result = sum(Γnm_matrix .* μtilde)
    return result
end

"""
`Lambda_nm` is integral of f(Ef)/(1-Ef^2)^2 * Γnm(Ef). Notice that all Ef is scaled to -1 to 1.

δ is the amount around ±1 to avoid.
"""
function Lambda_nm(n, m, E_f; δ=1e-2, beta=Inf, grid_N=100000)
    ff = fermiFunctions(E_f, beta)

    f(x) = ff(x) / (1 - x^2)^(3/2) * Γnm(n, m, x)

    x, w = gausschebyshevt(grid_N)
    idx = abs.(x).< 1-δ
    return dot(w[idx], f.(x[idx]))

end
