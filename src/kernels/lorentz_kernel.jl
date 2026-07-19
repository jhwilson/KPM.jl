"""
LorentzKernels(λ::Float64)

Returns function LorentzKernel(n, N) that evaluates the Lorentz kernel
`sinh(λ(1-n/N))/sinh(λ)` with parameter `λ > 0` at the n-th expansion
coefficient with N in total (NC). The damping acts as a Lorentzian lifetime
broadening only where the reflected `e^{-2λ}` term is negligible (λ ≳ 2);
see [`greens`](@ref) for the equivalent width.
"""
function LorentzKernels(λ::Float64)
    λ > 0 || throw(
        ArgumentError(
            "LorentzKernels: λ must be positive (got $λ); λ ≲ 2 also degrades the Lorentzian-broadening correspondence",
        ),
    )
    LorentzKernel(n::Integer, N::Integer) = sinh(λ*(1-n/N))/sinh(λ)
    return LorentzKernel
end
