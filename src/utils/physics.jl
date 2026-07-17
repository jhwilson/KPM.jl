#export sigmaMatrices, sigmaMatricesDot, fermiFunction, fermiFunctions
#export gammaMatrices
#export σx, σy, σz, σ0 
#export σ2dot, σ3dot


"""
    fermiFunction(E, E_f, beta)

calculate Fermi-Dirac function at energy E, Fermi energy μ and temperature β =1/T.
Input and output all Float64.
Infinite β only allowed when accessing fermi energy through fermiFunctions(). [For performance reason for now. TODO: allow β=Inf here withouth perf. reduction. ]

Allow sloppy use of type as long as convertion is available, if using keyword arguments. 
"""
function fermiFunction(E::Float64, E_f::Float64, beta::Float64)
    return 1/(exp((E-E_f)*beta)+1);
end
fermiFunction(; E, E_f, beta) = fermiFunction(Float64(E), Float64(E_f), Float64(beta))

"""
    fermiFunctions(E_f::Float64, beta::Float64)
returns a fermi function with given E_f and beta. 

Allow sloppy use of type as long as convertion is available, if using keyword arguments. 
"""
function fermiFunctions(E_f::Float64, beta::Float64)
    f(x) = fermiFunction(Float64(x), E_f, beta)
    g(x) = ((x < E_f) + (x <= E_f)) / 2
    if isinf(beta)
        return g
    else
        return f
    end
end
fermiFunctions(; E_f, beta) = fermiFunctions(Float64(E_f), Float64(beta))

"""
    fermi_window(E_f, beta)

Return the thermal window function `w(E) = -df/dE = beta*f(E)*(1-f(E))` for
the Fermi function at Fermi energy `E_f` and inverse temperature `beta`.
Throws `ArgumentError` for `beta <= 0` or `beta == Inf` (the window
degenerates to a delta function, which callers reject; the physically relevant
`T -> 0` thermoelectric statement is the Mott relation).
"""
function fermi_window(E_f::Float64, beta::Float64)
    isfinite(beta) && beta > 0 ||
        throw(ArgumentError("fermi_window: beta must be finite and positive."))
    function window(E)
        f = fermiFunction(Float64(E), E_f, beta)
        return beta * f * (1 - f)
    end
    return window
end
fermi_window(E_f, beta) = fermi_window(Float64(E_f), Float64(beta))
