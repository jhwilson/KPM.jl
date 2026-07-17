using KPM
using LinearAlgebra
using SparseArrays
using Random

function old_bdg_channel_moments(Hs, N::Integer,
                                 sites::AbstractVector{<:Integer},
                                 directed_bonds::Vector{Tuple{Int, Int}},
                                 NC::Integer; batch_size::Integer=64)
    site_columns = Dict{Int, Int}(Int(i) => c for (c, i) in pairs(sites))
    extraction = [Tuple{Int, Int}[] for _ in eachindex(sites)]
    for (bond_index, (i, j)) in pairs(directed_bonds)
        push!(extraction[site_columns[i]], (j, bond_index))
    end

    ns = length(sites)
    mu_rho = zeros(KPM.dt_real, NC, ns)
    mu_F = zeros(KPM.dt_cplx, NC, length(directed_bonds))
    iszero(ns) && return mu_rho, mu_F

    batch_capacity = min(Int(batch_size), ns)
    psi = zeros(KPM.dt_cplx, 2Int(N), batch_capacity, 2)
    for first_site in 1:batch_capacity:ns
        last_site = min(first_site + batch_capacity - 1, ns)
        B = last_site - first_site + 1
        psi_active = view(psi, :, 1:B, :)
        fill!(psi_active, zero(KPM.dt_cplx))
        psi_views = map(i -> view(psi_active, :, :, i), 1:2)

        for (c, cg) in enumerate(first_site:last_site)
            psi_active[sites[cg], c, 1] = one(KPM.dt_cplx)
        end

        function extract_moment!(m, slot)
            for (c, cg) in enumerate(first_site:last_site)
                i = sites[cg]
                mu_rho[m, cg] = real(slot[i, c])
                for (j, bond_index) in extraction[cg]
                    mu_F[m, bond_index] = conj(slot[j + N, c])
                end
            end
        end

        extract_moment!(1, psi_views[1])
        mul!(psi_views[2], Hs, psi_views[1])
        extract_moment!(2, psi_views[2])
        ip, ipp = 2, 1
        for m in 3:NC
            KPM.chebyshev_iter_single(Hs, psi_views[ipp], psi_views[ip])
            extract_moment!(m, psi_views[ipp])
            ip = 3 - ip
            ipp = 3 - ipp
        end
    end
    return mu_rho, mu_F
end

function random_hermitian(rng, N; dense=false)
    A = randn(rng, ComplexF64, N, N)
    H = (A + A') / (4N)
    dense ? H : sparse(H)
end

rng = Xoshiro(20260716)
N = 7
h_dense = random_hermitian(rng, N; dense=true)
h_sparse = sparse(h_dense)
D_dense = randn(rng, ComplexF64, N, N) / 10
D_sparse = sparse(D_dense .* (rand(rng, N, N) .< 0.35))
n = rand(rng, N)
U = rand(rng, N)
x = randn(rng, ComplexF64, 2N, 3)

assembly_cases = Any[
    ("dense/conjugate/dense-D", h_dense, D_dense, :conjugate, false),
    ("sparse/conjugate/sparse-D", h_sparse, D_sparse, :conjugate, false),
    ("dense/intervalley/diagonal-D", h_dense, Diagonal(randn(rng, ComplexF64, N)), :intervalley, true),
    ("Hermitian-dense/conjugate/sparse-D", Hermitian(copy(h_dense)), D_sparse, :conjugate, false),
    ("Hermitian-sparse/intervalley/dense-D", Hermitian(copy(h_sparse)), D_dense, :intervalley, true),
]

println("ASSEMBLY")
for (label, h, D, convention, assume) in assembly_cases
    op = KPM.BdGOperator(h; mu=0.17, U=U, n=n, D=D,
                         hole_convention=convention,
                         assume_intervalley=assume)
    y_mf = zeros(ComplexF64, 2N, 3)
    mul!(y_mf, op, x)
    y_asm = KPM.bdg_assemble(op) * x
    println(label, ": maxerr=", maximum(abs, y_mf - y_asm),
            " exact=", y_mf == y_asm,
            " hermitian=", ishermitian(KPM.bdg_assemble(op)))
end

op = KPM.BdGOperator(h_sparse; mu=-0.13, U=U, n=n, D=D_sparse,
                     hole_convention=:conjugate)
a = 1.2 * KPM.gershgorin_bound(op)
Hs = KPM.ScaledOperator(op, a, 0.0)
sites = [7, 2, 5, 1]
bonds = [(7, 7), (7, 1), (2, 2), (2, 5), (2, 5), (5, 2), (1, 7)]

println("MOMENTS")
for batch_size in (1, 2, 3, 4, 9), NC in (2, 3, 17)
    old = old_bdg_channel_moments(Hs, N, sites, bonds, NC; batch_size=batch_size)
    new = KPM.bdg_channel_moments(Hs, N, sites, bonds, NC; batch_size=batch_size)
    println("batch=", batch_size, " NC=", NC,
            " rho_exact=", old[1] == new[1],
            " F_exact=", old[2] == new[2],
            " rho_err=", maximum(abs, old[1] - new[1]),
            " F_err=", maximum(abs, old[2] - new[2]))
end

println("ALLOCATIONS")
KPM.bdg_channel_moments(Hs, N, sites, bonds, 17; batch_size=3)
alloc = @allocated KPM.bdg_channel_moments(Hs, N, sites, bonds, 17; batch_size=3)
println("new moments allocated bytes=", alloc)

println("EXTRACTION PERFORMANCE")
Nperf = 512
hperf = spdiagm(1 => fill(-0.3 + 0.1im, Nperf - 1),
                -1 => fill(-0.3 - 0.1im, Nperf - 1))
opperf = KPM.BdGOperator(hperf; mu=-0.2, U=0.0, n=zeros(Nperf),
                         Delta=fill(0.2 + 0.05im, Nperf),
                         hole_convention=:conjugate)
Hsperf = KPM.ScaledOperator(opperf, 1.2 * KPM.gershgorin_bound(opperf), 0.0)
sitesperf = collect(1:Nperf)
bondsperf = [(i, mod1(i + 1, Nperf)) for i in 1:Nperf]
old_bdg_channel_moments(Hsperf, Nperf, sitesperf, bondsperf, 64; batch_size=64)
KPM.bdg_channel_moments(Hsperf, Nperf, sitesperf, bondsperf, 64; batch_size=64)
GC.gc()
old_alloc = @allocated old_bdg_channel_moments(Hsperf, Nperf, sitesperf,
                                                bondsperf, 64; batch_size=64)
old_time = @elapsed old_bdg_channel_moments(Hsperf, Nperf, sitesperf,
                                            bondsperf, 64; batch_size=64)
GC.gc()
new_alloc = @allocated KPM.bdg_channel_moments(Hsperf, Nperf, sitesperf,
                                                bondsperf, 64; batch_size=64)
new_time = @elapsed KPM.bdg_channel_moments(Hsperf, Nperf, sitesperf,
                                            bondsperf, 64; batch_size=64)
println("old bytes=", old_alloc, " new bytes=", new_alloc,
        " old seconds=", old_time, " new seconds=", new_time)

struct MatrixFreeBlock{A}
    a::A
end
Base.size(x::MatrixFreeBlock) = size(x.a)
Base.size(x::MatrixFreeBlock, d::Integer) = size(x.a, d)
Base.eltype(x::MatrixFreeBlock) = eltype(x.a)
Base.adjoint(x::MatrixFreeBlock) = MatrixFreeBlock(adjoint(x.a))
LinearAlgebra.mul!(y::AbstractVecOrMat, x::MatrixFreeBlock,
                   v::AbstractVecOrMat, alpha::Number, beta::Number) =
    mul!(y, x.a, v, alpha, beta)

println("MATRIX-FREE STIFFNESS CPU BASELINE")
Nmf = 4
hmf = sparse([1, 2, 2, 3, 3, 4, 4, 1],
             [2, 1, 3, 2, 4, 3, 1, 4], fill(-0.4, 8), Nmf, Nmf)
Dmf = MatrixFreeBlock(Diagonal(fill(0.2 + 0.05im, Nmf)))
opmf = KPM.BdGOperator(hmf; mu=-0.1, U=0.0, n=zeros(Nmf), D=Dmf,
                       hole_convention=:conjugate)
posmf = reshape(collect(0.0:Nmf-1), :, 1)
dispmf(i, j) = [mod(posmf[i, 1] - posmf[j, 1] + Nmf / 2, Nmf) - Nmf / 2]
resmf = KPM.superfluid_stiffness(opmf, posmf, [0.0]; beta=5.0, eta=0.3,
                                 dir=1, disp=dispmf, NC=8, NR=2,
                                 volume=Nmf)
println("matrix-free CPU stiffness finite=", isfinite(resmf.Ds_over_pi),
        " assemblable=", KPM._bdg_assemblable(opmf))
