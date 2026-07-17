# KPM_TEST_GPU=1 loads CUDA before anything else so the extension activates
# and the whole suite exercises the GPU code paths (requires a functional
# GPU; used by the manual cluster validation, never by CI).
if get(ENV, "KPM_TEST_GPU", "") in ("1", "true")
    using CUDA
    @assert CUDA.functional() "KPM_TEST_GPU is set but CUDA is not functional"
end

using KPM
using Test

if get(ENV, "KPM_TEST_GPU", "") in ("1", "true")
    @assert KPM.whichcore() "KPM_TEST_GPU is set but the GPU device did not activate"
    @info "KPM test suite running with the GPU device active"
end

@testset "kernels/kernels.jl" begin
    include("test_kernel.jl")
end

@testset "util/chebyshev_iteration.jl" begin
    include("chebyshev_iteration_test.jl")
end

@testset "applications/dos.jl" begin
    include("dos_test.jl")
end

@testset "applications/conductivity.jl" begin
    include("cond_test.jl")
end

@testset "Kubo–Bastin vs exact diagonalization" begin
    include("kubo_bastin_test.jl")
end

@testset "BdG self-consistency (applications/bdg.jl)" begin
    include("bdg_test.jl")
end

@testset "Kitaev pairing operators (applications/bdg.jl)" begin
    include("kitaev_test.jl")
end

@testset "pairing channels (applications/bdg.jl)" begin
    include("pairing_channel_test.jl")
end

@testset "d-wave pairing channels (applications/bdg.jl)" begin
    include("dwave_test.jl")
end

@testset "Rashba pairing channels (applications/bdg.jl)" begin
    include("rashba_test.jl")
end

@testset "SSH embedding invariance (applications/stiffness.jl)" begin
    include("ssh_invariance_test.jl")
end

@testset "bond pairing stiffness (applications/stiffness.jl)" begin
    include("bond_stiffness_test.jl")
end

@testset "superfluid stiffness (applications/stiffness.jl)" begin
    include("stiffness_test.jl")
end

@testset "KPM.jl" begin
    include("integration_test.jl")
end

@testset "typed front end (frontend.jl)" begin
    include("frontend_test.jl")
end

@testset "optional CUDA extension migration" begin
    include("optional_cuda_test.jl")
end
