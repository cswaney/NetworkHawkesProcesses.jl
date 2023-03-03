using NetworkHawkesProcesses: Kernel, SquaredExponentialKernel, OrnsteinUhlenbeckKernel, PeriodicKernel
using Test

@testset "Kernel" begin

    kernel = SquaredExponentialKernel(1.0, 1.0)
    @test kernel(0.0, 0.0) == 1.0
    @test kernel(0.0, Inf) == 0.0

    kernel = OrnsteinUhlenbeckKernel(1.0, 1.0)
    @test kernel(0.0, 0.0) == 1.0
    @test kernel(0.0, Inf) == 0.0

    kernel = PeriodicKernel(1.0, 1.0, 1.0)
    @test kernel(0.0, 0.0) == 1.0
    @test kernel(0.0, 0.0) == kernel(0.0, 1.0) == kernel(0.0, -1.0)

end
