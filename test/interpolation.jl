using NetworkHawkesProcesses
using NetworkHawkesProcesses: LinearInterpolator, DiscreteLinearInterpolator, integrate
using Test

@testset "LinearInterpolator" begin

    x = collect(0.0:0.5:2pi)
    y = sin.(x)
    f = LinearInterpolator(x, y)
    @test f(0.0) == 0.0
    @test f(0.1) == sin(0.5) / .5 * .1
    @test_throws DomainError f(2pi)
    @test_throws DomainError f(-1)
    @test integrate(f) == sum(y[2:end] .+ y[1:end-1]) * 0.5 ^ 2

end
