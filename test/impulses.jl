using NetworkHawkesProcesses
using Distributions
using Test

@testset "UnivariateLogitNormalImpulseResponse" begin

    @test_throws DomainError UnivariateLogitNormalImpulseResponse(0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateLogitNormalImpulseResponse(0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateLogitNormalImpulseResponse(0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateLogitNormalImpulseResponse(0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0)
    @test_throws DomainError UnivariateLogitNormalImpulseResponse(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    
    model = UnivariateLogitNormalImpulseResponse(0.5, 2.0, 1.0)
    @test nparams(model) == 2
    @test params(model) == [0.5, 2.0]
    @test params!(model, [0.0, 1.0]) == [0.0, 1.0]
    @test_throws ArgumentError params!(model, [0.0])
    @test_throws DomainError params!(model, [0.0, 0.0])

    @test intensity(model, 0.0) == 0.0
    @test intensity(model, 1.0) == 0.0
    @test intensity(model, 0.5) == pdf(LogitNormal(0.0, 1.0), 0.5)


end