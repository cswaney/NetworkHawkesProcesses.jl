using NetworkHawkesProcesses
using Distributions
using Test

@testset "UnivariateWeightModel" begin
    
    @test_throws DomainError UnivariateWeightModel(-1.0, 1.0, 1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateWeightModel(1.0, 0.0, 1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateWeightModel(1.0, 1.0, 0.0, 1.0, 1.0)
    @test_throws DomainError UnivariateWeightModel(1.0, 1.0, 1.0, 0.0, 1.0)
    @test_throws DomainError UnivariateWeightModel(1.0, 1.0, 1.0, 1.0, 0.0)

    @test UnivariateWeightModel(1.0) isa UnivariateWeightModel{Float64}
    @test UnivariateWeightModel(1.0f0) isa UnivariateWeightModel{Float32}
    @test UnivariateWeightModel(Float16(1.0)) isa UnivariateWeightModel{Float16}

    @test nparams(UnivariateWeightModel(1.0)) == 1
    @test params(UnivariateWeightModel(1.0)) == [1.0]
    @test params!(UnivariateWeightModel(1.0), [2.0]) == [2.0]
    @test_throws ArgumentError params!(UnivariateWeightModel(1.0), [1.0, 1.0])
    @test_throws DomainError params!(UnivariateWeightModel(1.0), [-1.0])

    model = UnivariateWeightModel(1.0)
    events = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    parents = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    @test sufficient_statistics(model, (events, nothing), (nothing, parents)) == (10, 5)

    @test logprior(UnivariateWeightModel(1.0)) == pdf(Gamma(1.0, 1.0), 1.0)
end