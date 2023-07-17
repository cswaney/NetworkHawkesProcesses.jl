using NetworkHawkesProcesses
using Distributions
using Test

@testset "UnivariateGaussianImpulseResponse" begin

    @test_throws DomainError UnivariateGaussianImpulseResponse([0.1, 0.1, 0.1], 4, 1.0)
    @test_throws DomainError UnivariateGaussianImpulseResponse([0.5, 0.6, -0.1], 4, 1.0)
    @test_throws DomainError UnivariateGaussianImpulseResponse([1.1, 0.0, -0.1], 4, 1.0)
    @test_throws DomainError UnivariateGaussianImpulseResponse(ones(3) / 3, -1.0, ones(3), 4, 1.0, nothing)
    @test_throws DomainError UnivariateGaussianImpulseResponse(ones(3) / 3, 1.0, -ones(3), 4, 1.0, nothing)
    @test_throws DomainError UnivariateGaussianImpulseResponse(ones(3) / 3, 1.0, ones(3), 0, 1.0, nothing)
    @test_throws DomainError UnivariateGaussianImpulseResponse(ones(3) / 3, 1.0, ones(3), 1, 0.0, nothing)
    
    impulse = UnivariateGaussianImpulseResponse(ones(3) / 3, 4, 1.0)
    @test nbasis(impulse) == 3
    @test nlags(impulse) == 4
    @test nparams(impulse) == 3
    @test params(impulse) == ones(3) / 3
    @test params!(impulse, [0.25, 0.25, 0.50]) == [0.25, 0.25, 0.50]
    @test_throws ArgumentError params!(impulse, [0.0, 1.0])
    @test_throws DomainError params!(impulse, [0.1, 0.1, 0.1])
    @test_throws DomainError params!(impulse, [0.5, 0.6, -0.1])
    @test_throws DomainError params!(impulse, [1.1, 0.0, -0.1])

    # intensity

    # basis


end

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

@testset "UnivariateExponentialImpulseResponse" begin

    @test_throws DomainError UnivariateExponentialImpulseResponse(0.0, 1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateExponentialImpulseResponse(1.0, 0.0, 1.0, 1.0)
    @test_throws DomainError UnivariateExponentialImpulseResponse(1.0, 1.0, 0.0, 1.0)
    @test_throws DomainError UnivariateExponentialImpulseResponse(1.0, 1.0, 1.0, -1.0)
    
    @test isa(UnivariateExponentialImpulseResponse(1.0), UnivariateExponentialImpulseResponse{Float64})
    @test isa(UnivariateExponentialImpulseResponse(1.0f0), UnivariateExponentialImpulseResponse{Float32})

    model = UnivariateExponentialImpulseResponse(1.0)    
    @test ndims(model) == 1
    @test nparams(model) == 1
    @test params(model) == [1.0]
    @test_throws DomainError params!(model, [0.0])
    @test_throws ArgumentError params!(model, [1.0, 1.0])
    @test params!(model, [2.0]) == [2.0]
    @test_throws ArgumentError rand(model, -1)
    @test isa(rand(model, 10), Vector{Float64})
    
    
    model = UnivariateExponentialImpulseResponse(1.0)
    @test intensity(model, 0.0) == model.θ
    @test intensity(model, Inf) == 0.0
    @test intensity(model, 1.0) == pdf(Exponential(model.θ), 1.0)

    # sufficient_statistics
end