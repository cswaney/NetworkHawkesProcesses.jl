using NetworkHawkesProcesses
using NetworkHawkesProcesses: logprior
using LinearAlgebra: Diagonal, diag
using Test

@testset "IndependentWeightModel" begin

    W = Diagonal([-0.1, 0.1])
    nnodes = length(diag(W))
    @test_throws DomainError IndependentWeightModel(W)
    W = Diagonal([0.1, 0.1])
    @test_throws DomainError IndependentWeightModel(W, 0.0, 1.0, ones(nnodes), ones(nnodes))
    @test_throws DomainError IndependentWeightModel(W, 1.0, 0.0, ones(nnodes), ones(nnodes))
    @test_throws DomainError IndependentWeightModel(W, 1.0, 1.0, [0.0, 1.0], ones(nnodes))
    @test_throws DomainError IndependentWeightModel(W, 1.0, 1.0, ones(nnodes), [0.0, 1.0])
    @test_throws ArgumentError IndependentWeightModel(W, 1.0, 1.0, ones(nnodes + 1), ones(nnodes))
    @test_throws ArgumentError IndependentWeightModel(W, 1.0, 1.0, ones(nnodes), ones(nnodes + 1))

    # TODO tests with Float32
    # ...
    
    nnodes = 2
    W = Diagonal(0.1 * ones(nnodes))
    weights = IndependentWeightModel(W)
    @test size(weights) == nnodes
    @test nparams(weights) == nnodes
    @test params(weights) == 0.1 * ones(nnodes)
    @test_throws ArgumentError params!(weights, .1 * ones(nnodes * nnodes))
    params!(weights, 0.75 * ones(nnodes))
    @test params(weights) == .75 * ones(nnodes)
    @test logprior(weights) == -1.5
end