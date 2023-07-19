using NetworkHawkesProcesses
using NetworkHawkesProcesses: node_counts, nparams, range, length, nsteps
using NetworkHawkesProcesses: split_extract
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess
using NetworkHawkesProcesses: sufficient_statistics, integrated_intensity, update!
using Test

@testset "HomogeneousProcess" begin

    @test_throws DomainError HomogeneousProcess(-1 * ones(1), 1.0, 1.0)
    @test_throws DomainError HomogeneousProcess(ones(1), -1.0, 1.0)
    @test_throws DomainError HomogeneousProcess(ones(1), 1.0, -1.0)

    process = HomogeneousProcess(ones(3))
    @test ndims(process) == 3
    @test params(process) == ones(3)
    @test_throws ArgumentError params!(process, ones(1)) 
    @test_throws DomainError rand(process, -1.0)
    @test_throws DomainError rand(process, 1, -1.0)
    @test_throws BoundsError rand(process, 0, 1.0)
    @test_throws DomainError integrated_intensity(process, -1.0)

    nodes = [1, 1, 2, 2]
    parentnodes = [0, 1, 0, 2]
    nnodes = 2
    @test node_counts(nodes, parentnodes, nnodes) == [1, 1]

    nodes = []
    parentnodes = []
    nnodes = 2
    @test node_counts(nodes, parentnodes, nnodes) == [0, 0]

    nodes = [1, 1, 2, 2]
    parentnodes = [1, 2, 1, 2]
    nnodes = 2
    @test node_counts(nodes, parentnodes, nnodes) == [0, 0]

end

@testset "LogGaussianCoxProcess" begin
    
    data = ([], [], 1.)
    parents = ([], [])
    @test split_extract(data, parents, 1) == [([], [], 1.)]
    @test split_extract(data, parents, 2) == [([], [], 1.), ([], [], 1.)]
    
    data = ([0.1, 0.2, 0.3, 0.4], [1, 1, 2, 2], 1.0)
    parents = ([], [0, 1, 0, 2])
    @test split_extract(data, parents, 2) == [([.1], [1], 1.), ([.3], [2], 1.)]
    
    data = ([0.1, 0.2, 0.3, 0.4], [1, 1, 2, 2], 1.0)
    parents = ([], [1, 1, 2, 2])
    @test split_extract(data, parents, 2) == [([], [], 1.), ([], [], 1.)]


    kernel = SquaredExponentialKernel(1.0, 1.0)
    gp = GaussianProcess(kernel)
    x = collect(0.0:0.1:1.0)
    y = rand(gp, x)
    λ = [exp.(y)]
    @test_throws DomainError LogGaussianCoxProcess(collect(1.0:0.1:2.0), λ, kernel, 0.0)
    process = LogGaussianCoxProcess(x, λ, kernel, 0.0)
    @test ndims(process) == 1
    @test length(process) == 1.0
    @test nparams(process) == 11
    @test_throws ArgumentError params!(process, [0.0])

    ys = [rand(gp, x), rand(gp, x)]
    λ = [exp.(y) for y in ys]
    process = LogGaussianCoxProcess(x, λ, kernel, 0.0)
    @test ndims(process) == 2
    @test length(process) == 1.0
    @test nparams(process) == 22

end

@testset "DiscreteHomogeneousProcess" begin
    
    @test_throws DomainError DiscreteHomogeneousProcess([1.0, -1.0])
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 0.0, 1.0, ones(2), ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 0.0, ones(2), ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 1.0, [0.0, 1.0], ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 1.0, ones(2), [1.0, 0.0], 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 0.0)

    process = DiscreteHomogeneousProcess(ones(2), 0.5)
    @test intensity(process, 1:10) == 0.5 * ones(10, 2)
    @test intensity(process, 1, 1) == intensity(process, 2, 1) == 0.5
    @test_throws DomainError intensity(process, 0, 1)
    @test_throws DomainError intensity(process, 3, 1)
    @test_throws DomainError intensity(process, 1, 0)
    @test_throws DomainError intensity(process, -1:10)

    parents = [0 0 0 1 0 1 0 0 0 1; 2 0 0 0 0 0 0 0 0 0]
    @test sufficient_statistics(process, transpose(parents)) == ([3, 2], 10)

    @test integrated_intensity(process, 1.0) == [0.5, 0.5]
    @test integrated_intensity(process, 2.0) == [1.0, 1.0]
    @test_throws DomainError integrated_intensity(process, -1.0)
    @test_throws DomainError integrated_intensity(process, 0, 1.0)
    @test_throws DomainError integrated_intensity(process, 3, 1.0)
    @test_throws DomainError integrated_intensity(process, 1, -1.0)

    @test_throws ArgumentError update!(process, zeros(2, 10), zeros(2, 10, 1))

    @test_throws DomainError rand(process, -1)
end

@testset "UnivariateHomogeneousProcess" begin
    
    @test_throws DomainError UnivariateHomogeneousProcess(-1.0, 1.0, 1.0)
    @test_throws DomainError UnivariateHomogeneousProcess(1.0, 0.0, 1.0)
    @test_throws DomainError UnivariateHomogeneousProcess(1.0, 1.0, 0.0)

    @test typeof(UnivariateHomogeneousProcess(Float16(1))) == UnivariateHomogeneousProcess{Float16}
    @test_throws MethodError UnivariateHomogeneousProcess(Float32(1), 1.0, 1.0)
    @test typeof(UnivariateHomogeneousProcess{Float32}(Float16(1), 1., 1.0)) == UnivariateHomogeneousProcess{Float32}

    process = UnivariateHomogeneousProcess(1.0) # UnivariateHomogeneousProcess{Floata64}
    @test ndims(process) == 1
    @test params(process) == [1.0]
    @test params!(process, [2.0]) == [2.0]
    @test params!(process, 1.0) == [1.0]

    duration = 10.0
    parentnodes = [0, 1, 0, 0, 1, 1, 0, 1, 0]
    @test sufficient_statistics(process, ([], duration), ([], parentnodes)) == (5, 10.0)

    @test integrated_intensity(process, 10.0) == 10.0
    @test_throws DomainError integrated_intensity(process, -10.0)
    
    @test_throws DomainError intensity(process, -10.0)
    @test intensity(process, 1.0) == 1.0

    @test logprior(process) == -1.0
end

@testset "DiscreteUnivariateLogGaussianCoxProcess" begin

    kernel = SquaredExponentialKernel(1.0, 1.0)
    gp = GaussianProcess(kernel)
    baseline = DiscreteUnivariateLogGaussianCoxProcess(gp, 100.0, 10, 0.0, 2.5)

    @test nsteps(baseline) == length(range(baseline))
    @test nparams(baseline) == 11 # nsteps(baseline) + 1
    
    @test_throws ArgumentError rand(baseline, 41)
    @test_throws ArgumentError rand(baseline, 0)

    @test_throws DomainError intensity(baseline, 0)
    @test_throws DomainError intensity(baseline, 41)

    @test_throws DomainError intensity(baseline, 0:40)
    @test_throws DomainError intensity(baseline, 1:41)

    @test_throws ArgumentError loglikelihood(baseline, ones(Int, 41))

    y = log.(baseline.λ) .- baseline.m
    @test loglikelihood(baseline, ones(Int, 40)) == loglikelihood(baseline, ones(Int, 40), y)
end

