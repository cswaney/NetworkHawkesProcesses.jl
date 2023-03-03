using NetworkHawkesProcesses
using NetworkHawkesProcesses: node_counts
using NetworkHawkesProcesses: split_extract
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess
using Test

@testset "HomogeneousProcess" begin

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
    x = 0.0:0.1:1.0
    y = rand(gp, x)
    λ = [exp.(y)]
    process = LogGaussianCoxProcess(x, λ, kernel, 0.0)
    @test ndims(process) == 1
    @test length(process) == 1.0

    ys = [rand(gp, x), rand(gp, x)]
    λ = [exp.(y) for y in ys]
    process = LogGaussianCoxProcess(x, λ, kernel, 0.0)
    @test ndims(process) == 2
    @test length(process) == 1.0

end

@testset "DiscreteHomogeneousProcess" begin
    
    @test_throws DomainError DiscreteHomogeneousProcess([1.0, -1.0])
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 0.0, 1.0, ones(2), ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 0.0, ones(2), ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 1.0, [0.0, 1.0], ones(2), 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 1.0, 1.0, ones(2), [1.0, 0.0], 1.0)
    @test_throws DomainError DiscreteHomogeneousProcess(ones(2), 0.0)

    process = DiscreteHomogeneousProcess(ones(2), 0.5)
    @test intensity(process, 0.0:0.1:1.0) == 0.5 * ones(11, 2)
    @test intensity(process, 1, 0.0) == intensity(process, 2, 0.0) == 0.5
    @test_throws DomainError intensity(process, 0, 0.0)
    @test_throws DomainError intensity(process, 3, 0.0)
    @test_throws DomainError intensity(process, 1, -1.0)
    @test_throws DomainError intensity(process, -0.1:0.1:1.0)
end
