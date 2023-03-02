using NetworkHawkesProcesses: node_counts
using NetworkHawkesProcesses: split_extract
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

end