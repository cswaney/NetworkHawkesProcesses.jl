using NetworkHawkesProcesses: node_counts
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

# @testset begin "HomogeneousProcess"

#     process = HomogeneousProcess(1.0)
#     data = [
#         ([1.0, 5.0, 10.0], 10.0),
#         ([5.0], 5.0)
#     ]
#     @test sufficient_statistics(process, data) == [(3, 10.0), (1, 5.0)]

#     data = [([], 0.0)]
#     @test sufficient_statistics(process, data) == (0, 0.0)

#     data = [([], 10.0)]
#     @test sufficient_statistics(process, data) == (0, 10.0)

#     data = [([0.0], 10.0)]
#     @test sufficient_statistics(process, data) == (1, 10.0)

#     data = [([1.0], 0.0)] # error
#     @test sufficient_statistics(process, data) == ?
# end

# @testset begin "MultivariateHomogeneousProcess"

# data = ([0.008611226104070502, 0.11482329640222821, 0.18957278421725554, 0.22311355184680803, 0.4649685500267693, 0.6349894967246984, 0.7305679973656837], Int64[2, 1, 1, 2, 2, 1, 2], 1.0)

# @test sufficient_statistics(process, data) == ([3, 4], 1.0)

# end

# @testset "LogGaussianCoxProcess"

#     # loglikelihood

# end