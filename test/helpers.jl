using NetworkHawkesProcesses: cartesian
using Test

@testset "cartesian" begin

    data = ones(3, 4)

    indices = collect(eachindex(data))

    @test cartesian(indices[1], size(data)) == (1, 1)
    @test cartesian(indices[end], size(data)) == (3, 4)
    @test cartesian(indices[4], size(data)) == (1, 2)
    @test cartesian(indices[9], size(data)) == (3, 3)
    @test cartesian(indices[5], size(data)) == (2, 2)

end