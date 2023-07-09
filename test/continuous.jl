using NetworkHawkesProcesses
using Test

@testset "ContinuousUnivariateHawkesProcess" begin
    
    baseline = UnivariateHomogeneousProcess(0.5)
    impulse_response = UnivariateLogitNormalImpulseResponse(0.0, 1.0, 1.0)
    weight = UnivariateWeightModel(0.25)
    process = ContinuousUnivariateHawkesProcess(baseline, impulse_response, weight)

    @test isstable(process)
    @test params(process) == [0.5, 0.0, 1.0, 0.25]
    @test nparams(process) == 4
    @test_throws ArgumentError params!(process, ones(5))
    @test params!(process, ones(4)) == ones(4)
    @test !isstable(process)

    baseline = UnivariateHomogeneousProcess(0.5)
    impulse_response = UnivariateLogitNormalImpulseResponse(0.0, 1.0, 1.0)
    weight = UnivariateWeightModel(0.25)
    process = ContinuousUnivariateHawkesProcess(baseline, impulse_response, weight)

end