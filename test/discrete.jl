using NetworkHawkesProcesses
using Test

@testset "DiscreteIndependentHawkesProcess" begin

    nnodes = 2
    nbasis = 4
    nlags = 5
    dt = 1.0
    baseline = DiscreteHomogeneousProcess(0.1 * ones(nnodes), dt)
    weights = IndependentWeightModel(Diagonal(0.5 * ones(nnodes)))
    impulses = DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt)
    process = DiscreteStandardHawkesProcess(baseline, impulses, weights, dt)

    @test NetworkHawkesProcesses.effective_weights(process) == [
        0.125, 0.0, 0.0, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.0, 0.0, 0.125
    ]
    conn_weights, basis_weights = NetworkHawkesProcesses.weights(process, params(process))
    @test conn_weights == [0.5, 0.5]
    @test basis_weights == cat([
        [0.25 0.0; 0.0 0.25],
        [0.25 0.0; 0.0 0.25],
        [0.25 0.0; 0.0 0.25],
        [0.25 0.0; 0.0 0.25]
    ]..., dims=3)

end
