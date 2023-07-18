using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Random
using Statistics

# set random seed
Random.seed!(0);

# set hyperparameters
duration = 1000.0;
nnodes = 3;

# create a random process
process = ContinuousIndependentHawkesProcess([
    ContinuousUnivariateHawkesProcess(
        UnivariateHomogeneousProcess(rand()),
        UnivariateExponentialImpulseResponse(rand()),
        UnivariateWeightModel(rand())
    ) for _ in 1:nnodes
])
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = params(process)

# create random data
data = rand(process, duration)
println("Generated $(length(data[1])) events")

# estimate parameters via mle
res = mle!(process, data; verbose=true);
θmle = res.maximizer;
[θ θmle]

# convert to standard process
process = ContinuousStandardHawkesProcess(process)