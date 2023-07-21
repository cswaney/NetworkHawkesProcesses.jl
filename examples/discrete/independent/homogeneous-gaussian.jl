using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Random
using Statistics

# set random seed
Random.seed!(0);

# set hyperparameters
nbasis = 3;
nlags = 4;
ndims = 3;
duration = 1000;
dt = 2.0;

# create a random process
process = Independent{DiscreteHawkesProcess}([
    DiscreteUnivariateHawkesProcess(
        DiscreteUnivariateHomogeneousProcess(rand(), dt),
        UnivariateGaussianImpulseResponse(ones(nbasis) ./ nbasis, nlags, dt),
        UnivariateWeightModel(rand())
    ) for _ in 1:ndims
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

# reset parameters
params!(process, θ);

# estimate parameters via mcmc
res = mcmc!(process, data; verbose=true, joint=true);
θmcmc = mean(res.samples);
[θ θmcmc]


# convert to standard process
process = DiscreteStandardHawkesProcess(process)