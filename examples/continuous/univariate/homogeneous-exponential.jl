"""A simple synthetic example using a continuous-time univariate Hawkes process with exponential impulse response."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
duration = 1000.0; # 100.0 or 1000.0

# create a random process
baseline = UnivariateHomogeneousProcess(rand());
impulse_response = UnivariateExponentialImpulseResponse(rand());
weight = UnivariateWeightModel(rand());
process = ContinuousUnivariateHawkesProcess(baseline, impulse_response, weight);
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = copy(params(process));

# generate random data
data = NetworkHawkesProcesses.rand(process, duration);
println("Generated $(length(data[1])) events")

# estimate parameters via mle
res = mle!(process, data; verbose=true);
θmle = res.maximizer;
[θ θmle]

# reset parameters
params!(process, θ);

# estimate parameters via mcmc (this is very slow for exponential impulse response models)
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]