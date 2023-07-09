"""A simple synthetic example using a continuous-time univariate Hawkes process with logit-normal impulse response."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
duration = 1000.0; # 100.0 or 1000.0
Δtmax = 5.0;

# create a random process
baseline = UnivariateHomogeneousProcess(rand());
impulse_response = UnivariateLogitNormalImpulseResponse(rand(), rand(), Δtmax);
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

# estimate parameters via mcmc (NOTE: this would be *extremely* slow for exponential impulse response)
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]





#####################################
## ContinuousStandardHawkesProcess ##
#####################################

# set hyperparameters
nnodes = 1;
duration = 1000.0; # 100.0 or 1000.0
Δtmax = 5.0;

# create a random process
baseline = HomogeneousProcess([θ[1]]);
weights = DenseWeightModel(reshape([θ[4]], (1, 1)));
impulses = LogitNormalImpulseResponse(reshape([θ[2]], (1, 1)), reshape([θ[3]], (1, 1)), Δtmax);
process = ContinuousStandardHawkesProcess(baseline, impulses, weights);
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

# estimate parameters via mcmc (NOTE: this would be *extremely* slow for exponential impulse response)
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]