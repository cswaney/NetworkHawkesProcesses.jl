"""A simple synthetic example using a continuous-time standard process with logit-normal impulse response."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
nnodes = 2;
duration = 1000.0; # 100.0 or 1000.0
Δtmax = 1.0;
plink = 0.5;

# create a random process
baseline = HomogeneousProcess(rand(nnodes));
weights = DenseWeightModel(rand(nnodes, nnodes));
impulses = LogitNormalImpulseResponse(rand(nnodes, nnodes), rand(nnodes, nnodes), Δtmax);
network = NetworkHawkesProcesses.BernoulliNetworkModel(plink, nnodes);
links = NetworkHawkesProcesses.rand(network);
process = NetworkHawkesProcesses.ContinuousNetworkHawkesProcess(baseline, impulses, weights, links, network);
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = copy(params(process));

# generate random data
data = NetworkHawkesProcesses.rand(process, duration);
println("Generated $(length(data[1])) events")

# estimate parameters via mcmc (mle not available for discrete variables)
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]
