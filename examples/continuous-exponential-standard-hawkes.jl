"""A simple synthetic example using a continuous-time standard process with exponential impulse response.

Gibbs sampling is *extremely* slow for models using an exponential impulse response. Therefore, this example only applies maximum-likelihood estimation. On the other hand, maximum-likelihood estimation seems to be faster in this case than for models using the logit-normal impulse reponse with a similar number of events.
"""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
nnodes = 2;
duration = 1000.0; # 100.0 or 1000.0

# create a random process
baseline = HomogeneousProcess(rand(nnodes));
weights = DenseWeightModel(rand(nnodes, nnodes));
impulses = ExponentialImpulseResponse(rand(nnodes, nnodes));
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
