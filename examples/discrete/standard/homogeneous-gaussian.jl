"""A simple synthetic example using a continuous-time standard process with logit-normal impulse response."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
nnodes = 2;
nbasis = 3;
nlags = 4;
duration = 1000; # 100 or 1000
dt = 1.0;

# create a random process
baseline = DiscreteHomogeneousProcess(rand(nnodes), dt);
weights = DenseWeightModel(rand(nnodes, nnodes));
impulses = DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt);
process = DiscreteStandardHawkesProcess(baseline, impulses, weights, dt);
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = copy(params(process));

# generate random data
data = NetworkHawkesProcesses.rand(process, duration);
println("Generated $(sum(data)) events")

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

# reset parameters
params!(process, θ);

# estimate parameters via (mean-field) variational Bayes
res = vb!(process, data; max_steps=100, verbose=true);
qλ = mean.(NetworkHawkesProcesses.q(process.baseline))
qW = mean.(NetworkHawkesProcesses.q(process.weight_model))
qθ = mean.(NetworkHawkesProcesses.q(process.impulse_response))
qθ = reshape(transpose(cat(qθ..., dims=2)), size(process.impulse_response.θ))
θvb = [qλ; vec(qW .* qθ)];
[θ θvb]