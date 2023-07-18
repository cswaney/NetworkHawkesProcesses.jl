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
plink = 0.5;

# create a random process
baseline = DiscreteHomogeneousProcess(rand(nnodes), dt);
impulses = DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt);
weights = DenseWeightModel(rand(nnodes, nnodes));
network = BernoulliNetworkModel(plink, nnodes);
links = NetworkHawkesProcesses.rand(network);
process = DiscreteNetworkHawkesProcess(baseline, impulses, weights, links, network, dt);
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = copy(params(process));

# generate random data
data = NetworkHawkesProcesses.rand(process, duration);
println("Generated $(sum(data)) events")

# estimate parameters via mcmc
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]

# reset parameters
params!(process, θ)

# estimate parameters via (mean-field) variational Bayes
res = vb!(process, data; max_steps=100, verbose=true);
qλ = mean.(NetworkHawkesProcesses.q(process.baseline))
qW0 = mean.(NetworkHawkesProcesses.q(process.weight_model, 0))
qW1 = mean.(NetworkHawkesProcesses.q(process.weight_model, 1))
qθ = mean.(NetworkHawkesProcesses.q(process.impulse_response))
qθ = reshape(transpose(cat(qθ..., dims=2)), size(process.impulse_response.θ))
qA = Bool.(round.(mean.(Bernoulli.(process.weight_model.ρv))))
θvb = [qλ; vec(qA .* qW1 .* qθ)];
[θ θvb]
