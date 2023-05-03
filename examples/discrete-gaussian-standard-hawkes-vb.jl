"""A simple synthetic example using a discrete-time standard process with variational Bayes inference."""

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

# estimate parameters via (mean-field) variational Bayes
res = vb!(process, data; max_steps=10, verbose=true);
qλ = mean.(NetworkHawkesProcesses.q(process.baseline))
qW = mean.(NetworkHawkesProcesses.q(process.weights))
qθ = mean.(NetworkHawkesProcesses.q(process.impulses))
qθ = reshape(transpose(cat(qθ..., dims=2)), size(process.impulses.θ))
θvb = [qλ; vec(qW .* qθ)];
[θ θvb]