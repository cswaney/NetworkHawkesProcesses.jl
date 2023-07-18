"""A simple synthetic example using a continuous-time standard process with logit-normal impulse response."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
nnodes = 2;
nbasis = 3;
nlags = 4;
duration = 100; # 100 or 1000
dt = 1.0;
sigma = 1.0
eta = 1.0
bias = 0.0
nsteps = 10

# create a random process
kernel = SquaredExponentialKernel(sigma, eta);
gp = GaussianProcess(kernel);
baseline = DiscreteLogGaussianCoxProcess(gp, bias, duration, nsteps, nnodes, dt);
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
