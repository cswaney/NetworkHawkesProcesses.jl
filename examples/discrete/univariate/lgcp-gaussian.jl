"""A simple synthetic example using a continuous-time univariate process with logit-normal impulse response and log-Gaussian Cox baseline."""

using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess
using Statistics
using Random

# set random seed
Random.seed!(0);

# set hyperparameters
nbasis = 3;
nlags = 4;
duration = 1000.0; # 100.0 or 1000.0
dt = 1.0;
Δtmax = 2.0;
sigma = 1.0
eta = 1.0
bias = 0.0
nsteps = 10

# create a random process
kernel = SquaredExponentialKernel(sigma, eta);
gp = GaussianProcess(kernel);
baseline = DiscreteUnivariateLogGaussianCoxProcess(gp, duration, nsteps, bias, dt);
impulses = UnivariateGaussianImpulseResponse(ones(nbasis) ./ nbasis, nlags, dt);
weights = UnivariateWeightModel(rand());
process = DiscreteUnivariateHawkesProcess(baseline, impulses, weights);
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = copy(params(process));

# generate random data
data = NetworkHawkesProcesses.rand(process, Int(duration));
println("Generated $(sum(data)) events")

# estimate parameters via mle
res = mle!(process, data; verbose=true);
θmle = res.maximizer;
[θ θmle]

# reset parameters
params!(process, θ);

# estimate parameters via mcmc
res = mcmc!(process, data; verbose=true);
θmcmc = mean(res.samples);
[θ θmcmc]
