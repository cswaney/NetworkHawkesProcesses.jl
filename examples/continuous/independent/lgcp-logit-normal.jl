using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!
using Random
using Statistics

# set random seed
Random.seed!(1);

# set hyperparameters
ndims = 3;
duration = 100.0;
nstep = 10;
sigma = 1.0
eta = 1.0
bias = 0.0
Δtmax = 5.0;

# create a random process
kernel = SquaredExponentialKernel(sigma, eta);
gp = GaussianProcess(kernel);
process = Independent{ContinuousHawkesProcess}([
    ContinuousUnivariateHawkesProcess(
        UnivariateLogGaussianCoxProcess(gp, duration, nstep, bias),
        UnivariateLogitNormalImpulseResponse(rand(), rand(), Δtmax),
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
process = ContinuousStandardHawkesProcess(process)