using NetworkHawkesProcesses
using NetworkHawkesProcesses: params, params!, nsteps
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess
using Random
using Statistics

# set random seed
Random.seed!(0);

# set hyperparameters
nbasis = 3;
nlags = 4;
ndims = 3;
duration = 100;
dt = 2.0;
sigma = 1.0
eta = 1.0
bias = 0.0
nstep = 10

# create a random process
kernel = SquaredExponentialKernel(sigma, eta);
gp = GaussianProcess(kernel);
process = Independent{DiscreteHawkesProcess}([
    DiscreteUnivariateHawkesProcess(
        DiscreteUnivariateLogGaussianCoxProcess(gp, duration, nstep, bias, dt),
        UnivariateGaussianImpulseResponse(ones(nbasis) ./ nbasis, nlags, dt),
        UnivariateWeightModel(rand())
    ) for _ in 1:ndims
])
println("Process is stable? $(isstable(process))")

# save a copy of parameters
θ = params(process)

# create random data
data = rand(process, nsteps(process.list[1].baseline)) # dt != 1.0 => nsteps != duration
println("Generated $(sum(data)) events")

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
process = DiscreteStandardHawkesProcess(process)