using NetworkHawkesProcesses
using Gadfly

sigma = 1.0
eta = 1.0
kernel = NetworkHawkesProcesses.SquaredExponentialKernel(sigma, eta)
gp = NetworkHawkesProcesses.GaussianProcess(kernel)
bias = 0.0
duration = 10.0
nsteps = 10
nnodes = 2
baseline = NetworkHawkesProcesses.LogGaussianCoxProcess(gp, bias, duration, nsteps, nnodes)
data = rand(baseline, duration)
