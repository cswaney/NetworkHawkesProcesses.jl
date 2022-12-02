using NetworkHawkesProcesses

nnodes = 2
nbasis = 3
nlags = 4
weight = 0.1
duration = 1000
dt = 1.0

# DiscreteStandardHawkesProcess (DiscreteHomogeneousProcess, DiscreteGaussianImpulseResponse)
baseline = NetworkHawkesProcesses.DiscreteHomogeneousProcess(ones(nnodes), dt)
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
impulses = NetworkHawkesProcesses.DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt)
process = NetworkHawkesProcesses.DiscreteStandardHawkesProcess(baseline, impulses, weights, dt)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mle!(process, data; verbose=true, regularize=false)
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)

# DiscreteStandardHawkesProcess (DiscreteLogGaussianCoxProcess, DiscreteGaussianImpulseResponse)
# TODO

# DiscreteNetworkHawkesProcess (DiscreteHomogeneousProcess, DiscreteGaussianImpulseResponse)
baseline = NetworkHawkesProcesses.DiscreteHomogeneousProcess(ones(nnodes), dt)
impulses = NetworkHawkesProcesses.DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt)
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
network = NetworkHawkesProcesses.BernoulliNetworkModel(0.5, nnodes)
links = NetworkHawkesProcesses.rand(network)
process = NetworkHawkesProcesses.DiscreteNetworkHawkesProcess(baseline, impulses, weights, links, network, dt)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)

# DiscreteNetworkHawkesProcess (DiscreteLogGaussianCoxProcess, DiscreteGaussianImpulseResponse)
# TODO
