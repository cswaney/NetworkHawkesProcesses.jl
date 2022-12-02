using NetworkHawkesProcesses

nnodes = 2
weight = 0.1
duration = 1000.0
Δtmax = 1.0

# ContinuousStandardHawkesProcess (HomogeneousProcess, ExponentialImpulseResponse)
baseline = NetworkHawkesProcesses.HomogeneousProcess(ones(nnodes))
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
impulses = NetworkHawkesProcesses.ExponentialImpulseResponse(ones(nnodes, nnodes))
process = NetworkHawkesProcesses.ContinuousStandardHawkesProcess(baseline, impulses, weights)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mle!(process, data; verbose=true, regularize=true) # maximum a posteriori

# ContinuousStandardHawkesProcess (HomogeneousProcess, LogitNormalImpulseResponse)
baseline = NetworkHawkesProcesses.HomogeneousProcess(ones(nnodes))
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
impulses = NetworkHawkesProcesses.LogitNormalImpulseResponse(ones(nnodes, nnodes), ones(nnodes, nnodes), Δtmax)
process = NetworkHawkesProcesses.ContinuousStandardHawkesProcess(baseline, impulses, weights)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mle!(process, data; verbose=true, regularize=true) # maximum a posteriori
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)


# ContinuousStandardHawkesProcess (LogGaussianCoxProcess, ExponentialImpulseResponse)

# ContinuousStandardHawkesProcess (LogGaussianCoxProcess, LogitNormalImpulseResponse)


# ContinuousNetworkHawkesProcess (HomogeneousProcess, ExponentialProcess)
baseline = NetworkHawkesProcesses.HomogeneousProcess(ones(nnodes))
impulses = NetworkHawkesProcesses.ExponentialImpulseResponse(ones(nnodes, nnodes))
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
network = NetworkHawkesProcesses.BernoulliNetworkModel(0.5, nnodes)
links = NetworkHawkesProcesses.rand(network)
process = NetworkHawkesProcesses.ContinuousNetworkHawkesProcess(baseline, impulses, weights, links, network)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)

# ContinuousNetworkHawkesProcess (HomogeneousProcess, LogitNormalImpulseResponse)
baseline = NetworkHawkesProcesses.HomogeneousProcess(ones(nnodes))
impulses = NetworkHawkesProcesses.LogitNormalImpulseResponse(ones(nnodes, nnodes), ones(nnodes, nnodes), Δtmax)
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
network = NetworkHawkesProcesses.BernoulliNetworkModel(0.5, nnodes)
links = NetworkHawkesProcesses.rand(network)
process = NetworkHawkesProcesses.ContinuousNetworkHawkesProcess(baseline, impulses, weights, links, network)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)


# ContinuousNetworkHawkesProcess (LogGaussianCoxProcess, ExponentialProcess)
# ...
# res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)

# ContinuousNetworkHawkesProcess (LogGaussianCoxProcess, LogitNormalImpulseResponse)
# ...
# res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)