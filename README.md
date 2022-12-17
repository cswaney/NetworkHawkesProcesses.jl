# NetworkHawkesProcesses.jl
Mutually-exciting Hawkes processes in Julia.

## Description
This package implements methods to simulate and estimate mutually-exciting Hawkes processes with network structure as described in [Linderman, 2016](https://dash.harvard.edu/handle/1/33493391). It provides multiple inference procedures (including maximum-likelihood/maximum-a-posteriori estimation, Markov chain Monte Carlo sampling, and variational inference), a flexible set of model components, and an interface that allows users to develop custom models from new components.

Package features include:
- Supports continuous and discrete processes
- Uses modular design to support extensible components
- Implements simulation via Poisson thinning
- Provides multiple estimation/inference methods
- Supports a wide range of network specifications
- Supports non-homogeneous baselines
- Accelerates methods via Julia's built-in multithreading module

### Installation
```julia
julia> using Pkg; Pkg.add("NetworkHawkesProcesses")
```

### Usage
```julia
using NetworkHawkesProcesses
nnodes = 2
weight = 0.1
duration = 1000.0
Δtmax = 1.0
baseline = NetworkHawkesProcesses.HomogeneousProcess(ones(nnodes))
weights = NetworkHawkesProcesses.DenseWeightModel(weight .* ones(nnodes, nnodes))
impulses = NetworkHawkesProcesses.ExponentialImpulseResponse(ones(nnodes, nnodes))
process = NetworkHawkesProcesses.ContinuousStandardHawkesProcess(baseline, impulses, weights)
data = NetworkHawkesProcesses.rand(process, duration)
ll = NetworkHawkesProcesses.loglikelihood(process, data)
res = NetworkHawkesProcesses.mle!(process, data; verbose=true, regularize=true) # regularize => maximum a-priori estimation
```

## Roadmap
Beyond continued testing and documentation, we plan to add the following features in the future:
- stochastic variational inference
- advanced network models (e.g., latent distance networks)
- network models for weights
- time-varying network models
- exogenous covariates

## Contributing
Contributions and feedback are welcome. Please report issues and feature requests to our GitHub page, [https://github.com/cswaney/NetworkHawkesProcesses.jl](https://github.com/cswaney/NetworkHawkesProcesses.jl).
