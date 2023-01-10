[![Build Status](https://app.travis-ci.com/cswaney/NetworkHawkesProcesses.jl.svg?branch=master)](https://app.travis-ci.com/cswaney/NetworkHawkesProcesses.jl)

# NetworkHawkesProcesses.jl
Network Hawkes processes in Julia.

## Description
This package implements methods to simulate and estimate mutually-exciting Hawkes processes with network structure as described in [Linderman, 2016](https://dash.harvard.edu/handle/1/33493391). It allows researchers to construct models from a flexible set of model components, run inference from a list of compatible methods (including maximum-likelihood estimation, Markov chain Monte Carlo sampling, and variational inference), and explore results with visualization and diagnostic utilities. 

### Key Features
- Supports continuous and discrete processes
- Uses modular design to support extensible components
- Implements simulation via Poisson thinning
- Provides multiple estimation/inference methods
- Supports a wide range of network specifications
- Supports non-homogeneous baselines
- Accelerates methods via Julia's built-in multithreading module

## Installation
```julia
using Pkg;
Pkg.add("https://github.com/cswaney/NetworkHawkesProcesses.jl.git")
```

## Usage
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
res = NetworkHawkesProcesses.mle!(process, data; verbose=true, regularize=true)
```

## Roadmap
In addition to improved testing and documentation, we plan to add the following features in future releases:
- Support for (distributed) multiple-trial inference
- Support stochastic variational inference
- Implement additional network models (e.g., stochastic block and latent distance networks)
- Implement network models for weights
- Support for time-varying network models
- Support baselines processes with exogenous covariates

## Contributing
Contributions and feedback are welcome. Please report issues and feature requests to our GitHub page, [https://github.com/cswaney/NetworkHawkesProcesses.jl](https://github.com/cswaney/NetworkHawkesProcesses.jl).
