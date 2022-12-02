# NetworkHawkesProcesses.jl
Hawkes processes in Julia.


## Description
What is this project?

## Usage
How do you use it?

### Installation
```julia
julia> using Pkg; Pkg.add("NetworkHawkesProcesses")
```

### Usage
```julia
julia> using NetworkHawkesProcesses
```

#### Models
```julia
baseline = # ...
impulses = # ...
weights = # ...
process = NetworkHawkesProcesses.StandardHawkesProcess(baseline, impulses, weights)
```

#### Simulation
```julia
data = NetworkHawkesProcesses.rand(process, 100.)
```

#### Inference
```julia
res = NetworkHawkesProcesses.mle!(process, data)
res = NetworkHawkesProcesses.mcmc!(process, data)
```

#### Extensions
You can extend `NetworkHawkesProcesses` to support new models by creating new types that adhere to the interface of the model component you wish to replace.

For example, here's how to create a new `Network` model:
```julia
# TODO
```

### Examples
A few simple examples.

#### Continuous Processes

#### Discrete Processes

#### Visualization
How to view the results of simulation and inference?

## Contributing
How to contribute and report issues.

## Roadmap
What are the next steps?








## Continuous Hawkes Processes

### `ContinuousNetworkHawkesProcess`

#### Inference
- Maximum-likelihood
    - Baseline: any `BaselineProcess`.
    - Impulse: `LogitNormalProcess`.
    - Weights: `DenseWeightModel`.
    - Network: `Nothing` (connection matrix is `A` is taken as given and not estimated).
- Markov chain Monte Carlo
    - Baseline: any `BaselineProcess`.
    - Impulse: `LogitNormalProcess`.
    - Weights: `DenseWeightModel`.
    - Network: any `NetworkModel` or `Nothing` (if `Nothing`, then network model is ignored; connection matrix is `A` is taken as given and not estimated).


### `ContinuousStandardHawkesProcess`

#### Inference
- Maximum-likelihood
    - Baseline: any `BaselineProcess`.
    - Impulse: `ExponentialProcess`.
    - Weights: `DenseWeightModel`.
    - Network: `Nothing` (connection matrix is `A` is taken as given and not estimated).
- Markov chain Monte Carlo
    - Baseline: any `BaselineProcess`.
    - Impulse: `ExponentialProcess`.
    - Weights: `DenseWeightModel`.
    - Network: any `NetworkModel` or `Nothing` (if `Nothing`, then network model is ignored; connection matrix is `A` is taken as given and not estimated).


## Discrete Hawkes Processes

### `DiscreteNetworkHawkesProcess`
- Maximum-likelihood
    - Baseline: any `DiscreteProcess`.
    - Impulse: `NormalMixtureProcess`.
    - Weights: `DenseWeightModel`.
    - Network: `Nothing` (connection matrix is `A` is taken as given and not estimated).
- Markov chain Monte Carlo
    - Baseline: any `DiscreteProcess`.
    - Impulse: `NormalMixtureProcess`.
    - Weights: `DenseWeightModel`.
    - Network: any `NetworkModel` or `Nothing` (if `Nothing`, then network model is ignored; connection matrix is `A` is taken as given and not estimated).
- Variational Bayes
    - Baseline: any `DiscreteProcess`.
    - Impulse: `NormalMixtureProcess`.
    - Weights: `DenseWeightModel`.
    - Network: any `NetworkModel` or `Nothing` (if `Nothing`, then network model is ignored; connection matrix is `A` is taken as given and not estimated).