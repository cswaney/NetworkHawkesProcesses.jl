## Hawkes Processes

### Continuous Processes
```@docs
ContinuousStandardHawkesProcess
```

```@docs
ContinuousNetworkHawkesProcess
```

```@docs
rand(process::ContinuousHawkesProcess, duration::Float64)
```

```@docs
loglikelihood(process::ContinuousHawkesProcess, data)
```

```@docs
intensity(process::ContinuousHawkesProcess, data, times::Vector{Float64})
```

### Discrete Processes
```@docs
DiscreteStandardHawkesProcess
```

```@docs
DiscreteNetworkHawkesProcess
```

```@docs
rand(process::DiscreteHawkesProcess, steps::Int64)
```

```@docs
loglikelihood(process::DiscreteHawkesProcess, data)
```

```@docs
intensity(process::DiscreteHawkesProcess, convolved)
```


## Baseline Processes

### Continuous Processes
```@docs
HomogeneousProcess
```

```@docs
LogGaussianCoxProcess
```

### Discrete Processes
```@docs
DiscreteHomogeneousProcess
```

```@docs
DiscreteLogGaussianCoxProcess
```

## Impulse Response Models

### Continuous Models
```@docs
ExponentialImpulseResponse
```

```@docs
LogitNormalImpulseResponse
```

### Discrete Models
```@docs
DiscreteGaussianImpulseResponse
```


## Weight Models
```@docs
DenseWeightModel
```

```@docs
SparseWeightModel
```


## Network Models
```@docs
DenseNetworkModel
```

```@docs
BernoulliNetworkModel
```


## Inference
```@docs
mcmc!(process::HawkesProcess, data; nsteps=1000, verbose=false)
```

```@docs
vb!(process::HawkesProcess, data; max_steps::Int64=1_000, Δx_thresh=1e-6, Δq_thresh=1e-2)
```