## Hawkes Processes
```@docs
ContinuousUnivariateHawkesProcess
ContinuousStandardHawkesProcess
ContinuousNetworkHawkesProcess
DiscreteUnivariateHawkesProcess
DiscreteStandardHawkesProcess
DiscreteNetworkHawkesProcess
IndependentHawkesProcess
```

```@docs
ndims(process::HawkesProcess)
isstable(process::HawkesProcess)
nparams(process::HawkesProcess)
params(process::HawkesProcess)
params!(process::HawkesProcess, x)
rand(process::ContinuousHawkesProcess, duration::AbstractFloat)
rand(process::DiscreteHawkesProcess, duration::Integer)
loglikelihood(process::HawkesProcess, data)
logprior(process::HawkesProcess)
intensity(process::ContinuousHawkesProcess, data, time::AbstractFloat)
intensity(process::DiscreteHawkesProcess, data, time::Integer)
```

```@docs
NetworkHawkesProcesses.convolve(process::DiscreteHawkesProcess, data)
```


## Baseline Processes
```@docs
UnivariateHomogeneousProcess
HomogeneousProcess
UnivariateLogGaussianCoxProcess
UnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration::AbstractFloat, nsteps::Integer, m::T=0.0) where {T<:AbstractFloat}
LogGaussianCoxProcess
DiscreteUnivariateHomogeneousProcess
DiscreteHomogeneousProcess
DiscreteUnivariateLogGaussianCoxProcess
DiscreteUnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration, nsteps, m::T=0.0, dt::T=1.0) where {T<:AbstractFloat}
DiscreteLogGaussianCoxProcess
GaussianProcess
```


## Impulse Response Models
```@docs
UnivariateExponentialImpulseResponse
ExponentialImpulseResponse
UnivariateLogitNormalImpulseResponse
LogitNormalImpulseResponse
UnivariateGaussianImpulseResponse
GaussianImpulseResponse
```


## Weight Models
```@docs
UnivariateWeightModel
DenseWeightModel
SpikeAndSlabWeightModel
```


## Network Models
```@docs
DenseNetworkModel
BernoulliNetworkModel
```

```@docs

```


## Inference
```@docs
mle!
MaximumLikelihood
mcmc!
MarkovChainMonteCarlo
vb!
VariationalInference
```
