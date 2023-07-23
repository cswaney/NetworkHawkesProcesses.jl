"""
A Julia package for simulation and estimation of network Hawkes processes.

More information at https://github.com/NetworkHawkesProcesses
"""
module NetworkHawkesProcesses

import Base.ndims
import Base.rand
import Base.size
import Distributions.params
import Distributions.params!
using Statistics
using LinearAlgebra
using PDMats
using Optim
using DSP
using Distributions
using SpecialFunctions

abstract type HawkesProcess end

"""
    ndims(process::HawkesProcess)

Return the number of dimensions of `process`.
"""
function ndims(process::HawkesProcess) end

"""
    nparams(process::HawkesProcess)

Return the number of trainable parameters of `process`.
"""
function nparams(process::HawkesProcess) end

"""
    isstable(process::HawkesProcess)

Check if a process is stable. If `false`, the process may "blow up" (i.e., fail to generate a finite samples).
"""
function isstable(process::HawkesProcess) end

"""
    params(process::HawkesProcess)

Return a copy of the trainable parameters of `process` as a vector.
"""
function params(process::HawkesProcess) end

"""
    params!(process::HawkesProcess, x)

Set the trainable parameters of a process to `x`, where `x` is assumed to follow the same order as `params(process)`.
"""
function params!(process::HawkesProcess, x) end


include("utils/helpers.jl")
include("utils/interpolation.jl")
include("utils/gaussian.jl")
include("baselines.jl")
include("impulses.jl")
include("weights.jl")
include("networks.jl")
include("continuous.jl")
include("discrete.jl")
include("independent.jl")
include("parents.jl")
include("inference.jl")
include("plotting.jl")

export HawkesProcess,
       ContinuousHawkesProcess,
       ContinuousUnivariateHawkesProcess,
       ContinuousMultivariateHawkesProcess,
       ContinuousIndependentHawkesProcess,
       ContinuousStandardHawkesProcess,
       ContinuousNetworkHawkesProcess,
       DiscreteHawkesProcess,
       DiscreteUnivariateHawkesProcess,
       DiscreteStandardHawkesProcess,
       DiscreteNetworkHawkesProcess,
       Baseline,
       ContinuousBaseline,
       ContinuousUnivariateBaseline,
       UnivariateHomogeneousProcess,
       UnivariateLogGaussianCoxProcess,
       ContinuousMultivariateBaseline,
       HomogeneousProcess,
       LogGaussianCoxProcess,
       DiscreteBaseline,
       DiscreteUnivariateBaseline,
       DiscreteUnivariateHomogeneousProcess,
       DiscreteUnivariateLogGaussianCoxProcess,
       DiscreteMultivariateBaseline,
       DiscreteHomogeneousProcess,
       DiscreteLogGaussianCoxProcess,
       IndependentHawkesProcess,
       Independent,
       ImpulseResponse,
       ContinuousImpulseResponse,
       ContinuousUnivariateImpulseResponse,
       ContinuousMultivariateImpulseResponse,
       DiscreteImpulseResponse,
       DiscreteUnivariateImpulseResponse,
       DiscreteMultivariateImpulseResponse,
       UnivariateExponentialImpulseResponse,
       UnivariateLogitNormalImpulseResponse,
       LogitNormalImpulseResponse,
       ExponentialImpulseResponse,
       UnivariateGaussianImpulseResponse,
       DiscreteGaussianImpulseResponse,
       Weights,
       UnivariateWeightModel,
       DenseWeightModel,
       SpikeAndSlabWeightModel,
       DenseNetworkModel,
       BernoulliNetworkModel,
       isstable,
       nparams,
       params,
       params!,
       rand,
       intensity,
       loglikelihood,
       logprior,
       mle!,
       mcmc!,
       vb!

end # module
