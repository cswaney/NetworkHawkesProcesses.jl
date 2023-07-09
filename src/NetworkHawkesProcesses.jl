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

include("utils/helpers.jl")
include("utils/interpolation.jl")
include("utils/gaussian.jl")
include("baselines.jl")
include("impulses.jl")
include("weights.jl")
include("networks.jl")
include("continuous.jl")
include("discrete.jl")
include("parents.jl")
include("inference.jl")
include("plotting.jl")

export HawkesProcess,
       ContinuousHawkesProcess,
       ContinuousUnivariateHawkesProcess,
       ContinuousMultivariateHawkesProcess,
       ContinuousStandardHawkesProcess,
       ContinuousNetworkHawkesProcess,
       DiscreteHawkesProcess,
       DiscreteStandardHawkesProcess,
       DiscreteNetworkHawkesProcess,
       Baseline,
       UnivariateBaseline,
       UnivariateHomogeneousProcess,
       HomogeneousProcess,
       LogGaussianCoxProcess,
       DiscreteHomogeneousProcess,
       DiscreteLogGaussianCoxProcess,
       UnivariateImpulseResponse,
       UnivariateLogitNormalImpulseResponse,
       LogitNormalImpulseResponse,
       ExponentialImpulseResponse,
       DiscreteGaussianImpulseResponse,
       DenseNetworkModel,
       UnivariateWeightModel,
       DenseWeightModel,
       SpikeAndSlabWeightModel,
       DenseNetworkModel,
       BernoulliNetworkModel,
       params,
       params!,
       rand,
       isstable,
       intensity,
       loglikelihood,
       logprior,
       mle!,
       mcmc!,
       vb!

end # module
