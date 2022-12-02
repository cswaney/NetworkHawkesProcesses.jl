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
using Optim
using DSP
using Distributions

abstract type HawkesProcess end

include("utils/helpers.jl")
include("utils/interpolation.jl")
include("utils/gaussian.jl")
include("inference.jl")
include("baselines.jl")
include("impulses.jl")
include("weights.jl")
include("networks.jl")
include("continuous.jl")
include("discrete.jl")
include("parents.jl")

export ContinuousStandardHawkesProcess,
       ContinuousNetworkHawkesProcess,
       HomogeneousProcess,
       LogGaussianCoxProcess,
       LogitNormalImpulseResponse,
       ExponentialImpulseResponse,
       DenseNetworkModel,
       DenseWeightModel,
       BernoulliNetworkModel

end # module
