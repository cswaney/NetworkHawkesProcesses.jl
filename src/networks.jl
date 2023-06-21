abstract type Network end
import Base.size
import Base.rand

function size(network::Network) end
function params(network::Network) end
function rand(network::Network) end
function link_probability(network::Network) end
function resample!(network::Network, data) end
function loglikelihood(network::Network, data) end

"""
DenseNetwork

A fully-connected network model.

# Arguments
- `nnodes::Int64`: the network size.
"""
mutable struct DenseNetworkModel <: Network
    nnodes
end

size(network::DenseNetworkModel) = network.nnodes

params(network::DenseNetworkModel) = []

rand(network::DenseNetworkModel) = ones(network.nnodes, network.nnodes)

link_probability(network::DenseNetworkModel) = ones(network.nnodes, network.nnodes)

resample!(network::DenseNetworkModel, data) = ones(size(data))


"""
    BernoulliNetworkModel

A network model with independent Bernoulli distributed link probabilities.

# Arguments
- `ρ::Float64`: the probability of a connection between any two nodes (`ρ ∈ [0., 1.]`).
- ...
- `N::Int64`: the network size / number of nodes.
"""
mutable struct BernoulliNetworkModel <: Network
    ρ
    α
    β
    αv::Float64
    βv::Float64
    nnodes
end

BernoulliNetworkModel(ρ, nnodes) = BernoulliNetworkModel(ρ, 1.0, 1.0, 1.0, 1.0, nnodes)

size(network::BernoulliNetworkModel) = network.nnodes

params(network::BernoulliNetworkModel) = [network.ρ]

variational_params(network::BernoulliNetworkModel) = [network.αv, network.βv]

function rand(network::BernoulliNetworkModel)
    nnodes = size(network)
    return rand(Bernoulli(network.ρ), nnodes, nnodes)
end

function link_probability(network::BernoulliNetworkModel)
    nnodes = size(network)
    return network.ρ * ones(nnodes, nnodes)
end

resample!(network::BernoulliNetworkModel, data) = resample_connection_probability!(network, data)

function resample_connection_probability!(network::BernoulliNetworkModel, data)
    nlinks = sum(data)
    αn = network.α + nlinks
    βn = network.β + prod(size(data)) - nlinks
    network.ρ = rand(Beta(αn, βn))
    return copy(network.ρ)
end

function update!(network::BernoulliNetworkModel, ρ)
    α, β = update_link_probabilities!(network, ρ)
    return α, β
end

function update_link_probabilities!(network::BernoulliNetworkModel, ρ)
    network.αv = network.α + sum(ρ)
    network.βv = network.β + sum(1 .- ρ)
    return copy(network.αv), copy(network.βv)
end

function variational_log_expectation(network::BernoulliNetworkModel)
    return (digamma.(network.αv) .- digamma.(network.αv .+ network.βv)) .- (digamma.(network.βv) .- digamma.(network.αv .+ network.βv))
end

"""StochasticBlockNetworkModel"""


"""LatentDistanceNetworkModel"""
