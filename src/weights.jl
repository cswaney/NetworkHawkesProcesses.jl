import Base.rand
import Base.ndims

abstract type Weights end

function ndims(model::Weights) end
function nparams(model::Weights) end
function params(model::Weights) end
function params!(model::Weights, x) end
function rand(model::Weights) end
function logprior(model::Weights) end


abstract type MultivariateWeightModel <: Weights end

ndims(model::MultivariateWeightModel) = size(model.W, 1)
nparams(model::MultivariateWeightModel) = length(model.W)
params(model::MultivariateWeightModel) = copy(vec(model.W))

function params!(model::MultivariateWeightModel, x)
    length(x) == nparams(model) || throw(ArgumentError("length of parameter vector x ($(length(x))) should equal the number of model parameters ($(nparams(model)))"))

    model.W = reshape(x, size(model.W))

    return params(model)
end

function rand(model::MultivariateWeightModel)
    return rand.(Poisson.(model.W))
end

function rand(model::MultivariateWeightModel, row::Integer)
    return rand.(Poisson.(model.W[row, :]))
end

function rand(model::MultivariateWeightModel, row::Integer, col::Integer)
    return rand(Poisson(model.W[row, col]))
end

function sufficient_statistics(model::MultivariateWeightModel, data::Tuple, parents)
    """Calculate sufficient statistics for continuous-time data."""
    _, nodes, _ = data
    _, parentnodes = parents
    nnodes = ndims(model)
    Mn = node_counts(nodes, nnodes)
    Mnm = parent_counts(nodes, parentnodes, nnodes)

    return Mn, Mnm
end

function sufficient_statistics(model::MultivariateWeightModel, data::Matrix, parents)
    """Calculate sufficient statistics for discrete-time data."""
    nnodes = ndims(model)
    nbasis = div((size(parents)[3] - 1), nnodes)
    Mn = node_counts(data)
    Mnm = parent_counts(parents, nnodes, nbasis)

    return Mn, Mnm
end


"""
    UnivariateWeightModel(w, α0, β0, αv, βv)
    UnivariateWeightModel(w)

A simple univariate weight model with gamma prior, `w ~ Gamma(α0, β0)`.

### Arguments
- `w::AbstractFloat`: non-negative weight parameter
- `α0::AbstractFloat`: positive shape hyperparameter (default: 1.0)
- `β0::AbstractFloat`: positive rate hyperparameter (default: 1.0)
- `αv::AbstractFloat`: positive shape variational parameter (default: 1.0)
- `βv::AbstractFloat`: positive rate variational parameter (default: 1.0)
"""
mutable struct UnivariateWeightModel{T<:AbstractFloat} <: Weights
    w::T
    α0::T
    β0::T
    αv::T
    βv::T

    function UnivariateWeightModel{T}(w, α0, β0, αv, βv) where {T<:AbstractFloat}
        w < 0.0 && throw(DomainError(w, "weight parameter should be non-negative"))
        α0 > 0.0 || throw(DomainError(α0, "shape parameter α0 should be positive"))
        β0 > 0.0 || throw(DomainError(β0, "rate parameter β0 should be positive"))
        αv > 0.0 || throw(DomainError(αv, "shape parameter αv should be positive"))
        βv > 0.0 || throw(DomainError(βv, "rate parameter βv should be positive"))

        return new(w, α0, β0, αv, βv)
    end
end

UnivariateWeightModel(w::T, α0::T, β0::T, αv::T, βv::T) where {T<:AbstractFloat} = UnivariateWeightModel{T}(w, α0, β0, αv, βv)
UnivariateWeightModel(w::T) where {T<:AbstractFloat} = UnivariateWeightModel{T}(w, 1.0, 1.0, 1.0, 1.0)

function multivariate(model::UnivariateWeightModel, params)
    W = Matrix(Diagonal(cat(params...; dims=1)))

    return DenseWeightModel(W)
end

nparams(model::UnivariateWeightModel) = 1
params(model::UnivariateWeightModel) = [model.w]

function params!(model::UnivariateWeightModel, θ)
    length(θ) == 1 || throw(ArgumentError("length of parameter vector θ should equal the number of model parameters"))
    w = θ[1]
    w < 0.0 && throw(DomainError(w, "weight parameter w should be non-negative"))
    model.w = w

    return params(model)
end

rand(model::UnivariateWeightModel) = rand(Poisson(model.w))

function resample!(model::UnivariateWeightModel, data, parents)
    ntotal, nchild = sufficient_statistics(model, data, parents)
    α = model.α0 + nchild
    β = model.β0 + ntotal
    model.w = rand(Gamma(α, 1 / β))

    return params(model)
end

function sufficient_statistics(model::UnivariateWeightModel, data, parents)
    events, _ = data
    _, parentnodes = parents
    ntotal = length(events)
    nchild = mapreduce(x -> x == 1, +, parentnodes)

    return ntotal, nchild
end

function sufficient_statistics(model::UnivariateWeightModel, data::Vector{Int64}, parents)
    """Calculate sufficient statistics for discrete-time data. `parents` is the `T x (1 + B)` variational parameter for the auxillary parent variables."""
    Mn = sum(data) # sum all events
    Mnm = sum(parents[:, 2:end]) # sum endogenous events

    return Mn, Mnm
end

logprior(model::UnivariateWeightModel) = pdf(Gamma(model.α0, 1 / model.β0), model.w)

function update!(model::UnivariateWeightModel, data, parents)
    """Perform a variational inference update. `parents` is the `T x (1 + B)` variational parameter for the auxillary parent variables."""
    T = length(data)
    B = size(parents, 2) - 1
    α = 0.0
    β = 0.0
    for t = 1:T
        s = data[t]
        for b = 1:B
            α += s * parents[t, 1 + b]
        end
        β += s
    end
    model.αv = model.α0 .+ α
    model.βv = model.β0 .+ β

    return copy(model.αv), copy(model.βv)
end

variational_params(model::UnivariateWeightModel) = [model.αv, model.βv]

function variational_log_expectation(model::UnivariateWeightModel)
    return digamma(model.αv) - log(model.βv)
end

q(model::UnivariateWeightModel) = Gamma(model.αv, 1 / model.βv)


"""
    DenseWeightModel(W, κ, ν, κv, νv)
    DenseWeightModel(W)

A dense multivariate weight model to be used with standard Hawkes processes. The model assumes a fully-connected network with a shared gamma prior for each connection weight:

```
W[i, j] ~ Gamma(κ, 1 / ν)
```

for all `i, j`.

### Arguments
- `W::Matrix{T<:AbstractFloat}`: non-negataive weight parameters
- `κ::AbstractFloat`: shared, positive shape hyperparameter (default: 1.0)
- `ν::AbstractFloat`: shared, positive rate hyperparameter (default: 1.0)
- `κv::Matrix{T<:AbstractFloat}`: positive variational shape parameter (default: ones(size(W)))
- `νv::Matrix{T<:AbstractFloat}`: positive variational rate parameter (default: ones(size(W)))
"""
mutable struct DenseWeightModel{T<:AbstractFloat} <: MultivariateWeightModel
    W::Matrix{T}
    κ::T
    ν::T
    κv::Matrix{T}
    νv::Matrix{T}

    function DenseWeightModel{T}(W, κ, ν, κv, νv) where {T<:AbstractFloat}
        any(W .< 0.0) && throw(DomainError(W, "weight parameter W should be non-negative"))
        κ > 0.0 || throw(DomainError(κ, "shape hyperparameter κ should be positive"))
        ν > 0.0 || throw(DomainError(ν, "rate hyperparameter κ should be positive"))
        all(κv .> 0.0) || throw(DomainError(κv, "variational shape paramter κv should be positive"))
        all(νv .> 0.0) || throw(DomainError(νv, "variational shape paramter νv should be positive"))
        return new(W, κ, ν, κv, νv)
    end
end

function DenseWeightModel(W::Matrix{T}, κ::T, ν::T, κv::Matrix{T}, νv::Matrix{T}) where {T<:AbstractFloat}
    return DenseWeightModel{T}(W, κ, ν, κv, νv)
end

DenseWeightModel(W::Matrix{T}) where {T<:AbstractFloat} = DenseWeightModel(W, 1.0, 1.0, ones(size(W)), ones(size(W)))

function resample!(model::DenseWeightModel, data, parents)
    Mn, Mnm = sufficient_statistics(model, data, parents)
    κ = model.κ .+ Mnm
    ν = model.ν .+ Mn
    model.W = rand.(Gamma.(κ, 1 ./ ν))
end

function logprior(model::DenseWeightModel)
    return sum(log.(pdf.(Gamma(model.κ, 1 / model.ν), model.W)))
end

variational_params(model::DenseWeightModel) = [vec(model.κv); vec(model.νv)]

function update!(model::DenseWeightModel, data, parents)
    """Perform a variational inference update. `parents` is the `T x N x (1 + NB)` variational parameter for the auxillary parent variables."""
    N, T = size(data)
    B = div(size(parents)[3] - 1, N)
    κ = zeros(N, N)
    ν = zeros(N, N)
    for pidx = 1:N
        for cidx = 1:N
            for t = 1:T
                sp = data[pidx, t]
                sc = data[cidx, t]
                for b = 1:B
                    κ[pidx, cidx] += sc * parents[t, cidx, 1+(pidx-1)*B+b]
                end
                ν[pidx, cidx] += sp
            end
        end
    end
    model.κv = model.κ .+ κ
    model.νv = model.ν .+ ν
    return copy(model.κv), copy(model.νv)
end

function variational_log_expectation(model::DenseWeightModel, pidx, cidx)
    return digamma(model.κv[pidx, cidx]) - log(model.νv[pidx, cidx])
end

function q(model::DenseWeightModel)
    reshape([Gamma(κ, 1 / ν) for (κ, ν) in zip(vec(model.κv), vec(model.νv))], size(model.W))
end


"""
    SpikeAndSlabWeightModel(W, κ0, ν0, κ1, ν1, κv0, vv0, κv1, vv1, ρv)
    SpikeAndSlabWeightModel(W)

A spike-and-slab multivariate weight model to be used with network Hawkes processes. The model assumes a sparse network with a shared gamma-mixture prior for each connection weight:

```
A[i, j] ~ Bernoulli(ρ)
W[i, j | A[i, j] == 0] ~ Gamma(κ0, 1 / ν0)
W[i, j | A[i, j] == 1] ~ Gamma(κ1, 1 / ν1)
```

for all `i, j`. The default hyperparameter `(κ0, ν0) = (.1, 10.)` are set to induce an approximately sparse weight matrix (`mean(Gamma(.1, 1 / 10.)) == 0.01`).

### Arguments
- `W::Matrix{T<:AbstractFloat}`: non-negataive weight parameters
- `κ0::AbstractFloat`: shared, positive shape hyperparameter (default: 0.1)
- `ν0::AbstractFloat`: shared, positive rate hyperparameter (default: 10.0)
- `κ1::AbstractFloat`: shared, positive shape hyperparameter (default: 1.0)
- `ν1::AbstractFloat`: shared, positive rate hyperparameter (default: 1.0)
- `κv0::Matrix{T<:AbstractFloat}`: positive variational shape parameters (default: 0.1 * ones(size(W)))
- `νv0::Matrix{T<:AbstractFloat}`: positive variational rate parameters (default: 10 * ones(size(W)))
- `κv1::Matrix{T<:AbstractFloat}`: positive variational shape parameters (default: ones(size(W)))
- `νv1::Matrix{T<:AbstractFloat}`: positive variational rate parameters (default: ones(size(W)))
- `ρv::Matrix{T<:AbstractFloat}`: unit-range variational connection rate parameters (default: 0.5 * ones(size(W)))
"""
mutable struct SpikeAndSlabWeightModel{T<:AbstractFloat} <: MultivariateWeightModel
    W::Matrix{T}
    κ0::T
    ν0::T
    κ1::T
    ν1::T
    κv0::Matrix{T}
    νv0::Matrix{T}
    κv1::Matrix{T}
    νv1::Matrix{T}
    ρv::Matrix{T}

    function SpikeAndSlabWeightModel{T}(W, κ0, ν0, κ1, ν1, κv0, vv0, κv1, vv1, ρv) where {T<:AbstractFloat}
        # TODO: validate arguments...

        return new(W, κ0, ν0, κ1, ν1, κv0, vv0, κv1, vv1, ρv)
    end
end

function SpikeAndSlabWeightModel(W::Matrix{T}, κ0::T, ν0::T, κ1::T, ν1::T,
    κv0::Matrix{T}, vv0::Matrix{T}, κv1::Matrix{T}, vv1::Matrix{T},
    ρv::Matrix{T}) where {T<:AbstractFloat}
    
    return SpikeAndSlabWeightModel{T}(W, κ0, ν0, κ1, ν1, κv0, vv0, κv1, vv1, ρv)
end

function SpikeAndSlabWeightModel(W::Matrix{T}) where {T<:AbstractFloat}
    κ0 = 0.1
    ν0 = 10.
    κ1 = 1.
    ν1 = 1.
    κv0 = 0.1 .* ones(size(W))
    νv0 = 10. .* ones(size(W))
    κv1 = ones(size(W))
    νv1 = ones(size(W))
    ρv = 0.5 * ones(size(W))
    
    return SpikeAndSlabWeightModel{T}(W, κ0, ν0, κ1, ν1, κv0, νv0, κv1, νv1, ρv)
end

function resample!(model::SpikeAndSlabWeightModel, data, parents)
    Mn, Mnm = sufficient_statistics(model, data, parents)
    κ1 = model.κ1 .+ Mnm
    ν1 = model.ν1 .+ Mn
    model.W = rand.(Gamma.(κ1, 1 ./ ν1))
    return copy(model.W)
end

function logprior(model::SpikeAndSlabWeightModel)
    throw(ErrorException("Not implemented"))
end

variational_params(model::SpikeAndSlabWeightModel) = [vec(model.κv0); vec(model.νv0); vec(model.κv1); vec(model.νv1); vec(model.ρv)]

function update!(model::SpikeAndSlabWeightModel, data, parents)
    N, T = size(data)
    B = Int((size(parents, 3) - 1) / N)
    κ = zeros(N, N)
    ν = zeros(N, N)
    for pidx = 1:N
        for cidx = 1:N
            for t = 1:T
                sp = data[pidx, t]
                sc = data[cidx, t]
                for b = 1:B
                    κ[pidx, cidx] += sc * parents[t, cidx, 1+(pidx-1)*B+b]
                end
                ν[pidx, cidx] += sp
            end
        end
    end
    model.κv0 = model.κ0 .+ κ 
    model.νv0 = model.ν0 .+ ν
    model.κv1 = model.κ1 .+ κ
    model.νv1 = model.ν1 .+ ν
    return copy(model.κv0), copy(model.νv0), copy(model.κv1), copy(model.νv1)
end

function variational_log_expectation(model::SpikeAndSlabWeightModel, pidx, cidx)
    ElogW0 = digamma(model.κv0[pidx, cidx]) - log(model.νv0[pidx, cidx])
    ElogW1 = digamma(model.κv1[pidx, cidx]) - log(model.νv1[pidx, cidx])
    return (1 - model.ρv[pidx, cidx]) * ElogW0 + model.ρv[pidx, cidx] * ElogW1
end

function q(model::SpikeAndSlabWeightModel, a)
    """Spike-and-slab model is a mixture model, so `q` is a conditional distribution."""
    if a == 0
        reshape([Gamma(κ, 1 / ν) for (κ, ν) in zip(vec(model.κv0), vec(model.νv0))], size(model.W))
    elseif a == 1
        reshape([Gamma(κ, 1 / ν) for (κ, ν) in zip(vec(model.κv1), vec(model.νv1))], size(model.W))
    end
end
