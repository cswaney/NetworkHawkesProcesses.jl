abstract type Weights end
import Base.rand
import Base.size

size(model::Weights) = size(model.W)[1]

params(model::Weights) = copy(vec(model.W))

function params!(model::Weights, x)
    if length(x) != length(model.W)
        error("Parameter vector length does not match model parameter length.")
    else
        model.W = reshape(x, size(model.W))
    end
end

function rand(model::Weights)
    return rand.(Poisson.(model.W)) # NOTE: does not account for connection indicator
end

function rand(model::Weights, row::Int64)
    return rand.(Poisson.(model.W[row, :])) # NOTE: does not account for connection indicator
end

function rand(model::Weights, row::Int64, col::Int64)
    return rand(Poisson(model.W[row, col])) # NOTE: does not account for connection indicator
end

function sufficient_statistics(model::Weights, data::Tuple, parents)
    """Calculate sufficient statistics for continuous-time data."""
    _, nodes, _ = data
    _, parentnodes = parents
    nnodes = size(model)
    Mn = node_counts(nodes, nnodes)
    Mnm = parent_counts(nodes, parentnodes, nnodes)

    return Mn, Mnm
end

function sufficient_statistics(model::Weights, data::Matrix, parents)
    """Calculate sufficient statistics for discrete-time data."""
    ndims, _ = size(data)
    nbasis = div((size(parents)[3] - 1), ndims)
    Mn = node_counts(data)
    Mnm = parent_counts(parents, ndims, nbasis)

    return Mn, Mnm
end


"""DenseWeightModel"""
mutable struct DenseWeightModel <: Weights
    W
    κ
    ν
    κv
    νv
end

DenseWeightModel(W) = DenseWeightModel(W, 1.0, 1.0, ones(size(W)), ones(size(W)))

nparams(model::DenseWeightModel) = prod(size(model.W))

function resample!(model::DenseWeightModel, data, parents)
    Mn, Mnm = sufficient_statistics(model, data, parents)
    κ = model.κ .+ Mnm
    ν = model.ν .+ Mn
    model.W = rand.(Gamma.(κ, 1 ./ ν))
end

function logprior(model::DenseWeightModel)
    return sum(log.(pdf.(Gamma(model.κ, 1 / model.ν), model.W)))
end

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

variational_params(model::DenseWeightModel) = [vec(model.κv); vec(model.νv)]

function variational_log_expectation(model::DenseWeightModel, pidx, cidx)
    return digamma(model.κv[pidx, cidx]) - log(model.νv[pidx, cidx])
end

function q(model::DenseWeightModel)
    reshape([Gamma(κ, 1 / ν) for (κ, ν) in zip(vec(model.κv), vec(model.νv))], size(model.W))
end


"""SpikeAndSlabWeightModel"""
mutable struct SpikeAndSlabWeightModel <: Weights
    W
    κ0
    ν0
    κ1
    ν1
    κv0
    νv0
    κv1
    νv1
    ρv
end

function SpikeAndSlabWeightModel(W)
    κ0 = 0.1
    ν0 = 10.
    κ1 = 1.
    ν1 = 1.
    κv0 = 0.1 .* ones(size(W))
    νv0 = 10. .* ones(size(W))
    κv1 = ones(size(W))
    νv1 = ones(size(W))
    ρv = 0.5 * ones(size(W))
    SpikeAndSlabWeightModel(W, κ0, ν0, κ1, ν1, κv0, νv0, κv1, νv1, ρv)
end

nparams(model::SpikeAndSlabWeightModel) = prod(size(model.W))

variational_params(model::SpikeAndSlabWeightModel) = [vec(model.κv0); vec(model.νv0); vec(model.κv1); vec(model.νv1); vec(model.ρv)]

function resample!(model::SpikeAndSlabWeightModel, data, parents)
    Mn, Mnm = sufficient_statistics(model, data, parents)
    κ1 = model.κ1 .+ Mnm
    ν1 = model.ν1 .+ Mn
    model.W = rand.(Gamma.(κ1, 1 ./ ν1))
    return copy(model.W)
end

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



"""
    UnivariateWeightModel

"""
mutable struct UnivariateWeightModel{T<:AbstractFloat}
    w::T
    α0::T
    β0::T
    _α0::T
    _β0::T

    function UnivariateWeightModel{T}(w, α0, β0, _α0, _β0) where {T <: AbstractFloat}
        w < 0.0 && throw(DomainError(w, "weight parameter should be non-negative"))
        α0 > 0.0 || throw(DomainError(α0, "shape parameter α0 should be positive"))
        β0 > 0.0 || throw(DomainError(β0, "rate parameter β0 should be positive"))
        _α0 > 0.0 || throw(DomainError(_α0, "shape parameter _α0 should be positive"))
        _β0 > 0.0 || throw(DomainError(_β0, "rate parameter _β0 should be positive"))

        return new(w, α0, β0, _α0, _β0)
    end
end

UnivariateWeightModel(w::T, α0::T, β0::T, _α0::T, _β0::T) where {T<:AbstractFloat} = UnivariateWeightModel{T}(w, α0, β0, _α0, _β0)
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

function resample!(model, data, parents)
    ntotal, nchild = sufficient_statistics(model, data, parents)
    α = model.α0 + nchild
    β = model.β0 + ntotal
    model.w = rand(Gamma(α, 1 / β))

    return model.w
end

function sufficient_statistics(model::UnivariateWeightModel, data, parents)
    events, _ = data
    _, parentnodes = parents
    ntotal = length(events)
    nchild = mapreduce(x -> x == 1, +, parentnodes)

    return ntotal, nchild
end

# TODO: sufficient_statistics(model::UnivariateWeightModel, data::Matrix, parents)

logprior(model::UnivariateWeightModel) = pdf(Gamma(model.α0, 1 / model.β0), model.w)

# TODO: update!(model::UnivariateWeightModel, data, parents)

variational_params(model::UnivariateWeightModel) = [model._α0, model._β0]

function variational_log_expectation(model::UnivariateWeightModel)
    return digamma(model._α0) - log(model._β0)
end

q(model::UnivariateWeightModel) = Gamma(model._α0, 1 / model._β0)
