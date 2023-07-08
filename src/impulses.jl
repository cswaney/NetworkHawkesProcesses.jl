abstract type ImpulseResponse end
import Base.rand
import Base.size

function size(impulse::ImpulseResponse) end
function rand(impulse::ImpulseResponse, duration) end
function resample!(impulse::ImpulseResponse, data) end
function intensity(impulse::ImpulseResponse) end
function intensity(impulse::ImpulseResponse, parentnode, childnode, time) end


"""
ExponentialImpulseResponse

An exponential impulse response function with Gamma prior.

This model is a building block for Hawkes processes. Given a number of child events on node `j` attributed to node `i`, it generates event times according to a exponential distribution with rate parameter `θ[i, j]`.

For Bayesian inference we assume a uniform Gamma priors for `θ`:
    
    `θ[i, j] ~ Gamma(κ, ν) for all i, j`

The resulting model is only conjugate in the limit as the sampling duration approaches infinity, however. Thus, Bayesian inference for the exponential process is approximate.

# Arguments
- `θ::Float64`: the rate of the exponential distribution (i.e., `1 / scale`).
- `α`: shape parameter of Gamma prior for `θ` (default: 1.0).
- `β`: rate parameter of Gamma prior for `θ` (default: 1.0).
"""
mutable struct ExponentialImpulseResponse <: ImpulseResponse
    θ
    α
    β
    Δtmax
end

ExponentialImpulseResponse(θ) = ExponentialImpulseResponse(θ, 1.0, 1.0, Inf)

size(impulse::ExponentialImpulseResponse) = size(impulse.θ)[1]

params(impulse::ExponentialImpulseResponse) = copy(vec(impulse.θ))

function params!(impulse::ExponentialImpulseResponse, x)
    if length(x) != length(impulse.θ)
        error("Parameter vector length does not match model parameter length.")
    else
        nnodes = size(impulse)
        impulse.θ = reshape(x, nnodes, nnodes)
    end
    return nothing
end

function rand(impulse::ExponentialImpulseResponse, n::Int64)
    ts = rand.(Exponential.(1 ./ impulse.θ), n)
    return sort.(ts) # NOTE: assumes infinite duration; truncate results as required
end

function rand(impulse::ExponentialImpulseResponse, row::Int64, n::Int64)
    ts = rand.(Exponential.(1 ./ impulse.θ[row, :]), n)
    return sort.(ts) # NOTE: assumes infinite duration; truncate results as required
end

function rand(impulse::ExponentialImpulseResponse, row::Int64, col::Int64, n::Int64)
    ts = rand(Exponential(1 / impulse.θ[row, col]), n)
    return sort(ts) # NOTE: assumes infinite duration; truncate results as required
end

function resample!(impulse::ExponentialImpulseResponse, data, parents)
    Mnm, Xnm = sufficient_statistics(impulse, data, parents)
    αnm = impulse.α .+ Mnm  # = α (prior) for zero-counts
    βnm = impulse.β .+ Mnm .* Xnm  # Xnm returns 0 for zero-counts => β (prior)
    impulse.θ = rand.(Gamma.(αnm, 1 ./ βnm))
end

function sufficient_statistics(impulse::ExponentialImpulseResponse, data, parents)
    nnodes = size(impulse)
    events, nodes, _ = data
    parentindices, parentnodes = parents
    Mnm = parent_counts(nodes, parentnodes, nnodes)
    Xnm = duration_mean(events, nodes, parentindices, nnodes)
    return Mnm, Xnm
end

function duration_mean(events, nodes, parentindices, nnodes)
    Xnm = zeros(nnodes, nnodes)
    Mnm = zeros(nnodes, nnodes)
    for (event, node, parent) in zip(events, nodes, parentindices)
        if parent > 0
            parentnode = nodes[parent]
            parentevent = events[parent]
            Mnm[parentnode, node] += 1
            Xnm[parentnode, node] += event - parentevent
        end
    end
    return fillna!(Xnm ./ Mnm, 0)
end

function intensity(impulse::ExponentialImpulseResponse)
    ir = Matrix{Function}(undef, size(impulse.θ))
    for (idx, val) in enumerate(impulse.θ)
        ir[idx] = t -> pdf(Exponential(1 ./ impulse.θ[idx]), t)
    end
    return ir
end

function intensity(impulse::ExponentialImpulseResponse, parentnode, childnode, Δt)
    return pdf(Exponential(1 ./ impulse.θ[parentnode, childnode]), Δt)
end

function logprior(impulse::ExponentialImpulseResponse)
    return sum(log.(pdf.(Gamma(impulse.α, 1 / impulse.β), impulse.θ)))
end


"""
LogitNormalImpulseResponse

A logit-normal impulse response function.

This model is a building block for Hawkes process. Given a number of child events on node `j` attributed to node `i`, it generates event times according to a stretched logit-normal distribution with location parameter `μ[i, j]`, precision parameter `τ[i, j]`, and support `[0, Δtmax]`.
    
For Bayesian inference we assume a uniform normal-gamma prior for `μ` and `τ`:
    
    `τ[i, j] ~ Gamma(α0, β0)` for all i, j
    `μ[i, j] | σ[i, j] ~ Normal(μ[i, j], σ[i, j])` for all i, j
            
where `σ[i,, j] = 1 / sqrt(κ[i, j] * τ[i, j])` for all i, j.

# Arguments
- `μ::Float64`: the location of the logit-normal distribution.
- `τ::Float64`: the precision of the logit-normal distribution (i.e., `1 / σ^2`).
- `μμ::Float64`: the location of the normal prior for `μ` (default: 1.0).
- `κμ::Float64`: the precision multiplier of the normal prior for `μ` (default: 1.0).
- `α`: shape parameter of gamma prior for `τ` (default: 1.0).
- `β`: rate parameter of gamma prior for `τ` (default: 1.0).
- `Δtmax::Float64`: the upperbound of the process support, `[0, Δtmax]`.
"""
mutable struct LogitNormalImpulseResponse <: ImpulseResponse
    μ
    τ
    μμ
    κμ
    α0
    β0
    Δtmax
end

LogitNormalImpulseResponse(μ, τ, Δtmax) = LogitNormalImpulseResponse(μ, τ, 1.0, 1.0, 1.0, 1.0, Δtmax)

size(impulse::LogitNormalImpulseResponse) = size(impulse.μ)[1]

params(impulse::LogitNormalImpulseResponse) = [vec(impulse.μ); vec(impulse.τ)]

function params!(impulse::LogitNormalImpulseResponse, x)
    if length(x) != length(impulse.μ) + length(impulse.τ)
        error("Parameter vector length does not match model parameter length.")
    else
        nnodes = size(impulse)
        impulse.μ = reshape(x[1:length(impulse.μ)], nnodes, nnodes)
        impulse.τ = reshape(x[(length(impulse.μ)+1):end], nnodes, nnodes)
    end
end

function intensity(impulse::LogitNormalImpulseResponse)
    ir = Matrix{Function}(undef, size(impulse.μ))
    for idx in eachindex(impulse.μ)
        μ = impulse.μ[idx]
        τ = impulse.τ[idx]
        ir[idx] = t -> pdf(LogitNormal(μ, τ^(-1 / 2)), t ./ impulse.Δtmax)
    end
    return ir
end

function intensity(impulse::LogitNormalImpulseResponse, parentnode, childnode, Δt)
    μ = impulse.μ[parentnode, childnode]
    τ = impulse.τ[parentnode, childnode]
    return pdf(LogitNormal(μ, τ^(-1 / 2)), Δt ./ impulse.Δtmax)
end

function rand(impulse::LogitNormalImpulseResponse, n::Int64)
    μ = impulse.μ
    σ = 1 ./ sqrt.(impulse.τ)
    Δt = impulse.Δtmax
    ts = rand.(LogitNormal.(μ, σ), n) .* impulse.Δtmax
    return sort.(ts)
end

function rand(impulse::LogitNormalImpulseResponse, row::Int64, n::Int64)
    μ = impulse.μ[row, :]
    σ = 1 ./ sqrt.(impulse.τ[row, :])
    Δt = impulse.Δtmax
    ts = rand.(LogitNormal.(μ, σ), n) .* impulse.Δtmax
    return sort.(ts)
end

function rand(impulse::LogitNormalImpulseResponse, row::Int64, col::Int64, n::Int64)
    μ = impulse.μ[row, col]
    σ = 1 / sqrt(impulse.τ[row, col])
    Δt = impulse.Δtmax
    ts = rand(LogitNormal(μ, σ), n) .* impulse.Δtmax
    return sort(ts)
end

function resample!(impulse::LogitNormalImpulseResponse, data, parents)
    Mnm, Xnm, Vnm = sufficient_statistics(impulse, data, parents)
    αnm = impulse.α0 .+ Mnm ./ 2
    βnm = fillna!(Vnm ./ 2 .+ Mnm .* impulse.κμ ./ (Mnm .+ impulse.κμ) .* (Xnm .- impulse.μμ) .^ 2 ./ 2, impulse.β0)
    impulse.τ = rand.(Gamma.(αnm, 1 ./ βnm))
    κnm = impulse.κμ .+ Mnm
    μnm = fillna!((impulse.κμ .* impulse.μμ .+ Mnm .* Xnm) ./ (impulse.κμ .+ Mnm), impulse.μμ)
    σ = (1 ./ (κnm .* impulse.τ)) .^ (1 / 2)
    impulse.μ = rand.(Normal.(μnm, σ))
    return copy(impulse.μ), copy(impulse.τ)
end

function sufficient_statistics(impulse::LogitNormalImpulseResponse, data, parents)
    nnodes = size(impulse)
    Δtmax = impulse.Δtmax
    events, nodes, _ = data
    parentindices, parentnodes = parents
    Mnm = parent_counts(nodes, parentnodes, nnodes)
    Xnm = log_duration_sum(events, nodes, parentindices, nnodes, Δtmax) ./ Mnm
    any(isnan.(Xnm)) && @debug "`Xnm` contains `NaN`s. This typically means that there were no parent-child observations for some combination of nodes."
    Vnm = log_duration_variation(Xnm, events, nodes, parentindices, nnodes, Δtmax)
    return Mnm, Xnm, Vnm
end

log_duration(parent, child, Δtmax) = log((child - parent) / (Δtmax - (child - parent)))

function log_duration_sum(events, nodes, parents, nnodes, Δtmax)
    Xnm = zeros(nnodes, nnodes)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = nodes[parent]
            parent_event = events[parent]
            Xnm[parent_node, node] += log_duration(parent_event, event, Δtmax)
        end
    end
    return Xnm
end

function log_duration_variation(Xnm, events, nodes, parents, nnodes, Δtmax)
    Vnm = zeros(nnodes, nnodes)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = nodes[parent]
            parent_event = events[parent]
            Vnm[parent_node, node] += (log_duration(parent_event, event, Δtmax) - Xnm[parent_node, node])^2
        end
    end
    return Vnm
end

function logprior(impulse::LogitNormalImpulseResponse)
    lp = sum(log.(pdf.(Gamma.(impulse.α0, 1 / impulse.β0), impulse.τ)))
    σ = 1 ./ sqrt.(impulse.κμ .* impulse.τ)
    lp += sum(log.(pdf.(Normal.(impulse.μμ, σ), impulse.μ)))
    return lp
end


abstract type UnivariateImpulseResponse <: ImpulseResponse end


"""
    UnivariateLogitNormalImpulseResponse{T<:AbstractFloat}

A univariate logit-normal impulse response function.

This model is a building block for univariate Hawkes processes. Given a number of child events, it generates event times according to a stretched logit-normal distribution with location parameter `μ`, precision parameter `τ`, and support `[0, Δtmax]`.
    
For Bayesian inference we assume a uniform normal-gamma prior for `μ` and `τ`:
    
    `τ ~ Gamma(ατ, βτ)`
    `μ | σ ~ Normal(μ, σ)`
            
where `σ = 1 / sqrt(κ * τ)`.

# Arguments
- `μ::T`: location parameter of the logit-normal distribution
- `τ::T`: precision parameter of the logit-normal distribution (i.e., `1 / σ^2`)
- `μμ::T`: location of the normal-gamma prior (default: 0.0)
- `κμ::T`: precision multiplier of the normal-gamma prior (default: 1.0)
- `ατ::T`: shape parameter of the normal-gamma prior (default: 1.0)
- `βτ::T`: rate parameter of normal-gamma prior (default: 1.0)
- `Δtmax::Float64`: upperbound of the process support, `[0, Δtmax]`
"""
mutable struct UnivariateLogitNormalImpulseResponse{T<:AbstractFloat} <: UnivariateImpulseResponse
    μ::T
    τ::T
    μμ::T
    κμ::T
    ατ::T
    βτ::T
    Δtmax::T

    function UnivariateLogitNormalImpulseResponse{T}(μ, τ, μμ, κμ, ατ, βτ, Δtmax) where {T<:AbstractFloat}
        τ > 0.0 || throw(DomainError("UnivariateLogitNormalImpulseResponse: precision parameter τ should be positive"))
        κμ > 0.0 || throw(DomainError("UnivariateLogitNormalImpulseResponse: precision hyperparameter κμ should be positive"))
        ατ > 0.0 || throw(DomainError("UnivariateLogitNormalImpulseResponse: shape hyperparameter ατ should be positive"))
        βτ > 0.0 || throw(DomainError("UnivariateLogitNormalImpulseResponse: rate hyperparameter βτ should be positive"))
        Δtmax <= 0.0 && throw(DomainError("UnivariateLogitNormalImpulseResponse: process support Δtmax should be positive"))

        return new(μ, τ, μμ, κμ, ατ, βτ, Δtmax)
    end
end

function UnivariateLogitNormalImpulseResponse(μ::T, τ::T, μμ::T, κμ::T, ατ::T, βτ::T, Δtmax::T) where {T<:AbstractFloat}
    return UnivariateLogitNormalImpulseResponse{T}(μ, τ, μμ, κμ, ατ, βτ, Δtmax)
end

function UnivariateLogitNormalImpulseResponse(μ::T, τ::T, Δtmax::T) where {T<:AbstractFloat}
    return UnivariateLogitNormalImpulseResponse{T}(μ, τ, 0.0, 1.0, 1.0, 1.0, Δtmax)
end

nparams(model::UnivariateLogitNormalImpulseResponse) = 2
params(model::UnivariateLogitNormalImpulseResponse) = [model.μ, model.τ]

function params!(model::UnivariateLogitNormalImpulseResponse, θ)
    length(θ) == nparams(model) || throw(ArgumentError("params!: length of parameter vector θ should equal the number of model parameters"))
    μ, τ = θ
    τ > 0.0 || throw(DomainError("UnivariateLogitNormalImpulseResponse: precision parameter τ should be positive"))
    model.μ = μ
    model.τ = τ

    return params(model)
end

function intensity(model::UnivariateLogitNormalImpulseResponse)
    return t -> pdf(LogitNormal(model.μ, model.τ^(-1 / 2)), t ./ model.Δtmax)
end

function intensity(model::UnivariateLogitNormalImpulseResponse, Δt)
    return pdf(LogitNormal(model.μ, model.τ^(-1 / 2)), Δt ./ model.Δtmax)
end

function Base.rand(model::UnivariateLogitNormalImpulseResponse, n)
    σ = 1.0 / sqrt(model.τ)
    ts = rand(LogitNormal(model.μ, σ), n) .* model.Δtmax

    return sort(ts)
end

function resample!(model::UnivariateLogitNormalImpulseResponse, data, parents)
    n, xbar, vtot = sufficient_statistics(model, data, parents)
    if n == 0 # no child events => use priors
        @warn "resample!: no child events found"
        model.τ = rand(Gamma(model.ατ, 1 / model.βτ))
        σ = (1.0 / (model.κμ * model.τ))^(1 / 2)
        model.μ = rand(Normal(model.μμ, σ))
    else
        α = model.ατ + n / 2
        β = vtot / 2 + n * model.κμ / (n + model.κμ) * (xbar - model.μμ)^2 / 2 # TODO: why doesn't β update involve βτ?
        # β = isnan(β) ? model.βτ : β
        model.τ = rand(Gamma(α, 1.0 / β))
        κ = model.κμ + n
        μ = (model.κμ * model.μμ + n * xbar) / (model.κμ + n)
        # μ = isnan(μ) ? model.μμ : μ
        σ = (1.0 / (κ * model.τ))^(1 / 2)
        model.μ = rand(Normal(μ, σ))
    end

    return model.μ, model.τ
end

function sufficient_statistics(model::UnivariateLogitNormalImpulseResponse, data, parents)
    events, _ = data
    parentindices, _ = parents
    n = mapreduce(x -> x > 0, +, parentindices) # 0 => baseline event
    xbar = log_duration_sum(events, parentindices, model.Δtmax) / n
    vtot = log_duration_variation(xbar, events, parentindices, model.Δtmax)

    return n, xbar, vtot
end

function log_duration_sum(events, parentindices, Δtmax)
    xtot = 0
    for (event, parentindex) in zip(events, parentindices)
        if parentindex > 0 # baseline event => parentindex = 0
            parentevent = events[parentindex]
            xtot += log_duration(parentevent, event, Δtmax)
        end
    end

    return xtot
end

function log_duration_variation(xbar, events, parentindices, Δtmax)
    vtot = 0
    for (event, parentindex) in zip(events, parentindices)
        if parentindex > 0 # baseline event => parentindex = 0
            parentevent = events[parentindex]
            vtot += (log_duration(parentevent, event, Δtmax) - xbar)^2
        end
    end
    return vtot
end

function logprior(model::UnivariateLogitNormalImpulseResponse)
    ll = pdf(Gamma(model.ατ, 1 / model.βτ), model.τ)
    σ = 1 / sqrt(model.κμ * model.τ)
    ll += pdf(Normal(model.μμ, σ), model.μ)

    return ll
end



abstract type DiscreteImpulseResponse end

function basis(impulse::DiscreteImpulseResponse) end


"""
DiscreteGaussianImpulseResponse <: DiscreteImpulseResponse

If there are fewer basis functions than lags, then the means are evenly spaced between the endpoints of `[1, L]` such that the distance to the nearest mean or endpoint is the same everywhere. As a result, you can only have a means located at the endpoints if the number of basis functions is at least as great as the number of lags.
"""
mutable struct DiscreteGaussianImpulseResponse <: DiscreteImpulseResponse
    θ
    γ
    γv
    nlags
    dt
    ϕ
end

function DiscreteGaussianImpulseResponse(θ, nlags, dt=1.0)
    all(sum(θ, dims=3) .== 1.0) || error("Invalid discrete basis parameter.")
    γ = 1.0
    γv = ones(size(θ))
    impulse = DiscreteGaussianImpulseResponse(θ, γ, γv, nlags, dt, nothing)
    impulse.ϕ = intensity(impulse)
    return impulse
end

ndims(impulse::DiscreteGaussianImpulseResponse) = size(impulse.θ)[1]
nbasis(impulse::DiscreteGaussianImpulseResponse) = size(impulse.θ)[3]
nlags(impulse::DiscreteGaussianImpulseResponse) = impulse.nlags
nparams(impulse::DiscreteGaussianImpulseResponse) = prod(size(impulse.θ))

params(impulse::DiscreteGaussianImpulseResponse) = copy(vec(impulse.θ))

function params!(impulse::DiscreteGaussianImpulseResponse, x)
    if length(x) != length(impulse.θ)
        error("Parameter vector length does not match model parameter length.")
    elseif !all(isapprox.(sum(x, dims=3), 1.0))
        error("Basis weights must sum to 1.0 across each link.")
    else
        impulse.θ = reshape(x, size(impulse.θ))
    end
    return nothing
end

variational_params(impulse::DiscreteGaussianImpulseResponse) = copy(vec(impulse.γv))

function intensity(impulse::DiscreteGaussianImpulseResponse)
    """Calculate the `N x N x L` impulse-response of `p` for all parent-child-basis combinations."""
    nnodes = ndims(impulse)
    ϕ = hcat(basis(impulse)...)
    IR = zeros(nnodes, nnodes, impulse.nlags)
    for n = 1:nnodes
        IR[n, :, :] = impulse.θ[n, :, :] * transpose(ϕ)
    end
    return IR  # (N x N x B) x (B x L) = N x N x L
end

function basis(impulse::DiscreteGaussianImpulseResponse)
    """Construct basis functions with means evenly spaced on `[1, L]`. Returns a vector of `nbasis` length-`nlags` basis vectors.
    """
    L = nlags(impulse)
    B = nbasis(impulse)
    σ = L / (B - 1)
    if B < L
        μ = Array(LinRange(1, L, B + 2)[2:end-1]) # [x --- o --- o --- x]
    else
        μ = Array(LinRange(1, L, B)) # [o --- o --- o --- o]
    end
    lags = Array(1:L)
    ϕ = exp.(-1 / 2 * σ^-1 / 2 .* (lags .- transpose(μ)) .^ 2)
    return [ϕ[:, b] ./ (sum(ϕ[:, b]) .* impulse.dt) for b = 1:B]
end

function resample!(impulse::DiscreteGaussianImpulseResponse, parents)
    N = ndims(impulse)
    B = nbasis(impulse)
    γ = Array{Array{Int64,1},2}(undef, N, N)
    counts = reshape(sum(parents, dims=1), N, 1 + N * B)
    for parentchannel = 1:N
        for childchannel = 1:N
            start = 1 + (parentchannel - 1) * B + 1
            stop = start + B - 1
            γnm = counts[childchannel, start:stop]
            γ[parentchannel, childchannel] = impulse.γ .+ γnm
        end
    end
    θ = rand.(Dirichlet.(γ))
    impulse.θ = reshape(transpose(cat(θ..., dims=2)), (N, N, B))
    return copy(impulse.θ)
end

function update!(impulse::DiscreteGaussianImpulseResponse, data, parents)
    """Perform a variational inference update. `parents` is the `T x N x (1 + NB)` variational parameter for the auxillary parent variables."""
    N, T = size(data)
    _, _, B = size(impulse.θ)
    γ = zeros(N, N, B)
    for pidx = 1:N
        for cidx = 1:N
            for b = 1:B
                for t = 1:T
                    γ[pidx, cidx, b] += data[cidx, t] .* parents[t, cidx, 1+(pidx-1)*B+b]
                end
            end
        end
    end
    impulse.γv = impulse.γ .+ γ
    return copy(impulse.γv)
end

function variational_log_expectation(impulse::DiscreteGaussianImpulseResponse, pidx, cidx)
    return digamma.(impulse.γv[pidx, cidx, :]) .- digamma(sum(impulse.γv[pidx, cidx, :]))
end

function q(impulse::DiscreteGaussianImpulseResponse)
    idx = CartesianIndices(impulse.γv[:, :, 1])
    return reshape([Dirichlet(impulse.γv[idx[i], :]) for i in eachindex(impulse.γv[:, :, 1])], size(impulse.θ)[1:2])
end