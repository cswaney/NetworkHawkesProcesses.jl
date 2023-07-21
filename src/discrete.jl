import SpecialFunctions.loggamma
import Base.rand
import Base.ndims

abstract type DiscreteHawkesProcess <: HawkesProcess end

ndims(process::DiscreteHawkesProcess) = ndims(process.baseline)
nlags(process::DiscreteHawkesProcess) = nlags(process.impulse_response)

function isstable(process::DiscreteHawkesProcess) end
function nparams(process::DiscreteHawkesProcess) end
function params(process::DiscreteHawkesProcess) end
function params!(process::DiscreteHawkesProcess, x) end
function rand(process::DiscreteHawkesProcess, steps::Integer) end
function convolve(process::DiscreteHawkesProcess, data) end

"""
    intensity(process::DiscreteHawkesProcess, data)

Calculate the intensity of a discrete-time Hawkes process at all time steps represented by `data`.
"""
function intensity(process::DiscreteHawkesProcess, data)
    return intensity(process, convolve(process, data))
end

"""
    loglikelihood(process::DiscreteHawkesProcess, data)

Calculate the log likelihood of a discrete-time Hawkes process given `data`.
"""
function loglikelihood(process::DiscreteHawkesProcess, data)
    convolved = convolve(process, data)
    return loglikelihood(process, data, convolved)
end


"""
    DiscreteUnivariateHawkesProcess(baseline, impulse_response, weight_model)

A discrete, univariate Hawkes process.

### Arguments
- `baseline::DiscreteUnivariateBaseline`
- `impulse_response::DiscreteUnivariateImpulseResponse`
- `weight_model::UnivariateWeightModel`
"""
mutable struct DiscreteUnivariateHawkesProcess <: DiscreteHawkesProcess
    baseline::DiscreteUnivariateBaseline
    impulse_response::DiscreteUnivariateImpulseResponse
    weight_model::UnivariateWeightModel
end

isstable(process::DiscreteUnivariateHawkesProcess) = abs(process.weight_model.w) < 1.0

function nparams(process::DiscreteUnivariateHawkesProcess)
    # return nparams(process.baseline) + nparams(process.impulse_response) + nparams(process.weight_model)
    return length(params(process))
end

function params(process::DiscreteUnivariateHawkesProcess)
    # return [params(process.baseline); params(process.impulse_response); params(process.weight_model)]
    return [params(process.baseline); effective_weights(process)]
end

function params!(process::DiscreteUnivariateHawkesProcess, x)
    # nparams(process) == length(θ) || throw(ArgumentError("length of parameter vector θ ($length(θ)) should equal the number of model parameters ($nparams(process))"))
    # params!(process.baseline, θ[1:nparams(process.baseline)])
    # params!(process.impulse_response, θ[nparams(process.baseline)+1:end-1])
    # params!(process.weight_model, θ[end])

    nbaseline = nparams(process.baseline)
    connection_weights, basis_weights = weights(process, x)
    params!(process.baseline, x[1:nbaseline])
    params!(process.impulse_response, basis_weights)
    params!(process.weight_model, connection_weights)

    return params(process)
end

function effective_weights(process::DiscreteUnivariateHawkesProcess)
    return process.weight_model.w .* process.impulse_response.θ
end

function weights(process::DiscreteUnivariateHawkesProcess, x)
    """x = params(process) = [params(process.baseline); effective_weights(process)]"""
    nbaseline = nparams(process.baseline)
    weights = x[(nbaseline + 1):end]
    connection_weights = sum(weights)
    basis_weights = weights ./ connection_weights

    return connection_weights, basis_weights
end

"""
    rand(p::DiscreteUnivariateHawkesProcess, steps::Integer)

Sample a random sequence of events from a univariate discrete Hawkes process.

### Arguments
- `steps::Integer`: the number of time steps to sample.

### Returns
- `events::Vector{Int64}`: a length `steps` array of event counts.
"""
function Base.rand(process::DiscreteUnivariateHawkesProcess, steps::Integer)
    L = nlags(process)
    events = rand(process.baseline, steps)
    for t = 1:steps-1
        for _ = 1:events[t]
            max_steps = min(L, steps - t)
            for s = 1:max_steps
                λ = impulse_response(process, s)
                events[t + s] += rand(Poisson(λ .* process.baseline.dt))
            end
        end
    end
    return events
end

function impulse_response(process::DiscreteUnivariateHawkesProcess, lag::Integer)
    lag > 0 || throw(DomainError(lag, "lag should be positive"))

    return process.weight_model.w * process.impulse_response.ħ[lag]
end

function convolve(process::DiscreteUnivariateHawkesProcess, data)
    T = length(data)
    convolved = [conv(data, [0.0, u...])[1:T, :] for u in basis(process.impulse_response)]
    convolved = cat(convolved...; dims=2)
    return max.(convolved, 0.0)
end

function loglikelihood(process::DiscreteUnivariateHawkesProcess, data, convolved)
    λ = intensity(process, convolved)
    T, _ = size(convolved)
    ll = 0.0
    for t = 1:T
        s = data[t]
        ll += log(pdf(Poisson(λ[t]), s))
    end
    return ll
end

function intensity(process::DiscreteUnivariateHawkesProcess, convolved)
    T, B = size(convolved)
    λ = intensity(process.baseline, 1:T)
    for t = 1:T
        for b = 1:B
            shat = convolved[t, b]
            λ[t] += shat * bump(process, b)
        end
    end
    return λ
end

function bump(process::DiscreteUnivariateHawkesProcess, basis)
    w = process.weight_model.w
    θ = process.impulse_response.θ[basis]
    return w * θ * process.baseline.dt
end

function mle!(process::DiscreteUnivariateHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing, max_increase_steps=3)

    convolved = convolve(process, data)

    function objective(x)
        params!(process, x)
        return regularize ? -loglikelihood(process, data, convolved) - logprior(process) : -loglikelihood(process, data, convolved)
    end

    minloss = Inf
    outer_iter = 0
    converged = false
    increase_steps = 0
    steps = 0

    function status_update(o)
        if o.iteration == 0
            if verbose
                println("* iteration (n=$outer_iter)")
            end
            outer_iter += 1
            minloss = Inf
        end
        if o.iteration % 1 == 0
            if verbose
                println(" > step: $(o.iteration), loss: $(o.value), elapsed: $(o.metadata["time"])")
            end
        end
        if abs(o.value - minloss) < f_abstol
            converged = true
            steps = o.iteration
            println("\n* Status: f_abstol convergence criteria reached!")
            println("    elapsed: $(o.metadata["time"])")
            println("    final loss: $(o.value)")
            println("    min. loss: $(minloss)")
            println("    outer iterations: $outer_iter")
            println("    inner iterations: $(o.iteration)\n")
            return true
        elseif o.value > minloss
            increase_steps += 1
            if increase_steps >= max_increase_steps
                converged = true
                steps = o.iteration
                println("\n* Status: loss increase criteria reached!")
                println("    elapsed: $(o.metadata["time"])")
                println("    final loss: $(o.value)")
                println("    min. loss: $(minloss)")
                println("    outer iterations: $outer_iter")
                println("    inner iterations: $(o.iteration)\n")
                return true
            end
        else
            minloss = o.value
            increase_steps = 0
        end
        return false
    end

    guess = guess === nothing ? _rand_init_(process) : guess
    lower = fill(1e-6, size(guess))
    upper = fill(1e1, size(guess))
    optimizer = Fminbox(optimizer())
    options = Optim.Options(callback=status_update)
    res = optimize(objective, lower, upper, guess, optimizer, options)
    return MaximumLikelihood(
        res.minimizer,
        -res.minimum,
        steps,
        res.time_run,
        converged ? "success" : "failure"
    )
end

_rand_init_(process::DiscreteUnivariateHawkesProcess) = rand(nparams(process))

function resample!(process::DiscreteUnivariateHawkesProcess, data, convolved)
    parents = resample_parents(process, data, convolved)
    resample!(process.baseline, parents)
    resample!(process.weight_model, data, parents)
    resample!(process.impulse_response, parents)

    return params(process)
end

function variational_params(process::DiscreteUnivariateHawkesProcess)
    θ_baseline = variational_params(process.baseline)
    θ_impulse_response = variational_params(process.impulse_response)
    θ_weights = variational_params(process.weight_model)
    return [θ_baseline; θ_impulse_response; θ_weights]
end

function update!(process::DiscreteUnivariateHawkesProcess, data, convolved)
    parents = update_parents(process, convolved)
    update!(process.baseline, data, parents)
    update!(process.weight_model, data, parents)
    update!(process.impulse_response, data, parents)
    return variational_params(process)
end



abstract type DiscreteMultivariateHawkesProcess <: DiscreteHawkesProcess end

function impulse_response(process::DiscreteMultivariateHawkesProcess, parentnode, childnode, lag) end

"""
    rand(p::DiscreteMultivariateHawkesProcess, steps::Integer)

Sample a random sequence of events from a multivariate discrete Hawkes process.

### Arguments
- `steps::Integer`: the number of time steps to sample.

### Returns
- `events::Matrix{Int64}`: an `N x T` array of event counts.
"""
function rand(process::DiscreteMultivariateHawkesProcess, steps::Integer)
    N = ndims(process)
    L = nlags(process)
    events = rand(process.baseline, steps)
    for t = 1:steps-1
        for parentnode = 1:N
            for _ = 1:events[parentnode, t]
                max_steps = min(L, steps - t)
                for childnode in 1:N
                    for s = 1:max_steps
                        λ = impulse_response(process, parentnode, childnode, s)
                        events[childnode, t+s] += rand(Poisson(λ .* process.dt))
                    end
                end
            end
        end
    end
    return events
end

"""
    convolve(process::DiscreteMultivariateHawkesProcess, data)

Convolve `data` with a processes' basis functions.

The convolution of each basis function with `data` produces a `T x N` matrix. Stacking these matrices up across all `B` basis columns results in a `T x N x B` array.

# Arguments
- `data::Array{Int64,2}`: `N x T` array of event counts.

# Returns
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
"""
function convolve(process::DiscreteMultivariateHawkesProcess, data)
    _, T = size(data)
    convolved = [conv(transpose(data), [0.0, u...])[1:T, :] for u in basis(process.impulse_response)]
    convolved = cat(convolved..., dims=3)
    return max.(convolved, 0.0)
end

"""
    intensity(process::DiscreteMultivariateHawkesProcess, convolved)

Calculate the intensity at time all times `t ∈ [1, 2, ..., T]` given pre-convolved event data.

# Arguments
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.

# Returns
- `λ::Array{Float64,2}`: a `T x N` array of intensities conditional on the convolved event counts.
"""
function intensity(process::DiscreteMultivariateHawkesProcess, convolved)
    T, N, B = size(convolved)
    λ = intensity(process.baseline, 1:T)
    for t = 1:T
        for childnode = 1:N
            for parentnode = 1:N
                for b = 1:B
                    shat = convolved[t, parentnode, b]
                    λ[t, childnode] += shat * bump(process, parentnode, childnode, b)
                end
            end
        end
    end
    return λ
end

"""
    loglikelihood(process::DiscreteMultivariteHawkesProcess, data, convolved)

Calculate the log-likelihood of `data`. Providing `convolved` skips computation of the convolution step.

# Arguments
- `data::Array{Int64,2}`: `N x T` array of event counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.

# Returns
- `ll::Float64`: the log-likelihood of the event counts.
"""
function loglikelihood(process::DiscreteMultivariateHawkesProcess, data, convolved)
    λ = intensity(process, convolved)
    T, N, _ = size(convolved)
    ll = 0.0
    for t = 1:T
        for n = 1:N
            s = data[n, t]
            ll += log(pdf(Poisson(λ[t, n]), s))
        end
    end
    return ll
end

"""
    augmented_loglikelihood(process::DiscreteMultivariateHawkesProcess, parents, convolved)

Calculate the log-likelihood of the data given latent parent counts `parents`. The `parents` array contains all information required because summing across the last dimension gives the event count array.

### Arguments
- `parents::Array{Int64,3}`: an `T x N x (1 + N * B)` array of parent counts.
- `convolved::Array{Float64,3}`: an `T x N x B` array of event counts convolved with basis functions.

### Returns
- `ll::Float64`: the log-likelihood of the parent event counts.
"""
function augmented_loglikelihood(process::DiscreteMultivariateHawkesProcess, parents, convolved)
    T, N, B = size(convolved)
    λ0 = intensity(process.baseline, 1:T)
    ll = 0.0
    for t = 1:T
        for childchannel = 1:N
            λ = λ0[t, childchannel]
            ll += log(pdf(Poisson(λ), parents[t, childchannel, 1]))
            for parentchannel = 1:N
                for b = 1:B
                    parentindex = 1 + (parentchannel - 1) * B + b
                    shat = convolved[t, parentchannel, b]
                    λ = shat * bump(process, parentchannel, childchannel, b)
                    ll += log(pdf(Poisson(λ), parents[t, childchannel, parentindex]))
                end
            end
        end
    end
    return ll
end


"""
    DiscreteStandardHawkesProcess(baseline, impulse_response, weight_model, dt)

A standard discrete-time Hawkes processes with time step `dt`. Equivalent to a discrete network Hawkes process with a fully-connected network.

### Arguments
- `baseline::DiscreteMultivariateBaseline`
- `impulse_response::DiscreteMultivariateImpulseResponse`
- `weight_model::MultivariateWeightModel`
- `dt::Float64`
"""
mutable struct DiscreteStandardHawkesProcess <: DiscreteMultivariateHawkesProcess
    baseline::DiscreteMultivariateBaseline
    impulse_response::DiscreteMultivariateImpulseResponse
    weight_model::MultivariateWeightModel
    dt::Float64

    function DiscreteStandardHawkesProcess(baseline, impulse_response, weight_model, dt)
        (baseline.dt != dt || impulse_response.dt != dt) && throw(ArgumentError("baseline and impulse response time step must match process time step."))

        return new(baseline, impulse_response, weight_model, dt)
    end
end

isstable(p::DiscreteStandardHawkesProcess) = maximum(abs.(eigvals(p.weight_model.W))) < 1.0

function nparams(process::DiscreteStandardHawkesProcess)
    return nparams(process.baseline) + nparams(process.impulse_response) + nparams(process.weight_model)
end

function params(process::DiscreteStandardHawkesProcess)
    return [params(process.baseline); effective_weights(process)]
end

function params!(process::DiscreteStandardHawkesProcess, x)
    """Set the trainable parameters of a process from a vector."""
    nbaseline = nparams(process.baseline)
    connection_weights, basis_weights = weights(process, x)
    params!(process.baseline, x[1:nbaseline])
    params!(process.impulse_response, basis_weights)
    params!(process.weight_model, connection_weights)
    
    return params(process)
end

function effective_weights(process::DiscreteStandardHawkesProcess)
    return vec(process.weight_model.W .* process.impulse_response.θ)
end

function weights(process::DiscreteStandardHawkesProcess, x)
    nbaseline = nparams(process.baseline)
    ndim = ndims(process)
    nbase = nbasis(process.impulse_response)
    weights = reshape(x[nbaseline+1:end], ndim, ndim, nbase)
    connection_weights = reshape(sum(weights, dims=3), ndim, ndim)
    basis_weights = weights ./ connection_weights
    return connection_weights, basis_weights
end

function impulse_response(process::DiscreteStandardHawkesProcess, parentnode, childnode, lag)
    return process.weight_model.W[parentnode, childnode] * process.impulse_response.ϕ[parentnode, childnode, lag]
end

function mle!(process::DiscreteStandardHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing, max_increase_steps=3)

    convolved = convolve(process, data)

    function objective(x)
        params!(process, x)
        return regularize ? -loglikelihood(process, data, convolved) - logprior(process) : -loglikelihood(process, data, convolved)
    end

    # function gradient!(g, x)
    #     params!(process, x)
    #     g .= regularize ? -d_loglikelihood(process, data, convolved) - d_logprior(process) : -d_loglikelihood(process, data, convolved)
    # end

    # OPTIMIZATION
    # function fg!(f, g, x)
    #     params!(process, x)
    #     calculate_intensity!(process, convolved)
    #     g .= regularize ? -d_loglikelihood(process, data) - d_logprior(process) : -d_loglikelihood(process, data)
    # end
    # optimize(Optim.only_fg!(fg!), ...) # instead of optimize(f, g!, ...)

    minloss = Inf
    outer_iter = 0
    converged = false
    increase_steps = 0
    steps = 0

    function status_update(o)
        if o.iteration == 0
            if verbose
                println("* iteration (n=$outer_iter)")
            end
            outer_iter += 1
            minloss = Inf
        end
        if o.iteration % 1 == 0
            if verbose
                println(" > step: $(o.iteration), loss: $(o.value), elapsed: $(o.metadata["time"])")
            end
        end
        if abs(o.value - minloss) < f_abstol
            converged = true
            steps = o.iteration
            println("\n* Status: f_abstol convergence criteria reached!")
            println("    elapsed: $(o.metadata["time"])")
            println("    final loss: $(o.value)")
            println("    min. loss: $(minloss)")
            println("    outer iterations: $outer_iter")
            println("    inner iterations: $(o.iteration)\n")
            return true
        elseif o.value > minloss
            increase_steps += 1
            if increase_steps >= max_increase_steps
                converged = true
                steps = o.iteration
                println("\n* Status: loss increase criteria reached!")
                println("    elapsed: $(o.metadata["time"])")
                println("    final loss: $(o.value)")
                println("    min. loss: $(minloss)")
                println("    outer iterations: $outer_iter")
                println("    inner iterations: $(o.iteration)\n")
                return true
            end
        else
            minloss = o.value
            increase_steps = 0
        end
        return false
    end

    guess = guess === nothing ? _rand_init_(process) : guess # or _default_init_(process, data)
    lower = fill(1e-6, size(guess))
    upper = fill(1e1, size(guess))
    optimizer = Fminbox(optimizer())
    options = Optim.Options(callback=status_update)
    res = optimize(objective, lower, upper, guess, optimizer, options)
    # res = optimize(objective, gradient!, lower, upper, guess, optimizer, options)
    return MaximumLikelihood(
        res.minimizer,
        -res.minimum,
        steps,
        res.time_run,
        converged ? "success" : "failure"
    )
end


function d_loglikelihood(process::DiscreteStandardHawkesProcess, data, convolved)
    λ = intensity(process, convolved)
    T, N, B = size(convolved)
    dλ0 = zeros(nparams(process.baseline))
    dη = zeros(size(process.impulse_response.θ)) # η := W .* θ
    for t = 1:T
        for m = 1:N
            dλ0[m] += data[m, t] / λ[t, m] - 1
            for n = 1:N
                for b = 1:B
                    dη[n, m, b] += (data[m, t] / λ[t, m] - 1) * convolved[t, m, b]
                end
            end
        end
    end
    return [dλ0; vec(dη)]
end

function logprior(process::DiscreteStandardHawkesProcess)
    lp = 0.0
    lp += sum(loggamma(process.baseline.λ, process.baseline.αλ, process.baseline.βλ))
    lp += sum(loggamma(process.W, process.κ1, process.ν1))
    lp += sum(logdirichlet(process.θ, process.γ))
    return lp
end

function d_logprior(process::DiscreteStandardHawkesProcess)
    dλ = d_loggamma(process.baseline.λ, process.baseline.αλ, process.baseline.βλ)
    dη = d_loggamma(process.W, process.κ1, process.ν1) ./ process.θ
    dη .+= d_logdirichlet(process.θ, process.γ) ./ process.W
    return [dλ; vec(dη)]
end

function loggamma(x, α, β)
    return (α - 1) * log.(x) - β * x
end

function d_loggamma(x, α, β)
    return (α - 1) ./ x .- β
end

function logdirichlet(x, γ)
    return sum((γ - 1) * log.(x))
end

function d_logdirichlet(x, γ)
    return (γ - 1) ./ x
end

function _rand_init_(process::DiscreteStandardHawkesProcess; max_attempts=100, scale=10)
    attempts = 0
    while attempts < max_attempts
        x = rand(length(params(process)))
        x[nparams(process.baseline)+1:end] ./= scale
        W, _ = weights(process, x)
        if maximum(abs.(eigvals(W))) < 1.0
            return x
        end
        attempts += 1
    end
    error("Random initialization reached max attempts.")
end

function resample!(process::DiscreteStandardHawkesProcess, data, convolved)
    parents = resample_parents(process, data, convolved)
    resample!(process.baseline, parents)
    resample!(process.weight_model, data, parents)
    resample!(process.impulse_response, parents)
    return params(process)
end

function variational_params(process::DiscreteStandardHawkesProcess)
    θ_baseline = variational_params(process.baseline)
    θ_impulse_response = variational_params(process.impulse_response)
    θ_weights = variational_params(process.weight_model)
    return [θ_baseline; θ_impulse_response; θ_weights]
end

function update!(process::DiscreteStandardHawkesProcess, data, convolved)
    parents = update_parents(process, convolved)
    update!(process.baseline, data, parents)
    update!(process.weight_model, data, parents)
    update!(process.impulse_response, data, parents)
    return variational_params(process)
end

function bump(process::DiscreteStandardHawkesProcess, parentchannel, childchannel, basis)
    w = process.weight_model.W[parentchannel, childchannel]
    θ = process.impulse_response.θ[parentchannel, childchannel, basis]
    return w * θ * process.dt
end


"""
    DiscreteNetworkHawkesProcess(baseline, impulse_response, weight_model, adjacency_matrix, network, dt)

A discrete-time network Hawkes process with time step `dt`.

Network Hawkes processes allow for sparse (i.e., partially-connected) networks. The binary `adjacency_matrix` defines network connections generated by the probabilistic `network` model, where `adjacency_matrix[i, j]` represents a connection from node `i` to node `j`.

### Arguments
- `baseline::DiscreteMultivariateBaseline`
- `impulse_response::DiscreteMultivariateImpulseResponse`
- `weight_model::MultivariateWeightModel`
- `adjacency_matrix::Matrix{Bool}`
- `network::Network`
- `dt::Float64`
"""
struct DiscreteNetworkHawkesProcess <: DiscreteMultivariateHawkesProcess
    baseline::DiscreteMultivariateBaseline
    impulse_response::DiscreteMultivariateImpulseResponse
    weight_model::MultivariateWeightModel
    adjacency_matrix::Matrix
    network::Network
    dt::Float64

    function DiscreteNetworkHawkesProcess(baseline, impulse_response, weight_model, adjacency_matrix, network, dt)
        (baseline.dt != dt || impulse_response.dt != dt) && throw(ArgumentError("baseline and impulse response time step must match process time step."))

        return new(baseline, impulse_response, weight_model, adjacency_matrix, network, dt)
    end
end

isstable(p::DiscreteNetworkHawkesProcess) = maximum(abs.(eigvals(p.adjacency_matrix .* p.weight_model.W))) < 1.0

function nparams(process::DiscreteNetworkHawkesProcess)
    return (nparams(process.baseline) + nparams(process.impulse_response)
        + nparams(process.weight_model) + length(process.adjacency_matrix))
end

function params(process::DiscreteNetworkHawkesProcess)
    """Return a copy of a processes' trainable parameters as a vector."""
    return [params(process.baseline); effective_weights(process)]
end

function params!(process::DiscreteNetworkHawkesProcess)
    throw(ErrorException("Not implemented"))
end

function impulse_response(process::DiscreteNetworkHawkesProcess, parentnode, childnode, lag)
    a = process.adjacency_matrix[parentnode, childnode]
    w = process.weight_model.W[parentnode, childnode]
    ϕ = process.impulse_response.ϕ[parentnode, childnode, lag]
    return a * w * ϕ
end

function effective_weights(process::DiscreteNetworkHawkesProcess)
    return vec(process.adjacency_matrix .* process.weight_model.W .* process.impulse_response.θ)
end

function resample!(process::DiscreteNetworkHawkesProcess, data, convolved)
    parents = resample_parents(process, data, convolved)
    resample!(process.baseline, parents)
    resample!(process.weight_model, data, parents) # TODO: remove `data` from arguments
    resample!(process.impulse_response, parents)
    resample_adjacency_matrix!(process, data, convolved)
    resample!(process.network, process.adjacency_matrix)
    return params(process)
end

function resample_adjacency_matrix!(process::DiscreteNetworkHawkesProcess, data, convolved)
    """
        sample_adjacency_matrix!(p::DiscreteHawkesProcess, events, T)

    Draw a sample form the conditional posterior of the adjacency matrix and update its current value.

    # Arguments
    - `events::Array{Int64,2}`: `N x T` array of event counts.
    - `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
    """
    if Threads.nthreads() > 1
        @debug "using multi-threaded adjacency matrix sampler"
        Threads.@threads for childnode = 1:ndims(process)
            resample_column!(process, data, convolved, childnode)
        end
    else
        for childnode = 1:ndims(process)
            resample_column!(process, data, convolved, childnode)
        end
    end
    return copy(process.adjacency_matrix)
end

function resample_column!(process::DiscreteNetworkHawkesProcess, data, convolved, childnode)
    ρ = link_probability(process.network)
    for parentnode = 1:ndims(process)
        ll0 = conditional_loglikelihood(process, data, convolved, 0, parentnode, childnode) # P(A[pidx, cidx] = 0)
        ll0 += log(1 - ρ[parentnode, childnode])
        ll1 = conditional_loglikelihood(process, data, convolved, 1, parentnode, childnode) # P(A[pidx, cidx] = 1)
        ll1 += log(ρ[parentnode, childnode])
        lZ = logsumexp(ll0, ll1)
        process.adjacency_matrix[parentnode, childnode] = rand(Bernoulli(exp(ll1 - lZ)))
    end
    return process.adjacency_matrix[:, childnode]
end

function conditional_loglikelihood(process::DiscreteNetworkHawkesProcess, data, convolved, value, pidx, cidx)
    ll = 0.0
    T, N, B = size(convolved)
    λ0 = intensity(process.baseline, 1:T)
    for t = 1:T
        λ = λ0[t, cidx]
        for parentnode = 1:N
            w = process.weight_model.W[parentnode, cidx]
            a = parentnode == pidx ? value : process.adjacency_matrix[parentnode, cidx]
            for b = 1:B
                shat = convolved[t, parentnode, b]
                θ = process.impulse_response.θ[parentnode, cidx, b]
                λ += shat * a * w * θ * process.dt
            end
        end
        ll += log(pdf(Poisson(λ), data[cidx, t]))
    end
    return ll
end

function variational_params(process::DiscreteNetworkHawkesProcess)
    θ_baseline = variational_params(process.baseline)
    θ_impulse_response = variational_params(process.impulse_response)
    θ_weights = variational_params(process.weight_model)
    θ_network = variational_params(process.network)
    return [θ_baseline; θ_impulse_response; θ_weights; θ_network]
end

function update!(process::DiscreteNetworkHawkesProcess, data, convolved)
    parents = update_parents(process, convolved)
    update!(process.baseline, data, parents)
    update!(process.weight_model, data, parents)
    update!(process.impulse_response, data, parents)
    update_adjacency_matrix(process)
    update!(process.network, process.weight_model.ρv)
    return variational_params(process)
end

function update_adjacency_matrix(p::DiscreteNetworkHawkesProcess)
    """Perform a variational inference update. For ρ ~ Beta(α, β), we have 1 - ρ ~ Beta(β, α), which implies that E[log (1 - ρ)] = digamma(β) - digamma(β + α)."""
    logodds = variational_log_expectation(p.network)
    logodds += p.weight_model.κ1 * log(p.weight_model.ν1) - loggamma(p.weight_model.κ1)
    logodds = logodds .+ loggamma.(p.weight_model.κv1) .- p.weight_model.κv1 .* log.(p.weight_model.νv1)
    logodds .-= p.weight_model.κ0 * log(p.weight_model.ν0) - loggamma(p.weight_model.κ0)
    logodds .-= loggamma.(p.weight_model.κv0) .- p.weight_model.κv0 .* log.(p.weight_model.νv0)
    logits = exp.(logodds)
    p.weight_model.ρv = logits ./ (logits .+ 1)
    return copy(p.weight_model.ρv)
end

function bump(process::DiscreteNetworkHawkesProcess, parentchannel, childchannel, basis)
    a = process.adjacency_matrix[parentchannel, childchannel]
    w = process.weight_model.W[parentchannel, childchannel]
    θ = process.impulse_response.θ[parentchannel, childchannel, basis]
    return a * w * θ * process.dt
end
