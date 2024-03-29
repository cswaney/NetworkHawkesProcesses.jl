import SpecialFunctions.loggamma

abstract type DiscreteHawkesProcess <: HawkesProcess end

ndims(process::DiscreteHawkesProcess) = ndims(process.baseline)
nlags(process::DiscreteHawkesProcess) = nlags(process.impulses)
function isstable(p::DiscreteHawkesProcess) end

"""
    rand(p::DiscreteHawkesProcess, steps)

Sample a random sequence of events from a discrete Hawkes process.

# Arguments
- `T::Int64`: the number of time steps to sample.

# Returns
- `S::Array{Int64,2}`: an `N x T` array of event counts.
"""
function rand(process::DiscreteHawkesProcess, steps::Int64)
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
    augmented_loglikelihood(process::DiscreteHawkesProcess, parents, convolved)

Calculate the log-likelihood of the data given latent parent counts `parents`. The `parents` array contains all information required about events because summing across the last dimension gives the event count array.

# Arguments
- `parents::Array{Int64,3}`: `T x N x (1 + N * B)` array of parent counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.

# Returns
- `ll::Float64`: the log-likelihood of the parent event counts.
"""
function augmented_loglikelihood(process::DiscreteHawkesProcess, parents, convolved)
    T, N, B = size(convolved)
    λ0 = intensity(p.baseline, 1:T)
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
    loglikelihood(process::DiscreteHawkesProcess, data)
    loglikelihood(process::DiscreteHawkesProcess, data, convolved)

Calculate the log-likelihood of `data`. Providing `convolved` skips computation of the convolution step.

# Arguments
- `data::Array{Int64,2}`: `N x T` array of event counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.

# Returns
- `ll::Float64`: the log-likelihood of the event counts.
"""
function loglikelihood(process::DiscreteHawkesProcess, data)
    convolved = convolve(process, data)
    return loglikelihood(process, data, convolved)
end

function loglikelihood(process::DiscreteHawkesProcess, data, convolved)
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
    intensity(process::DiscreteHawkesProcess, convolved)

Calculate the intensity at time all times `t ∈ [1, 2, ..., T]` given pre-convolved event data.

# Arguments
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.

# Returns
- `λ::Array{Float64,2}`: a `T x N` array of intensities conditional on the convolved event counts.
"""
function intensity(process::DiscreteHawkesProcess, convolved)
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

intensity(process::DiscreteHawkesProcess, data::Matrix) = intensity(process, convolve(process, data))

"""
    convolve(process::DiscreteHawkesProcess, data)

Convolve `data` with a processes' basis functions.

The convolution of each basis function with `data` produces a `T x N` matrix. Stacking these matrices up across all `B` basis columns results in a `T x N x B` array.

# Arguments
- `data::Array{Int64,2}`: `N x T` array of event counts.

# Returns
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
"""
function convolve(process::DiscreteHawkesProcess, data)
    _, T = size(data)
    convolved = [conv(transpose(data), [0.0, u...])[1:T, :] for u in basis(process.impulses)]
    convolved = cat(convolved..., dims=3)
    return max.(convolved, 0.0)
end


"""
    DiscreteStandardHawkesProcess(baseline, impulses, weights, dt)

A standard discrete Hawkes processes with time step `dt`.

Equivalent to a discrete network Hawkes process with a fully-connected network.
"""
mutable struct DiscreteStandardHawkesProcess <: DiscreteHawkesProcess
    baseline::DiscreteBaseline
    impulses::DiscreteImpulseResponse
    weights::Weights
    dt::Float64
    function DiscreteStandardHawkesProcess(baseline, impulses, weights, dt)
        baseline.dt != dt || impulses.dt != dt && error("Baseline and impulse response time step must match process time step.")
        return new(baseline, impulses, weights, dt)
    end
end

isstable(p::DiscreteStandardHawkesProcess) = maximum(abs.(eigvals(p.weights.W))) < 1.0

function effective_weights(process::DiscreteStandardHawkesProcess)
    return vec(process.weights.W .* process.impulses.θ)
end

function params(process::DiscreteStandardHawkesProcess)
    """Return a copy of a processes' trainable parameters as a vector."""
    # return [params(process.baseline); params(process.impulses); params(process.weights)]
    return [params(process.baseline); effective_weights(process)]
end

function weights(process::DiscreteStandardHawkesProcess, x)
    nbaseline = nparams(process.baseline)
    ndim = ndims(process)
    nbase = nbasis(process.impulses)
    weights = reshape(x[nbaseline+1:end], ndim, ndim, nbase)
    connection_weights = reshape(sum(weights, dims=3), ndim, ndim)
    basis_weights = weights ./ connection_weights
    return connection_weights, basis_weights
end

function params!(process::DiscreteStandardHawkesProcess, x)
    """Set the trainable parameters of a process from a vector."""
    nbaseline = nparams(process.baseline)
    connection_weights, basis_weights = weights(process, x)
    params!(process.baseline, x[1:nbaseline])
    params!(process.impulses, basis_weights)
    params!(process.weights, connection_weights)
    params(process)
end

function variational_params(process::DiscreteStandardHawkesProcess)
    θ_baseline = variational_params(process.baseline)
    θ_impulses = variational_params(process.impulses)
    θ_weights = variational_params(process.weights)
    return [θ_baseline; θ_impulses; θ_weights]
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
    dη = zeros(size(process.impulses.θ)) # η := W .* θ
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
    resample!(process.weights, data, parents)
    resample!(process.impulses, parents)
    return params(process)
end

function update!(process::DiscreteStandardHawkesProcess, data, convolved)
    parents = update_parents(process, convolved)
    update!(process.baseline, data, parents)
    update!(process.weights, data, parents)
    update!(process.impulses, data, parents)
    return variational_params(process)
end

function impulse_response(process::DiscreteStandardHawkesProcess, parentnode, childnode, lag)
    return process.weights.W[parentnode, childnode] * process.impulses.ϕ[parentnode, childnode, lag]
end

function bump(process::DiscreteStandardHawkesProcess, parentchannel, childchannel, basis)
    w = process.weights.W[parentchannel, childchannel]
    θ = process.impulses.θ[parentchannel, childchannel, basis]
    return w * θ * process.dt
end


"""
    DiscreteNetworkHawkesProcess(baseline, impulses, weights, adjacency_matrix, network, dt)

A discrete network Hawkes process with time step `dt`.

Network Hawkes processes allow for partially-connected networks. The binary `adjacency_matrix` defines network connections generated by the probabilistic `network` model, where `adjacency_matrix[i, j]` represents a connection from node `i` to node `j`.
"""
struct DiscreteNetworkHawkesProcess <: DiscreteHawkesProcess
    baseline::DiscreteBaseline
    impulses::DiscreteImpulseResponse
    weights::Weights
    adjacency_matrix::Matrix
    network::Network
    dt::Float64
end

isstable(p::DiscreteNetworkHawkesProcess) = maximum(abs.(eigvals(p.adjacency_matrix .* p.weights.W))) < 1.0

function params(process::DiscreteNetworkHawkesProcess)
    """Return a copy of a processes' trainable parameters as a vector."""
    ϕ = params(process.network)
    λ0 = params(process.baseline)
    W = params(process.weights)
    θ = params(process.impulses)
    A = copy(vec(process.adjacency_matrix))
    return [ϕ; λ0; W; θ; A]
end

function resample!(process::DiscreteNetworkHawkesProcess, data, convolved)
    parents = resample_parents(process, data, convolved)
    resample!(process.baseline, parents)
    resample!(process.weights, data, parents) # TODO: remove `data` from arguments
    resample!(process.impulses, parents)
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
            w = process.weights.W[parentnode, cidx]
            a = parentnode == pidx ? value : process.adjacency_matrix[parentnode, cidx]
            for b = 1:B
                shat = convolved[t, parentnode, b]
                θ = process.impulses.θ[parentnode, cidx, b]
                λ += shat * a * w * θ * process.dt
            end
        end
        ll += log(pdf(Poisson(λ), data[cidx, t]))
    end
    return ll
end

function update_adjacency_matrix(p::DiscreteNetworkHawkesProcess)
    """Perform a variational inference update. For ρ ~ Beta(α, β), we have 1 - ρ ~ Beta(β, α), which implies that E[log (1 - ρ)] = digamma(β) - digamma(β + α)."""
    logodds = variational_log_expectation(p.network)
    logodds .+= p.weights.κ1 * log(p.weights.ν1) - loggamma(p.weights.κ1)
    logodds .+= loggamma.(p.weights.κv1) .- p.weights.κv1 .* log(p.weights.νv1)
    logodds .-= p.weights.κ0 * log(p.weights.ν0) - loggamma(p.weights.κ0)
    logodds .-= loggamma.(p.weights.κv0) .- p.weights.κv0 .* log(p.weights.νv0)
    logits = exp.(logodds)
    ρ = logits ./ (logits .+ 1)
    return ρ
end

function update!(process::DiscreteNetworkHawkesProcess, data, convolved)
    parents = update_parents(process, convolved)
    update!(process.baseline, parents)
    update!(process.weights, parents)
    update!(process.impulses, parents)
    update_adjacency_matrix(process)
    update_network!(process.network, process.ρv) # or process.weights.ρv? We are assuming a spike-and-slab type model—should we enforce that?
    return variational_params(process)
end

function impulse_response(process::DiscreteNetworkHawkesProcess, parentnode, childnode, lag)
    a = process.adjacency_matrix[parentnode, childnode]
    w = process.weights.W[parentnode, childnode]
    ϕ = process.impulses.ϕ[parentnode, childnode, lag]
    return a * w * ϕ
end

function bump(process::DiscreteNetworkHawkesProcess, parentchannel, childchannel, basis)
    a = process.adjacency_matrix[parentchannel, childchannel]
    w = process.weights.W[parentchannel, childchannel]
    θ = process.impulses.θ[parentchannel, childchannel, basis]
    return a * w * θ * process.dt
end