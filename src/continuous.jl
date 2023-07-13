abstract type ContinuousHawkesProcess <: HawkesProcess end

ndims(process::ContinuousHawkesProcess) = ndims(process.baseline)

"""
    rand(process::ContinuousHawkesProcess, duration)

Sample a random sequence of events from a continuous Hawkes process.

# Arguments
- `duration::Float64`: the sample duration.

# Returns
- `data::Tuple{Vector{Float64},Vector{Int64},Float64}`: a tuple containing a vector of events, a vector of nodes associated with each event, and the duration of the data sample.
"""
function rand(process::ContinuousHawkesProcess, duration::AbstractFloat)
    nnodes = ndims(process)
    events = Array{Array{Float64,1},1}(undef, nnodes)
    for parentnode = 1:nnodes
        events[parentnode] = Array{Float64,1}()
    end
    for node = 1:nnodes
        baseline_events = rand(process.baseline, node, duration)
        append!(events[node], baseline_events)
        for event in baseline_events
            _generate_children!_(events, event, node, process, duration)
        end
    end
    times = Array{Float64,1}()
    nodes = Array{Int64,1}()
    for (idx, col) in enumerate(events)
        append!(times, col)
        append!(nodes, idx * ones(length(col)))
    end
    idx = sortperm(times)
    return times[idx], nodes[idx], duration
end

function truncate(events, t0)
    """Truncate a list of sorted events at `t0`."""
    isempty(events) && return events
    idx = length(events)
    while events[idx] > t0
        idx -= 1
        idx == 0 && return typeof(events)()
    end
    return events[1:idx]
end

"""
    loglikelihood(process::ContinuousHawkesProcess, data)

Calculate the log-likelihood of `data`.

# Arguments
- `data::Tuple{Vector{Float64},Vector{Int64},Float64}`: a tuple containing a vector of events, a vector of nodes associated with each event, and the duration of the data sample.
- `recursive::Bool`: use recursive formulation, if possible.

# Returns
- `ll::Float64`: the log-likelihood of the data.
"""
function loglikelihood(process::ContinuousHawkesProcess, data) end

"""
    intensity(process::ContinuousHawkesProcess, data, times)

Calculate the intensity of `process` at `times` given `data`.

# Arguments
- `data::Tuple{Vector{Float64},Vector{Int64},Float64}`: a tuple containing a vector of events, a vector of nodes associated with each event, and the duration of the data sample.
- `times::Vector{Float64}`: a vector times where the process intensity will be computed.

# Returns
- `λ::Vector{Float64}`: a `len(times)` array of intensities conditional on `data` and `process`.
"""
function intensity(process::ContinuousHawkesProcess, data, times::Vector{Float64})
    λs = zeros(length(times), ndims(process))
    for (i, t0) in enumerate(times)
        λs[i, :] = intensity(process, data, t0)
    end
    return λs
end

function intensity(process::ContinuousHawkesProcess, data, time::Float64)
    events, nodes, _ = data
    nnodes = ndims(process)
    idx = time - process.impulses.Δtmax .< events .< time
    λ = zeros(nnodes)
    for childnode = 1:nnodes
        for (parenttime, parentnode) in zip(events[idx], nodes[idx])
            Δt = time - parenttime
            λ[childnode] += impulse_response(process, parentnode, childnode, Δt)
        end
    end
    return intensity(process.baseline, time) .+ λ
end

function impulse_response(process::ContinuousHawkesProcess, parentnode, childnode, Δt) end


"""
    ContinuousStandardHawkesProcess(baseline, impulses, weights)

A continuous standard Hawkes processes.

Equivalent to a continuous network Hawkes process with a fully-connected network.
"""
struct ContinuousStandardHawkesProcess <: ContinuousHawkesProcess
    baseline::Baseline
    impulses::ImpulseResponse
    weights::Weights
end

isstable(p::ContinuousStandardHawkesProcess) = maximum(abs.(eigvals(p.weights.W))) < 1.0

function params(process::ContinuousStandardHawkesProcess)
    """Return a copy of a processes' trainable parameters as a vector."""
    return [params(process.baseline); params(process.impulses); params(process.weights)]
end

function params!(process::ContinuousStandardHawkesProcess, x)
    """Set the trainable parameters of a process from a vector."""
    nbaseline = length(params(process.baseline))
    nweights = length(params(process.weights))
    nimpulses = length(params(process.impulses))
    params!(process.baseline, x[1:nbaseline])
    params!(process.impulses, x[(nbaseline+1):(nbaseline+nimpulses)])
    params!(process.weights, x[(nbaseline+nimpulses+1):(nbaseline+nimpulses+nweights)])
end

function _generate_children!_(events, parentevent, parentnode, process::ContinuousStandardHawkesProcess, duration)
    t0 = parentevent
    nnodes = ndims(process)
    for childnode = 1:nnodes
        nchildren = rand(process.weights, parentnode, childnode)
        childevents = t0 .+ rand(process.impulses, parentnode, childnode, nchildren)
        append!(events[childnode], truncate(childevents, duration))
        for event in childevents
            _generate_children!_(events, event, childnode, process, duration)
        end
    end
end

function mle!(process::ContinuousStandardHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing)

    function objective(x)
        params!(process, x)
        return regularize ? -loglikelihood(process, data) - logprior(process) : -loglikelihood(process, data)
    end

    minloss = Inf
    outer_iter = 0
    converged = false
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
        else
            minloss = o.value
        end
        return false
    end

    guess = guess === nothing ? _rand_init_(process) : guess # or _default_init_(process, data)
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

_rand_init_(process::ContinuousStandardHawkesProcess) = rand(length(params(process)))

function resample!(process::ContinuousStandardHawkesProcess, data)
    parents = resample_parents(process, data)
    resample!(process.baseline, data, parents)
    resample!(process.weights, data, parents)
    resample!(process.impulses, data, parents)
    return params(process)
end

function loglikelihood(process::ContinuousStandardHawkesProcess, data; recursive=true)

    if typeof(process.impulses) == ExponentialImpulseResponse && recursive
        return recursive_loglikelihood(process, data)
    end

    events, nodes, duration = data
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    for parentnode in nodes
        ll -= sum(process.weights.W[parentnode, :])
    end
    if Threads.nthreads() > 1
        ll = Threads.Atomic{Float64}(ll)
        @debug "using multi-threaded loglikelihood calculation"
        Threads.@threads for childindex = 1:length(events)
            childtime = events[childindex]
            childnode = nodes[childindex]
            λtot = total_intensity(process, events, nodes, childindex, childtime, childnode)
            Threads.atomic_add!(ll, log(λtot))
        end
        return ll.value
    else
        for (childindex, (childtime, childnode)) in enumerate(zip(events, nodes))
            λtot = total_intensity(process, events, nodes, childindex, childtime, childnode)
            ll += log(λtot)
        end
        return ll
    end
end

function recursive_loglikelihood(process::ContinuousStandardHawkesProcess, data)
    events, nodes, duration = data
    nnodes = ndims(process)
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    for parentnode in nodes
        ll -= sum(process.weights.W[parentnode, :]) # approximate (exact requires `cdf`)
    end
    partialsums = zeros(nnodes, nnodes)
    parenttimes = zeros(nnodes)
    for (childtime, childnode) in zip(events, nodes)
        λtot = intensity(process.baseline, childnode, childtime)
        if parenttimes[childnode] > 0.0
            Δt = childtime - parenttimes[childnode]
            next = vec(exp.(-Δt .* process.impulses.θ[childnode, :]))
            partialsums[childnode, :] .= next .* (1 .+ partialsums[childnode, :])
        end
        for parentnode = 1:nnodes
            parenttime = parenttimes[parentnode]
            if parenttime > 0.0
                if parentnode == childnode
                    r = partialsums[parentnode, childnode]
                else
                    Δt = childtime - parenttime
                    next = exp(-Δt * process.impulses.θ[parentnode, childnode])
                    r = next * (1 + partialsums[parentnode, childnode])
                end
                w = process.weights.W[parentnode, childnode]
                λtot += w * process.impulses.θ[parentnode, childnode] * r
            end
        end
        ll += log(λtot)
        parenttimes[childnode] = childtime
    end
    return ll
end

function logprior(process::ContinuousStandardHawkesProcess)
    """Compute the log prior probability of the trainable model parameters."""
    lp = logprior(process.baseline)
    lp += logprior(process.weights)
    lp += logprior(process.impulses)
    return lp
end

function total_intensity(process::ContinuousStandardHawkesProcess, events, nodes, index, time, node)
    """Calculate the total intensity on `node` for the `index`-th event occurring at `time`."""
    λtot = intensity(process.baseline, node, time)
    index == 1 && return λtot
    parentindex = index - 1
    while events[parentindex] > time - process.impulses.Δtmax
        parenttime = events[parentindex]
        parentnode = nodes[parentindex]
        Δt = time - parenttime
        λtot += impulse_response(process, parentnode, node, Δt)
        parentindex -= 1
        parentindex == 0 && break
    end
    return λtot
end

function impulse_response(process::ContinuousStandardHawkesProcess, parentnode, childnode, Δt)
    w = process.weights.W[parentnode, childnode]
    return w * intensity(process.impulses, parentnode, childnode, Δt)
end


"""
    ContinuousNetworkHawkesProcess(baseline, impulses, weights, adjacency_matrix, network)

A continuous network Hawkes process.

Network Hawkes processes allow for partially-connected networks. The binary `adjacency_matrix` defines network connections generated by the probabilistic `network` model, where `adjacency_matrix[i, j]` represents a connection from node `i` to node `j`.
"""
struct ContinuousNetworkHawkesProcess <: ContinuousHawkesProcess
    baseline::Baseline
    impulses::ImpulseResponse
    weights::Weights
    adjacency_matrix::Matrix
    network::Network
end

isstable(p::ContinuousNetworkHawkesProcess) = maximum(abs.(eigvals(p.adjacency_matrix .* p.weights.W))) < 1.0

function params(process::ContinuousNetworkHawkesProcess)
    """Return a copy of a processes' trainable parameters as a vector."""
    ϕ = params(process.network)
    λ0 = params(process.baseline)
    W = params(process.weights)
    θ = params(process.impulses)
    A = copy(vec(process.adjacency_matrix))
    return [ϕ; λ0; W; θ; A]
end

function _generate_children!_(events, parentevent, parentnode, process::ContinuousNetworkHawkesProcess, duration)
    t0 = parentevent
    nnodes = ndims(process)
    for childnode = 1:nnodes
        if process.adjacency_matrix[parentnode, childnode] == 1
            nchildren = rand(process.weights, parentnode, childnode)
            childevents = t0 .+ rand(process.impulses, parentnode, childnode, nchildren)
            append!(events[childnode], truncate(childevents, duration))
            for event in childevents
                _generate_children!_(events, event, childnode, process, duration)
            end
        end
    end
end

function resample!(process::ContinuousNetworkHawkesProcess, data)
    parents = resample_parents(process, data)
    resample!(process.baseline, data, parents)
    resample!(process.weights, data, parents)
    resample!(process.impulses, data, parents)
    resample_adjacency_matrix!(process, data)
    resample!(process.network, process.adjacency_matrix)
    return params(process)
end

function loglikelihood(process::ContinuousNetworkHawkesProcess, data; recursive=true)
    if typeof(process.impulses) == ExponentialImpulseResponse && recursive
        return recursive_loglikelihood(process, data)
    end

    events, nodes, duration = data
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    for parentnode in nodes
        ll -= sum(process.adjacency_matrix[parentnode, :] .* process.weights.W[parentnode, :])
        # ll -= sum(effective_weights(process, parentnode))
    end
    if Threads.nthreads() > 1
        ll = Threads.Atomic{Float64}(ll)
        @debug "using multi-threaded loglikelihood calculation"
        Threads.@threads for childindex = 1:length(events)
            childtime = events[childindex]
            childnode = nodes[childindex]
            λtot = total_intensity(process, events, nodes, childindex, childtime, childnode)
            Threads.atomic_add!(ll, log(λtot))
        end
        return ll.value
    else
        for (childindex, (childtime, childnode)) in enumerate(zip(events, nodes))
            λtot = total_intensity(process, events, nodes, childindex, childtime, childnode)
            ll += log(λtot)
        end
        return ll
    end
end

function total_intensity(process::ContinuousNetworkHawkesProcess, events, nodes, index, time, node)
    """Calculate the total intensity on `node` for the `index`-th event occurring at `time`."""
    λtot = intensity(process.baseline, node, time)
    index == 1 && return λtot
    parentindex = index - 1
    while events[parentindex] > time - process.impulses.Δtmax
        parenttime = events[parentindex]
        parentnode = nodes[parentindex]
        Δt = time - parenttime
        λtot += impulse_response(process, parentnode, node, Δt)
        parentindex -= 1
        parentindex == 0 && break
    end
    return λtot
end

function recursive_loglikelihood(process::ContinuousNetworkHawkesProcess, data)
    events, nodes, duration = data
    nnodes = ndims(process)
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    for parentnode in nodes
        ll -= sum(process.weights.W[parentnode, :]) # approximate (exact requires `cdf`)
    end
    partialsums = zeros(nnodes, nnodes)
    parenttimes = zeros(nnodes)
    for (childtime, childnode) in zip(events, nodes)
        λtot = intensity(process.baseline, childnode, childtime)
        if parenttimes[childnode] > 0.0
            Δt = childtime - parenttimes[childnode]
            next = vec(exp.(-Δt .* process.impulses.θ[childnode, :]))
            partialsums[childnode, :] .= next .* (1 .+ partialsums[childnode, :])
        end
        for parentnode = 1:nnodes
            parenttime = parenttimes[parentnode]
            if parenttime > 0.0
                if parentnode == childnode
                    r = partialsums[parentnode, childnode]
                else
                    Δt = childtime - parenttime
                    next = exp(-Δt * process.impulses.θ[parentnode, childnode])
                    r = next * (1 + partialsums[parentnode, childnode])
                end
                weff = effective_weight(process, parentnode, childnode)
                λtot += weff * process.impulses.θ[parentnode, childnode] * r
            end
        end
        ll += log(λtot)
        parenttimes[childnode] = childtime
    end
    return ll
end

function resample_adjacency_matrix!(process::ContinuousNetworkHawkesProcess, data)
    """
        Sample the conditional posterior distribution of the adjacency matrix, `A`.

        # Implementation Note
        Each column of `A` is sampled *jointly* and independently from all other columns. The calculation uses multiple threads, if available.

        # Arguments
        - `events::Array{Array{Float64,1},1}`: an array of event time arrays in `[0, T]`.
        - `nodes::Array{Array{Int64,1},1}`: an array of node arrays in `{1, ..., N}`.
        - `durations::Array{Float64,1}`: an array of observation lengths.
    """
    _, nodes, _ = data
    nnodes = ndims(process)
    parentcounts = node_counts(nodes, nnodes)
    linkprob = link_probability(process.network)
    if Threads.nthreads() > 1
        @debug "using multi-threaded adjacency matrix sampler"
        Threads.@threads for childnode = 1:nnodes
            resample_column!(process, childnode, data, parentcounts, linkprob)
        end
    else
        for childnode = 1:nnodes
            resample_column!(process, childnode, data, parentcounts, linkprob)
        end
    end
end

function resample_column!(process::ContinuousNetworkHawkesProcess, node, data, nodecounts, linkprob)
    """Resample the `node`-th column of the adjacency matrix."""
    events, nodes, duration = data
    for parentnode = 1:ndims(process)
        process.adjacency_matrix[parentnode, node] = 0.0
        ll0 = -integrated_intensity(process, node, nodecounts, duration)
        ll0 += sum_log_intensity(process, node, events, nodes)
        ll0 += log(1.0 - linkprob[parentnode, node])
        process.adjacency_matrix[parentnode, node] = 1.0
        ll1 = -integrated_intensity(process, node, nodecounts, duration)
        ll1 += sum_log_intensity(process, node, events, nodes)
        ll1 += log(linkprob[parentnode, node])
        Z = logsumexp(ll0, ll1)
        process.adjacency_matrix[parentnode, node] = rand(Bernoulli(exp(ll1 - Z)))
    end
end

function integrated_intensity(process::ContinuousNetworkHawkesProcess, node, nodecounts, duration)
    """Calculate integrated intensity on `node`. `nodecounts` holds the number of events on each node."""
    I = integrated_intensity(process.baseline, node, duration)
    for parentnode = 1:ndims(process)
        a = process.adjacency_matrix[parentnode, node]
        w = process.weights.W[parentnode, node]
        I += a * w * nodecounts[parentnode]
    end
    return I
end

function sum_log_intensity(process::ContinuousNetworkHawkesProcess, node, events, nodes)
    """Sum the log total intensity over all events occurring on `node`."""
    S = 0.0
    for (index, (childevent, childnode)) in enumerate(zip(events, nodes))
        childnode != node && continue
        λtot = intensity(process.baseline, childnode, childevent)
        index == 1 && continue
        parentindex = index - 1
        while events[parentindex] > childevent - process.impulses.Δtmax
            parentevent = events[parentindex]
            parentnode = nodes[parentindex]
            Δt = childevent - parentevent
            λtot += impulse_response(process, parentnode, childnode, Δt)
            parentindex -= 1
            parentindex == 0 && break
        end
        S += log(λtot)
    end
    return S
end

function impulse_response(process::ContinuousNetworkHawkesProcess, parentnode, childnode, Δt)
    a = process.adjacency_matrix[parentnode, childnode]
    w = process.weights.W[parentnode, childnode]
    return a * w * intensity(process.impulses, parentnode, childnode, Δt)
end

function effective_weight(process::ContinuousNetworkHawkesProcess, parentnode, childnode)
    a = process.adjacency_matrix[parentnode, childnode]
    w = process.weights.W[parentnode, childnode]
    return a * w
end

function effective_weights(process::ContinuousNetworkHawkesProcess, parentnode)
    a = process.adjacency_matrix[parentnode, :]
    w = process.weights.W[parentnode, :]
    return a .* w
end


"""
    ContinuousUnivariateHawkesProcess

A univariate Hawkes processes composed of baseline and impulse response models. 
"""
mutable struct ContinuousUnivariateHawkesProcess <: ContinuousHawkesProcess
    baseline::UnivariateBaseline
    impulse_response::UnivariateImpulseResponse
    weight::UnivariateWeightModel
end

isstable(process::ContinuousUnivariateHawkesProcess) = abs(process.weight.w) < 1.0

function params(process::ContinuousUnivariateHawkesProcess)
    return [params(process.baseline); params(process.impulse_response); process.weight.w]
end

nparams(process::ContinuousUnivariateHawkesProcess) = length(params(process))

function params!(process::ContinuousUnivariateHawkesProcess, θ)
    nparams(process) == length(θ) || throw(ArgumentError("params!: length of parameter vector θ ($length(θ)) should equal the number of model parameters ($nparams(process))"))
    params!(process.baseline, θ[1:nparams(process.baseline)])
    params!(process.impulse_response, θ[nparams(process.baseline) + 1:end-1])
    params!(process.weight, θ[end])

    return params(process)
end

function Base.rand(process::ContinuousUnivariateHawkesProcess, duration::AbstractFloat)
    events = Float64[]
    baseline_events, _ = rand(process.baseline, duration)
    append!(events, baseline_events)
    for event in baseline_events
        _generate_children!_(events, event, process, duration)
    end

    return sort(events), duration
end

function _generate_children!_(events, parentevent, process::ContinuousUnivariateHawkesProcess, duration)
    t0 = parentevent
    nchildren = rand(Poisson(process.weight.w))
    childevents = t0 .+ rand(process.impulse_response, nchildren)
    append!(events, truncate(childevents, duration))
    for event in childevents
        _generate_children!_(events, event, process, duration)
    end
end

function loglikelihood(process::ContinuousUnivariateHawkesProcess, data; recursive=true)
    
    if (process.impulse_response isa UnivariateExponentialImpulseResponse) && recursive
        return recursive_loglikelihood(process, data)
    end

    events, duration = data
    
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    ll -= process.weight.w * length(events) # integrated impulse responses

    if Threads.nthreads() > 1
        ll = Threads.Atomic{Float64}(ll)
        @debug "using multi-threaded loglikelihood calculation"
        Threads.@threads for childindex = 1:length(events)
            childtime = events[childindex]
            λtot = total_intensity(process, events, childindex, childtime)
            Threads.atomic_add!(ll, log(λtot))
        end
        
        return ll.value
    else
        for (childindex, childtime) in enumerate(events)
            λtot = total_intensity(process, events, childindex, childtime)
            ll += log(λtot)
        end
        
        return ll
    end
end

function recursive_loglikelihood(process::ContinuousUnivariateHawkesProcess, data)
    # TODO: this method only works if process.impulse_response isa UnivariateExponentialImpulseResponse
    events, duration = data
    
    ll = 0.0
    ll -= sum(integrated_intensity(process.baseline, duration))
    ll -= process.weight.w * length(events) # approximate (exact requires `cdf`)

    partialsum = 0.0
    parenttime = 0.0
    for childtime in events
        λtot = intensity(process.baseline, childtime)
        if parenttime > 0.0
            Δt = childtime - parenttime
            partialsum = exp(-Δt * process.impulse_response.θ) * (1 + partialsum)
            λtot += process.weight.w * process.impulse_response.θ * partialsum
        end
        ll += log(λtot)
        parenttime = childtime
    end

    return ll
end

function logprior(process::ContinuousUnivariateHawkesProcess)
    ll = logprior(process.baseline)
    ll += logprior(process.impulse_response)
    ll += logprior(process.weight)

    return ll
end

function intensity(process::ContinuousUnivariateHawkesProcess, data, time::AbstractFloat)
    events, _ = data
    idx = time - process.impulse_response.Δtmax .< events .< time
    λ = intensity(process.baseline, time)
    for parenttime in events[idx]
        Δt = time - parenttime
        λ += impulse_response(process, Δt)
    end

    return λ
end

intensity(process::ContinuousUnivariateHawkesProcess,
    data, times::Vector{AbstractFloat}) = map(t -> intensity(process, data, t), times)

function total_intensity(process::ContinuousUnivariateHawkesProcess,
    events::Vector{T}, index::Int, time::AbstractFloat) where {T<:AbstractFloat}
    """Calculate the total intensity of the `index`-th event occurring at `time`."""
    λtot = intensity(process.baseline, time)
    index == 1 && return λtot
    parentindex = index - 1
    while events[parentindex] > time - process.impulse_response.Δtmax
        parenttime = events[parentindex]
        Δt = time - parenttime
        λtot += impulse_response(process, Δt)
        parentindex -= 1
        parentindex == 0 && break
    end
    return λtot
end

impulse_response(process::ContinuousUnivariateHawkesProcess,
    Δt::AbstractFloat) = process.weight.w * intensity(process.impulse_response, Δt)

function mle!(process::ContinuousUnivariateHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing)

    function objective(x)
        params!(process, x)
        return regularize ? -loglikelihood(process, data) - logprior(process) : -loglikelihood(process, data)
    end

    minloss = Inf
    outer_iter = 0
    converged = false
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
        else
            minloss = o.value
        end
        return false
    end

    # TODO: the random guess and lower bounds don't seem right for all parameters, e.g. logit-normal impulse response mean can be negative, so this creates bias. More generally, we can't guaruntee these are good guesses and bounds for any model that might permit MLE estimation. (This also needs to be fixed for ContinuousStandardHawkesProcess).
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

_rand_init_(process::ContinuousUnivariateHawkesProcess) = rand(nparams(process))

function resample!(process::ContinuousUnivariateHawkesProcess, data)
    parents = resample_parents(process, data)
    resample!(process.baseline, data, parents)
    resample!(process.impulse_response, data, parents)
    resample!(process.weight, data, parents)

    return params(process)
end





abstract type ContinuousMultivariateHawkesProcess <: ContinuousHawkesProcess end

"""
    ContinuousIndependentHawkesProcess

A container type for an collection of independent univariate Hawkes processes having identical component types. Provides convenience functions to construct non-independent multivariate processes.

Methods for independent processes *cannot*, in general, assume uniformity of the constituent univariate processes. For example, the default constructor does not prevent the construction of an independent process from two univariate processes with different `impulse_response` types or having the type with different fields (e.g., `Δtmax`). 

# Example
ndims = 3;
Δtmax = 1.0;
list = [
    ContinuousUnivariateHawkesProcess(
        UnivariateHomogeneousProcess(rand()),
        UnivariateLogitNormalImpulseResponse(rand(), rand(), Δtmax),
        UnivariateWeightModel(rand())
    ) for _ in 1:ndims
];
process = ContinuousIndependentHawkesProcess(list);
"""
struct ContinuousIndependentHawkesProcess <: ContinuousMultivariateHawkesProcess
    list::Vector{ContinuousUnivariateHawkesProcess}
end

function ContinuousIndependentHawkesProcess(baseline, impulse_response, weight, n)
    return ContinuousIndependentHawkesProcess(
        [ContinuousUnivariateHawkesProcess(deepcopy(baseline), deepcopy(impulse_response), deepcopy(weight)) for _ in 1:n]
    )
end

function ContinuousStandardHawkesProcess(process::ContinuousIndependentHawkesProcess)
    """Assumes uniformity of the constituent processes non-trainable parameter fields."""
    baseline = multivariate(process.list[1].baseline,
        params.([p.baseline for p in process.list]))
    impulse_response = multivariate(process.list[1].impulse_response,
        params.([p.impulse_response for p in process.list]))
    weights = multivariate(process.list[1].weight,
        params.([p.weight for p in process.list]))

    return ContinuousStandardHawkesProcess(baseline, impulse_response, weights)
end

ndims(process::ContinuousIndependentHawkesProcess) = length(process.list)

isstable(process::ContinuousIndependentHawkesProcess) = all(isstable.(process.list))

nparams(process::ContinuousIndependentHawkesProcess) = mapreduce(nparams, +, process.list)

params(process::ContinuousIndependentHawkesProcess) = mapreduce(params, vcat, process.list)

function params!(process::ContinuousIndependentHawkesProcess, x)
    length(x) == nparams(process) || throw(ArgumentError("Length of parameter vector x ($(length(x))) should equal the number of model parameters ($nparams(process))"))

    for (p, x) in zip(process.list, partition(x, process))
        params!(p, x)
    end

    return params(process)
end

function Base.rand(process::ContinuousIndependentHawkesProcess, duration::AbstractFloat)    
    data = [rand(p, duration) for p in process.list]
    events = mapreduce(d -> d[1], vcat, data)
    nodes = mapreduce(((i, d), ) -> i * ones(Int, length(d[1])), vcat, enumerate(data))
    idx = sortperm(events)

    return events[idx], nodes[idx], duration
end

function loglikelihood(process::ContinuousIndependentHawkesProcess, data; recurvise=true)
    """Expects data to be separated by process, i.e., data = separate(data, process)."""
    return mapreduce(((p, d), ) -> loglikelihood(p, d), +, zip(process.list, data))
end

function logprior(process::ContinuousIndependentHawkesProcess)
    return mapreduce(p -> logprior(p), *, process.list)
end

# function intensity(process::ContinuousIndependentHawkesProcess, data, time::AbstractFloat)
#     return [
#         intensity(p, d, time) for (p, d) in zip(process.list, split(data, process))
#     ]
# end

# function intensity(process::ContinuousIndependentHawkesProcess, data, times::Vector{AbstractFloat})
#     # TODO

#     return λ
# end

# function impulse_response(process::ContinuousIndependentHawkesProcess, Δt::AbstractFloat)
#     # TODO

#     return ir
# end

function mle!(process::ContinuousIndependentHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing, joint=false)
    
    data = split(data, process)
    guess = isnothing(guess) ? _rand_init_(process) : guess # or _default_init_(process, data)
    
    if !joint
        # Note: not parallelized over nodes b/c `loglikelihood` is already using all available threads (much higher benefit)
        guesses = partition(guess, process)
        res = [
                mle!(p, d; 
                    optimizer=optimizer,
                    verbose=verbose,
                    f_abstol=f_abstol,
                    regularize=regularize, guess=θ) for (p, d, θ) in zip(process.list, data, guesses)
            ]

        return MaximumLikelihood(
            reduce(vcat, [r.maximizer for r in res]),
            reduce(+, [r.maximum for r in res]),
            reduce(+, [r.steps for r in res]),
            reduce(+, [r.elapsed for r in res]),
            all(mapreduce(s -> s == "success", vcat, [r.status for r in res]))
        )
    else
        # Joint optimization (this seems to be generally a slower option)
        function objective(x)
            params!(process, x)
            return regularize ? -loglikelihood(process, data) - logprior(process) : -loglikelihood(process, data)
        end

        minloss = Inf
        outer_iter = 0
        converged = false
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
            else
                minloss = o.value
            end
            return false
        end

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
end

_rand_init_(process::ContinuousIndependentHawkesProcess) = mapreduce(_rand_init_, vcat, process.list)

function resample!(process::ContinuousIndependentHawkesProcess, data)
    """Expects data to be separated by process, i.e., data = separate(data, process)."""
    for (p, d) in zip(process.list, data)
        parents = resample_parents(p, d)
        resample!(p, d, parents)
    end

    return params(process)
end

function mcmc!(process::ContinuousIndependentHawkesProcess, data; kwargs)
    data = separate(data, process)

    if Distributions.nprocs() > 1
        res = pmap(((i, p), ) -> mcmc!(p, data[i]; kwargs...), enumerate(process.list))
    else
        res = MarkovChainMonteCarlo(process)
        start_time = time()
        while res.steps < nsteps
            resample!(process, data)
            push!(res.samples, params(process))
            res.steps += 1
            if res.steps % log_freq == 0 && verbose
                res.elapsed = time() - start_time
                println(" > step: $(res.steps), elapsed: $(res.elapsed)")
            end
        end
        res.elapsed = time() - start_time
        return res
    end

    return res
end

function partition(x, process)
    """Partition a parameter vector into slices based the number of parameters in each constituent univariate process."""
    length(x) == nparams(process) || throw(ArgumentError("Length of the parameter vector ($(length(x))) should equal the number of process parameters ($(nparams(process)))."))

    partitions = []
    idx = 1
    for p in process.list
        push!(partitions, x[idx:idx+nparams(p)-1])
        idx += nparams(p)
    end

    return partitions
end

function split(data, process)
    """Split combined data into univariate samples."""
    events, nodes, duration = data

    return [(events[nodes .== i], duration) for i = 1:ndims(process)]
end