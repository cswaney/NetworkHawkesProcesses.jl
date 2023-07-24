abstract type ContinuousHawkesProcess <: HawkesProcess end

"""
    rand(process::ContinuousHawkesProcess, duration::AbstractFloat)

Sample a random sequence of events from a continuous Hawkes process.

If `process` is univariate, returns a tuple containing the sequence of event times and sample duration, `(events::Vector{Float64}, duration)`. Else, returns a tuple containing the sequences of event times and nodes and sample duration, `(events::Vector{Float64}, nodes::{Int64}, duration)`.
"""
function rand(process::ContinuousHawkesProcess, duration::AbstractFloat) end


"""
    intensity(process::ContinuousHawkesProcess, data, time::AbstractFloat)

Calculate the intensity of a continuous-time Hawkes process at `time` given `data`.

### Arguments
- `data`: a tuple containing sample data in the format returned by `rand(process, duration)`, e.g., `(events, nodes, duration)` for a multivariate process.

# Returns
- `λ::Float64`: the intensity of the process.
"""
function intensity(process::ContinuousHawkesProcess, data, time::AbstractFloat) end


"""
    ContinuousUnivariateHawkesProcess <: ContinuousHawkesProcess

A continuous Hawkes processes composed of univariate component models.

### Example
```jldoctest; output = false
baseline = UnivariateHomogeneousProcess(0.1)
impulse_response = UnivariateLogitNormalImpulseResponse(0.0, 1.0, 2.0)
weight_model = UnivariateWeightModel(.25)
process = ContinuousUnivariateHawkesProcess(baseline, impulse_response, weight_model)

# output
ContinuousUnivariateHawkesProcess(UnivariateHomogeneousProcess{Float64}(0.1, 1.0, 1.0), UnivariateLogitNormalImpulseResponse{Float64}(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0), UnivariateWeightModel{Float64}(0.25, 1.0, 1.0, 1.0, 1.0))
```
"""
mutable struct ContinuousUnivariateHawkesProcess <: ContinuousHawkesProcess
    baseline::ContinuousUnivariateBaseline
    impulse_response::ContinuousUnivariateImpulseResponse
    weight::UnivariateWeightModel
end

ndims(process::ContinuousUnivariateDistribution) = 1

isstable(process::ContinuousUnivariateHawkesProcess) = abs(process.weight.w) < 1.0

function params(process::ContinuousUnivariateHawkesProcess)
    return [params(process.baseline); params(process.impulse_response); process.weight.w]
end

nparams(process::ContinuousUnivariateHawkesProcess) = length(params(process))

function params!(process::ContinuousUnivariateHawkesProcess, θ)
    nparams(process) == length(θ) || throw(ArgumentError("params!: length of parameter vector θ ($length(θ)) should equal the number of model parameters ($nparams(process))"))
    params!(process.baseline, θ[1:nparams(process.baseline)])
    params!(process.impulse_response, θ[nparams(process.baseline)+1:end-1])
    params!(process.weight, θ[end])

    return params(process)
end

function rand(process::ContinuousUnivariateHawkesProcess, duration::AbstractFloat)
    """
        rand(process::ContinuousUnivariateHawkesProcess, duration::AbstractFloat)
    
    Sample a random sequence of events from a continuous, univariate Hawkes process.
    
    ### Arguments
    - `duration::AbstractFloat`: the sample duration.
    
    ### Returns
    - `data::Tuple{Vector{Float64},Float64}`: a tuple containing the sequence of event times and sample duration.
    """

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
    nchildren = rand(process.weight)
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

ndims(process::ContinuousMultivariateHawkesProcess) = ndims(process.baseline)

function rand(process::ContinuousMultivariateHawkesProcess, duration::AbstractFloat)
    """
        rand(process::ContinuousMultivariateHawkesProcess, duration::AbstractFloat)
    
    Sample a random sequence of events from a continuous, multivariate Hawkes process.
    
    ### Arguments
    - `duration::AbstractFloat`: the sample duration.
    
    ### Returns
    - `data::Tuple{Vector{Float64},Vector{Int64},Float64}`: a tuple containing the sequences of event times and nodes and sample duration.
    """

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

function intensity(process::ContinuousMultivariateHawkesProcess, data, times::Vector{Float64})
    """
        intensity(process::ContinuousMultivariateHawkesProcess, data, times)
    
    Calculate the intensity of `process` at `times` given `data`.
    
    # Arguments
    - `data::Tuple{Vector{Float64},Vector{Int64},Float64}`: a tuple containing a vector of events, a vector of nodes associated with each event, and the duration of the data sample.
    - `times::Vector{Float64}`: a vector times where the process intensity will be computed.
    
    # Returns
    - `λ::Vector{Float64}`: a `len(times)` array of intensities conditional on `data` and `process`.
    """
    λs = zeros(length(times), ndims(process))
    for (i, t0) in enumerate(times)
        λs[i, :] = intensity(process, data, t0)
    end
    return λs
end

function intensity(process::ContinuousMultivariateHawkesProcess, data, time::Float64)
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


"""
    ContinuousStandardHawkesProcess <: ContinuousMultivariateHawkesProcess

A continuous standard Hawkes processes.

Equivalent to a continuous network Hawkes process with a fully-connected network.

### Example
```jldoctest; output = false
baseline = HomogeneousProcess([.1, .1])
weight_model = DenseWeightModel([.1 .2; .3 .4])
impulse_response = LogitNormalImpulseResponse(zeros(2, 2), ones(2, 2), 2.0)
process = ContinuousStandardHawkesProcess(baseline, impulse_response, weight_model)

# output
ContinuousStandardHawkesProcess(HomogeneousProcess{Float64}([0.1, 0.1], 1.0, 1.0), LogitNormalImpulseResponse([0.0 0.0; 0.0 0.0], [1.0 1.0; 1.0 1.0], 1.0, 1.0, 1.0, 1.0, 2.0), DenseWeightModel{Float64}([0.1 0.2; 0.3 0.4], 1.0, 1.0, [1.0 1.0; 1.0 1.0], [1.0 1.0; 1.0 1.0]))
```
"""
struct ContinuousStandardHawkesProcess <: ContinuousMultivariateHawkesProcess
    baseline::ContinuousMultivariateBaseline
    impulses::ContinuousMultivariateImpulseResponse
    weights::Weights
end

isstable(p::ContinuousStandardHawkesProcess) = maximum(abs.(eigvals(p.weights.W))) < 1.0

function params(process::ContinuousStandardHawkesProcess)
    return [params(process.baseline); params(process.impulses); params(process.weights)]
end

function params!(process::ContinuousStandardHawkesProcess, x)
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
    ContinuousNetworkHawkesProcess <: ContinuousMultivariateHawkesProcess

A continuous network Hawkes process.

Network Hawkes processes allow for partially-connected networks. The binary `adjacency_matrix` defines network connections generated by the probabilistic `network` model, where `adjacency_matrix[i, j]` represents a connection from node `i` to node `j`.

### Example
```jldoctest; output = false
baseline = HomogeneousProcess([.1, .1])
weight_model = DenseWeightModel([.1 .2; .3 .4])
impulse_response = LogitNormalImpulseResponse(zeros(2, 2), ones(2, 2), 2.0)
network = NetworkHawkesProcesses.BernoulliNetworkModel(.5, 2)
links = [1 0; 0 0]
process = ContinuousNetworkHawkesProcess(baseline, impulse_response, weight_model, links, network)

# output
ContinuousNetworkHawkesProcess(HomogeneousProcess{Float64}([0.1, 0.1], 1.0, 1.0), LogitNormalImpulseResponse([0.0 0.0; 0.0 0.0], [1.0 1.0; 1.0 1.0], 1.0, 1.0, 1.0, 1.0, 2.0), DenseWeightModel{Float64}([0.1 0.2; 0.3 0.4], 1.0, 1.0, [1.0 1.0; 1.0 1.0], [1.0 1.0; 1.0 1.0]), [1 0; 0 0], BernoulliNetworkModel(0.5, 1.0, 1.0, 1.0, 1.0, 2))
```
"""
struct ContinuousNetworkHawkesProcess <: ContinuousMultivariateHawkesProcess
    baseline::ContinuousMultivariateBaseline
    impulses::ContinuousMultivariateImpulseResponse
    weights::Weights
    adjacency_matrix::Matrix
    network::Network
end

isstable(p::ContinuousNetworkHawkesProcess) = maximum(abs.(eigvals(p.adjacency_matrix .* p.weights.W))) < 1.0

function params(process::ContinuousNetworkHawkesProcess)
    ϕ = params(process.network)
    λ0 = params(process.baseline)
    W = params(process.weights)
    θ = params(process.impulses)
    A = copy(vec(process.adjacency_matrix))
    return [ϕ; λ0; W; θ; A]
end

function params!(process::ContinuousNetworkHawkesProcess, x)
    throw(ErrorException("params! not implemented for processes that do not support maximum-likelihood estimation."))
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
