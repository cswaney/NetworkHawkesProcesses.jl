

# const UnivariateHawkesProcess = Union{ContinuousHawkesProcess,DiscreteUnivariateHawkesProcess}
const Components = Union{ContinuousHawkesProcess,DiscreteHawkesProcess}

"""
    IndependentHawkesProcess{T<:Union{ContinuousHawkesProcess,DiscreteHawkesProcess}} <: HawkesProcess

A container type for an collection of independent, discrete-time Hawkes processes.

### Details
This type is provided for convenience to fit a multivariate Hawkes processes with no interactions between nodes (a useful baseline for comparison): it is equivalent to fitting `K` univariate processes (with identical component types) independently. However, methods for independent processes *do not*, in general, assume uniformity of the constituent univariate processes. For example, the default constructor allows for a list of processes with different `impulse_response` types or having the same type with different fields (e.g., `Δtmax`). 

### Example
ndims = 2
nbasis = 3
nlags = 4
dt = 2.0
list = [
    DiscreteUnivariateHawkesProcess(
        DiscreteUnivariateHomogeneousProcess(rand(), dt),
        UnivariateGaussianImpulseResponse(rand(nbasis), nlags, dt),
        UnivariateWeightModel(rand())
    ) for _ in 1:ndims
];
process = IndependentHawkesProcess(list);
"""
struct IndependentHawkesProcess{T<:Components} <: HawkesProcess
    list::Vector{T}
end

const Independent{T} = IndependentHawkesProcess{T}

function ContinuousStandardHawkesProcess(process::IndependentHawkesProcess{ContinuousHawkesProcess})
    """Assumes uniformity of the constituent processes non-trainable parameter fields."""
    baseline = multivariate(process.list[1].baseline,
        params.([p.baseline for p in process.list]))
    impulse_response = multivariate(process.list[1].impulse_response,
        params.([p.impulse_response for p in process.list]))
    weights = multivariate(process.list[1].weight,
        params.([p.weight for p in process.list]))

    return ContinuousStandardHawkesProcess(baseline, impulse_response, weights)
end

function DiscreteStandardHawkesProcess(process::IndependentHawkesProcess{DiscreteHawkesProcess})
    """Assumes uniformity of the constituent processes non-trainable parameter fields."""
    baseline = multivariate(process.list[1].baseline,
        params.([p.baseline for p in process.list]))
    impulse_response = multivariate(process.list[1].impulse_response,
        params.([p.impulse_response for p in process.list]))
    weights = multivariate(process.list[1].weight_model,
        params.([p.weight_model for p in process.list]))
    dt = process.list[1].baseline.dt

    return DiscreteStandardHawkesProcess(baseline, impulse_response, weights, dt)
end

ndims(process::IndependentHawkesProcess) = length(process.list)
isstable(process::IndependentHawkesProcess) = all(isstable.(process.list))
nparams(process::IndependentHawkesProcess) = mapreduce(nparams, +, process.list)
params(process::IndependentHawkesProcess) = mapreduce(params, vcat, process.list)

function params!(process::IndependentHawkesProcess, x)
    length(x) == nparams(process) || throw(ArgumentError("Length of parameter vector x ($(length(x))) should equal the number of model parameters ($nparams(process))"))

    for (p, x) in zip(process.list, partition(x, process))
        params!(p, x)
    end

    return params(process)
end

function Base.rand(process::IndependentHawkesProcess{ContinuousHawkesProcess}, duration::AbstractFloat)
    data = [rand(p, duration) for p in process.list]
    events = mapreduce(d -> d[1], vcat, data)
    nodes = mapreduce(((i, d),) -> i * ones(Int, length(d[1])), vcat, enumerate(data))
    idx = sortperm(events)

    return events[idx], nodes[idx], duration
end

function Base.rand(process::IndependentHawkesProcess{DiscreteHawkesProcess}, duration::Integer)
    return transpose(hcat([rand(p, duration) for p in process.list]...))
end

function loglikelihood(process::IndependentHawkesProcess, splits; recursive=true)
    """Expects data to be split by process, i.e., splits = split(data, process)."""
    return mapreduce(((p, d),) -> loglikelihood(p, d; recursive=recursive), +, zip(process.list, splits))
end

function logprior(process::IndependentHawkesProcess)
    return mapreduce(p -> logprior(p), *, process.list)
end

function intensity(process::IndependentHawkesProcess, data, time::AbstractFloat)
    # return [
    #     intensity(p, d, time) for (p, d) in zip(process.list, split(data, process))
    # ]
    throw(ErrorException("method not implemented"))
end

function intensity(process::IndependentHawkesProcess, data, times::Vector{AbstractFloat})
    # TODO

    throw(ErrorException("method not implemented"))
    # return λ
end

function mle!(process::IndependentHawkesProcess, data; optimizer=BFGS, verbose=false, f_abstol=1e-6, regularize=false, guess=nothing, joint=false)

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

_rand_init_(process::IndependentHawkesProcess) = mapreduce(_rand_init_, vcat, process.list)

function resample!(process::IndependentHawkesProcess, splits)
    """Expects data to be split by process, i.e., splits = split(data, process)."""
    for (p, d) in zip(process.list, splits)
        if process isa Independent{ContinuousHawkesProcess}
            resample!(p, d)
        else
            resample!(p, d...)
        end
    end

    return params(process)
end

function mcmc!(process::IndependentHawkesProcess, data; nsteps=1000, log_freq=100, verbose=false, joint=false)

    if process isa Independent{ContinuousHawkesProcess}
        splits = split(data, process)
    else
        splits = split_convolve(data, process)
    end

    if !joint # this is probably slower, in general
        kwargs = Dict(
            :nsteps => nsteps,
            :log_freq => log_freq,
            :verbose => verbose
        )
        res = [mcmc!(p, d; kwargs...) for (p, d) in zip(process.list, splits)]
        # TODO: combine results...
    else
        start_time = time()
        res = MarkovChainMonteCarlo(process)
        while res.steps < nsteps
            resample!(process, splits)
            push!(res.samples, params(process))
            res.steps += 1
            if res.steps % log_freq == 0 && verbose
                res.elapsed = time() - start_time
                println(" > step: $(res.steps), elapsed: $(res.elapsed)")
            end
        end
        res.elapsed = time() - start_time
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

function split(data, process::Independent{ContinuousHawkesProcess})
    """Split combined data into univariate samples."""
    events, nodes, duration = data

    return [(events[nodes.==i], duration) for i = 1:ndims(process)]
end

"""
    split(data, process::Independent{DiscreteHawkesProcess})

Split combined data into univariate samples."""
split(data, process::Independent{DiscreteHawkesProcess}) = collect(eachrow(data))

"""
    split_convolve(data, process::Independent{DiscreteHawkesProcess})

Split combined data into univariate samples and convolve each sample with its corresponding process."""
function split_convolve(data, process::Independent{DiscreteHawkesProcess})
    return [(d, convolve(process.list[i], d)) for (i, d) in enumerate(eachrow(data))]
end
