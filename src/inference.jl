"""
    MaximumLikelihood

A container to store the results of maximum likelihood estimation.
"""
mutable struct MaximumLikelihood
    maximizer
    maximum
    steps
    elapsed
    status
end

function Base.show(io::IO, res::MaximumLikelihood)
    print(io, "\n* Status: $(res.status)\n    steps: $(res.steps)\n    elapsed: $(res.elapsed)\n    loss: $(res.maximum)")
end


"""
    MarkovChainMonteCarlo

A container to store the results of Markov chain Monte Carlo (Gibbs) sampling.
"""
mutable struct MarkovChainMonteCarlo
    samples
    steps
    elapsed
    status
end

function MarkovChainMonteCarlo(process::HawkesProcess)
    samples = Vector{typeof(params(process))}()
    steps = 0
    elapsed = 0.0
    status = "incomplete"
    return MarkovChainMonteCarlo(samples, steps, elapsed, status)
end

function show(io::IO, res::MarkovChainMonteCarlo)
    print(io, "\n* Status: complete\n    steps: $(res.steps)\n    elapsed: $(res.elapsed)")
end

# TODO - initialize parameters (random or fitted)
"""
    mcmc!(process::HawkesProcess, data; kwargs...)

Perform Markov chain Monte Carlo (Gibbs) sampling.
"""
function mcmc!(process::HawkesProcess, data; nsteps=1000, verbose=false)
    res = MarkovChainMonteCarlo(process)
    start_time = time()
    if isa(process, DiscreteHawkesProcess)
        convolved = convolve(process, data)
    end
    while res.steps < nsteps
        if isa(process, DiscreteHawkesProcess)
            resample!(process, data, convolved)
        else
            resample!(process, data)
        end
        push!(res.samples, params(process))
        res.steps += 1
        if res.steps % 10 == 0 && verbose
            res.elapsed = time() - start_time
            println(" > step: $(res.steps), elapsed: $(res.elapsed)")
        end
    end
    res.elapsed = time() - start_time
    return res
end


"""
    VariationalInference

A container to store the results of variational inference.
"""
mutable struct VariationalInference
    trace
    steps
    elapsed
    status
end

function VariationalInference(process::HawkesProcess)
    trace = Vector{typeof(params(process))}()
    steps = 0
    elapsed = 0.0
    status = "incomplete"
    VariationalInference(trace, steps, elapsed, status)
end

function show(io::IO, res::VariationalInference)
    print(io, "\n* Status: complete\n    steps: $(res.steps)\n    elapsed: $(res.elapsed)")
end

function convergence_criteria(res::VariationalInference)
    trace = res.trace
    step = res.step
    ??x_max = maximum([maximum(abs.(x[step] .- x[step-1])) for x in trace])  # ??x = || x[i + 1] - x[i] ||_Inf < ??_x
    ??q_max = 0.0 # ??q = || q[i + 1] - q[i] ||_KL < ??_q
    q_new = Gamma.(trace.????[step], 1 ./ trace.????[step])
    q_prev = Gamma.(trace.????[step-1], 1 ./ trace.????[step-1])
    ??q_max = max(??q_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    _, N, B = size(trace.??[1])
    q_new = reshape(Dirichlet.(eachrow(reshape(trace.??[step], N^2, B))), N, N)
    q_prev = reshape(Dirichlet.(eachrow(reshape(trace.??[step-1], N^2, B))), N, N)
    ??q_max = max(??q_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    if haskey(trace, :??)
        q_new = MixtureModel([Gamma(trace.??1[step], 1 ./ trace.??1[step]), Gamma(trace.??0[step], 1 ./ trace.??0[step])], [trace.??[step], 1 - trace.??[step]])
        q_prev = MixtureModel([Gamma(trace.??1[step-1], 1 ./ trace.??1[step-1]), Gamma(trace.??0[step-1], 1 ./ trace.??0[step-1])], [trace.??[step-1], 1 - trace.??[step-1]])
        ??q_max = max(??q_max, kldivergence(q_new, q_prev))
        q_new = Binomial.(1, trace.??[step])
        q_prev = Binomial.(1, trace.??[step-1])
        ??q_max = max(??q_max, maximum(abs.(kldivergence(q_new, q_prev))))
    else
        q_new = Gamma.(trace.??1[step], 1 ./ trace.??1[step])
        q_prev = Gamma.(trace.??1[step-1], 1 ./ trace.??1[step-1])
        ??q_max = max(??q_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    end
    if haskey(trace, :????) && haskey(trace, :????)
        q_new = Beta(trace.????[step], 1 ./ trace.????[step])
        q_prev = Beta(trace.????[step-1], 1 ./ trace.????[step-1])
        ??q_max = max(??q_max, maximum(abs.(kldivergence(q_new, q_prev))))
    end
    return ??x_max, ??q_max
end

"""
    vb!(process::DiscreteHawkesProcess, data; kwargs...)

Perform mean-field variational inference.

The variational distribution takes the form q(??0)q(??)q(W)q(A)q(??), where:
- q(??0) = Gamma(??, ??)
- q(??) = Dir(??)
- q(W) = Gamma(kappa , ??)
- q(A) = Bern(??)
- q(??) = Mult(u)

# Arguments
-
-

# Keyword Arguments
- `max_steps::Int64`: maximum number of updates to perform.
- `??x_thresh::Float64`: ?
- `??q_thresh::Float64`: ?

# Return
- `res::`: a struct containing results of the inference routine
"""
function vb!(process::HawkesProcess, data; max_steps::Int64=1_000, ??x_thresh=1e-6, ??q_thresh=1e-2)
    # TODO set initial variational parameters guess
    convolved = PointProcesses.convolve(process, data)
    res = VariationalInferenceResult(process)
    converged = false
    for step = 1:max_steps
        update!(process, data, convolved)
        push!(res, variational_params(process))
        if step > 1
            ??x_max, ??q_max = convergence_criteria(res)
            println(" > iter: $i/$max_steps, ??x_max=$(??x_max), ??q_max=$(??q_max)")
            if ??x_max < ??x_thresh
                converged = "??x_max"
            elseif ??q_max < ??q_thresh
                converged = "??q_max"
            end
            if converged != false
                println(" ** convergence criteria $converged < ?? reached **")
                break
            end
        end
    end
    return res
end

"""
    svi!(process::DiscreteHawkesProcess, data; kwargs...)

Perform stochastic mean-field variational inference.

Explain how this differs from `vb!` here...
"""
function svi!(process::HawkesProcess, data) end