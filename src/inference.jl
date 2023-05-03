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
function mcmc!(process::HawkesProcess, data; nsteps=1000, log_freq=100, verbose=false)
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
        if res.steps % log_freq == 0 && verbose
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
    step
    elapsed
    status
end

function VariationalInference(process::HawkesProcess)
    trace = Vector{typeof(params(process))}()
    step = 0
    elapsed = 0.0
    status = "incomplete"
    VariationalInference(trace, step, elapsed, status)
end

function Base.show(io::IO, res::VariationalInference)
    print(io, "\n* Status: $(res.status)\n    step: $(res.step)\n    elapsed: $(res.elapsed)")
end

function convergence_criteria(res::VariationalInference)
    trace = res.trace
    step = res.step
    Δx_max = maximum([maximum(abs.(x[step] .- x[step-1])) for x in trace])  # Δx = || x[i + 1] - x[i] ||_Inf < ϵ_x
    Δq_max = 0.0 # Δq = || q[i + 1] - q[i] ||_KL < ϵ_q
    q_new = Gamma.(trace.αλ[step], 1 ./ trace.βλ[step])
    q_prev = Gamma.(trace.αλ[step-1], 1 ./ trace.βλ[step-1])
    Δq_max = max(Δq_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    _, N, B = size(trace.γ[1])
    q_new = reshape(Dirichlet.(eachrow(reshape(trace.γ[step], N^2, B))), N, N)
    q_prev = reshape(Dirichlet.(eachrow(reshape(trace.γ[step-1], N^2, B))), N, N)
    Δq_max = max(Δq_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    if haskey(trace, :ρ)
        q_new = MixtureModel([Gamma(trace.κ1[step], 1 ./ trace.ν1[step]), Gamma(trace.κ0[step], 1 ./ trace.ν0[step])], [trace.ρ[step], 1 - trace.ρ[step]])
        q_prev = MixtureModel([Gamma(trace.κ1[step-1], 1 ./ trace.ν1[step-1]), Gamma(trace.κ0[step-1], 1 ./ trace.ν0[step-1])], [trace.ρ[step-1], 1 - trace.ρ[step-1]])
        Δq_max = max(Δq_max, kldivergence(q_new, q_prev))
        q_new = Binomial.(1, trace.ρ[step])
        q_prev = Binomial.(1, trace.ρ[step-1])
        Δq_max = max(Δq_max, maximum(abs.(kldivergence(q_new, q_prev))))
    else
        q_new = Gamma.(trace.κ1[step], 1 ./ trace.ν1[step])
        q_prev = Gamma.(trace.κ1[step-1], 1 ./ trace.ν1[step-1])
        Δq_max = max(Δq_max, maximum(abs.(kldivergence.(q_new, q_prev))))
    end
    if haskey(trace, :αρ) && haskey(trace, :βρ)
        q_new = Beta(trace.αρ[step], 1 ./ trace.βρ[step])
        q_prev = Beta(trace.αρ[step-1], 1 ./ trace.βρ[step-1])
        Δq_max = max(Δq_max, maximum(abs.(kldivergence(q_new, q_prev))))
    end
    return Δx_max, Δq_max
end

"""
    vb!(process::DiscreteHawkesProcess, data; kwargs...)

Perform mean-field variational inference.

The variational distribution takes the form q(λ0)q(θ)q(W)q(A)q(ω), where:
- q(λ0) = Gamma(α, β)
- q(θ) = Dir(γ)
- q(W) = Gamma(kappa , ν)
- q(A) = Bern(ρ)
- q(ω) = Mult(u)

# Arguments
-
-

# Keyword Arguments
- `max_steps::Int64`: maximum number of updates to perform.
- `Δx_thresh::Float64`: ?
- `Δq_thresh::Float64`: ?

# Return
- `res::`: a struct containing results of the inference routine
"""
function vb!(process::HawkesProcess, data; max_steps::Int64=1_000, Δx_thresh=1e-6, Δq_thresh=1e-2, verbose=false)
    # TODO set initial variational parameters guess
    convolved = convolve(process, data)
    res = VariationalInference(process)
    converged = false
    while res.step < max_steps
        update!(process, data, convolved)
        push!(res.trace, variational_params(process))
        res.step += 1
        if res.step > 1
            # Δx_max, Δq_max = convergence_criteria(res)
            # if verbose
            #     println(" > iter: $i/$max_steps, Δx_max=$(Δx_max), Δq_max=$(Δq_max)")
            # end
            # if Δx_max < Δx_thresh
            #     converged = "Δx_max"
            # elseif Δq_max < Δq_thresh
            #     converged = "Δq_max"
            # end
            # if converged != false
            #     println(" ** convergence criteria $converged < ϵ reached **")
            #     res.status = "converged"
            #     return res
            # end
        end
    end
    println(" ** maximum steps reached **")
    return res
end

"""
    svi!(process::DiscreteHawkesProcess, data; kwargs...)

Perform stochastic mean-field variational inference.

Explain how this differs from `vb!` here...
"""
function svi!(process::HawkesProcess, data) end