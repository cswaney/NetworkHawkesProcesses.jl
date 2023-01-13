abstract type Baseline end

import Base.rand
import Base.size
import Base.length

function size(process::Baseline) end
function params(process::Baseline) end
function params!(process::Baseline, x) end
function rand(process::Baseline, duration) end
function resample!(process::Baseline) end
function integrated_intensity(process::Baseline, duration) end
function integrated_intensity(process::Baseline, node, duration) end
function intensity(process::Baseline, node, time) end


"""
HomogeneousProcess

A homogeneous Poisson process with constant intensity λ ~ Gamma(α0, β0).

# Arguments
- `λ`: constant intensity parameter.
- `α0`: shape parameter of Gamma prior for Bayesian inference (default: 1.0).
- `β0`: rate parameter of Gamma prior for Bayesian inference (default: 1.0).
"""
mutable struct HomogeneousProcess <: Baseline
    λ
    α0
    β0
end

HomogeneousProcess(λ) = HomogeneousProcess(λ, 1.0, 1.0)

size(process::HomogeneousProcess) = length(process.λ)
ndims(process::HomogeneousProcess) = length(process.λ)

params(process::HomogeneousProcess) = copy(process.λ)

function params!(process::HomogeneousProcess, x)
    if length(x) != length(process.λ)
        error("Parameter vector length does not match model parameter length.")
    else
        process.λ .= x
    end
end

function rand(process::HomogeneousProcess, duration)
    nnodes = size(process)
    events = Array{Array{Float64,1},1}(undef, nnodes)
    nodes = Array{Array{Int64,1},1}(undef, nnodes)
    for parentnode = 1:nnodes
        n = rand(Poisson(process.λ[parentnode] * duration))
        events[parentnode] = rand(Uniform(0, duration), n)
        nodes[parentnode] = parentnode * ones(Int64, n)
    end
    events = vcat(events...)
    nodes = vcat(nodes...)
    idx = sortperm(events)
    return events[idx], Vector{Int64}(nodes[idx]), duration
end

function rand(process::HomogeneousProcess, node, duration)
    n = rand(Poisson(process.λ[node] * duration))
    return sort(rand(Uniform(0, duration), n))
end

function resample!(process::HomogeneousProcess, data, parents)
    counts, duration = sufficient_statistics(process, data, parents)
    α = process.α0 .+ counts
    β = process.β0 .+ duration
    process.λ = rand.(Gamma.(α, 1 / β))
end

function sufficient_statistics(process::HomogeneousProcess, data, parents)
    _, nodes, duration = data
    _, parentnodes = parents
    nnodes = size(process)
    counts = node_counts(nodes, parentnodes, nnodes)
    return counts, duration
end

function node_counts(nodes, parentnodes, nnodes)
    """Count the number of baseline events on each node."""
    counts = zeros(nnodes)
    for (node, parentnode) in zip(nodes, parentnodes)
        if parentnode == 0
            counts[node] += 1
        end
    end
    return counts
end

function integrated_intensity(process::HomogeneousProcess, duration)
    """Calculate the integral of the intensity."""
    return process.λ .* duration
end

function integrated_intensity(process::HomogeneousProcess, node, duration)
    """Calculate the integral of the intensity."""
    return process.λ[node] .* duration
end

function intensity(process::HomogeneousProcess, time::Float64)
    return process.λ
end

function intensity(process::HomogeneousProcess, node::Int64, time::Float64)
    return process.λ[node]
end

function logprior(process::HomogeneousProcess)
    return sum(log.(pdf.(Gamma.(process.α0, 1 / process.β0), process.λ)))
end


"""
LogGaussianCoxProcess

A log Gaussian Cox process constructed from a realization of a Gaussian process at fixed gridpoints.

The model is

    y ~ GP(0, K)
    λ(t) = exp(m + y(t))
    s ~ PP(λ(t))

For an arbitrary set of gridpoints, `x[1], ..., x[N]`, a corresponding sample of the Gaussian process, `y[1], ..., y[N]`, has a `N(0, Σ)` distribution, where

    Σ[i, j] = K(x[i], x[j])

The process is sampled by interpolating between intensity values `λ[1], ..., λ[N]`.

# Arguments
- `x::Vector{Vector{Float64}}`: strictly increasing vectors of sampling gridpoints.
- `λ::Vector{Vector{Float64}}`: non-negative intensity vectors, ie, `λ[i] = λ([x[i])`.
- `Σ::Vector{Matrix{Float64}}`: apositive-definite variance matrices.
- `m::Vector{Float64}`: intensity offsets equal to `log(λ0)` of homogeneous processes.
"""
struct LogGaussianCoxProcess <: Baseline
    x
    λ
    Σ::Vector{Matrix}
    m
end

function LogGaussianCoxProcess(x, λ, K::Function, m)
    """Construct a LGCP process from a kernel function `K`."""
    all([xk[1] == 0.0 for xk in x]) || error("All sampling grids must start at 0.")
    all([xk[end] == x[1][end] for xk in x]) || error("All sampling grids must end at the same time.")
    nnodes = length(x)
    Σ = Vector{Matrix{Float64}}(undef, nnodes)
    for k = 1:nnodes
        n = length(x[k])
        L = zeros(n, n)
        for i = 1:n
            for j = 1:i
                xi = x[k][i]
                xj = x[k][j]
                L[i, j] = K(xi, xj)
            end
        end
        Σ[k] = posdef!(Symmetric(L, :L))
    end
    return LogGaussianCoxProcess(x, λ, Σ, m)
end

function LogGaussianCoxProcess(gp::GaussianProcess, m, T, n, k)
    ms = fill(m, k)
    xs = [collect(range(0.0, length=n + 1, stop=T)) for _ in 1:k]
    ys = Vector{Vector{Float64}}()
    Σs = Vector{Matrix{Float64}}()
    for x in xs
        y, Σ = rand(gp, x)
        push!(ys, y)
        push!(Σs, Σ)
    end
    λs = [exp.(m .+ y) for (m, y) in zip(ms, ys)]
    return LogGaussianCoxProcess(xs, λs, Σs, ms)
end

function LogGaussianCoxProcess(nnodes, duration)
    kernel = NetworkHawkesProcesses.SquaredExponentialKernel(1.0, 1.0)
    gp = NetworkHawkesProcesses.GaussianProcess(kernel)
    return NetworkHawkesProcesses.LogGaussianCoxProcess(gp, 0.0, duration, 10, nnodes)
end

size(process::LogGaussianCoxProcess) = length(process.x)
ndims(process::LogGaussianCoxProcess) = length(process.x)
length(process::LogGaussianCoxProcess) = process.x[1][end] # process.x[1][1] == 0.
params(process::LogGaussianCoxProcess) = vcat(process.λ...)

function params!(process::LogGaussianCoxProcess, x)
    if length(x) != sum(length.(process.λ))
        error("Parameter vector length does not match model parameter length.")
    else
        nnodes = length(process.λ)
        start_index = 1
        for node = 1:nnodes
            npoints = length(process.λ[node])
            stop_index = start_index + npoints - 1
            process.λ[node] .= x[start_index:stop_index]
            start_index += npoints
        end
    end
end

function rand(process::LogGaussianCoxProcess, duration)
    length(process) != duration && error("Provided sample duration does not match process duration.")
    nnodes = size(process)
    events = Array{Array{Float64,1},1}(undef, nnodes)
    nodes = Array{Array{Int64,1},1}(undef, nnodes)
    for node = 1:nnodes
        f = LinearInterpolator(process.x[node], process.λ[node]) # OPTIMIZATION: pre-compute on `set!`
        events[node] = rejection_sample(f, rand(Poisson(integrate(f))))
        nodes[node] = node * ones(Int64, length(events[node]))
    end
    events = vcat(events...)
    nodes = vcat(nodes...)
    idx = sortperm(events)
    return events[idx], Vector{Int64}(nodes[idx]), duration
end

function resample!(process::LogGaussianCoxProcess, data, parents; method=:elliptical_slice)
    nnodes = size(process)
    data = split_extract(data, parents, nnodes)
    if Threads.nthreads() > 1
        @debug "using multi-threaded log Gaussian Cox process sampler"
        Threads.@threads for node in 1:nnodes
            resample_node!(process, data, node; method=method)
        end
    else
        for node = 1:nnodes
            resample_node!(process, data, node; method=method)
        end
    end
    return copy(process.λ)
end

function split_extract(data, parents, nnodes)
    """Extract data attributed to the baseline process and split by node."""
    indices = [Vector{Int64}() for _ in 1:nnodes]
    events, nodes, duration = data
    _, parentnodes = parents
    for (index, (node, parentnode)) in enumerate(zip(nodes, parentnodes))
        if parentnode == 0
            push!(indices[node], index)
        end
    end
    return [(events[idx], nodes[idx], duration) for idx in indices]
end

function resample_node!(process::LogGaussianCoxProcess, data, node; method=:elliptical_slice)
    init_y = log.(process.λ[node]) .- process.m[node]
    if method == :metropolis_hastings
        y = metropolis_hastings(process, data, node, init_y)
    elseif method == :elliptical_slice
        y = elliptical_slice(process, data, node, init_y)
    end
    process.λ[node] = exp.(process.m[node] .+ y)
end

function loglikelihood(process::LogGaussianCoxProcess, data, node, y)
    events, _, _ = data[node]
    f = LinearInterpolator(process.x[node], exp.(process.m[node] .+ y))
    ll = 0.0
    ll -= integrate(f)
    ll += sum(log.(f.(events)))
    return ll
end

function metropolis_hastings(process::LogGaussianCoxProcess, data, node, y0; step_size=0.1, max_attempts=max_attempts)
    """
        metropolis_hastings(p::LogGaussianCoxProcess, data, y0; step_size)
    
    Resample the posterior of a log Gaussian Cox process (LGCP) via Metropolis-Hastings [Neal, 1999].
    
    # Arguments
    - `process`: log Gaussian Cox process.
    - `data`: observed data generated by the Poisson process.
    - `y0`: current sample of the latent Gaussian process.
    
    # Keyword Arguments
    - `step_size`: step size parameter.
    """
    y = y0
    lly = loglikelihood(process, data, node, y)
    attempts = 0
    while max_attempts < max_attempts
        attempts += 1
        nu = rand(MvNormal(process.Σ[node]))
        ynew = sqrt(1 - step_size * step_size) .* y + step_size .* nu
        lly_new = loglikelihood(process, data, node, ynew)
        ratio = exp(lly_new - lly)
        if rand() < min(1, ratio)
            @debug "Metropolis-Hastings sampling attempts: $attempts"
            return ynew
        end
    end
    error("Metropolis-Hastings sampling reached maximum attempts.")
end

function elliptical_slice(p::LogGaussianCoxProcess, data, node, y; max_attempts=100)
    """
        elliptical_slice(p::LogGaussianCoxProcess, data, y0)
    
    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].
    
    # Arguments
    - `data`: observed data generated by the Poisson process.
    - `y`: initial sample of the latent Gaussian process.
    """
    attempts = 1
    v = rand(MvNormal(p.Σ), dim(p))
    u = rand()
    lly = loglikelihood(p, data, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(p, data, ynew)
    if lly_new >= lly
        @debug "Elliptical slice sampling attempts: $attempts"
        return ynew
    end
    while attempts < max_attempts
        attempts += 1
        if theta < 0.0
            theta_min = theta
        else
            theta_max = theta
        end
        theta = theta_min + (theta_max - theta_min) * rand()
        ynew = y .* cos(theta) .+ v .* sin(theta)
        lly_new = loglikelihood(p, data, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    error("Elliptical slice sampling reached maximum attempts.")
end

function intensity(p::LogGaussianCoxProcess, time::Float64)
    return [LinearInterpolator(p.x[n], p.λ[n])(time) for n in 1:ndims(p)]
end

function intensity(p::LogGaussianCoxProcess, node::Int64, time::Float64)
    return LinearInterpolator(p.x[node], p.λ[node])(time)
end


abstract type DiscreteBaseline end


"""
    DiscreteHomogeneousProcess

A discrete, homogeneous Poisson process.

The model supports Bayesian inference of the probabilistic model:

    λ[i] ~ Gamma(λ[i] | α0, β0) (i = 1, ..., N)
    x[i, t] ~ Poisson(x[i, t] | λ[i] * dt) (t = 1, ..., T)

# Arguments
- `λ::Vector{Float64}`: a vector of intensities.
- `α0::Float64`: the shape parameter of the Gamma prior.
- `β0::Float64`: the inverse-scale (i.e., rate) parameter of the Gamma prior.
- `dt::Float64`: the physical time represented by each time step, `t`.
"""
mutable struct DiscreteHomogeneousProcess <: DiscreteBaseline
    λ::Vector{Float64}
    α0::Float64
    β0::Float64
    αv::Vector{Float64}
    βv::Vector{Float64}
    dt::Float64
end

function DiscreteHomogeneousProcess(λ, dt=1.0)
    α0 = 1.0
    β0 = 1.0
    αv = ones(size(λ))
    βv = ones(size(λ))
    return DiscreteHomogeneousProcess(λ, α0, β0, αv, βv, dt)
end

ndims(p::DiscreteHomogeneousProcess) = length(p.λ)

params(p::DiscreteHomogeneousProcess) = copy(p.λ)

nparams(p::DiscreteHomogeneousProcess) = length(p.λ)

function params!(p::DiscreteHomogeneousProcess, x)
    if length(x) != length(p.λ)
        error("Parameter vector length does not match model parameter length.")
    else
        p.λ .= x
    end
end

variational_params(p::DiscreteHomogeneousProcess) = [copy(p.αv); copy(p.βν)]

function rand(p::DiscreteHomogeneousProcess, T::Int64)
    return vcat(transpose(rand.(Poisson.(p.λ .* p.dt), T))...)
end

intensity(p::DiscreteHomogeneousProcess, ts) = Matrix(transpose(repeat(p.λ, 1, length(ts)))) .* p.dt

intensity(p::DiscreteHomogeneousProcess, node, time) = p.λ[node] .* p.dt

function resample!(p::DiscreteHomogeneousProcess, parents)
    Mn, T = sufficient_statistics(p, parents[:, :, 1])
    α = p.α0 .+ Mn
    β = p.β0 + T * p.dt
    p.λ = vec(rand.(Gamma.(α, 1 ./ β)))
    return copy(p.λ)
end

function sufficient_statistics(p::DiscreteHomogeneousProcess, data)
    T, _ = size(data)
    Mn = sum(data, dims=[1])
    return Mn, T
end

function integrated_intensity(process::DiscreteHomogeneousProcess, duration)
    """Calculate the integral of the intensity."""
    return process.λ .* process.dt .* duration
end

function integrated_intensity(process::DiscreteHomogeneousProcess, node, duration)
    """Calculate the integral of the intensity on a single node."""
    return process.λ[node] * process.dt * duration
end

function logprior(process::DiscreteHomogeneousProcess)
    return sum(log.(pdf.(Gamma.(process.α0, 1 / process.β0), process.λ)))
end

function update!(process::DiscreteHomogeneousProcess, data, parents)
    """Perform a variational inference update. `parents` is the `T x N x (1 + NB)` variational parameter for the auxillary parent variables.
    """
    N, T = size(data)
    process.αv = process.α0 .+ sum(parents[:, :, 1] .* transpose(data), dims=1)
    process.βv = 1 ./ process.β0 .+ T .* process.dt .* ones(N)
    return vec(process.αv), copy(process.βv)
end


"""
    DiscreteLogGaussianCoxProcess

A discrete log Gaussian Cox process constructed from a realization of a Gaussian process at fixed gridpoints.

The probabilistic model is:

    y ~ GP(0, K)
    λ[t] = exp(m + y[t])
    s ~ PP(λ[t])

For an arbitrary set of gridpoints, `x[1] = 1, ..., x[N] = T`, a corresponding sample of the Gaussian process, `y[1], ..., y[N]`, has a `N(0, Σ)` distribution, where

    Σ[i, j] = K(x[i], x[j])

The process is sampled by interpolating between intensity values `λ[1], ..., λ[N]`.

# Arguments
- `x::Vector{Int64}`: a strictly increasing vector of sampling gridpoints.
- `λ::Matrix{Float64}`: a non-negative, `T x N` intensity matrix, ie, `λ[i, n] = λ_n([x[i])`.
- `Σ::Matrix{Float64}`: a positive-definite variance matrix.
- `m::Float64`: intensity offset equal to `log(λ0)` of a homogeneous process.
"""
mutable struct DiscreteLogGaussianCoxProcess <: DiscreteBaseline
    x::Array{Int64,1}
    λ::Array{Float64,2}
    Σ::Array{Float64,2}
    m::Float64
    dt::Float64
end

function DiscreteLogGaussianCoxProcess(x::Vector{Int64}, λ::Vector{Float64}, K::Function, m::Float64, dt::Float64)
    n = length(x)
    Σ = zeros(n, n)
    for i = 1:n
        for j = 1:i
            xi = x[i]
            xj = x[j]
            Σ[i, j] = K(xi, xj)
        end
    end
    Σ = posdef!(Symmetric(Σ, :L))
    return DiscreteLogGaussianCoxProcess(x, λ, Σ, m, dt)
end

import Base.range
ndims(p::DiscreteLogGaussianCoxProcess) = size(p.λ)[2]
start(p::DiscreteLogGaussianCoxProcess) = p.x[1]
stop(p::DiscreteLogGaussianCoxProcess) = p.x[end]
range(p::DiscreteLogGaussianCoxProcess) = p.x[1]:p.dt:p.x[end]
duration(p::DiscreteLogGaussianCoxProcess) = length(range(p))
params(p::DiscreteLogGaussianCoxProcess) = copy(vec(p.λ))

function rand(p::DiscreteLogGaussianCoxProcess, T::Int64)
    T == duration(p) || error("Sample length does not match process duration.")
    ts = range(p)
    λ = intensity(p, ts)
    return Matrix(transpose(rand.(Poisson.(λ))))
end

intensity(p::DiscreteLogGaussianCoxProcess, t) = DiscreteLinearInterpolator(p.x, p.λ .* p.dt)(t)

integrated_intensity(p::DiscreteLogGaussianCoxProcess) = integrate(DiscreteLinearInterpolator(p.x, p.λ .* p.dt), p.dt)

function loglikelihood(p::DiscreteLogGaussianCoxProcess, data, node)
    """
       loglikelihood(p::DiscreteLogGaussianCoxProcess, data)
    
    Compute the approximate likelihood of a discrete, inhomogeneous Poisson process.
    
    # Arguments
    - `data::Array{Int64,2}`: `N x duration(p)` event counts array.
    """
    ts = range(p)
    λ = intensity(p, ts)
    ll = 0.0
    for t in ts
        ll += log(pdf(Poisson(λ[t, node]), data[node, t]))
    end
    return ll
end

function loglikelihood(process::DiscreteLogGaussianCoxProcess, data, node, y)
    """
       loglikelihood(p::DiscreteLogGaussianCoxProcess, data, node, y)
    
    Compute the approximate likelihood of a discrete, inhomogeneous Poisson process with intensity `λ(s) = exp(m + y(s))`.
    
    # Arguments
    - `data::Array{Int64,2}`: `duration(p) x N` event counts array.
    - `y::Array{Float64,2}`: `length(p.λ) x N` Gaussian process sample.
    """
    ts = range(process)
    λ = DiscreteLinearInterpolator(p.x[node], exp.(process.m[node] .+ y[node]))(ts)
    ll = 0.0
    for t in ts
        ll += log(pdf(Poisson(λ[t, node]), data[node, t]))
    end
    return ll
end

function resample!(process::DiscreteLogGaussianCoxProcess, parents; method=:elliptical_slice)
    nnodes = ndims(process)
    data = transpose(parents[:, :, 1])
    if Threads.nthreads() > 1
        @debug "using multi-threaded log Gaussian Cox process sampler"
        Threads.@threads for node in 1:nnodes
            resample_node!(process, data, node; method=method)
        end
    else
        for node = 1:nnodes
            resample_node!(process, data, node; method=method)
        end
    end
    return copy(process.λ)
end

function resample_node!(process::DiscreteLogGaussianCoxProcess, data, node; method=:elliptical_slice)
    init_y = log.(process.λ[node]) .- process.m[node]
    if method == :metropolis_hastings
        y = metropolis_hastings(process, data, node, init_y)
    elseif method == :elliptical_slice
        y = elliptical_slice(process, data, node, init_y)
    end
    process.λ[node] = exp.(process.m[node] .+ y)
end

function metropolis_hastings(process::DiscreteLogGaussianCoxProcess, data, node, y; step_size=0.1, max_attempts=100)
    """
        metropolis_hastings(p::DiscreteLogGaussianCoxProcess, s, y0; eps)
    
    Sample the posterior of a log Gaussian Cox process (LGCP) via Metropolis-Hastings [Neal, 1999].
    
    # Arguments
    `data::Array{Int64,2}`: a `duration(p) x N` array of event counts generated by a Poisson process.
    `y::Array{Float64,1}`: a `length(p.λ) x N` sample of the latent Gaussian process.
    `eps::Float64`: the step size parameter.
    """
    lly = loglikelihood(process, data, node, y)
    attempts = 0
    while attempts < max_attempts
        attempts += 1
        nu = rand(MvNormal(p.Σ), dim(p))
        ynew = sqrt(1 - step_size * step_size) .* y .+ step_size .* nu
        lly_new = loglikelihood(p, data, node, ynew)
        ratio = exp(lly_new - lly)
        if rand() < min(1, ratio)
            return ynew
        end
    end
    error("Metropolis-Hastings algorithm reached maximum attempts.")
end

function elliptical_slice(p::DiscreteLogGaussianCoxProcess, data, node, y; max_attempts=100)
    """
        elliptical_slice(p::DiscreteLogGaussianCoxProcess, data, y0)
    
    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].
    
    # Arguments
    `data::Array{Int64,1}`: an array of event counts generated by a Poisson process.
    `y::Array{Float64,1}`: a sample of the latent Gaussian process.
    """
    attempts = 1
    v = rand(MvNormal(p.Σ), dim(p))
    u = rand()
    lly = loglikelihood(p, data, node, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(p, data, node, ynew)
    if lly_new >= lly
        @debug "Elliptical slice sampling attempts: $attempts"
        return ynew
    end
    while attempts < max_attempts
        attempts += 1
        if theta < 0.0
            theta_min = theta
        else
            theta_max = theta
        end
        theta = theta_min + (theta_max - theta_min) * rand()
        ynew = y .* cos(theta) .+ v .* sin(theta)
        lly_new = loglikelihood(p, data, node, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    error("Elliptical slice sampling reached maximum attempts.")
end
