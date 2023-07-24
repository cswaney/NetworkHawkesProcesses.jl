import Base.length
import Base.range
import Base.rand
import Base.ndims

abstract type Baseline end

function ndims(process::Baseline) end
function nparams(process::Baseline) end
function params(process::Baseline) end
function params!(process::Baseline, x) end
function intensity(process::Baseline, time::AbstractFloat) end
function integrated_intensity(process::Baseline, duration::AbstractFloat) end
function logprior(process::Baseline) end


abstract type ContinuousBaseline <: Baseline end

function rand(process::ContinuousBaseline, duration::AbstractFloat) end


abstract type ContinuousUnivariateBaseline <: ContinuousBaseline end

ndims(process::ContinuousUnivariateBaseline) = 1
function multivariate(process::ContinuousUnivariateBaseline, x) end


abstract type ContinuousMultivariateBaseline <: ContinuousBaseline end

function rand(process::ContinuousMultivariateBaseline, node::Integer, duration::AbstractFloat) end
function intensity(process::ContinuousMultivariateBaseline, node::Integer, time::AbstractFloat) end
function integrated_intensity(process::ContinuousMultivariateBaseline, node::Integer, duration::AbstractFloat) end


abstract type DiscreteBaseline <: Baseline end

function rand(process::DiscreteBaseline, duration::Integer) end
function intensity(process::DiscreteBaseline, time::Integer) end
function intensity(process::DiscreteBaseline, times::AbstractVector{<:Integer}) end


abstract type DiscreteUnivariateBaseline <: DiscreteBaseline end

ndims(process::DiscreteUnivariateBaseline) = 1
function multivariate(process::DiscreteUnivariateBaseline, x) end


abstract type DiscreteMultivariateBaseline <: DiscreteBaseline end

function intensity(process::DiscreteMultivariateBaseline, node::Integer, time::Integer) end
function integrated_intensity(process::DiscreteMultivariateBaseline, node::Integer, duration::Integer) end


"""
    UnivariateHomogeneousProcess{T<:AbstractFloat} <: ContinuousUnivariateBaseline

A continuous-time, univariate homogeneous Poisson process with constant intensity `λ ~ Gamma(α0, β0)`.

### Arguments
- `λ::T`: constant, non-negative intensity parameter.
- `α0::T`: shape hyperparameter of the Gamma prior (default: 1.0).
- `β0::T`: rate hyperparameter of the Gamma prior (default: 1.0).

### Examples
```jldoctest; output = false
baseline = UnivariateHomogeneousProcess(1.0, 1.0, 1.0)

# output
UnivariateHomogeneousProcess{Float64}(1.0, 1.0, 1.0)
```

```jldoctest; output = false
baseline = UnivariateHomogeneousProcess(1.0) # use default hyperparameters

# output
UnivariateHomogeneousProcess{Float64}(1.0, 1.0, 1.0)
```
"""
mutable struct UnivariateHomogeneousProcess{T <: AbstractFloat} <: ContinuousUnivariateBaseline
    λ::T
    α0::T
    β0::T

    function UnivariateHomogeneousProcess{T}(λ, α0, β0) where {T <: AbstractFloat}
        λ < 0.0 && throw(DomainError(λ, "intensity parameter λ should be non-negative"))
        α0 > 0.0 || throw(DomainError(α0, "shape parameter α0 should be positive"))
        β0 > 0.0 || throw(DomainError(β0, "rate parameter β0 should be positive"))
        return new(λ, α0, β0)
    end
end

function UnivariateHomogeneousProcess(λ::T, α0::T, β0::T) where {T <: AbstractFloat}
    return UnivariateHomogeneousProcess{T}(λ, α0, β0)
end

UnivariateHomogeneousProcess(λ::T) where {T <: AbstractFloat} = UnivariateHomogeneousProcess{T}(λ, 1.0, 1.0)

function multivariate(process::UnivariateHomogeneousProcess, params)
    λ = cat(params...; dims=1) # params = params.([p1, p2, ...])

    return HomogeneousProcess(λ)
end

nparams(process::UnivariateHomogeneousProcess) = 1
params(process::UnivariateHomogeneousProcess) = [process.λ]

function params!(process::UnivariateHomogeneousProcess, θ)
    length(θ) == 1 || throw(ArgumentError("params!: length of parameter vector θ should equal the number of model parameters"))
    λ = θ[1]
    λ < 0.0 && throw(DomainError(λ, "intensity parameter λ should be non-negative"))
    process.λ = λ

    return params(process)
end

function Base.rand(process::UnivariateHomogeneousProcess, duration::AbstractFloat)
    duration < 0.0 && throw(DomainError(duration, "duration should be non-negative"))
    n = rand(Poisson(process.λ * duration))
    events = rand(Uniform(0, duration), n)

    return events, duration
end

function resample!(process::UnivariateHomogeneousProcess, data, parents)
    counts, duration = sufficient_statistics(process, data, parents)
    α = process.α0 + counts
    β = process.β0 + duration
    process.λ = rand(Gamma(α, 1 / β))

    return process.λ
end

function sufficient_statistics(process::UnivariateHomogeneousProcess, data, parents)
    _, duration = data
    _, parentnodes = parents
    counts = mapreduce(x -> x == 0, +, parentnodes)

    return counts, duration
end

function integrated_intensity(process::UnivariateHomogeneousProcess, duration::AbstractFloat)
    """Calculate the integral of the intensity."""
    duration < 0.0 && throw(DomainError("integrated_intensity: duration should be non-negative"))

    return process.λ .* duration
end

function intensity(process::UnivariateHomogeneousProcess, time::AbstractFloat)
    time < 0.0 && throw(DomainError("time should be non-negative"))
    
    return process.λ
end

function logprior(process::UnivariateHomogeneousProcess)
    return log(pdf(Gamma(process.α0, 1 / process.β0), process.λ))
end


"""
    HomogeneousProcess{T<:AbstractFloat} <: ContinuousMultivariateBaseline

A continuous-time, multivariate homogeneous Poisson process with constant intensity `λ[k] ~ Gamma(α0, β0)` for all `k = 1,..., ndims(process)`.



### Arguments
- `λ::Vector{T}`: constant, non-negative intensity parameter.
- `α0::T`: shape hyperparameter of the Gamma prior (default: 1.0).
- `β0::T`: rate hyperparameter of the Gamma prior (default: 1.0).

### Examples
```jldoctest; output = false
baseline = HomogeneousProcess(ones(Float32, 2), 1.0f0, 1.0f0)

# output
HomogeneousProcess{Float32}(Float32[1.0, 1.0], 1.0f0, 1.0f0)
```

```jldoctest; output = false
baseline = HomogeneousProcess(ones(3)) # use default hyperparameters

# output
HomogeneousProcess{Float64}([1.0, 1.0, 1.0], 1.0, 1.0)
```
"""
mutable struct HomogeneousProcess{T<:AbstractFloat} <: ContinuousMultivariateBaseline
    λ::Vector{T}
    α0::T
    β0::T
    
    function HomogeneousProcess{T}(λ, α0, β0) where T <: AbstractFloat
        any(λ .< 0) && throw(DomainError(λ, "intensity parameter λ must be non-negative"))
        α0 > 0 || throw(DomainError(α0, "shape parameter α0 must be positive"))
        β0 > 0 || throw(DomainError(β0, "rate parameter β0 must be positive"))
        return new(λ, α0, β0)
    end
end

HomogeneousProcess(λ::Vector{T}, α0::T, β0::T) where {T<:AbstractFloat} = HomogeneousProcess{T}(λ, α0, β0)
HomogeneousProcess(λ::Vector{T}) where {T<:AbstractFloat} = HomogeneousProcess{T}(λ, 1.0, 1.0)

ndims(process::HomogeneousProcess) = length(process.λ)
nparams(process::HomogeneousProcess) = length(process.λ)
params(process::HomogeneousProcess) = copy(process.λ)

function params!(process::HomogeneousProcess, x)
    if length(x) != length(process.λ)
        throw(ArgumentError("Parameter vector length ($(length(x))) does not match model parameter length ($(length(process.λ)))."))
    else
        process.λ .= x
    end
end

function Base.rand(process::HomogeneousProcess, duration::AbstractFloat)
    """
        rand(process::HomogeneousProcess, duration)
    
    Sample a random sequence of events from a homogeneous Poisson process.
    
    # Arguments
    - `duration`: the sampling duration.
    
    # Returns
    - `data::tuple{Vector{Float64},Vector{Int64},T}`: sampled events, nodes and duration data.
    
    # Example
    ```julia
    p = HomogeneousProcess(ones(2))
    events, nodes, duration = rand(p, 100.0)
    ````
    """
    duration < 0.0 && throw(DomainError("Sampling duration must be non-negative ($(duration))"))
    nnodes = ndims(process)
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
    return events[idx], nodes[idx], duration
end

function rand(process::HomogeneousProcess, node::Integer, duration::AbstractFloat)
    """
        rand(process::HomogeneousProcess, node, duration)
    
    Sample a random sequence of events from a single node of a homogeneous Poisson process.
    
    # Arguments
    - `node`: the node to sample.
    - `duration`: the sampling duration.
    
    # Returns
    - `data::Vector{Float64}`: sampled events data.
    """
    duration < 0.0 && throw(DomainError("Sampling duration must be non-negative ($(duration))"))
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
    nnodes = ndims(process)
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

function integrated_intensity(process::HomogeneousProcess, duration::AbstractFloat)
    """Calculate the integral of the intensity."""
    duration < 0 && throw(DomainError("duration must be non-negative"))
    return process.λ .* duration
end

function integrated_intensity(process::HomogeneousProcess, node::Integer, duration::AbstractFloat)
    """Calculate the integral of the intensity on a single node."""
    duration < 0 && throw(DomainError("duration must be non-negative"))
    return process.λ[node] .* duration
end

function intensity(process::HomogeneousProcess, time::AbstractFloat)
    time < 0 && throw(DomainError("time must be non-negative"))
    return process.λ
end

function intensity(process::HomogeneousProcess, node::Integer, time::AbstractFloat)
    time < 0 && throw(DomainError("time must be non-negative"))
    return process.λ[node]
end

function logprior(process::HomogeneousProcess)
    """Calculate the log prior probably of process parameters given hyperparameters."""
    return sum(log.(pdf.(Gamma.(process.α0, 1 / process.β0), process.λ)))
end


"""
    UnivariateLogGaussianCoxProcess{T<:AbstractFloat} <: ContinuousUnivariateBaseline

A continuous-time, univariate log Gaussian Cox process.

### Details
A log Gaussian Cox process is a Poisson process whose intensity is an exponentiated realization of a Gaussian process. The data generating process is

```math
\\begin{aligned}
    y &\\sim \\mathcal{GP}(0, K) \\\\
    λ(t) &= \\exp(m + y(t)) \\\\
    s &\\sim \\mathcal{PP}(λ(t)),
\\end{aligned}
```

where the Gaussian process is the *prior* distribution over ``y`` having the *function* ``K`` and scalar ``m`` as hyperparameters.

Sampling a log Gaussian Cox process works as follows. First, we select a sequence of gridpoints, ``x_{1:T}``. Next, we draw a sample of the Gaussian process ``\\mathcal{GP}(0, K)``,

```math
y_{1:T} \\sim \\mathcal{N}(0, \\Sigma),
```

where

```math
\\Sigma_{t, s} = K(x_t, x_s)
```

Finally, the intensity of the process is calculated via linear interpolation of the values ``\\lambda_{1:T} = \\exp(m + y_{1:T})`` and samples are drawn using rejection sampling.

### Arguments
- `x::Vector{T}`: a sequence of gridpoints ``x_1 = 0, \\dots, x_N``.
- `λ::Vector{T}`: a non-negative intensity parameter.
- `Σ::PDMat{T}`: a positive definite covariance hyperparameter.
- `m::T`: an offset hyperparameter (default: 0.0).

### Examples
```julia
kernel = SquaredExponentialKernel(1.0, 1.0)
gp = GaussianProcess(kernel)
x = collect(0.0:1.0:10.)
Σ = cov(gp, x)
y = rand(gp, x; sigma=Σ)
λ = exp.(0.0 .+ y)
baseline = UnivariateLogGaussianCoxProcess(x, λ, Σ, 0.0)
# or
baseline = UnivariateLogGaussianCoxProcess(gp, 10.0, 10, 0.0)
```

### References
- [J. Møller, A. R. Syversveen, and R. P. Waagepetersen. Log Gaussian Cox Processes. Scandinavian journal of statistics, 25(3):451–482, 1998.](https://onlinelibrary.wiley.com/doi/10.1111/1467-9469.00115)
"""
struct UnivariateLogGaussianCoxProcess{T<:AbstractFloat} <: ContinuousUnivariateBaseline
    x::Vector{T}
    λ::Vector{T}
    Σ::PDMat{T}
    m::T

    function UnivariateLogGaussianCoxProcess{T}(x, λ, Σ, m) where {T<:AbstractFloat}
        x[1] == 0.0 || throw(ArgumentError("Grid values x should start at 0.0"))
        any(λ .< 0.0) && throw(DomainError(λ, "Intensity parameter λ should be non-negative"))

        return new(x, λ, Σ, m)
    end
end

UnivariateLogGaussianCoxProcess(
    x::Vector{T},
    λ::Vector{T},
    Σ::PDMat{T},
    m::T
) where {T<:AbstractFloat} = UnivariateLogGaussianCoxProcess{T}(x, λ, Σ, m)

UnivariateLogGaussianCoxProcess(
    x::Vector{T},
    λ::Vector{T},
    K::Kernel,
    m::T
) where {T<:AbstractFloat} = UnivariateLogGaussianCoxProcess{T}(x, λ, K(x), m)

"""
    UnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration::AbstractFloat, nsteps::Integer, m::AbstractFloat)

Construct a continuous-time, univariate log Gaussian Cox process whose intensity is sampled from a Gaussian process at `nsteps` gridpoints evenly distributed between `0.0` and `duration`.

### Example
```julia
kernel = NetworkHawkesProcesses.SquaredExponentialKernel(1.0, 1.0)
gp = NetworkHawkesProcesses.GaussianProcess(kernel)
baseline = UnivariateLogGaussianCoxProcess(gp, 100.0, 10)
```
"""
function UnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration::AbstractFloat, nsteps::Integer, m::T=0.0) where {T<:AbstractFloat}
    duration > 0.0 || throw(DomainError(duration, "Duration should be positive"))
    nsteps > 0 || throw(DomainError(nsteps, "Number of steps should be  a positive integer"))

    x = Vector(range(start=0.0, stop=duration, length=nsteps+1))
    Σ = cov(gp, x)
    y = rand(gp, x; sigma=Σ)
    λ = exp.(m .+ y)

    return UnivariateLogGaussianCoxProcess{T}(x, λ, Σ, m)
end

multivariate(p::UnivariateLogGaussianCoxProcess, params) = LogGaussianCoxProcess(p.x, params, p.Σ, p.m)

length(process::UnivariateLogGaussianCoxProcess) = process.x[end]
nparams(process::UnivariateLogGaussianCoxProcess) = length(process.λ)
params(process::UnivariateLogGaussianCoxProcess) = copy(process.λ)

function params!(process::UnivariateLogGaussianCoxProcess, x)
    length(x) != nparams(process) && throw(ArgumentError("Parameter vector length ($(length(x))) should equal the number of model parameters ($(nparams(process)))."))
    any(x .< 0.) && throw(DomainError(x, "Intensity parameter should be non-negative"))


    process.λ .= x

    return params(process)
end

function rand(process::UnivariateLogGaussianCoxProcess, duration::AbstractFloat)
    length(process) != duration && throw(ArgumentError("Sample duration should equal process duration."))

    f = LinearInterpolator(process.x, process.λ)
    events = rejection_sample(f, rand(Poisson(integrate(f))))

    return events, duration
end

function loglikelihood(process::UnivariateLogGaussianCoxProcess, data, y)
    events, _ = data
    f = LinearInterpolator(process.x, exp.(process.m .+ y))

    return sum(log.(f.(events))) - integrate(f)
end

function resample!(process::UnivariateLogGaussianCoxProcess, data, parents; sampler=elliptical_slice)
    _, parentnodes = parents
    events, duration = data
    backgroundevents = events[parentnodes .== 0]
    init_y = log.(process.λ) .- process.m
    y = sampler(process, (backgroundevents, duration), init_y)
    process.λ .= exp.(process.m .+ y)

    return params(process)
end

function metropolis_hastings(process::UnivariateLogGaussianCoxProcess, data, y0; step_size=0.1, max_attempts=max_attempts)
    """
        metropolis_hastings(p::UnivariateLogGaussianCoxProcess, data, y0; step_size)

    Resample the posterior of a log Gaussian Cox process (LGCP) via Metropolis-Hastings [Neal, 1999].

    ### Arguments
    - `process`: log Gaussian Cox process.
    - `data`: observed data generated by the Poisson process.
    - `y0`: current sample of the latent Gaussian process.

    ### Keyword Arguments
    - `step_size`: step size parameter.
    """
    y = y0
    lly = loglikelihood(process, data, y)
    attempts = 0
    while max_attempts < max_attempts
        attempts += 1
        nu = rand(MvNormal(process.Σ))
        ynew = sqrt(1 - step_size * step_size) .* y + step_size .* nu
        lly_new = loglikelihood(process, data, ynew)
        ratio = exp(lly_new - lly)
        if rand() < min(1, ratio)
            @debug "Metropolis-Hastings sampling attempts: $attempts"
            return ynew
        end
    end
    error("Metropolis-Hastings sampling reached maximum attempts.")
end

function elliptical_slice(process::UnivariateLogGaussianCoxProcess, data, y; max_attempts=100)
    """
        elliptical_slice(p::UnivariateLogGaussianCoxProcess, data, y0)

    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].

    # Arguments
    - `data`: observed data generated by the Poisson process.
    - `y`: initial sample of the latent Gaussian process.
    """
    attempts = 1
    v = rand(MvNormal(process.Σ))
    u = rand()
    lly = loglikelihood(process, data, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(process, data, ynew)
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
        lly_new = loglikelihood(process, data, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    throw(ErrorException("Elliptical slice sampling reached maximum attempts."))
end

function intensity(p::UnivariateLogGaussianCoxProcess, time::AbstractFloat)
    return LinearInterpolator(p.x, p.λ)(time)
end

integrated_intensity(p::UnivariateLogGaussianCoxProcess, duration::AbstractFloat) = integrate(LinearInterpolator(p.x, p.λ))



"""
    LogGaussianCoxProcess{T<:AbstractFloat}(x, λ, Σ, m) <: ContinuousMultivariateBaseline

An ``N``-dimensional continuous-time log Gaussian Cox process.

### Details
The process consists of ``N`` *independent* log Gaussian Cox processes with a shared Gaussian process prior, ``\\mathcal{GP}(0, K)``, which is sampled along a common sequence of gridpoints, ``x_{1:T}``.

See [`UnivariateLogGaussianCoxProcess`](@ref) for details of the log Gaussian Cox data generating process.

### Arguments
- `x::Vector{T}`: a sequence of gridpoints ``x_1 = 0, \\dots, x_N``.
- `λ::Vector{Vector{T}}`: non-negative intensity parameters.
- `Σ::PDMat{T}`: a positive definite covariance hyperparameter.
- `m::T`: an offset hyperparameter (default: 0.0).

### Examples
```julia
kernel = SquaredExponentialKernel(1.0, 1.0)
gp = GaussianProcess(kernel)
x = collect(0.0:1.0:10.)
Σ = cov(gp, x)
y = [rand(gp, x; sigma=Σ), rand(gp, x; sigma=Σ)]
λ = [exp.(0.0 .+ y[1], exp.(0.0 .+ y[2])]
baseline = LogGaussianCoxProcess(x, λ, Σ, 0.0)
# or
baseline = LogGaussianCoxProcess(gp, 10.0, 10, 0.0)
```
"""
struct LogGaussianCoxProcess{T<:AbstractFloat} <: ContinuousMultivariateBaseline
    x::Vector{T}
    λ::Vector{Vector{T}}
    Σ::PDMat{T}
    m::T
    function LogGaussianCoxProcess{T}(x, λ, Σ, m) where {T<:AbstractFloat}
        x[1] == 0.0 || throw(DomainError(x, "Grid points x must start at 0."))
        return new(x, λ, PDMat(Σ), m)
    end
end

LogGaussianCoxProcess(
    x::Vector{T},
    λ::Vector{Vector{T}},
    Σ::PDMat{T},
    m::T
) where {T<:AbstractFloat} = LogGaussianCoxProcess{T}(x, λ, Σ, m)

LogGaussianCoxProcess(
    x::Vector{T},
    λ::Vector{Vector{T}},
    K::Kernel,
    m::T
) where {T<:AbstractFloat} = LogGaussianCoxProcess{T}(x, λ, K(x), m)

function LogGaussianCoxProcess(gp::GaussianProcess, m, T, n, k)
    """Construct a LGCP with random intensity given a Gaussian process."""
    x = Vector(range(0.0, length=n+1, stop=T))
    Σ = cov(gp, x)
    ys = [rand(gp, x; sigma=Σ) for _ in 1:k]
    λ = [exp.(m .+ y) for y in ys]
    return LogGaussianCoxProcess(x, λ, Σ, m)
end

ndims(process::LogGaussianCoxProcess) = length(process.λ)
length(process::LogGaussianCoxProcess) = process.x[end]
nparams(process::LogGaussianCoxProcess) = sum(length.(process.λ))
params(process::LogGaussianCoxProcess) = vcat(process.λ...)

function params!(process::LogGaussianCoxProcess, x)
    if length(x) != sum(length.(process.λ))
        throw(ArgumentError("Parameter vector length ($(length(x))) does not match model parameter length ($(sum(length.(process.λ))))."))
    end
    nnodes = length(process.λ)
    start_index = 1
    for node = 1:nnodes
        npoints = length(process.λ[node])
        stop_index = start_index + npoints - 1
        process.λ[node] .= x[start_index:stop_index]
        start_index += npoints
    end
end

function rand(process::LogGaussianCoxProcess, duration::AbstractFloat)
    """
        rand(process::LogGaussianCoxProcess, duration)
    
    Sample a random sequence of events from a log Gaussian Cox process.
    
    # Arguments
    - `duration`: the sampling duration.
    
    # Returns
    - `data::tuple{Vector{Float64},Vector{Int64},T}`: sampled events, nodes and duration data.
    
    # Example
    ```julia
    kernel = SquaredExponentialKernel(1.0, 1.0)
    gp = GaussianProcess(kernel)
    x = collect(0.0:0.1:10.0)
    y = rand(gp, x)
    λ = [exp.(y)]
    p = LogGaussianCoxProcess(x, λ, kernel, 0.0)
    events, nodes, duration = rand(p, 10.0)
    """
    length(process) != duration && throw(ArgumentError("Sample duration does not match process duration."))
    nnodes = ndims(process)
    events = Array{Array{Float64,1},1}(undef, nnodes)
    nodes = Array{Array{Int64,1},1}(undef, nnodes)
    for node = 1:nnodes
        f = LinearInterpolator(process.x, process.λ[node])
        events[node] = rejection_sample(f, rand(Poisson(integrate(f))))
        nodes[node] = node * ones(Int64, length(events[node]))
    end
    events = vcat(events...)
    nodes = vcat(nodes...)
    idx = sortperm(events)
    return events[idx], nodes[idx], duration
end

function rand(process::LogGaussianCoxProcess, node::Integer, duration::AbstractFloat)
    """
        rand(process::LogGaussianCoxProcess, node, duration)
    
    Sample a random sequence of events from a single node of a log Gaussian Cox process.
    
    # Arguments
    - `node`: the node to sample.
    - `duration`: the sampling duration.
    
    # Returns
    - `data::Vector{Float64}`: sampled events data.
    """
    length(process) != duration && throw(ArgumentError("Sample duration does not match process duration."))
    f = LinearInterpolator(process.x, process.λ[node])
    return rejection_sample(f, rand(Poisson(integrate(f))))
end

function resample!(process::LogGaussianCoxProcess, data, parents; sampler=elliptical_slice)
    nnodes = ndims(process)
    data = split_extract(data, parents, nnodes)
    if Threads.nthreads() > 1
        @debug "using multi-threaded log Gaussian Cox process sampler"
        Threads.@threads for node in 1:nnodes
            resample_node!(process, data, node; sampler=sampler)
        end
    else
        for node = 1:nnodes
            resample_node!(process, data, node; sampler=sampler)
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

function resample_node!(process::LogGaussianCoxProcess, data, node; sampler=:elliptical_slice)
    init_y = log.(process.λ[node]) .- process.m
    y = sampler(process, data, node, init_y)
    process.λ[node] = exp.(process.m .+ y)
end

function loglikelihood(process::LogGaussianCoxProcess, data, node, y)
    events, _, _ = data[node]
    f = LinearInterpolator(process.x, exp.(process.m .+ y))
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

function elliptical_slice(process::LogGaussianCoxProcess, data, node, y; max_attempts=100)
    """
        elliptical_slice(p::LogGaussianCoxProcess, data, y0)

    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].

    # Arguments
    - `data`: observed data generated by the Poisson process.
    - `y`: initial sample of the latent Gaussian process.
    """
    attempts = 1
    v = rand(MvNormal(process.Σ))
    u = rand()
    lly = loglikelihood(process, data, node, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(process, data, node, ynew)
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
        lly_new = loglikelihood(process, data, node, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    error("Elliptical slice sampling reached maximum attempts.")
end

function intensity(p::LogGaussianCoxProcess, time::AbstractFloat)
    return [LinearInterpolator(p.x, p.λ[n])(time) for n in 1:ndims(p)]
end

function intensity(p::LogGaussianCoxProcess, node::Integer, time::AbstractFloat)
    return LinearInterpolator(p.x, p.λ[node])(time)
end

integrated_intensity(p::LogGaussianCoxProcess, duration::AbstractFloat) = integrate.([LinearInterpolator(p.x, p.λ[k]) for k in 1:ndims(p)])

integrated_intensity(p::LogGaussianCoxProcess, node::Integer, duration::AbstractFloat) = integrate(LinearInterpolator(p.x, p.λ[node]))


"""
    DiscreteUnivariateHomogeneousProcess{T<:AbstractFloat} <: DiscreteUnivariateBaseline

A univariate, discrete-time homogeneous Poisson process that supports Bayesian inference for the probabilistic model:

```math
\\begin{aligned}
\\lambda &\\sim \\text{Ga}(\\lambda \\ | \\ \\alpha_0, \\beta_0) \\\\
x_t &\\sim \\text{Poisson}(x_t \\ | \\ \\lambda \\cdot dt)
\\end{aligned}
```

The model supports mean-field variational inference via a ``\\text{Ga}(\\alpha^v, \\beta^v)`` variational distribution.

### Arguments
- `λ::AbstractFloat`: a constant, non-negative intensity parameter.
- `α0::AbstractFloat`: a positive shape hyperparameter (default: `1.0`).
- `β0::AbstractFloat`: a positive rate hyperparameter (default: `1.0`).
- `αv::AbstractFloat`: a positive shape variational parameter (default: `1.0`).
- `βv::AbstractFloat`: a positive rate variational parameter (default: `1.0`).
- `dt::AbstractFloat`: the duration of each time step.

### Examples
```jldoctest; output = false
baseline = DiscreteUnivariateHomogeneousProcess(1.0, 1.0, 1.0, 1.0, 1.0, 2.0)

# output
DiscreteUnivariateHomogeneousProcess{Float64}(1.0, 1.0, 1.0, 1.0, 1.0, 2.0)
```

```jldoctest; output = false
baseline = DiscreteUnivariateHomogeneousProcess(1.0, 2.0) # use default hyperparameters

# output
DiscreteUnivariateHomogeneousProcess{Float64}(1.0, 1.0, 1.0, 1.0, 1.0, 2.0)
```
"""
mutable struct DiscreteUnivariateHomogeneousProcess{T<:AbstractFloat} <: DiscreteUnivariateBaseline
    λ::AbstractFloat
    α0::AbstractFloat
    β0::AbstractFloat
    αv::AbstractFloat
    βv::AbstractFloat
    dt::AbstractFloat

    function DiscreteUnivariateHomogeneousProcess{T}(λ, α0, β0, αv, βv, dt) where {T<:AbstractFloat}
        # TODO: validate arguments...

        return new(λ, α0, β0, αv, βv, dt)
    end
end

function DiscreteUnivariateHomogeneousProcess(λ::T, α0::T, β0::T, αv::T, βv::T, dt::T) where {T<:AbstractFloat}
    return DiscreteUnivariateHomogeneousProcess{T}(λ, α0, β0, αv, βv, dt)
end

function DiscreteUnivariateHomogeneousProcess(λ::T, dt::T) where {T<:AbstractFloat}
    return DiscreteUnivariateHomogeneousProcess{T}(λ, 1.0, 1.0, 1.0, 1.0, dt)
end

function multivariate(process::DiscreteUnivariateHomogeneousProcess, params)
    λ = cat(params...; dims=1) # params = params.([p1, p2, ...]

    return DiscreteHomogeneousProcess(λ, process.dt)
end

nparams(process::DiscreteUnivariateHomogeneousProcess) = 1
params(process::DiscreteUnivariateHomogeneousProcess) = [process.λ]

function params!(process::DiscreteUnivariateHomogeneousProcess, λ)
    length(λ) == nparams(process) || throw(ArgumentError(""))
    λ[1] < 0.0 && throw(DomainError(λ, ""))

    process.λ = λ[1]

    return params(process)
end

function rand(process::DiscreteUnivariateHomogeneousProcess, duration::Integer)
    duration < 0 && throw(DomainError(duration, "Number of time steps should be non-negative"))

    return rand(Poisson(process.λ .* process.dt), duration)
end

function intensity(p::DiscreteUnivariateHomogeneousProcess, t::Integer)
    t < 0.0 && throw(DomainError(t, "time should be non-negative"))

    return p.λ * p.dt
end

function intensity(process::DiscreteUnivariateHomogeneousProcess, times::AbstractVector{T}) where {T<:Integer}
    any(times .< 0) && throw(DomainError(times, "Times should be non-negative"))

    return fill(process.λ * process.dt, length(times))
end

function integrated_intensity(process::DiscreteUnivariateHomogeneousProcess, duration::AbstractFloat)
    duration < 0.0 && throw(DomainError(duration, "duration shouls be non-negative"))

    return process.λ * process.dt * duration
end

function resample!(process::DiscreteUnivariateHomogeneousProcess, parents)
    N, T = sufficient_statistics(process, parents[:, 1])
    α = process.α0 + N
    β = process.β0 + T * process.dt
    process.λ = rand(Gamma(α, 1 / β))

    return params(process)
end

function sufficient_statistics(process::DiscreteUnivariateHomogeneousProcess, data)
    T = length(data)
    N = sum(data)

    return N, T
end

function logprior(process::DiscreteUnivariateHomogeneousProcess)
    return pdf(Gamma(process.α0, 1 / process.β0), process.λ)
end

variational_params(p::DiscreteUnivariateHomogeneousProcess) = [p.αv, p.βv]

function update!(process::DiscreteUnivariateHomogeneousProcess, data, parents)
    """Perform a variational inference update. `parents` is the `T x (1 + B)` variational parameter for the auxillary parent variables.
    """
    length(data) != size(parents, 1) && throw(ArgumentError("data and parent dimensions do not conform"))

    T = length(data)

    process.αv = process.α0 .+ sum(parents[:, 1] .* data)
    process.βv = 1 / process.β0 + T * process.dt

    return process.αv, process.βv
end

function variational_log_expectation(process::DiscreteUnivariateHomogeneousProcess)
    return digamma(process.αv) - log(process.βv)
end

function q(p::DiscreteUnivariateHomogeneousProcess)
    return Gamma(p.αv, 1 / p.βv)
end


"""
    DiscreteHomogeneousProcess{T<:AbstractFloat} <: DiscreteMultivariateBaseline

A discrete, homogeneous Poisson process that supports Bayesian inference for the probabilistic model:

```math
\\begin{aligned}
\\lambda_n &\\sim \\text{Ga}(\\lambda_n \\ | \\ \\alpha_0, \\beta_0) \\\\
x_{n,t} &\\sim \\text{Poisson}(x_{n,t} \\ | \\ \\lambda_n \\cdot dt)
\\end{aligned}
```

The model supports mean-field variational inference via a ``\\text{Ga}(\\alpha_n^v, \\beta_n^v)`` variational distribution for each node ``n``.

### Arguments
- `λ::AbstractFloat`: a constant, non-negative intensity parameter.
- `α0::AbstractFloat`: a positive shape hyperparameter (default: `1.0`).
- `β0::AbstractFloat`: a positive rate hyperparameter (default: `1.0`).
- `αv::AbstractFloat`: a positive shape variational parameter (default: `1.0`).
- `βv::AbstractFloat`: a positive rate variational parameter (default: `1.0`).
- `dt::AbstractFloat`: the duration of each time step.

### Examples
```jldoctest; output = false
baseline = DiscreteHomogeneousProcess([0.5, 0.25], 2.0)

# output
DiscreteHomogeneousProcess{Float64}([0.5, 0.25], 1.0, 1.0, [1.0, 1.0], [1.0, 1.0], 2.0)
```
"""
mutable struct DiscreteHomogeneousProcess{T<:AbstractFloat} <: DiscreteMultivariateBaseline
    λ::Vector{T}
    α0::T
    β0::T
    αv::Vector{T}
    βv::Vector{T}
    dt::T

    function DiscreteHomogeneousProcess{T}(λ, α0, β0, αv, βv, dt) where {T <: AbstractFloat}
        any(λ .< 0) && throw(DomainError(λ, "DiscreteHomogeneousProcess: intensity parameter λ must be non-negative"))
        α0 > 0 || throw(DomainError(α0, "DiscreteHomogeneousProcess: shape parameter α0 must be positive"))
        β0 > 0 || throw(DomainError(β0, "DiscreteHomogeneousProcess: rate parameter β0 must be positive"))
        all(αv .> 0) || throw(DomainError(αv, "DiscreteHomogeneousProcess: shape parameter αv must be positive"))
        all(βv .> 0) || throw(DomainError(βv, "DiscreteHomogeneousProcess: rate parameter βv must be positive"))
        dt > 0.0 || throw(DomainError(dt, "DiscreteHomogeneousProcess: time step dt must be non-negative"))
        return new(λ, α0, β0, αv, βv, dt)
    end
end

function DiscreteHomogeneousProcess(λ::Vector{T}, α0::T, β0::T, αv::Vector{T}, βv::Vector{T}, dt=1.0) where {T<:AbstractFloat}
    return DiscreteHomogeneousProcess{T}(λ, α0, β0, αv, βv, dt)
end

function DiscreteHomogeneousProcess(λ::Vector{T}, dt=1.0) where {T<:AbstractFloat}
    α0 = 1.0
    β0 = 1.0
    αv = ones(size(λ))
    βv = ones(size(λ))
    return DiscreteHomogeneousProcess{T}(λ, α0, β0, αv, βv, dt)
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

variational_params(p::DiscreteHomogeneousProcess) = [copy(p.αv); copy(p.βv)]

function rand(p::DiscreteHomogeneousProcess, T::Integer)
    """
        rand(process::DiscreteHomogeneousProcess, T::Integer)
    
    Sample a random sequence of events from a discrete homogeneous Poisson process.
    
    # Arguments
    - `T::Integer`: the sampling duration as a number of discrete time steps.
    
    # Returns
    - `data::Matrix{Int64}`: an `nchannels x nsteps` matrix of sampled events.
    
    # Example
    ```julia
    p = DiscreteHomogeneousProcess(ones(2))
    data = rand(p, 100)
    ````
    """
    T < 0 && throw(DomainError(T, "Number of time steps must be non-negative"))
    return vcat(transpose(rand.(Poisson.(p.λ .* p.dt), T))...)
end

function intensity(p::DiscreteHomogeneousProcess, t::Integer)
    t < 1 && throw(DomainError(t, "Time should be a positive integer"))
    
    return p.λ .* p.dt
end

function intensity(p::DiscreteHomogeneousProcess, ts::AbstractVector{T}) where {T<:Integer}
    any(ts .< 1) && throw(DomainError(ts, "Times must be positive integers"))

    return Matrix(transpose(repeat(p.λ, 1, length(ts)))) .* p.dt
end

function intensity(p::DiscreteHomogeneousProcess, node::Integer, time::Integer)
    (node < 1 || node > ndims(p)) && throw(DomainError(node, "Node should be between one and the number process dimensions"))
    time < 1 && throw(DomainError(time, "Time should be a positive integer"))
    
    return p.λ[node] .* p.dt
end

function resample!(p::DiscreteHomogeneousProcess, parents)
    Mn, T = sufficient_statistics(p, parents[:, :, 1])
    α = p.α0 .+ Mn
    β = p.β0 + T * p.dt
    p.λ .= vec(rand.(Gamma.(α, 1 ./ β)))
    return copy(p.λ)
end

function sufficient_statistics(p::DiscreteHomogeneousProcess, data)
    T, _ = size(data)
    Mn = sum(data, dims=[1])
    return vec(Mn), T
end

function integrated_intensity(process::DiscreteHomogeneousProcess, duration::AbstractFloat)
    """Calculate the integral of the intensity."""
    duration < 0.0 && throw(DomainError(duration, "duration must be non-negative"))
    return process.λ .* process.dt .* duration
end

function integrated_intensity(process::DiscreteHomogeneousProcess, node::Integer, duration::AbstractFloat)
    """Calculate the integral of the intensity on a single node."""
    (node < 1 || node > ndims(process)) && throw(DomainError(node, "node must be between one and ndims"))
    duration < 0.0 && throw(DomainError(duration, "duration must be non-negative"))
    return process.λ[node] * process.dt * duration
end

function logprior(process::DiscreteHomogeneousProcess)
    return sum(log.(pdf.(Gamma.(process.α0, 1 / process.β0), process.λ)))
end

function update!(process::DiscreteHomogeneousProcess, data, parents)
    """Perform a variational inference update. `parents` is the `T x N x (1 + NB)` variational parameter for the auxillary parent variables.
    """
    size(data) != size(parents)[[2, 1]] && throw(ArgumentError("update!: data and parent dimensions do not conform"))
    N, T = size(data)
    process.αv = process.α0 .+ vec(sum(parents[:, :, 1] .* transpose(data), dims=1))
    process.βv = 1 ./ process.β0 .+ T .* process.dt .* ones(N)
    return vec(process.αv), copy(process.βv)
end

function variational_log_expectation(process::DiscreteHomogeneousProcess, cidx)
    return digamma(process.αv[cidx]) - log(process.βv[cidx])
end

function q(process::DiscreteHomogeneousProcess)
    return [Gamma(α, 1 / β) for (α, β) in zip(process.αv, process.βv)]
end


"""
    DiscreteUnivariateLogGaussianCoxProcess{T<:AbstractFloat} <: DiscreteUnivariateBaseline

A univariate, discrete-time log Gaussian Cox process.

### Details
The intensity of the process at time step ``t`` is

```math
f^{\\lambda}_x((t - 1) \\cdot dt) \\cdot dt
```

where ``f^{\\lambda}_x`` represents the linear interpolation of the intensity values ``λ_{1:N}`` through gridpoints ``x_{1:N}``. In words, we assign the value of the linear interpolation evaluated at the *left* endpoint of the time step interval, ``[(t-1) \\cdot dt, t \\cdot dt]``, scaled by the length of the time step, ``dt``. The length of each time step is generally smaller than the distance between consecutive ``x_t`` values and the total number of time steps is the number of time steps whose *right* endpoint lies within the domain ``[x_1, x_N]``, ``T = x_N \\div dt``.

See [`UnivariateLogGaussianCoxProcess`](@ref) for details of the log Gaussian Cox data generating process.

### Arguments
- `x::Vector{T}`: a sequence of gridpoints ``x_1 = 0, \\dots, x_N``.
- `λ::Vector{T}`: non-negative intensity parameters.
- `Σ::PDMat{T}`: a positive definite covariance hyperparameter.
- `m::T`: an offset hyperparameter (default: `0.0`).
- `dt::AbstractFloat`: positive time step (default: `1.0`).
"""
mutable struct DiscreteUnivariateLogGaussianCoxProcess{T<:AbstractFloat} <: DiscreteUnivariateBaseline
    x::Vector{T}
    λ::Vector{T}
    Σ::PDMat{T}
    m::T
    dt::T

    function DiscreteUnivariateLogGaussianCoxProcess{T}(x, λ, Σ, m, dt) where {T<:AbstractFloat}
        x[1] == 0.0 || throw(ArgumentError("Sampling gridpoints should start from 0.0."))
        any(λ .< 0.0) && throw(DomainError(λ, "Intensity parameter should be non-negative"))
        dt > 0.0 || throw(DomainError(dt, "Time step should be positive"))

        return new(x, λ, Σ, m, dt)
    end
end

function DiscreteUnivariateLogGaussianCoxProcess(x::Vector{T}, λ::Vector{T}, K::Function, m::T = 0.0, dt::T = 1.0) where {T<:AbstractFloat}
    x[1] == 0.0 || error("Grid points must start at 0.")

    return DiscreteUnivariateLogGaussianCoxProcess{T}(x, λ, K(x), m, dt)
end

"""
    DiscreteUnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration, nsteps, m, dt)

Construct a discrete-time, univariate log Gaussian Cox process by randomly sampling a Gaussian process along gridpoints `x = range(start=0.0, stop=duration, length=nsteps+1)`.
"""
function DiscreteUnivariateLogGaussianCoxProcess(gp::GaussianProcess, duration, nsteps, m::T = 0.0, dt::T = 1.0) where {T<:AbstractFloat}
    rem(duration, nsteps) == 0 || throw(ArgumentError("Duration ($duration) should be divisible by number of steps ($nsteps)."))

    x = collect(range(start=0.0, stop=duration, length=nsteps + 1))
    Σ = cov(gp, x)
    y = rand(gp, x; sigma=Σ)
    λ = exp.(m .+ y)

    return DiscreteUnivariateLogGaussianCoxProcess{T}(x, λ, Σ, m, dt)
end

multivariate(p::DiscreteUnivariateLogGaussianCoxProcess, params) = DiscreteLogGaussianCoxProcess(p.x, hcat(params...), p.Σ, p.m, p.dt)

function range(p::DiscreteUnivariateLogGaussianCoxProcess)
    """
        range(p::DiscreteUnivariateLogGaussianCoxProcess)
    
    Calculate the sequence of *left* endpoints for time steps whose *right* endpoints lie within the domain `[x[1], x[end]]` (i.e., the left endpoint of all time steps supported by the process).
    """

    return Base.range(start=0.0, stop=p.x[end] - p.dt, step=p.dt)
end

nsteps(p::DiscreteUnivariateLogGaussianCoxProcess)::Int = div(p.x[end], p.dt) # == length(range(p))

params(p::DiscreteUnivariateLogGaussianCoxProcess) = copy(vec(p.λ))

nparams(p::DiscreteUnivariateLogGaussianCoxProcess) = length(p.λ)

function params!(process::DiscreteUnivariateLogGaussianCoxProcess, x)
    if length(x) != length(process.λ)
        error("Parameter vector length does not match model parameter length.")
    else
        process.λ .= reshape(x, size(process.λ))
    end
end

function Base.rand(p::DiscreteUnivariateLogGaussianCoxProcess, T::Integer)
    T > nsteps(p) && throw(ArgumentError("Sample duration should not exceed process duration."))
    T < 1 && throw(ArgumentError("Sample duration should be a positive integer."))

    λ = intensity(p, 1:T)

    return rand.(Poisson.(λ))
end

function intensity(p::DiscreteUnivariateLogGaussianCoxProcess, t::Integer)
    t > nsteps(p) && throw(DomainError(t, "Time should not exceed process duration ($(nsteps(p)))"))
    return LinearInterpolator(p.x, p.λ * p.dt)(p.dt * (t - 1))
end

function intensity(p::DiscreteUnivariateLogGaussianCoxProcess, ts::AbstractVector{<:Integer})
    any(ts .> nsteps(p)) && throw(DomainError(ts, "Time should not exceed process duration ($(nsteps(p)))"))
    return LinearInterpolator(p.x, p.λ * p.dt).(p.dt .* (vec(ts) .- 1))
end

function loglikelihood(p::DiscreteUnivariateLogGaussianCoxProcess, data::Vector{Int})
    length(data) == nsteps(p) || throw(ArgumentError("Length of data ($(length(data))) should equal process duration ($(nsteps(p)))."))

    λ = intensity(p, 1:nsteps(p))

    return sum(log.(pdf.(Poisson.(λ), data)))
end

function loglikelihood(p::DiscreteUnivariateLogGaussianCoxProcess, data::Vector{Int}, y::Vector{<:AbstractFloat})
    """
       loglikelihood(p::DiscreteUnivariateLogGaussianCoxProcess, data, y)
    
    Compute the log likelihood of a discrete, univariate log Gaussian Cox process with intensity `λ[t] = exp(m + y[t])`.
    
    ### Arguments
    - `data::Vector{Int}`: length-`nsteps(p)` event counts array.
    - `y::Vector{<:AbstractFloat}`: a length-`length(p.λ)` Gaussian process sample.
    """
    λ = LinearInterpolator(p.x, exp.(p.m .+ y) .* p.dt).(range(p))

    return sum(log.(pdf.(Poisson.(λ), data)))
end

function resample!(process::DiscreteUnivariateLogGaussianCoxProcess, parents; sampler=elliptical_slice)
    backgroundevents = parents[:, 1]

    init_y = log.(process.λ) .- process.m
    y = sampler(process, backgroundevents, init_y)
    process.λ .= exp.(process.m .+ y)

    return copy(process.λ)
end

function metropolis_hastings(process::DiscreteUnivariateLogGaussianCoxProcess, data, y; step_size=0.1, max_attempts=100)
    """
        metropolis_hastings(process, data, y; kwargs)

    Sample the posterior of a log Gaussian Cox process (LGCP) via Metropolis-Hastings [Neal, 1999].

    ### Arguments
    `data::Array{Int64,2}`: a `nsteps(p) x N` array of event counts generated by a Poisson process.
    `y::Array{Float64,1}`: a `length(p.λ) x N` sample of the latent Gaussian process.
    `step_size::Float64`: the step size parameter.
    `max_attempts::Int64`: the maximum number of attempts to draw a improved sample. 
    """
    lly = loglikelihood(process::DiscreteUnivariateLogGaussianCoxProcess, data, y)
    attempts = 0
    while attempts < max_attempts
        attempts += 1
        nu = rand(MvNormal(process.Σ))
        ynew = sqrt(1 - step_size * step_size) .* y .+ step_size .* nu
        lly_new = loglikelihood(process, data, ynew)
        ratio = exp(lly_new - lly)
        if rand() < min(1, ratio)
            return ynew
        end
    end
    error("Metropolis-Hastings algorithm reached maximum attempts.")
end

function elliptical_slice(process::DiscreteUnivariateLogGaussianCoxProcess, data, y; max_attempts=100)
    """
        elliptical_slice(process, data, y; kwargs)

    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].

    ### Arguments
    `data::Array{Int64,1}`: an array of event counts generated by a Poisson process.
    `y::Array{Float64,1}`: a sample of the latent Gaussian process.
    `max_attempts::Int64`: the maximum number of attempts to draw a improved sample. 
    """
    attempts = 1
    v = rand(MvNormal(process.Σ))
    u = rand()
    lly = loglikelihood(process, data, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(process, data, ynew)
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
        lly_new = loglikelihood(process, data, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    error("Elliptical slice sampling reached maximum attempts.")
end


"""
    DiscreteLogGaussianCoxProcess{T<:AbstractFloat} <: DiscreteMultivariateBaseline

An ``N``-dimensional, discrete-time log Gaussian Cox process.

### Details
See [`DiscreteUnivariateLogGaussianCoxProcess`](@ref) for details.

### Arguments
- `x::Vector{T}`: a sequence of gridpoints ``x_1 = 0, \\dots, x_N``.
- `λ::Matrix{T}`: non-negative, `length(x) x ndims(process)` intensity parameter.
- `Σ::PDMat{T}`: a positive definite covariance hyperparameter.
- `m::T`: an offset hyperparameter (default: `0.0`).
- `dt::AbstractFloat`: positive time step (default: `1.0`).
"""
mutable struct DiscreteLogGaussianCoxProcess <: DiscreteMultivariateBaseline
    x::Vector{Float64}
    λ::Matrix{Float64}
    Σ::Matrix{Float64}
    m::Float64
    dt::Float64
end

function DiscreteLogGaussianCoxProcess(x, λ, K::Function, m, dt)
    x[1] == 0.0 || error("Grid points must start at 0.")
    return DiscreteLogGaussianCoxProcess(x, λ, K(x), m, dt)
end

function DiscreteLogGaussianCoxProcess(gp::GaussianProcess, m, T, n, k, dt)
    rem(T, n) == 0 || error("Duration must be divisible by number of steps.")
    x = collect(range(0.0, length=n+1, stop=T))
    Σ = cov(gp, x)
    ys = [rand(gp, x; sigma=Σ) for _ in 1:k]
    λ = hcat([exp.(m .+ y) for y in ys]...)
    return DiscreteLogGaussianCoxProcess(x, λ, Σ, m, dt)
end


ndims(p::DiscreteLogGaussianCoxProcess) = size(p.λ, 2)
range(p::DiscreteLogGaussianCoxProcess) = Base.range(start=p.x[1], stop=p.x[end] - p.dt, step=p.dt)
nsteps(p::DiscreteLogGaussianCoxProcess)::Int = div(p.x[end], p.dt)
params(p::DiscreteLogGaussianCoxProcess) = copy(vec(p.λ))
nparams(p::DiscreteLogGaussianCoxProcess) = length(p.λ)

function params!(process::DiscreteLogGaussianCoxProcess, x)
    if length(x) != length(process.λ)
        error("Parameter vector length does not match model parameter length.")
    else
        process.λ .= reshape(x, size(process.λ))
    end
end

function Base.rand(p::DiscreteLogGaussianCoxProcess, T::Integer)
    T == nsteps(p) || throw(ArgumentError("Sample duration should equal process duration."))
    T < 1 && throw(DomainError("Sample duration should be a positive integer"))

    λ = intensity(p, 1:T)
    
    return Matrix(rand.(Poisson.(λ))')
end

function intensity(p::DiscreteLogGaussianCoxProcess, t::Integer)
    t > nsteps(p) && throw(DomainError(t, "Time step ($t) should not exceed process duration ($(nsteps(p)))"))

    return [LinearInterpolator(p.x, p.λ[:, n] * p.dt)(p.dt * (t - 1)) for n in 1:ndims(p)]
end

function intensity(p::DiscreteLogGaussianCoxProcess, node::Integer, t::Integer)
    t > nsteps(p) && throw(DomainError(t, "Time step ($t) should not exceed process duration ($(nsteps(p)))"))

    return LinearInterpolator(p.x, p.λ[:, node] * p.dt)(p.dt * (t - 1))
end

function intensity(p::DiscreteLogGaussianCoxProcess, ts::AbstractVector{<:Integer})
    any(ts .> nsteps(p)) && throw(DomainError(ts, "Time steps should not exceed process duration ($(nsteps(p)))"))

    λ = zeros(length(ts), ndims(p))
    for n in 1:ndims(p)
        λ[:, n] = LinearInterpolator(p.x, p.λ[:, n] * p.dt).(p.dt .* (vec(ts) .- 1))
    end

    return λ
end

function intensity(p::DiscreteLogGaussianCoxProcess, node::Integer, ts::AbstractVector{<:Integer})
    any(ts .> nsteps(p)) && throw(DomainError(ts, "Time steps should not exceed process duration ($(nsteps(p)))"))

    return LinearInterpolator(p.x, p.λ[:, node] * p.dt).(p.dt .* (vec(ts) .- 1))
end

integrated_intensity(p::DiscreteLogGaussianCoxProcess, duration) = vec(sum(intensity(p, 1:nsteps(p)), dims=1))

function loglikelihood(p::DiscreteLogGaussianCoxProcess, data, node)
    """
       loglikelihood(p::DiscreteLogGaussianCoxProcess, data)

    Compute the approximate likelihood of a discrete, inhomogeneous Poisson process.

    # Arguments
    - `data::Array{Int64,2}`: `N x nsteps(p)` event counts array.
    """
    λ = intensity(p, node, 1:nsteps(p))
    ll = 0.0
    for t in 1:length(times)
        ll += log(pdf(Poisson(λ[t]), data[node, t]))
    end
    return ll
end

function loglikelihood(p::DiscreteLogGaussianCoxProcess, data, node, y)
    """
       loglikelihood(p::DiscreteLogGaussianCoxProcess, data, node, y)

    Compute the approximate likelihood of a discrete, inhomogeneous Poisson process with intensity `λ(s) = exp(m + y(s))`.

    # Arguments
    - `data::Array{Int64,2}`: `nsteps(p) x N` event counts array.
    - `y::Vector{Float64}`: `length(p.λ)` Gaussian process sample for the given node.
    """
    times = range(p)
    λ = LinearInterpolator(p.x, exp.(p.m .+ y) .* p.dt).(times)
    ll = 0.0
    for t in 1:length(times)
        ll += log(pdf(Poisson(λ[t]), data[node, t]))
    end
    return ll
end

function resample!(process::DiscreteLogGaussianCoxProcess, parents; sampler=elliptical_slice)
    nnodes = ndims(process)
    data = transpose(parents[:, :, 1])
    if Threads.nthreads() > 1
        @debug "using multi-threaded log Gaussian Cox process sampler"
        Threads.@threads for node in 1:nnodes
            resample_node!(process, data, node; sampler=sampler)
        end
    else
        for node = 1:nnodes
            resample_node!(process, data, node; sampler=sampler)
        end
    end
    return copy(process.λ)
end

function resample_node!(process::DiscreteLogGaussianCoxProcess, data, node; sampler=elliptical_slice)
    init_y = log.(process.λ[:, node]) .- process.m
    y = sampler(process, data, node, init_y)
    process.λ[:, node] = exp.(process.m .+ y)
end

function metropolis_hastings(process, data, node, y; step_size=0.1, max_attempts=100)
    """
        metropolis_hastings(process, data, node, y; kwargs)

    Sample the posterior of a log Gaussian Cox process (LGCP) via Metropolis-Hastings [Neal, 1999].

    # Arguments
    `data::Array{Int64,2}`: a `nsteps(p) x N` array of event counts generated by a Poisson process.
    `node::Int64`: the node of the process to sample.
    `y::Array{Float64,1}`: a `length(p.λ) x N` sample of the latent Gaussian process.
    `step_size::Float64`: the step size parameter.
    `max_attempts::Int64`: the maximum number of attempts to draw a improved sample. 
    """
    lly = loglikelihood(process, data, node, y)
    attempts = 0
    while attempts < max_attempts
        attempts += 1
        nu = rand(MvNormal(process.Σ))
        ynew = sqrt(1 - step_size * step_size) .* y .+ step_size .* nu
        lly_new = loglikelihood(process, data, node, ynew)
        ratio = exp(lly_new - lly)
        if rand() < min(1, ratio)
            return ynew
        end
    end
    error("Metropolis-Hastings algorithm reached maximum attempts.")
end

function elliptical_slice(process, data, node, y; max_attempts=100)
    """
        elliptical_slice(process, data, node, y; kwargs)

    Sample the posterior of a log Gaussian Cox process (LGCP) via elliptical slicing [Murray et al., 2010].

    # Arguments
    `data::Array{Int64,1}`: an array of event counts generated by a Poisson process.
    `node::Int64`: the node of the process to sample.
    `y::Array{Float64,1}`: a sample of the latent Gaussian process.
    `max_attempts::Int64`: the maximum number of attempts to draw a improved sample. 
    """
    attempts = 1
    v = rand(MvNormal(process.Σ))
    u = rand()
    lly = loglikelihood(process, data, node, y) + log(u)
    theta = 2 * pi * rand()
    theta_min = theta - 2 * pi
    theta_max = theta
    ynew = y .* cos(theta) .+ v .* sin(theta)
    lly_new = loglikelihood(process, data, node, ynew)
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
        lly_new = loglikelihood(process, data, node, ynew)
        if lly_new >= lly
            @debug "Elliptical slice sampling attempts: $attempts"
            return ynew
        end
    end
    error("Elliptical slice sampling reached maximum attempts.")
end
