# TODO - A single Interpolator struct can replace the continuous and discrete versions here...

abstract type Interpolator end

nnodes(f::Interpolator) = length(f.x)
nsteps(f::Interpolator) = length(f.x) - 1

function (f::Interpolator)(x) end


"""
    LinearInterpolator

A linear interpolation between values `y` evaluated at gridpoints `x`.

# Arguments
- `x::Vector{Float64}`: a vector of gridpoints
- `y::Vector{Float64}`: a vector of function values
"""
struct LinearInterpolator <: Interpolator
    x::Vector{Float64}
    y::Vector{Float64}
    I::Union{Float64,Missing}
end

LinearInterpolator(x, y) = LinearInterpolator(x, y, missing)

function interpolate(f::LinearInterpolator, x0)
    """Evaluate the linear interpolation at a point."""
    (x0 < f.x[1] || x0 > f.x[end]) && error("Value is outside function support")
    for i in 1:nsteps(f)
        if x0 >= f.x[i] && x0 < f.x[i+1]
            return (f.y[i+1] * (x0 - f.x[i]) + f.y[i] * (f.x[i+1] - x0)) / (f.x[i+1] - f.x[i])
        end
    end
    return f.y[end]
end

(f::LinearInterpolator)(x0) = interpolate(f, x0)

function integrate(f::LinearInterpolator)
    """Integrate a linear interpolation approximately using the trapezoidal rule."""
    !ismissing(f.I) && return f.I
    I = 0.0
    for i in 1:nsteps(f)
        I += 0.5 * (f.y[i] + f.y[i+1]) * (f.x[i+1] - f.x[i])
    end
    return I
end

function rejection_sample(f::LinearInterpolator, n::Int)
    """Sample `n` draws from the univariate distribution given by `p(x) = f(x) / Z`, where `Z` is a normalization constant."""
    xmin, xmax = minimum(f.x), maximum(f.x)
    ymin, ymax = minimum(f.y), maximum(f.y)
    U = Product([Uniform(xmin, xmax), Uniform(ymin, ymax)])
    s = []
    while length(s) < n
        x0, y0 = rand(U, n)
        if y0 < f(x0)
            push!(s, x0)
        end
    end
    return Vector{Float64}(s)
end

rejection_sample(f::LinearInterpolator) = rejection_sample(f, 1)


"""
    DiscreteLinearInterpolator

A discrete linear interpolater between values `y` evaluated at gridpoints `x`.

# Arguments
- `x::Vector{Float64}`: a vector of gridpoints
- `y::Vector{Float64}`: a vector of function values
"""
struct DiscreteLinearInterpolator <: Interpolator
    x::Array{Int64,1}
    y::Array{Float64,2}
    I::Union{Float64,Missing}
end

DiscreteLinearInterpolator(x, y) = DiscreteLinearInterpolator(x, y, missing)

function interpolate(f::DiscreteLinearInterpolator, x0::Int64)
    """
        interpolate(f::DiscreteLinearInterpolator, x0)
    
    Evaluate the linear interpolation at a point `x0 ∈ [x[1], x[N]]`.
    """
    (x0 < f.x[1] || x0 > f.x[end]) && error("Value is outside function support")
    ndim = size(f.y)[2]
    y0 = zeros(ndim)
    for k in 1:ndim
        for i in 1:nsteps(f)
            if x0 >= f.x[i] && x0 < f.x[i+1]
                y0[k] = (f.y[i+1, k] * (x0 - f.x[i]) + f.y[i, k] * (f.x[i+1] - x0)) / (f.x[i+1] - f.x[i])
                break
            end
            y0[k] = f.y[end, k]
        end
    end
    return y0
end

function interpolate(f::DiscreteLinearInterpolator, xs)
    ndim = size(f.y)[2]
    nobs = length(xs)
    y0 = zeros(nobs, ndim)
    for t in 1:nobs
        x0 = xs[t]
        (x0 < f.x[1] || x0 > f.x[end]) && error("Value $x0 is outside function support")
        for k in 1:ndim
            for i in 1:nsteps(f)
                if x0 >= f.x[i] && x0 < f.x[i+1]
                    y0[t, k] = (f.y[i+1, k] * (x0 - f.x[i]) + f.y[i, k] * (f.x[i+1] - x0)) / (f.x[i+1] - f.x[i])
                    break
                end
                y0[t, k] = f.y[end, k]
            end
        end
    end
    return y0
end

(f::DiscreteLinearInterpolator)(x0) = interpolate(f, x0)

function integrate(f::DiscreteLinearInterpolator, dt)
    """Integrate (sum) the discrete interpolated function with step size `dt`."""
    return sum(f(collect(f.x[1]:dt:f.x[end])))
end