abstract type Interpolator end

nnodes(f::Interpolator) = length(f.x)
nsteps(f::Interpolator) = length(f.x) - 1

function (f::Interpolator)(x) end


struct LinearInterpolator <: Interpolator
    """
        LinearInterpolator
    
    A linear interpolation between values `y` evaluated at grid points `x`.
    
    # Arguments
    - `x::Vector{Float64}`: a vector of grid points
    - `y::Vector{Float64}`: a vector of function values
    - `I::Union{Float64,Missing}`: the integral of the interpolation
    """
    x::Vector{Float64}
    y::Vector{Float64}
    I::Union{Float64,Missing}
end

LinearInterpolator(x, y) = LinearInterpolator(x, y, missing)

function interpolate(f::LinearInterpolator, x0)
    """Evaluate the linear interpolator at a point."""
    (x0 < f.x[1] || x0 > f.x[end]) && throw(DomainError(x0, "Value is outside interpolation support $((f.x[1], f.x[end]))"))
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
