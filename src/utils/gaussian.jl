import Base.rand
import Statistics.cov

abstract type Kernel end

function (kernel::Kernel)(x, y) end

function (kernel::Kernel)(x::Vector)
    n = length(x)
    Σ = zeros(n, n)
    for i = 1:n
        for j = 1:i
            Σ[i, j] = kernel(x[i], x[j])
        end
    end

    return PDMat(posdef!(Symmetric(Σ, :L)))
end

struct SquaredExponentialKernel <: Kernel
    σ
    η
end

(kernel::SquaredExponentialKernel)(x, y) = kernel.σ^2 * exp(-((x - y) / kernel.η)^2 / 2)

struct OrnsteinUhlenbeckKernel <: Kernel
    σ
    η
end

(kernel::OrnsteinUhlenbeckKernel)(x, y) = kernel.σ^2 * exp(-abs(x - y) / kernel.η)

struct PeriodicKernel <: Kernel
    σ
    η
    θ # period
end

(kernel::PeriodicKernel)(x, y) = kernel.σ^2 * exp(-2 * (sin(π * abs(x - y) / kernel.θ) / kernel.η)^2)


"""
GaussianProcess

A 1-dimensional Gaussian process.

Defines a distribution over trajectories `f(x)` on a domain `[a, b]` such that:

    E[f(x)] = mu(x)
    cov[f(x), f(y)] = kernel(x, y)

for all `x, y ∈ [a, b]`. Drawing a discretized sample `f = [f(x[1]), ..., f(x[n])]` is equivalent to sampling a multivariate Gaussian:

    f ~ N(μ, Σ)
    μ[i] = mu(x[i])
    Σ[i, j] = kernel(x[i], x[j])

# Arguments
- `mu`: mean function
- `kernel`: covariance function
"""
struct GaussianProcess
    mu::Function
    kernel::Kernel
end

GaussianProcess(kernel) = GaussianProcess(zero, kernel)

cov(gp::GaussianProcess, x) = gp.kernel(x)

function rand(gp::GaussianProcess, x; sigma=nothing)
    """Sample a Gaussian process along grid points `x`. Supplying `sigma` skips covariance computation."""
    n = length(x)
    mu = zeros(n)
    for i = 1:n
        mu[i] = gp.mu(x[i])
    end
    if isnothing(sigma)
        sigma = cov(gp, x)
    end
    return rand(MvNormal(mu, sigma))
end
