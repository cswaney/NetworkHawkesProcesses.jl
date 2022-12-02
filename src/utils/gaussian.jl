import Base.rand
import Statistics.cov

abstract type Kernel end

struct SquaredExponentialKernel <: Kernel
    σ
    η
end

function cov(kernel::SquaredExponentialKernel, x, y)
    return kernel.σ^2 * exp(-((x - y) / kernel.η)^2 / 2)
end

struct OrnsteinUhlenbeckKernel <: Kernel
    σ
    η
end

function cov(kernel::OrnsteinUhlenbeckKernel, x, y)
    return kernel.σ^2 * exp(-abs(x - y) / kernel.η)
end

struct PeriodicKernel <: Kernel
    σ
    η
    θ # period
end

function cov(kernel::PeriodicKernel, x, y)
    kernel.σ^2 * exp(-2 * (sin(π * abs(x - y) / kernel.θ) / kernel.η)^2)
end


"""
GaussianProcess

A 1-dimensional Gaussian process.

# Arguments
- `mu`: mean function
- `kernel`: covariance function
"""
struct GaussianProcess
    mu # x -> mu(x)
    kernel::Kernel
end

GaussianProcess(kernel) = GaussianProcess(zero, kernel)

function rand(gp::GaussianProcess, x)
    """Sample a Gaussian process along grid points `x`."""
    n = length(x)
    mu = zeros(n)
    for i = 1:n
        mu[i] = gp.mu(x[i])
    end
    sigma = zeros(n, n)
    for i = 1:n
        for j = 1:i
            xi = x[i]
            xj = x[j]
            sigma[i, j] = cov(gp.kernel, xi, xj)
        end
    end
    sigma = posdef!(Symmetric(sigma, :L))
    return rand(MvNormal(mu, sigma)), sigma
end
