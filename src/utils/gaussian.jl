import Base.rand
import Statistics.cov

abstract type Kernel end

function (kernel::Kernel)(x, y) end

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
            sigma[i, j] = gp.kernel(x[i], x[j])
        end
    end
    sigma = posdef!(Symmetric(sigma, :L))
    return rand(MvNormal(mu, sigma)), sigma
end
