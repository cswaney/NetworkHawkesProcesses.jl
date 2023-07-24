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

A univariate Gaussian process.

### Details
Defines a distribution over trajectories ``f(x)`` on domain ``[a, b]`` such that:

```math
\\begin{aligned}
\\mathbb{E} \\left[f(x)\\right] &= \\mu(x) \\\\
\\text{cov} \\left[f(x), f(y)\\right] &= K(x, y)
\\end{aligned}
```

for all ``x, y \\in [a, b]``.
    
Drawing a discretized sample ``f_{1:N}`` is equivalent to sampling a multivariate Gaussian:

```math
\\begin{aligned}
f &\\sim \\mathcal{N}(\\mu, \\Sigma) \\\\
\\mu_i &= \\mu(x_i) \\\\
\\Sigma_{i, j} &= K(x_i, x_j)
\\end{aligned}
```

### Arguments
- `mu`: a mean function ``x \\rightarrow \\mu(x)`` (default: `zero`).
- `kernel`: a covariance function ``x, y \\rightarrow K(x, y)``.

### Examples
```jldoctest
kernel = SquaredExponentialKernel(1.0, 1.0)
gp = GaussianProcess(kernel)

# output
GaussianProcess(zero, SquaredExponentialKernel(1.0, 1.0))
```

### References
- [https://en.wikipedia.org/wiki/Gaussian_process](https://en.wikipedia.org/wiki/Gaussian_process)
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
