function posdef!(sigma; maxiter=3)
    iter = 0
    while iter < maxiter
        eps = 2 * minimum(eigvals(sigma))
        eps > 0.0 && return sigma
        sigma[diagind(sigma)] .-= eps
        iter += 1
    end
    println("WARNING: failed to make sigma positive definite")
    return sigma
end

function logsumexp(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function fillna!(X, value)
    for index in eachindex(X)
        if isnan(X[index])
            X[index] = value
        end
    end
    return X
end