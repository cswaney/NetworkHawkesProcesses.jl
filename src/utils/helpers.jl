function posdef!(sigma)
    eps = 2 * minimum(eigvals(sigma))
    eps > 0.0 && return sigma
    idx = diagind(sigma)
    sigma[idx] .-= eps
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