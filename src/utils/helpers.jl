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

function node_counts(nodes, nnodes)
    """Count the number of events on each node."""
    cnts = zeros(nnodes)
    for node in nodes
        cnts[node] += 1
    end
    return cnts
end

function node_counts(data::Matrix)
    """Count the number of events on each node data generated by a discrete process."""
    return sum(data, dims=2)
end