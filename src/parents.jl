function resample_parents(process::ContinuousHawkesProcess, data)
    """Resample latent parent variables given model parameters and event data."""
    events, nodes, _ = data
    parents = Vector{Int64}(undef, length(events))
    parentnodes = Vector{Int64}(undef, length(events))
    if Threads.nthreads() > 1
        @debug "using multi-threaded parent sampler"
        Threads.@threads for index in eachindex(events)
            event = events[index]
            node = nodes[index]
            parent, parentnode = resample_parent(process, event, node, index, events, nodes)
            parents[index] = parent
            parentnodes[index] = parentnode
        end
    else
        for (index, (event, node)) in enumerate(zip(events, nodes))
            parent, parentnode = resample_parent(process, event, node, index, events, nodes)
            parents[index] = parent
            parentnodes[index] = parentnode
        end
    end
    return parents, parentnodes
end

function resample_parent(process::ContinuousHawkesProcess, event, node, index, events, nodes)
    if index == 1
        return 0, 0
    end
    λs = []
    parentindices = []
    parentindex = index - 1
    while events[parentindex] > event - process.impulses.Δtmax
        parenttime = events[parentindex]
        parentnode = nodes[parentindex]
        append!(λs, impulse_response(process, parentnode, node, event - parenttime))
        append!(parentindices, parentindex)
        parentindex -= 1
        parentindex == 0 && break
    end
    append!(λs, intensity(process.baseline, node, event))
    append!(parentindices, 0)
    index = rand(Categorical(λs ./ sum(λs)))
    parent = parentindices[index]
    parentnode = parent > 0 ? nodes[parent] : 0
    return parent, parentnode
end

function get_parentnodes(nodes, parents)
    """Look-up the node each parent occurred on. Returns zero for parent events with value zero (i.e., the "parent event" is the baseline process)."""
    return Vector{Int64}([get_parentnode(nodes, p) for p in parents])
end

function get_parentnode(nodes, parent)
    if parent == 0
        return 0
    else
        return nodes[parent]
    end
end

function node_counts(nodes, nnodes)
    """Count the number of events on each node."""
    cnts = zeros(nnodes)
    for node in nodes
        cnts[node] += 1
    end
    return cnts
end

function parent_counts(nodes, parentnodes, nnodes)
    """Count events on each node attributed to a parent on each other node."""
    cnts = zeros(nnodes, nnodes)
    for (node, parentnode) in zip(nodes, parentnodes)
        if parentnode > 0  # 0 => background event
            cnts[parentnode, node] += 1
        end
    end
    return cnts
end


function resample_parents(process::DiscreteHawkesProcess, data, convolved)
    """Resample latent parent variables given model parameters, event data, and pre-computed convolutions."""
    T, N, B = size(convolved)
    parents = zeros(Int64, T, N, 1 + N * B)
    # λ0 = intensity(p.baseline, 1:T)
    if Threads.nthreads() > 1
        @debug "using multi-threaded parent sampler"
        Threads.@threads for idx in eachindex(data)
            childnode, t = cartesian(idx, size(data))
            parents[t, childnode, :] = resample_parent(process, childnode, t, data, convolved)
        end
    else
        for t = 1:T
            for childnode = 1:N
                parents[t, childnode, :] = resample_parent(process, childnode, t, data, convolved)
            end
        end
    end
    return parents
end

function resample_parent(process::DiscreteHawkesProcess, node, time, data, convolved)
    nevents = data[node, time]
    λ0 = intensity(process.baseline, node, time)
    _, N, B = size(convolved)
    λ = zeros(B, N)
    for parentnode = 1:N
        for b = 1:B
            shat = convolved[time, parentnode, b]
            λ[b, parentnode] = shat * bump(process, parentnode, node, b)
        end
    end
    μ = [λ0, vec(λ)...]
    μ ./= sum(μ)
    return rand(Multinomial(nevents, μ))
end

function node_counts(data::Matrix)
    """Count the number of events on each node data generated by a discrete process."""
    return sum(data, dims=2)
end

function parent_counts(parents::Array{Int64,3}, ndims, nbasis)
    counts = zeros(ndims, ndims)
    for parentchannel = 1:ndims
        for childchannel = 1:ndims
            start = 1 + (parentchannel - 1) * nbasis + 1
            stop = start + nbasis - 1
            counts[parentchannel, childchannel] = sum(parents[:, childchannel, start:stop])
        end
    end
    return counts
end

function update_parents(p::DiscreteHawkesProcess, convolved::Array{Float64,3}, α, β, κ0, ν0, κ1, ν1, γ, ρ)
    """Perform variational inference update on auxillary parent variables ("local context"). The required variational parameters are α, β, κ, ν, and γ."""
    T, N, B = size(convolved)
    is_mixture = !isa(p.network, DenseNetwork)
    u = zeros(T, N, 1 + N * B)
    if Threads.nthreads() > 1
        @debug "using multi-threaded parent updater"
        Threads.@threads for tidx = 1:T
            u[tidx, cidx, 1] = update_parent_kernel(0, cidx, nothing, α, β, κ0, ν0, κ1, ν1, γ, ρ, is_mixture)
            for pidx = 1:N
                start = 1 + (pidx - 1) * B + 1
                stop = start + B - 1
                shat = convolved[tidx, pidx, :]
                u[tidx, cidx, start:stop] = update_parent_kernel(pidx, cidx, shat, α, β, κ0, ν0, κ1, ν1, γ, ρ, is_mixture)
            end
        end
    else
        for tidx = 1:T
            for cidx = 1:N
                u[tidx, cidx, 1] = update_parent_kernel(0, cidx, nothing, α, β, κ0, ν0, κ1, ν1, γ, ρ, is_mixture)
                for pidx = 1:N
                    start = 1 + (pidx - 1) * B + 1
                    stop = start + B - 1
                    shat = convolved[tidx, pidx, :]
                    u[tidx, cidx, start:stop] = update_parent_kernel(pidx, cidx, shat, α, β, κ0, ν0, κ1, ν1, γ, ρ, is_mixture)
                end
            end
        end
    end
    Z = sum(u, dims=3)
    return u ./ Z
end

function update_parent(updater, pidx, cidx, shat, α, β, κ0, ν0, κ1, ν1, γ, ρ, is_mixture=false)
    # 
    if pidx == 0
        # Elogλ0 = digamma(α[cidx]) - log(β[cidx])
        # return exp(Elogλ0)
        return expectation(updater.baseline, cidx)
    elseif is_mixture
        # Elogθ = digamma.(γ[pidx, cidx, :]) .- digamma(sum(γ[pidx, cidx, :]))
        # ElogW0 = digamma(κ0[pidx, cidx]) - log(ν0[pidx, cidx])
        # ElogW1 = digamma(κ1[pidx, cidx]) - log(ν1[pidx, cidx])
        # ElogW = (1 - ρ[pidx, cidx]) * ElogW0 + ρ[pidx, cidx] * ElogW1
        # return shat .* exp.(Elogθ .+ ElogW)
        Elogθ = expectation(updater.impulses, pidx, cidx)
        ElogW = expectation(updater.weights, pidx, cidx, false) # κ0, ν0
        return shat .* exp.(Elogθ .+ ElogW)
    else
        # Elogθ = digamma.(γ[pidx, cidx, :]) .- digamma(sum(γ[pidx, cidx, :]))
        # ElogW = digamma(κ1[pidx, cidx]) - log(ν1[pidx, cidx])
        # return shat .* exp.(Elogθ .+ ElogW)
        Elogθ = expectation(update.impulses, pidx, cidx)
        ElogW = expectation(updater.weights, pidx, cidx, true) # κ1, ν1
        return shat .* exp.(Elogθ .+ ElogW)
    end
end