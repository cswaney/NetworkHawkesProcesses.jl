using Gadfly
palette = Gadfly.Scale.default_discrete_colors(6)[[1, 6, 4, 5]];

function plot(process::ContinuousHawkesProcess, data; start=0.0, stop=nothing, path=nothing)
    events, nodes, duration = data
    if !isnothing(stop)
        idx = filter(t -> events[t] < stop, 1:length(events))
        events = events[idx]
        nodes = nodes[idx]
    else
        stop = duration
    end
    ts = Base.range(start=start, stop=stop, length=10001)
    λ = transpose(hcat([intensity(process, data, t) for t in ts]...))
    layers = []
    for i in 1:size(λ)[2]
        append!(layers, layer(x=ts, y=λ[:, i], Geom.line, color=[i]))
        xs = [t for (t, c) in zip(events, nodes) if c == i]
        λs = [intensity(process, data, t)[i] for t in xs];
        append!(layers, layer(x=xs, y=λs, Geom.point, color=[i]))
    end
    p = Gadfly.plot(layers...,
        Scale.color_discrete_manual(palette...),
        Guide.colorkey(title=""),
        Guide.xlabel("t"),
        Guide.ylabel("λ(t)"),
        Coord.cartesian(xmin=0.0, xmax=stop, ymin=0.0)
    )
    if !isnothing(path)
        img = SVG(path, 8.5inch, 4inch)
        draw(img, p)
    else
        return p
    end
end

function plot(process::DiscreteHawkesProcess, data; start=1, stop=nothing, path=nothing)
    _, duration = size(data)
    stop = isnothing(stop) ? duration : stop
    ts = Base.range(start=start, stop=stop)
    λ = intensity(process, data)
    layers = []
    for i in 1:size(λ)[2]
        append!(layers, layer(x=ts, y=λ[start:stop, i], Geom.line, color=[i]))
        append!(layers, layer(x=ts, y=data[i, start:stop], Geom.point, color=[i]))
    end
    p = Gadfly.plot(layers...,
        Scale.color_discrete_manual(palette...),
        Guide.colorkey(title=""),
        Guide.xlabel("t"),
        Guide.ylabel("λ(t)"),
        Coord.cartesian(xmin=1, xmax=stop, ymin=0.0)
    )
    if !isnothing(path)
        img = SVG(path, 8.5inch, 4inch)
        draw(img, p)
    else
        return p
    end
end

function plot(impulses::ImpulseResponse, index; tmin=0.0, tmax=1.0, path=nothing)
    x = tmin:0.001:tmax
    λ = intensity(impulses)[index...]
    y = λ.(x)
    p = Gadfly.plot(x=x, y=y, Geom.line, Guide.xlabel("Δt"), Guide.ylabel("ħ(Δt)"))
    if !isnothing(path)
        img = SVG(path, 8.5inch, 4inch)
        draw(img, p)
    else
        return p
    end
end

function plot(impulses::DiscreteImpulseResponse, index; path=nothing)
    x = 1:impulses.nlags
    y = intensity(impulses)[index..., :]
    p = Gadfly.plot(
        x=x,
        y=y,
        Geom.point,
        Coord.cartesian(xmin=1, xmax=impulses.nlags, ymin=0.0),
        Guide.xlabel("Δt"), Guide.ylabel("ħ(Δt)"))
    if !isnothing(path)
        img = SVG(path, 8.5inch, 4inch)
        draw(img, p)
    else
        return p
    end
end
