using NetworkHawkesProcesses
using Statistics

# Continuous Hawkes Processes

## Standard Hawkes Process

### Exponential
nnodes = 2;
duration = 1000.0;
baseline = HomogeneousProcess([1.0, 2.0]);
weights = DenseWeightModel([0.1 0.2; 0.2 0.1]);
impulses = ExponentialImpulseResponse(ones(nnodes, nnodes));
process = ContinuousStandardHawkesProcess(baseline, impulses, weights);
isstable(process)
data = rand(process, duration);
NetworkHawkesProcesses.plot(process, data; stop=10.0, path="/Users/colinswaney/GitHub/NetworkHawkesProcesses.jl/docs/src/assets/img/continuous-exponential-data.svg")
res = mle!(process, data; verbose=true, regularize=true);
NetworkHawkesProcesses.params(process)
ll = loglikelihood(process, data)
NetworkHawkesProcesses.plot(process.impulses, (1, 1), tmax=3.0)

### Logit-Normal
Δtmax = 1.0;
baseline = HomogeneousProcess([1.0, 2.0]);
weights = DenseWeightModel([0.1 0.2; 0.2 0.1]);
impulses = LogitNormalImpulseResponse(ones(nnodes, nnodes), ones(nnodes, nnodes), Δtmax);
process = ContinuousStandardHawkesProcess(baseline, impulses, weights);
isstable(process)
data = rand(process, duration);
NetworkHawkesProcesses.plot(process, data; stop=10.0)
res = mle!(process, data; verbose=true, regularize=true);
NetworkHawkesProcesses.params(process)
ll = loglikelihood(process, data)
NetworkHawkesProcesses.plot(process.impulses, (1, 1), tmax=Δtmax)


## Network Hawkes Process
baseline = HomogeneousProcess([1.0, 2.0]);
weights = DenseWeightModel([0.1 0.2; 0.2 0.1]);
impulses = LogitNormalImpulseResponse(ones(nnodes, nnodes), ones(nnodes, nnodes), Δtmax);
network = NetworkHawkesProcesses.BernoulliNetworkModel(0.5, nnodes);
links = [1 0; 1 1]; # links = NetworkHawkesProcesses.rand(network)
process = NetworkHawkesProcesses.ContinuousNetworkHawkesProcess(baseline, impulses, weights, links, network);
isstable(process)
data = rand(process, duration);
NetworkHawkesProcesses.plot(process, data; stop=10.0, path="/Users/colinswaney/GitHub/NetworkHawkesProcesses.jl/docs/src/assets/img/continuous-logit-normal-data.svg")
res = NetworkHawkesProcesses.mcmc!(process, data; verbose=true)
mean(res.samples) # NOTE: some of the impulse response parameters seem off... links okay?
ll = NetworkHawkesProcesses.loglikelihood(process, data)
NetworkHawkesProcesses.plot(process.impulses, (1, 1), tmax=Δtmax)


# TODO
# 1. Compare mcmc! results w/ and w/o network


# Discrete Hawkes Processes
nnodes = 2;
nbasis = 3;
nlags = 4;
duration = 1000;
dt = 1.0;
baseline = DiscreteHomogeneousProcess([1.0, 2.0], dt);
weights = DenseWeightModel([0.1 0.2; 0.2 0.1]);
impulses = DiscreteGaussianImpulseResponse(ones(nnodes, nnodes, nbasis) ./ nbasis, nlags, dt);
process = DiscreteStandardHawkesProcess(baseline, impulses, weights, dt);
isstable(process)
data = rand(process, duration);
NetworkHawkesProcesses.plot(process, data; stop=100, path="/Users/colinswaney/GitHub/NetworkHawkesProcesses.jl/docs/src/assets/img/discrete-data.svg")
res = mle!(process, data; verbose=true, regularize=false);
NetworkHawkesProcesses.params(process)
# res = mcmc!(process, data; verbose=true);
# mean(res.samples)
# res = vb!(process, data; verbose=true, regularize=false);
ll = loglikelihood(process, data)
NetworkHawkesProcesses.plot(process.impulses, (1, 1))


# TODO
# 1. Compare mcmc! and vb! results w/ and w/o network













layers = []
D = process.impulses.nlags
for (b, ϕ) in enumerate(NetworkHawkesProcesses.basis(process.impulses))
    push!(layers, layer(x=1:D, y=ϕ, Geom.point, color=[b]))
end
λ = intensity(process.impulses)[1, 1, :]
push!(layers, layer(x=1:D, y=λ, Geom.line))
labels = string.(round.(process.impulses.θ[1, 1, :], digits=2))
p = plot(layers...,
    Guide.colorkey(title="θ[b]", labels=labels),
    Scale.color_discrete,
    Guide.xlabel("d"),
    Guide.ylabel("ħ[d]"),
    Coord.cartesian(xmin=1, xmax=D, ymin=0.0)
)
img = SVG("/Users/colinswaney/GitHub/NetworkHawkesProcesses.jl/docs/src/assets/img/discrete-gaussian-impulse-response.svg", 8.5inch, 4inch)
draw(img, p)