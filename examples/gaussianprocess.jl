using Gadfly
using NetworkHawkesProcesses
using NetworkHawkesProcesses: SquaredExponentialKernel, OrnsteinUhlenbeckKernel, PeriodicKernel, cov

kern = SquaredExponentialKernel(1., 1.)
xs = -3.:0.01:3.
ys = [cov(kern, 0., x) for x in xs]
plot(x=xs, y=ys, Geom.line)

kern = OrnsteinUhlenbeckKernel(1., 1.)
xs = -3.:0.01:3.
ys = [cov(kern, 0., x) for x in xs]
plot(x=xs, y=ys, Geom.line)

kern = PeriodicKernel(1., 1., 1.)
xs = -3.:0.01:3
ys = [cov(kern, 0., x) for x in xs]
plot(x=xs, y=ys, Geom.line)
