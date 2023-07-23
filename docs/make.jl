using Pkg
# Pkg.activate("..")
Pkg.activate(".")
using Documenter, NetworkHawkesProcesses
using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess

DocMeta.setdocmeta!(NetworkHawkesProcesses, :DocTestSetup, :(using NetworkHawkesProcesses; using NetworkHawkesProcesses: SquaredExponentialKernel, GaussianProcess); recursive=true)

makedocs(
    sitename="NetworkHawkesProcesses.jl",
    modules=[NetworkHawkesProcesses],
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ]
)

deploydocs(
    repo="github.com/cswaney/NetworkHawkesProcesses.jl.git",
)
