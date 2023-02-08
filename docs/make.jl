using Pkg
# Pkg.activate("..")
Pkg.activate(".")
using Documenter, NetworkHawkesProcesses

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
