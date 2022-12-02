using Pkg
Pkg.activate("..")
using Documenter, NetworkHawkesProcesses

makedocs(
    sitename="NetworkHawkesProcesses.jl",
    modules=[NetworkHawkesProcesses],
    pages=[
        "index.md",
        "tutorial.md",
        "examples.md",
        "api.md",
    ]
)

# deploydocs(
#     repo = "github.com/cswaney/NetworkHawkesProcesses.jl.git",
# )
