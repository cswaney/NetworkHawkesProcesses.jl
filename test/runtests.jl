using NetworkHawkesProcesses
using Test
using Random
using Statistics

Random.seed!(0)

@testset verbose = true "NetworkHawkesProcesses.jl" begin

    @testset "Baselines" begin
        include("baselines.jl")
    end

    # @static if VERSION == v"1.6"
    #     using Documenter
    #     @testset "Docs" begin
    #         DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
    #         doctest(Flux)
    #     end
    # end

end