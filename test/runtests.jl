using NetworkHawkesProcesses
using Test
using Random
using Statistics

Random.seed!(0)

@testset verbose = true "NetworkHawkesProcesses.jl" begin

    @testset "baselines" begin
        include("baselines.jl")
    end

    @testset "impulse responses" begin
        include("impulses.jl")
    end

    @testset "weights" begin
        include("weights.jl")
    end

    @testset "gaussian" begin
        include("gaussian.jl")
    end

    @testset "interpolation" begin
        include("interpolation.jl")
    end

    # @static if VERSION == v"1.6"
    #     using Documenter
    #     @testset "Docs" begin
    #         DocMeta.setdocmeta!(Flux, :DocTestSetup, :(using Flux); recursive=true)
    #         doctest(Flux)
    #     end
    # end

end