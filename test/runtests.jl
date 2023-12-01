using LagrangeRelaxedShortestPaths
using JuliaFormatter
using Aqua
using JET
using Test

@testset verbose = true "LagrangeRelaxedShortestPaths.jl" begin

    # Code format
    @testset "Code formatting" begin
        @test format(LagrangeRelaxedShortestPaths; verbose=false, overwrite=false)
    end

    if VERSION >= v"1.9"
        @testset "Code quality" begin
            Aqua.test_all(LagrangeRelaxedShortestPaths; ambiguities=false)
        end

        @testset "Code linting" begin
            JET.test_package(LagrangeRelaxedShortestPaths; target_defined_modules=true)
        end
    end
end
