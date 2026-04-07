using TestItems

@testmodule ScenarioHelpers begin
    using Pkg
    export activate_scenario, GPUEnv_root, scenarios_root

    const GPUEnv_root = normpath(joinpath(@__DIR__, ".."))
    const scenarios_root = joinpath(GPUEnv_root, "test_scenarios")

    function activate_scenario(name::AbstractString)
        root = joinpath(scenarios_root, name)
        Pkg.activate(root)
        Pkg.develop(path = GPUEnv_root)
    end
end

@testitem "TestTarget scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg

    activate_scenario("TestTarget")
    Pkg.test()
    @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
end

@testitem "TestProject scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg

    activate_scenario("TestProject")
    Pkg.test()
    @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
end

@testitem "TestWorkspace scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg

    if VERSION >= v"1.12"
        activate_scenario("TestWorkspace")
        Pkg.test()
        @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
    end
end

@testitem "Benchmarking scenario" setup = [ScenarioHelpers] begin
    using Test

    if VERSION >= v"1.12"
        activate_scenario("Benchmarking/benchmark")
        include(joinpath(scenarios_root, "Benchmarking", "benchmark", "benchmark.jl"))
        @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
    end
end

@testitem "DeeplyNestedWorkspace scenario" setup = [ScenarioHelpers] begin
    using Test, Pkg

    if VERSION >= v"1.12"
        @testset "DeeplyNestedWorkspace" begin
            activate_scenario("DeeplyNestedWorkspace")
            Pkg.test()
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        end
        @testset "DeeplyNestedWorkspace/SecondLevel" begin
            activate_scenario("DeeplyNestedWorkspace/SecondLevel")
            Pkg.test()
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        end
        @testset "DeeplyNestedWorkspace/SecondLevel/ThirdLevel" begin
            activate_scenario("DeeplyNestedWorkspace/SecondLevel/ThirdLevel")
            Pkg.test()
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        end
        @testset "DeeplyNestedWorkspace/SecondLevel2" begin
            activate_scenario("DeeplyNestedWorkspace/SecondLevel2")
            Pkg.test()
            @test true # Pkg.test() will throw an error if tests fail, so reaching this line means the test passed
        end
    end
end
