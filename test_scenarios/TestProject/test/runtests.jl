using GPUEnv
using Test

GPUEnv.activate(backends_to_test = [:JLArrays])

# After activate, JLArrays is loaded; query available backends
ts = @testset "TestProject" begin
    backends = GPUEnv.gpu_backends()
    @test length(backends) == 1 # Only JLArrays should be available
    @test backends[1].name == :JLArray

    # Smoke test: perform a simple operation and check results
    backend = backends[1]
    x_cpu = Float32.(1:10)
    x_gpu = backend(x_cpu)
    @test isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6)

    # Test if a dependency from parent project is available (JSON is a dependency of the root project, but not of this package)
    using JSON
    @test JSON.json(Dict("status" => "ok")) == "{\"status\":\"ok\"}"
end

ts.anynonpass && exit(1)
