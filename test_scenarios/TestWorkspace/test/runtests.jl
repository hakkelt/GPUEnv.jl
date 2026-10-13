using GPUEnv
using Test

GPUEnv.activate()

@testset "TestWorkspace" begin
    backends = GPUEnv.gpu_backends()
    @test !isempty(backends)

    for backend in backends
        @testset "$(backend.name)" begin
            array_type = gpu_wrapper(backend, Float32)
            gpu_type = backend.array_type
            @test array_type === gpu_type

            x_cpu = Float32.(1:64)
            x_gpu = to_gpu(backend, x_cpu)
            @test isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6)

            x = gpu_allocate(backend, Float32, 1000)
            @test size(x) == (1000,)
            @test eltype(x) == Float32
            @test isa(x, gpu_type)

            x_zeros = gpu_zeros(backend, Float32, 1000)
            @test all(x_zeros .== 0)
            @test isa(x_zeros, gpu_type)

            x_ones = gpu_ones(backend, Float32, 1000)
            @test all(x_ones .== 1)
            @test isa(x_ones, gpu_type)

            x_randn = gpu_randn(backend, Float32, 1000)
            @test isa(x_randn, gpu_type)
        end
    end

    # Test if a dependency from parent project is available (JSON is a dependency of the root project, but not of this package)
    using JSON
    @test JSON.json(Dict("status" => "ok")) == "{\"status\":\"ok\"}"
end
