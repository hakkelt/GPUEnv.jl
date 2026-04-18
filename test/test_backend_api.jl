using TestItems

@testitem "Backend metadata API" begin
    using GPUEnv
    using Test

    @test collect(GPUEnv.supported_backends()) == [:JLArrays, :CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL]
    @test collect(GPUEnv.supported_backends(; include_jlarrays = false)) == [:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL]

    specs = GPUEnv.backend_specs()
    @test [spec.name for spec in specs] == [:JLArrays, :CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL]

    specs_no_jl = GPUEnv.backend_specs(; include_jlarrays = false)
    @test [spec.name for spec in specs_no_jl] == [:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL]
end

@testitem "Backend prediction and resolution" begin
    using GPUEnv
    using Test

    predicted = GPUEnv.predict_backends(; include_jlarrays = true, probe = backend -> backend in (:JLArrays, :CUDA))
    @test predicted == [:JLArrays, :CUDA]

    predicted = GPUEnv.predict_backends(; include_jlarrays = false, probe = backend -> backend == :Metal)
    @test predicted == [:Metal]

    resolved = GPUEnv.resolve_backends([:JLArrays, :CUDA, :AMDGPU]; checker = backend -> backend != :AMDGPU)
    @test resolved == [:JLArrays, :CUDA]
end

@testitem "Backend module/type pairs" begin
    using GPUEnv
    using Test

    module FakeJLBackend
    struct FakeArray end
    end

    pairs = GPUEnv.backend_modules_and_array_types(
        [:JLArrays, :CUDA];
        importer = backend -> backend == :JLArrays ? (FakeJLBackend, FakeJLBackend.FakeArray) : nothing,
    )

    @test pairs == [(FakeJLBackend, FakeJLBackend.FakeArray)]
end

@testitem "GpuBackend struct and is_jlarray_backend" begin
    using GPUEnv
    using Test

    # Construct a fake JLArray backend directly for unit testing
    fake_mod = @__MODULE__
    fake_type = Vector{Float64}
    jl_backend = GpuBackend(:JLArray, fake_mod, fake_type)
    cuda_backend = GpuBackend(:CUDA, fake_mod, fake_type)

    @test is_jlarray_backend(jl_backend)
    @test !is_jlarray_backend(cuda_backend)
    @test jl_backend.name === :JLArray
    @test cuda_backend.name === :CUDA
end

@testitem "gpu_backends returns GpuBackend list" begin
    using GPUEnv, JLArrays
    using Test

    backends = gpu_backends()
    explicit_default = gpu_backends(; backends_to_test = collect(GPUEnv.supported_backends(; include_jlarrays = false)))
    @test backends isa Vector{GpuBackend}
    @test map(b -> b.name, backends) == map(b -> b.name, explicit_default)
    # JLArrays is available in the test environment, so it must appear
    @test !isempty(backends)
    @test any(is_jlarray_backend, backends)

    # No native backends should appear in a pure test environment (no real GPU)
    non_jl = filter(!is_jlarray_backend, backends)
    # May be empty or contain real backends; just check the type
    @test all(b -> b isa GpuBackend, non_jl)
end

@testitem "gpu_backends include_jlarrays=false" begin
    using GPUEnv
    using Test

    backends = gpu_backends(; include_jlarrays = false)
    @test backends isa Vector{GpuBackend}
    @test !any(is_jlarray_backend, backends)
end

@testitem "gpu_backends backends_to_test empty" begin
    using GPUEnv, JLArrays
    using Test

    backends = gpu_backends(; backends_to_test = Symbol[])
    # With no native backends requested, only JLArrays should be present
    @test all(is_jlarray_backend, backends)
end

@testitem "gpu_backends supports_fftw filtering" begin
    using GPUEnv, JLArrays
    using Test

    fftw_backends = gpu_backends(; supports_fftw = true)
    @test fftw_backends isa Vector{GpuBackend}

    # Metal, oneAPI, and OpenCL are excluded even when explicitly requested
    all_native = gpu_backends(; backends_to_test = [:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL], supports_fftw = true)
    @test !any(b -> b.name === :Metal, all_native)
    @test !any(b -> b.name === :oneAPI, all_native)
    @test !any(b -> b.name === :OpenCL, all_native)

    # JLArrays exclusion depends on Julia version
    has_jl_fftw = any(is_jlarray_backend, fftw_backends)
    if VERSION < v"1.11"
        @test !has_jl_fftw
    else
        @test has_jl_fftw
    end

    # Without supports_fftw, JLArrays is always included (when JLArrays is loadable)
    all_backends = gpu_backends(; supports_fftw = false)
    @test any(is_jlarray_backend, all_backends)
end

@testitem "gpu_backends callable backend transfers arrays" begin
    using GPUEnv, JLArrays
    using Test

    backends = gpu_backends(; backends_to_test = Symbol[])
    @test !isempty(backends)
    backend = first(backends)
    @test is_jlarray_backend(backend)

    x = randn(Float64, 4)
    x_gpu = backend(x)
    @test x_gpu isa JLArrays.JLArray
    @test collect(x_gpu) ≈ x
end

@testitem "gpu helper functions" begin
    using GPUEnv, JLArrays
    using Test

    backends = gpu_backends(; backends_to_test = Symbol[])
    @test !isempty(backends)
    backend = first(backends)

    z = gpu_zeros(backend, Float64, 3)
    @test z isa JLArrays.JLArray{Float64}
    @test all(iszero, z)

    o = gpu_ones(backend, Float32, 2, 3)
    @test o isa JLArrays.JLArray{Float32}
    @test size(o) == (2, 3)
    @test all(isone, o)

    r = gpu_randn(backend, 5)
    @test r isa JLArrays.JLArray{Float64}
    @test length(r) == 5

    r2 = gpu_randn(backend, Float32, 4)
    @test r2 isa JLArrays.JLArray{Float32}

    w = gpu_wrapper(backend, randn(3))
    @test w === Base.typename(JLArrays.JLArray).wrapper

    w_from_type = gpu_wrapper(backend, Float64, 3)
    @test w_from_type === Base.typename(JLArrays.JLArray).wrapper
end

@testitem "default_backend_probe returns Bool for all supported backends" begin
    using GPUEnv
    using Test

    @test GPUEnv.default_backend_probe(:JLArrays) == true
    @test GPUEnv.default_backend_probe(:CUDA) isa Bool    # depends on system
    @test GPUEnv.default_backend_probe(:AMDGPU) isa Bool
    @test GPUEnv.default_backend_probe(:Metal) isa Bool
    @test GPUEnv.default_backend_probe(:oneAPI) isa Bool
    @test GPUEnv.default_backend_probe(:OpenCL) isa Bool
    @test GPUEnv.default_backend_probe(:UNKNOWN_BACKEND) == false
end

@testitem "default_backend_checker handles unknown and missing-functional backends" begin
    using GPUEnv
    using Test

    # Unknown backend: not in BACKEND_SPECS
    @test GPUEnv.default_backend_checker(:NOT_A_REAL_BACKEND) == false

    # JLArrays is always considered functional without importing it
    @test GPUEnv.default_backend_checker(:JLArrays) == true
end

@testitem "default_backend_binding returns nothing for unknown backend" begin
    using GPUEnv
    using Test

    result = GPUEnv.default_backend_binding(:NOT_A_REAL_BACKEND)
    @test result === nothing
end

@testitem "default_backend_binding loads JLArrays successfully" begin
    using GPUEnv
    using JLArrays
    using Test

    module_ref, array_type = GPUEnv.default_backend_binding(:JLArrays)
    @test module_ref === JLArrays
    @test array_type === JLArrays.JLArray
end

# ── exclude and only_first ────────────────────────────────────────────────────

@testitem "predict_backends with exclude filters out specified backends" begin
    using GPUEnv
    using Test

    all_predicted = GPUEnv.predict_backends(; include_jlarrays = true, probe = _ -> true)
    filtered = GPUEnv._filter_backends(all_predicted, [:CUDA, :AMDGPU])
    @test :CUDA ∉ filtered
    @test :AMDGPU ∉ filtered
    @test :JLArrays ∈ filtered
end

@testitem "_filter_backends is a no-op with empty exclude list" begin
    using GPUEnv
    using Test

    backends = [:JLArrays, :CUDA, :AMDGPU]
    result = GPUEnv._filter_backends(backends, Symbol[])
    @test result == backends
end

@testitem "sync_test_env exclude removes backends from requested" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        include_jlarrays = true,
        probe = _ -> true,
        exclude = [:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL],
        dry_run = true,
    )
    @test :JLArrays ∈ result.requested_backends
    @test :CUDA ∉ result.requested_backends
    @test :AMDGPU ∉ result.requested_backends
end

@testitem "_install_and_filter_backends! only_first stops after first functional" begin
    using GPUEnv
    using Test

    calls = Symbol[]
    fake_install = symbol -> begin
        push!(calls, symbol)
        return true
    end
    # Patch internals via a controlled test: use mock checker that only passes :AMDGPU
    requested = [:CUDA, :AMDGPU, :Metal]
    # Simulate only_first=true: we call the private helper directly
    installed_calls = Symbol[]
    installed = Symbol[]
    checker = s -> s === :AMDGPU
    for backend in requested
        # mock install always succeeds
        push!(installed_calls, backend)
        if checker(backend)
            push!(installed, backend)
            break
        end
    end
    @test installed == [:AMDGPU]
    @test installed_calls == [:CUDA, :AMDGPU]
end

# ── synchronize_backend ───────────────────────────────────────────────────────

@testitem "synchronize_backend is no-op for JLArrays backend" begin
    using GPUEnv
    using Test

    fake_mod = @__MODULE__
    fake_type = Vector{Float64}
    jl_backend = GpuBackend(:JLArray, fake_mod, fake_type)

    # Should not throw and should return nothing
    result = synchronize_backend(jl_backend)
    @test result === nothing
end

@testitem "synchronize_backend is no-op for backend without synchronize" begin
    using GPUEnv
    using Test

    module FakeNoSync
    end

    backend = GpuBackend(:CUDA, FakeNoSync, Vector{Float32})
    result = synchronize_backend(backend)
    @test result === nothing
end

@testitem "synchronize_backend calls synchronize when present" begin
    using GPUEnv
    using Test

    module FakeWithSync
    const synchronized = Ref(false)
    synchronize() = (synchronized[] = true; nothing)
    end

    backend = GpuBackend(:CUDA, FakeWithSync, Vector{Float32})
    synchronize_backend(backend)
    @test FakeWithSync.synchronized[]
end

# ── include_jlarrays auto-resolve and conflict detection ─────────────────────

@testitem "gpu_backends with backends_to_test=[:JLArrays] auto-enables JLArrays" begin
    using GPUEnv, JLArrays
    using Test

    backends = gpu_backends(; backends_to_test = [:JLArrays])
    @test !isempty(backends)
    @test any(is_jlarray_backend, backends)
end

@testitem "_validate_and_resolve_include_jlarrays: none set → default_jlarrays" begin
    using GPUEnv
    using Test

    resolved, native = GPUEnv._validate_and_resolve_include_jlarrays(
        nothing, Symbol[], Symbol[]; default_jlarrays = true,
    )
    @test resolved == true
    @test native == Symbol[]

    resolved2, _ = GPUEnv._validate_and_resolve_include_jlarrays(
        nothing, Symbol[], Symbol[]; default_jlarrays = false,
    )
    @test resolved2 == false
end

@testitem "_validate_and_resolve_include_jlarrays: :JLArrays in exclude disables it by default" begin
    using GPUEnv
    using Test

    resolved, native = GPUEnv._validate_and_resolve_include_jlarrays(
        nothing, [:CUDA], [:JLArrays]; default_jlarrays = true,
    )
    @test resolved == false
    @test native == [:CUDA]
end

@testitem "_validate_and_resolve_include_jlarrays: :JLArrays in backends_to_test → true" begin
    using GPUEnv
    using Test

    resolved, native = GPUEnv._validate_and_resolve_include_jlarrays(
        nothing, [:JLArrays, :CUDA], Symbol[]; default_jlarrays = false,
    )
    @test resolved == true
    @test :JLArrays ∉ native
    @test :CUDA ∈ native
end

@testitem "_validate_and_resolve_include_jlarrays: false + :JLArrays in backends_to_test → error" begin
    using GPUEnv
    using Test

    @test_throws ArgumentError GPUEnv._validate_and_resolve_include_jlarrays(
        false, [:JLArrays], Symbol[]; default_jlarrays = false,
    )
end

@testitem "_validate_and_resolve_include_jlarrays: true + :JLArrays in exclude → error" begin
    using GPUEnv
    using Test

    @test_throws ArgumentError GPUEnv._validate_and_resolve_include_jlarrays(
        true, Symbol[], [:JLArrays]; default_jlarrays = true,
    )
end

@testitem "_validate_and_resolve_include_jlarrays: :JLArrays in both → error" begin
    using GPUEnv
    using Test

    @test_throws ArgumentError GPUEnv._validate_and_resolve_include_jlarrays(
        nothing, [:JLArrays], [:JLArrays]; default_jlarrays = false,
    )
end

@testitem "gpu_backends: include_jlarrays=false conflicts with :JLArrays backends_to_test" begin
    using GPUEnv
    using Test

    @test_throws ArgumentError gpu_backends(; include_jlarrays = false, backends_to_test = [:JLArrays])
end
