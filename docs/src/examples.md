# Examples

This page shows the three main use cases for GPUEnv:

1. Create an overlay environment for test suites.
2. Create an overlay environment for GPU benchmarks.
3. Predict available backends and allocate arrays through a unified API.

## 1) Test-suite overlays

### Target-based test dependencies (`[extras]` / `[targets]`)

`Project.toml`:

```toml
name = "MyPkg"
uuid = "..."

[extras]
GPUEnv = "78a0b619-6146-4252-b244-0f81c54be577"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["GPUEnv", "Test"]
```

`test/runtests.jl`:

```julia
using GPUEnv
using Test

GPUEnv.activate(backends_to_test = [:JLArrays])

@testset "MyPkg" begin
    backend = first(gpu_backends())
    x = backend(randn(Float32, 32))
    @test size(x) == (32,)
end
```

### Separate `test/Project.toml` (no workspace)

`test/Project.toml`:

```toml
[deps]
GPUEnv = "78a0b619-6146-4252-b244-0f81c54be577"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```

`test/runtests.jl` is the same as above.

### Separate `test/Project.toml` with workspace

Root `Project.toml`:

```toml
[workspace]
projects = ["test"]
```

`test/Project.toml` and `test/runtests.jl` follow the same pattern as the
non-workspace test-project case.

## 2) Benchmark overlays

Run benchmarks from the benchmark environment and skip JLArrays by default:

```julia
using GPUEnv

GPUEnv.activate(include_jlarrays = false, only_first = true)

backends = gpu_backends(; include_jlarrays = false)
if isempty(backends)
    println("No functional native backend found; skipping benchmark run.")
else
    backend = first(backends)
    x = gpu_randn(backend, Float32, 1024)
    synchronize_backend(backend)
end
```

This keeps benchmark environments lightweight while still enabling available
native GPU backends.

## 3) Backend prediction + unified allocation

Downstream packages can suggest backend packages and allocate arrays through the
same API regardless of backend:

```julia
using GPUEnv

predicted = predict_backends()
if :CUDA in predicted && "CUDA" ∉ keys(Pkg.project().dependencies)
    @info "CUDA backend looks available, but CUDA.jl is not a declared dependency. Consider adding CUDA.jl to your Project.toml to enable it."
end

for backend in gpu_backends() # this will load CUDA if available
    x = gpu_zeros(backend, Float32, 64, 64)
    y = gpu_randn(backend, Float32, 64, 64)
    z = gpu_ones(backend, Float32, 64, 64)
    synchronize_backend(backend)
end
```

`gpu_zeros`, `gpu_ones`, and `gpu_randn` provide a small backend-agnostic layer
for downstream code that supports multiple GPU implementations.
