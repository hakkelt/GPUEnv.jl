# GPUEnv

GPUEnv is for downstream packages that support multiple GPU backends but do not
want to instantiate all of them up front. It creates a GPU overlay environment
on top of the active project, lets Pkg resolve the optional backend packages
that make sense on the current machine, and keeps the parent environment small.

The package is built around three workflows:

1. Test overlays with JLArrays plus any real backends available on the host.
2. Benchmark overlays that use native backends only.
3. Backend prediction and a tiny common allocation layer.

GPUEnv supports unnamed and temporary active projects and can also overlay an
explicit directory that contains a `Project.toml`.

## What `activate()` function does?

1. Reads the active project (or the explicit `path/Project.toml` when `path` is given).
2. Creates a temporary or persisted GPU overlay environment.
3. Predicts which GPU backends are worth attempting on the current host.
4. Installs those backends and keeps only the ones whose runtime checks pass.

## Supported backends

| Symbol     | Package   | Array type  |
|:-----------|:----------|:------------|
| `:JLArrays`| JLArrays  | `JLArray`   |
| `:CUDA`    | CUDA      | `CuArray`   |
| `:AMDGPU`  | AMDGPU    | `ROCArray`  |
| `:Metal`   | Metal     | `MtlArray`  |
| `:oneAPI`  | oneAPI    | `oneArray`  |
| `:OpenCL`  | OpenCL    | `CLArray`   |

Backend prediction prefers direct host hints such as Linux device nodes and
PCI vendor IDs, Windows video-controller names, and macOS display metadata.
Command-line tools like `nvidia-smi`, `rocminfo`, `sycl-ls`, and `clinfo`
remain fallback probes.

## Test overlays

This is the primary workflow. In tests you usually want JLArrays for portable
coverage and real GPU backends whenever they are available.

```julia
# test/runtests.jl
using GPUEnv
using Test

GPUEnv.activate(; include_jlarrays = true, persist = true)

for backend in gpu_backends(; include_jlarrays = true)
    x = gpu_ones(backend, Float32, 64, 64)
    y = gpu_ones(backend, Float32, 64, 64) .* 2
    @test Array(x + y) == 3f0 .* ones(64, 64)
end
```

This pattern is useful for packages that want CPU-only CI coverage via JLArrays
while still exercising CUDA, AMDGPU, Metal, oneAPI, or OpenCL when those are
available.

## Benchmark overlays

Benchmarks should usually skip JLArrays and use the first functional native
backend, or skip cleanly when no real GPU backend is available.

```julia
# benchmark/gpu_benchmark.jl
using GPUEnv
using BenchmarkTools

GPUEnv.activate(; include_jlarrays = false, only_first = true)

backends = gpu_backends(; include_jlarrays = false)
if isempty(backends)
    println("No functional native GPU backend found; skipping benchmark run.")
else
    backend = first(backends)
    x = gpu_randn(backend, Float32, 1024)
    y = gpu_randn(backend, Float32, 1024)
    @btime begin
        $x .+ $y
        synchronize_backend($backend)
    end
end
```

The benchmark project only needs `GPUEnv` and the package being benchmarked.
GPUEnv arranges the optional GPU backend packages at runtime.

## Backend-agnostic helpers

Downstream packages can use the same small helper layer across CUDA, AMDGPU,
Metal, oneAPI, OpenCL, and JLArrays.

```julia
using GPUEnv

predicted = predict_backends()
@show predicted

for backend in gpu_backends(; include_jlarrays = true)
    x = gpu_zeros(backend, Float32, 64, 64)
    y = gpu_ones(backend, Float32, 64, 64)
    z = gpu_randn(backend, Float32, 64, 64)
    @show backend.name typeof(z)
end
```

## Active project overlay

Call `activate()` without arguments to overlay the currently active project
with available GPU backends. Works for unnamed and temporary projects too:

```julia
using GPUEnv

GPUEnv.activate()   # overlays the active project
backend = first(gpu_backends())
x = backend(randn(Float32, 128))
```

## Compat and dependency resolution

GPUEnv reuses Pkg for actual resolution, instantiation, and precompilation. The
custom logic is mainly in constructing the overlay project:

- parent and workspace dependency tables are merged into nested `test/` or `benchmark/` projects when needed;
- existing compat entries are preserved;
- backend compat is only added when the overlay does not already define it;
- optional backend installation failures are treated as skippable;
- base-project resolution failures still surface as real errors.

## Narrowing backends

```julia
# Skip specific backends
GPUEnv.activate(; exclude = [:OpenCL, :Metal])

# Install only the first functional backend found
GPUEnv.activate(; only_first = true)
```

## Working with backends after activation

```jldoctest
julia> using GPUEnv

julia> fake_loader = backend -> backend == :JLArrays ? (Main, Int) : nothing;

julia> GPUEnv.backend_modules_and_array_types([:JLArrays, :CUDA]; importer = fake_loader)
1-element Vector{Tuple{Module, DataType}}:
 (Main, Int64)
```

Query and use the available backends after `activate` has run:

```julia
backends = gpu_backends()                               # all functional native backends
backends_fftw = gpu_backends(; supports_fftw = true)   # only FFTW-compatible
backends_jl = gpu_backends(; include_jlarrays = true)  # also include JLArrays mock

for backend in backends
    x = gpu_randn(backend, Float32, 128, 128)
    @show backend.name, typeof(x)
    synchronize_backend(backend)   # no-op for backends without synchronize()
end
```

`backends_to_test` narrows the backend probe/query set. Use it when you want
to restrict to a specific subset such as CUDA and AMDGPU only.

## Why not only Pkg?

Pkg already provides the heavy lifting, and GPUEnv intentionally relies on it.
What Pkg does not provide by itself is a way for downstream packages to say
"take this active test or benchmark project, add only the GPU backends that are
useful here, and keep the original project clean." GPUEnv is that thin overlay
layer.

## sync_test_env versus activate

Use `sync_test_env` when you want a [`SyncResult`](@ref) describing the GPU
environment but do not want to leave that environment active. Use `activate`
when the synchronized GPU environment should stay active for the rest of the
current Julia session.

## JLArrays opt-in

JLArrays is disabled by default. Enable it explicitly when a CPU-side GPU mock
is needed (e.g., in pure CI environments without real GPU hardware):

```julia
GPUEnv.activate(; include_jlarrays = true)
```

## Persistent environments

See [Persistence](@ref) for details on why persisting the environment speeds up
repeated test runs and how to configure it.

