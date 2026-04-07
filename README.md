# GPUEnv

[![Julia CI](https://github.com/hakkelt/GPUEnv.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/hakkelt/GPUEnv.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/hakkelt/GPUEnv.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/GPUEnv.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://hakkelt.github.io/GPUEnv.jl/)
[![Runic](https://img.shields.io/badge/formatter-Runic-blue.svg)](https://github.com/fredrikekre/Runic.jl)

GPUEnv is a utility package for downstream Julia projects that support more
than one GPU backend but do not want to instantiate all of them all the time.
It builds a temporary or persisted overlay environment on top of the active
project, asks Pkg to resolve only the relevant backend packages, and leaves the
parent project itself unchanged.

The package is aimed at three concrete workflows:

1. Test overlays that should run with JLArrays and any real backends available on the host.
2. Benchmark overlays that should prefer real GPU backends and skip JLArrays.
3. Small backend-agnostic helpers such as `gpu_backends`, `gpu_zeros`, `gpu_ones`, and `gpu_randn`.

GPUEnv supports unnamed and temporary active projects and can also overlay an
explicit directory that contains a `Project.toml`.

## Why this package exists

Pkg already knows how to resolve, instantiate, and precompile environments, and
GPUEnv relies on those operations directly. The missing piece is conditional GPU
backend activation for downstream packages: Test and benchmark projects often should not depend permanently on every backend package because that would make the parent environment unnecessarily large and slow to resolve. Instead, they want to ask "which backends are actually worth trying on this machine?" and only add those to the test or benchmark environment. GPUEnv handles that overlay step, then hands the resulting environment back to
Pkg for the actual dependency resolution and installation work.

## Supported backends

`JLArrays`, `CUDA`, `AMDGPU`, `Metal`, `oneAPI`, `OpenCL`

Backend prediction prefers direct host hints first: Linux device nodes and PCI
vendor IDs, Windows video-controller names, and macOS display information.
Command-line tools such as `nvidia-smi`, `rocminfo`, `sycl-ls`, and `clinfo`
remain fallbacks.

## Use case 1: test overlays

This is the main use case. A downstream package can keep its ordinary test
environment small and ask GPUEnv to add JLArrays plus any real backends that
actually work on the current machine.

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

## Use case 2: benchmark overlays

Benchmarks usually should not include JLArrays because the CPU-side mock is not
representative. In that case, ask GPUEnv for native backends only and skip the
run when none is available.

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

## Use case 3: backend prediction and unified allocation

Downstream code can query the installed backends and allocate arrays through a
small common interface instead of branching on CUDA versus AMDGPU versus Metal
everywhere.

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

## Compat and failure behavior

GPUEnv preserves the source project's dependency information and only appends
missing backend dependencies. For nested projects it also merges parent or
workspace dependency tables so that `test/` and `benchmark/` environments can
still see the packages they rely on.

- Existing compat entries from the source, parent, or workspace project are kept.
- GPUEnv only inserts its own backend compat entry when the overlay does not already define one.
- If a backend package cannot be resolved or installed, that backend is skipped with a warning.
- If the base project itself is not resolvable, overlay creation fails because Pkg cannot instantiate an invalid environment.

That means GPUEnv is tolerant of optional backend failures, but it does not try
to hide real dependency problems in the parent environment.

## Persistence

By default the overlay environment is temporary. That is fine for one-off runs,
but repeated test runs might pay the full resolve and precompile cost every time.

Passing `persist = true` writes the overlay to `gpu_env/` under the overlay
root, records a compact snapshot of the source environment, and reuses the
overlay until the source project or manifest changes.

```julia
GPUEnv.activate(; include_jlarrays = true, persist = true)
```

If the persisted environment lives inside a Git repository, add `gpu_env/` to
`.gitignore`. GPUEnv warns when a persisted overlay inside a repository is not
ignored.

## API summary

- `activate(...)` synchronizes the overlay and leaves it active.
- `sync_test_env(...)` performs the same synchronization work but restores the previous environment before returning.
- `gpu_backends(...)` returns the functional backends available in the current session.
- `predict_backends(...)` returns the backends that look worth trying on this host before installation.
- `gpu_zeros`, `gpu_ones`, `gpu_randn`, and `gpu_wrapper` provide small backend-agnostic helpers.

## AI Usage Disclaimer

This README and the rest of the repository were generated with the assistance of AI tools.
The code is carefully reviewed and tested, but users should be aware that AI-generated content may contain inaccuracies or require adjustments to fit specific use cases.
