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
vendor IDs, and Apple silicon on macOS (Metal.jl does not support Intel Macs).
When those are inconclusive, and always on Windows, it looks for the vendor
driver library file on the loader search path (`libcuda`/`nvcuda`,
`libamdhip64`, `libze_loader`) without loading it, so no vendor command-line
tools are needed. OpenCL is predicted from hardware hints only, because its ICD
loader is installed on many machines without a GPU.

Not every device supports `Float64`: Metal never does, and some Intel GPUs
(oneAPI) and OpenCL devices do not either. `supports_float64(backend)` asks the
device, and `gpu_backends(; require_float64 = true)` keeps only the backends
that do.

Backends with an AbstractFFTs implementation (`gpu_backends(; supports_fftw = true)`)
are CUDA, AMDGPU, Metal (Metal.jl 1.10 or newer) and JLArrays on Julia 1.11 or
newer. `require_float64` defaults to `supports_fftw`, so this returns the
backends that transform `Float64` data like FFTW does; Metal, whose FFTs take
`Float32`/`Float16` data only, is added with
`gpu_backends(; supports_fftw = true, require_float64 = false)`.

Run `report_backends()` to see the hardware hints, driver libraries, installed
packages, functional checks, `Float64` support and FFT support for every backend
on the current host.

## Use case 1: test overlays

This is the main use case. A downstream package can keep its ordinary test
environment small and ask GPUEnv to add JLArrays plus any real backends that
actually work on the current machine.

```julia
# test/runtests.jl
using GPUEnv
using Test

GPUEnv.activate(; include_cpu_backends = true, persist = true)

for backend in gpu_backends(; include_cpu_backends = true)
    x = gpu_ones(backend, Float32, 64, 64)
    y = gpu_ones(backend, Float32, 64, 64) .* 2
    @test Array(x + y) == 3f0 .* ones(64, 64)
end
```

This pattern is useful for packages that want CPU-only CI coverage via JLArrays
while still exercising CUDA, AMDGPU, Metal, oneAPI, or OpenCL when those are
available.

## Use case 2: benchmark overlays

Benchmarks usually should not include CPU backends: the JLArrays mock and
OpenCL on a CPU device (such as the PoCL runtime) are not representative. In
that case, set `include_cpu_backends = false` to get GPU backends only, and skip
the run when none is available.

```julia
# benchmark/gpu_benchmark.jl
using GPUEnv
using BenchmarkTools

GPUEnv.activate(; include_cpu_backends = false, only_first = true)

backends = gpu_backends(; include_cpu_backends = false)
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

for backend in gpu_backends(; include_cpu_backends = true)
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
overlay until the source project or manifest changes. A backend that fails its
functional check is removed from the overlay with a warning; later runs add back
and re-check only the backends that are missing, so an overlay first built on a
machine without a working GPU picks up the GPU once one is available.

```julia
GPUEnv.activate(; include_cpu_backends = true, persist = true)
```

If the persisted environment lives inside a Git repository, add `gpu_env/` to
`.gitignore`. GPUEnv warns when a persisted overlay inside a repository is not
ignored.

## GPUSelect.jl as an alternative

[GPUSelect.jl](https://github.com/SimonDanisch/GPUSelect.jl) solves a related
problem, and may be the better fit when you write a script or an application
rather than a package test suite. The two differ in what they set up:

- **One backend versus all of them.** GPUSelect picks the single backend that
  matches the hardware and returns its KernelAbstractions backend (`Backend()`)
  or array type (`Storage()`). GPUEnv gives you every functional backend on the
  host, plus JLArrays by default, so the same tests run on each of them.
- **Where the backend is installed.** GPUSelect's `auto_install!()` adds the
  backend to the active project once, and `@load_backend()` or a preference
  picks it in later sessions. GPUEnv never edits your `Project.toml`; it builds
  a temporary or persisted overlay environment on top of it and resolves the
  backends there.
- **Verification.** GPUEnv installs the predicted backends, runs each one's
  functional check, and drops those that fail (for example CUDA on a node
  with the driver but no GPU). GPUSelect trusts the detected driver and falls
  back to the CPU when loading fails.
- **Backends.** GPUSelect supports CUDA, AMDGPU, Metal, oneAPI and the Vulkan
  backend Lava.jl, with KernelAbstractions' `CPU()` as fallback. GPUEnv supports
  CUDA, AMDGPU, Metal, oneAPI, OpenCL and JLArrays, and can filter backends by
  FFT support. Neither needs vendor command-line tools.
- **Dependencies.** GPUSelect depends on KernelAbstractions, Adapt and
  Preferences and works with KernelAbstractions backends. GPUEnv depends only on
  standard libraries and works with array types.

GPUEnv's driver-library detection and `report_backends` are adapted from
GPUSelect.jl's `find_gpu_pkgid` and `test_backends` (MIT License, Copyright (c)
2026 Simon Danisch, ReynKo, and contributors; see `THIRD_PARTY_NOTICES.md`).

## Upgrading from 0.2

- `include_jlarrays` is now `include_cpu_backends` (also the `SyncResult`
  field). Setting it to `false` also drops OpenCL when it has no working
  non-CPU device. OpenCL now prefers a GPU device over a CPU device whatever
  the setting.
- `gpu_backends(; supports_fftw = true)` keeps returning only backends that
  transform `Float64` data; Metal needs `require_float64 = false`.
- A persisted overlay retries a failed backend only when the Julia version, the
  GPUEnv version or the backend's latest release changes; pass
  `retry_failed = true` to force it.

## API summary

- `activate(...)` synchronizes the overlay and leaves it active.
- `sync_test_env(...)` performs the same synchronization work but restores the previous environment before returning.
- `gpu_backends(...)` returns the functional backends available in the current session.
- `predict_backends(...)` returns the backends that look worth trying on this host before installation.
- `report_backends()` prints what GPUEnv detects on this host and in the active environment.
- `supports_float64(backend)` tells whether the backend's device supports `Float64`.
- `gpu_zeros`, `gpu_ones`, `gpu_randn`, and `gpu_wrapper` provide small backend-agnostic helpers.

## AI Usage Disclaimer

This README and the rest of the repository were generated with the assistance of AI tools.
The code is carefully reviewed and tested, but users should be aware that AI-generated content may contain inaccuracies or require adjustments to fit specific use cases.
