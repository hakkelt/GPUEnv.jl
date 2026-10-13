# GPUEnv.jl

GPUEnv is a utility for packages that support multiple GPU backends but do not
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

Run [`report_backends`](@ref) to see the hardware hints, driver libraries, installed
packages, functional checks, `Float64` support and FFT support for every backend
on the current host.

## Test overlays

This is the primary workflow. In tests you usually want JLArrays for portable
coverage and real GPU backends whenever they are available.

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

## Benchmark overlays

Benchmarks should usually skip CPU backends (the JLArrays mock and OpenCL on a
CPU device such as PoCL) and use the first functional GPU backend, or skip
cleanly when none is available.

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

## Backend-agnostic helpers

Downstream packages can use the same small helper layer across CUDA, AMDGPU,
Metal, oneAPI, OpenCL, and JLArrays.

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

## Active project overlay

Call `activate()` without arguments to overlay the currently active project
with available GPU backends. Works for unnamed and temporary projects too:

```julia
using GPUEnv

GPUEnv.activate()   # overlays the active project
backend = first(gpu_backends())
x = gpu_randn(backend, Float32, 128)
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
backends_fftw = gpu_backends(; supports_fftw = true)   # only AbstractFFTs-capable, with Float64
backends_f64 = gpu_backends(; require_float64 = true)  # only devices that support Float64
backends_jl = gpu_backends(; include_cpu_backends = true)  # also include JLArrays mock

for backend in backends
    x = gpu_randn(backend, Float32, 128, 128)
    @show backend.name, typeof(x)
    synchronize_backend(backend)   # no-op for backends without synchronize()
end
```

`backends_to_test` narrows the backend probe/query set. Use it when you want
to restrict to a specific subset such as CUDA and AMDGPU only.

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

## CPU backends

Two backends can run on the CPU: the JLArrays mock, and OpenCL when its only
working device is a CPU, as with the PoCL runtime. They are useful in CI without
GPU hardware, but not for benchmarks. `include_cpu_backends` controls both: with
`include_cpu_backends = false`, JLArrays is left out and OpenCL counts as
functional only on a non-CPU device.

OpenCL can offer several devices, for example a CPU runtime next to a GPU
driver. GPUEnv tries GPU devices first, then other non-CPU devices, and CPU
devices last (only when CPU backends are allowed), and makes the first device
that runs a test kernel the current OpenCL device. OpenCL.jl keeps the current
device per task: GPUEnv selects it again whenever its own functions (`to_gpu`,
`gpu_zeros`, `synchronize_backend`, ...) reach OpenCL, but code that calls
OpenCL directly in a newly spawned task should select it with `cl.device!`.

```julia
GPUEnv.activate(; include_cpu_backends = false)  # GPU backends only
```

`include_cpu_backends` replaces the `include_jlarrays` keyword of GPUEnv 0.2.

## Persistent environments

See [Persistence](@ref) for details on why persisting the environment speeds up
repeated test runs and how to configure it.

