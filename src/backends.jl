"""Metadata describing a supported GPU test backend.

Each backend records its package name, UUID, module name, and primary array type.
"""
struct BackendSpec
    name::Symbol
    package::String
    uuid::String
    module_name::Symbol
    array_type_name::Symbol
end

const BACKEND_SPECS = Dict(
    :JLArrays => BackendSpec(:JLArrays, "JLArrays", "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb", :JLArrays, :JLArray),
    :CUDA => BackendSpec(:CUDA, "CUDA", "052768ef-5323-5732-b1bb-66c8b64840ba", :CUDA, :CuArray),
    :AMDGPU => BackendSpec(:AMDGPU, "AMDGPU", "21141c5a-9bdb-4563-92ae-f87d6854732e", :AMDGPU, :ROCArray),
    :Metal => BackendSpec(:Metal, "Metal", "dde4c033-4e86-420c-a63e-0dd931031962", :Metal, :MtlArray),
    :oneAPI => BackendSpec(:oneAPI, "oneAPI", "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b", :oneAPI, :oneArray),
    :OpenCL => BackendSpec(:OpenCL, "OpenCL", "08131aa3-fb12-5dee-8b74-c09406e224a2", :OpenCL, :CLArray),
)

const NATIVE_BACKENDS = (:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL)

# Native backends that implement the AbstractFFTs interface (`plan_fft` etc.)
# for their array type, mapped to the first package version that does so.
# Metal gained it in v1.10 (MPSGraph, Float32/Float16 only); oneAPI and OpenCL
# do not provide it.
const FFTW_NATIVE_BACKENDS = Dict(
    :CUDA => v"0",
    :AMDGPU => v"0",
    :Metal => v"1.10",
)

"""A resolved GPU test backend with its loaded module and primary array type.

Use [`gpu_backends`](@ref) to obtain a list of `GpuBackend` instances for the
current host. A `GpuBackend` is callable: `backend(array)` transfers `array`
to the backend's device and returns the corresponding GPU array.

The `name` field holds the backend identifier. For JLArrays it is `:JLArray`
(the array-type name); for native backends it is the module name (for example
`:CUDA`). Use [`is_jlarray_backend`](@ref) to distinguish JLArrays from native
backends.
"""
struct GpuBackend
    name::Symbol
    module_ref::Module
    array_type
end

"""
    is_jlarray_backend(backend::GpuBackend) -> Bool

Return whether `backend` is the JLArrays (CPU-side GPU mock) backend.

# Arguments
- `backend::GpuBackend`: the backend to check
"""
is_jlarray_backend(backend::GpuBackend) = backend.name === :JLArray

"""
    synchronize_backend(backend::GpuBackend)

Wait for all pending GPU operations on `backend` to complete.

Delegates to `backend.module_ref.synchronize()` when the backend exposes a
`synchronize` function. Backends that do not provide this symbol are a silent
no-op, so this function is safe to call unconditionally in device-agnostic code.

# Arguments
- `backend::GpuBackend`: the GPU backend to synchronize
"""
function synchronize_backend(backend::GpuBackend)
    is_jlarray_backend(backend) && return nothing
    if backend.name == :OpenCL
        barrier_func = getfield(backend.module_ref, :barrier)
        fence_const = getfield(backend.module_ref, :CLK_LOCAL_MEM_FENCE)
        Base.invokelatest(barrier_func, fence_const)
        return nothing
    else
        isdefined(backend.module_ref, :synchronize) || return nothing
        Base.invokelatest(getfield(backend.module_ref, :synchronize))
        return nothing
    end
end

"""
    to_gpu(backend::GpuBackend, array)

Transfer `array` to the device represented by `backend`.

For the JLArrays backend, this calls `JLArrays.jl(array)`. For native backends,
it calls `backend.array_type(array)` directly.

# Arguments
- `backend::GpuBackend`: the GPU backend to use for device transfer
- `array`: the CPU array to transfer
"""
function to_gpu(backend::GpuBackend, array)
    if is_jlarray_backend(backend)
        jl_fn = getfield(backend.module_ref, :jl)
        return Base.invokelatest(jl_fn, array)
    end
    return backend.array_type(array)
end

"""A summary of a synchronized GPU test environment."""
struct SyncResult
    environment_path::String
    project_path::String
    source_project_path::String
    requested_backends::Vector{Symbol}
    installed_backends::Vector{Symbol}
    functional_backends::Vector{Symbol}
    include_jlarrays::Bool
    persisted::Bool
    base_environment_kind::Symbol
    warned_about_gitignore::Bool
    dry_run::Bool
end

function Base.show(io::IO, result::SyncResult)
    return print(
        io,
        "SyncResult(env=",
        result.environment_path,
        ", backends=",
        result.functional_backends,
        ", persisted=",
        result.persisted,
        ", dry_run=",
        result.dry_run,
        ")",
    )
end

"""
    supported_backends(; include_jlarrays=true)

Return the backend symbols supported by this package.

# Arguments
- `include_jlarrays::Bool=true`: whether to include the JLArrays mock backend

# Examples
```jldoctest
julia> using GPUEnv

julia> GPUEnv.supported_backends()
(:JLArrays, :CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL)

julia> GPUEnv.supported_backends(; include_jlarrays = false)
(:CUDA, :AMDGPU, :Metal, :oneAPI, :OpenCL)
```
"""
supported_backends(; include_jlarrays::Bool = true) =
    include_jlarrays ? (:JLArrays, NATIVE_BACKENDS...) : NATIVE_BACKENDS

"""
    backend_specs(; include_jlarrays=true)

Return metadata entries for supported backends.

# Arguments
- `include_jlarrays::Bool=true`: whether to include metadata for the JLArrays backend

# Examples
```jldoctest
julia> using GPUEnv

julia> [spec.name for spec in GPUEnv.backend_specs()]
6-element Vector{Symbol}:
 :JLArrays
 :CUDA
 :AMDGPU
 :Metal
 :oneAPI
 :OpenCL
```
"""
function backend_specs(; include_jlarrays::Bool = true)
    return [BACKEND_SPECS[backend] for backend in supported_backends(; include_jlarrays)]
end

"""
    predict_backends(; include_jlarrays=true, probe=default_backend_probe,
                     backends_to_test=collect(NATIVE_BACKENDS))

Predict which GPU backends should be attempted on this host.

`probe` is a pure function that receives a backend symbol and returns `true`
when the backend should be installed. The default probe uses direct host hints
first and falls back to command-based checks without importing GPU packages.

# Arguments
- `include_jlarrays::Bool=true`: whether to include JLArrays in the prediction
- `probe::Function=default_backend_probe`: function to determine whether each backend should be tested
- `backends_to_test::AbstractVector{Symbol}`: which native backends to probe (default: all supported native backends)

# Examples
```jldoctest
julia> using GPUEnv

julia> GPUEnv.predict_backends(; include_jlarrays = false, probe = backend -> backend == :CUDA)
1-element Vector{Symbol}:
 :CUDA
```
"""
function predict_backends(
        ;
        include_jlarrays::Bool = true,
        probe::Function = default_backend_probe,
        backends_to_test::AbstractVector{Symbol} = collect(NATIVE_BACKENDS),
    )
    predicted = Symbol[]
    include_jlarrays && push!(predicted, :JLArrays)
    hardware = probe === default_backend_probe ? detect_gpu_hardware() : nothing
    for backend in backends_to_test
        should_install = hardware === nothing ? probe(backend) : probe(backend; hardware)
        should_install && push!(predicted, backend)
    end
    return predicted
end

"""
    backend_modules_and_array_types(backends=collect(supported_backends());
                                    importer=default_backend_binding)

Return backend module and array-type pairs for `backends`.

The default behavior attempts to import each backend module and fetch its array
type from that module. Backends that cannot be loaded are skipped.

# Arguments
- `backends::AbstractVector{Symbol}`: the backends to load (default: all supported backends)
- `importer::Function=default_backend_binding`: function to load each backend module and extract its array type

# Examples
```jldoctest
julia> using GPUEnv

julia> fake_loader = backend -> backend == :JLArrays ? (Main, Int) : nothing;

julia> GPUEnv.backend_modules_and_array_types([:JLArrays, :CUDA]; importer = fake_loader)
1-element Vector{Tuple{Module, DataType}}:
 (Main, Int64)
```
"""
function backend_modules_and_array_types(
        backends::AbstractVector{Symbol} = collect(supported_backends());
        importer::Function = default_backend_binding,
    )
    bindings = Tuple{Module, DataType}[]
    for backend in backends
        binding = importer(backend)
        binding === nothing && continue
        push!(bindings, binding)
    end
    return bindings
end

"""
    resolve_backends(backends; checker=default_backend_checker)

Filter `backends` by runtime availability.

`checker` is a pure function that receives a backend symbol and returns `true`
when the installed backend is usable.

# Arguments
- `backends::AbstractVector{Symbol}`: the backends to check for runtime availability
- `checker::Function=default_backend_checker`: function to test each backend's runtime availability
"""
function resolve_backends(backends::AbstractVector{Symbol}; checker::Function = default_backend_checker)
    resolved = Symbol[]
    for backend in backends
        checker(backend) && push!(resolved, backend)
    end
    return resolved
end

# Validate the relationship between include_jlarrays, backends_to_test, and exclude.
# Returns (resolved_include_jlarrays::Bool, native_backends::Vector{Symbol}) where
# native_backends has :JLArrays stripped (it is handled via include_jlarrays).
# `default_jlarrays` is the fallback when include_jlarrays is nothing and no
# :JLArrays appears in backends_to_test.
function _validate_and_resolve_include_jlarrays(
        include_jlarrays::Union{Nothing, Bool},
        backends_to_test::AbstractVector{Symbol},
        exclude::AbstractVector{Symbol};
        default_jlarrays::Bool,
    )
    jl_in_backends = :JLArrays ∈ backends_to_test
    jl_in_exclude = :JLArrays ∈ exclude

    if jl_in_backends && jl_in_exclude
        throw(
            ArgumentError(
                ":JLArrays appears in both `backends_to_test` and `exclude`; " *
                    "remove it from one of them",
            )
        )
    end

    if include_jlarrays === false && jl_in_backends
        throw(
            ArgumentError(
                ":JLArrays is listed in `backends_to_test` but `include_jlarrays = false`; " *
                    "either remove :JLArrays from `backends_to_test` or set `include_jlarrays = true`",
            )
        )
    end

    if include_jlarrays === true && jl_in_exclude
        throw(
            ArgumentError(
                ":JLArrays is in `exclude` but `include_jlarrays = true`; " *
                    "either remove :JLArrays from `exclude` or set `include_jlarrays = false`",
            )
        )
    end

    resolved::Bool = if include_jlarrays !== nothing
        include_jlarrays
    elseif jl_in_backends
        true
    elseif jl_in_exclude
        false
    else
        default_jlarrays
    end

    native_backends = filter(!=(:JLArrays), collect(backends_to_test))

    # Validate that every requested backend is either :JLArrays (handled above)
    # or a recognised native backend.
    for b in native_backends
        if b ∉ NATIVE_BACKENDS
            known = join(vcat([:JLArrays], collect(NATIVE_BACKENDS)), ", ")
            throw(
                ArgumentError(
                    "backend :$b is not recognised; known backends are: $known",
                )
            )
        end
    end

    return resolved, native_backends
end

"""
    gpu_backends(; include_jlarrays=nothing,

Return [`GpuBackend`](@ref) instances available on the current host.

`backends_to_test` restricts which native GPU backends are probed (default:
all native backends supported by GPUEnv). Pass an explicit list such as
`[:CUDA, :AMDGPU]` if you want to narrow the query set. Set
`supports_fftw = true` to include only backends that support FFTW-based
operators, that is, backends implementing the AbstractFFTs interface for their
array type: JLArrays requires Julia 1.11+, Metal requires Metal.jl 1.10+ (and
transforms only `Float32`/`Float16` data, real or complex), and oneAPI / OpenCL
are always excluded because they do not provide an FFT backend.

# Arguments
- `include_jlarrays::Union{Nothing,Bool}=nothing`: whether to include the JLArrays mock backend.
  When `nothing` (default), behaves as `true` — JLArrays is included whenever it is loaded.
  Set to `true` automatically when `:JLArrays` appears in `backends_to_test`.
  Providing an explicit value that contradicts `backends_to_test` throws an `ArgumentError`.
- `backends_to_test::AbstractVector{Symbol}`: which native backends to probe (default: all supported)
- `supports_fftw::Bool=false`: if true, only return backends supporting FFTW-based operations

# Examples
```jldoctest
julia> using GPUEnv

julia> backends = GPUEnv.gpu_backends(; backends_to_test = Symbol[]);

julia> eltype(backends) <: GPUEnv.GpuBackend
true

julia> fftw_backends = GPUEnv.gpu_backends(; backends_to_test = Symbol[], supports_fftw = true);

julia> typeof(fftw_backends) <: AbstractVector
true
```
"""
function gpu_backends(
        ;
        include_jlarrays::Union{Nothing, Bool} = nothing,
        backends_to_test::AbstractVector{Symbol} = collect(NATIVE_BACKENDS),
        supports_fftw::Bool = false,
    )
    # gpu_backends defaults to include_jlarrays=true (historic default).
    # _validate_and_resolve_include_jlarrays with default_jlarrays=true implements this.
    resolved_jlarrays, native_backends = _validate_and_resolve_include_jlarrays(
        include_jlarrays, backends_to_test, Symbol[]; default_jlarrays = true,
    )

    backends = GpuBackend[]

    if resolved_jlarrays && (!supports_fftw || VERSION >= v"1.11")
        binding = _loaded_binding(:JLArrays)
        if binding !== nothing
            push!(backends, GpuBackend(:JLArray, binding...))
        end
    end

    fftw_native = supports_fftw ? filter(b -> haskey(FFTW_NATIVE_BACKENDS, b), native_backends) : native_backends
    for backend in resolve_backends(collect(fftw_native))
        binding = _loaded_binding(backend)
        binding === nothing && continue
        supports_fftw && !_supports_fftw(backend, binding[1]) && continue
        push!(backends, GpuBackend(nameof(binding[1]), binding...))
    end

    return backends
end

# Whether the loaded `module_ref` of `backend` implements AbstractFFTs for its arrays.
function _supports_fftw(backend::Symbol, module_ref::Module)
    backend === :JLArrays && return VERSION >= v"1.11"
    min_version = get(FFTW_NATIVE_BACKENDS, backend, nothing)
    min_version === nothing && return false
    version = pkgversion(module_ref)
    return version === nothing || version >= min_version
end

"""
    gpu_allocate(backend::GpuBackend, T, dims...)

Allocate an uninitialized array of type `T` and dimensions `dims` on `backend`.
Backends that do not expose their own uninitialized constructor fall back to
creating the CPU array first and then transferring it with `backend(array)`.

# Arguments
- `backend::GpuBackend`: the GPU backend to use for allocation
- `T::Type`: the element type for the array
- `dims...`: dimensions of the array
"""
function gpu_allocate(backend::GpuBackend, ::Type{T}, dims...) where {T}
    return _backend_allocate(backend, :undef, T, dims...; fallback = () -> to_gpu(backend, Array{T}(undef, dims...)))
end

"""
    gpu_zeros(backend::GpuBackend, T, dims...)

Return a zero-filled array on `backend`.

Backends that do not expose their own `zeros` constructor fall back to creating
the CPU array first and then transferring it with `backend(array)`.

# Arguments
- `backend::GpuBackend`: the GPU backend to use
- `T::Type`: the element type for the array
- `dims...`: dimensions of the array
"""
function gpu_zeros(backend::GpuBackend, ::Type{T}, dims...) where {T}
    return _backend_allocate(backend, :zeros, T, dims...; fallback = () -> to_gpu(backend, zeros(T, dims...)))
end

"""
    gpu_ones(backend::GpuBackend, T, dims...)

Return a one-filled array on `backend`.

# Arguments
- `backend::GpuBackend`: the GPU backend to use
- `T::Type`: the element type for the array
- `dims...`: dimensions of the array
"""
function gpu_ones(backend::GpuBackend, ::Type{T}, dims...) where {T}
    return _backend_allocate(backend, :ones, T, dims...; fallback = () -> to_gpu(backend, ones(T, dims...)))
end

"""
    gpu_randn(backend::GpuBackend, dims...)

Return a GPU array of random normal values with `Float64` elements.

# Arguments
- `backend::GpuBackend`: the GPU backend to use
- `dims...`: dimensions of the array
"""
gpu_randn(backend::GpuBackend, dims...) = gpu_randn(backend, Float64, dims...)

"""
    gpu_randn(backend::GpuBackend, T, dims...)

Return a GPU array of random normal values with element type `T`.

# Arguments
- `backend::GpuBackend`: the GPU backend to use
- `T::Type`: the element type for the array
- `dims...`: dimensions of the array
"""
function gpu_randn(backend::GpuBackend, ::Type{T}, dims...) where {T}
    return _backend_allocate(backend, :randn, T, dims...; fallback = () -> to_gpu(backend, randn(T, dims...)))
end

"""
    gpu_wrapper(backend::GpuBackend, array)

Return the parametric wrapper type used by `backend` for `array`.

This is useful for APIs that expect a storage wrapper such as `CuArray` or
`JLArray` rather than a concrete array type like `CuArray{Float32, 2}`.

# Arguments
- `backend::GpuBackend`: the GPU backend to query
- `array`: a representative array to determine the wrapper type from

# Examples
```julia
using GPUEnv

backends = gpu_backends()
isempty(backends) && error("No functional backend found on this host")

backend = first(backends)
wrapper = gpu_wrapper(backend, Float32, 64, 64)
```
"""
gpu_wrapper(backend::GpuBackend, array) = Base.typename(typeof(to_gpu(backend, array))).wrapper

"""
    gpu_wrapper(backend::GpuBackend, T, dims...)

Return the parametric wrapper type used by `backend` for `T` and `dims`.

This overload is convenient when you need to pass an `array_type` keyword to an
API without first constructing a CPU array for transfer.

# Arguments
- `backend::GpuBackend`: the GPU backend to query
- `T::Type`: the element type
- `dims...`: dimensions that would be used
"""
gpu_wrapper(backend::GpuBackend, ::Type{T}, dims...) where {T} =
    Base.typename(typeof(gpu_zeros(backend, T, dims...))).wrapper

"""
    default_backend_probe(backend; hardware=detect_gpu_hardware(),
                          fallback=_driver_library_probe) -> Bool

Return whether `backend` is likely available on the current host.

This first checks direct host-level hardware hints and, when they are
inconclusive, looks for the backend's driver library (for example `libcuda`,
`libamdhip64` or `libze_loader`). It does not import any GPU packages and does
not need vendor command-line tools such as `nvidia-smi`.

# Arguments
- `backend::Symbol`: the backend name to check (`:CUDA`, `:AMDGPU`, etc.)
- `hardware::AbstractDict{Symbol, Bool}`: cached hardware detection results
- `fallback::Function`: function to use when hardware hints are not conclusive
"""
function default_backend_probe(
        backend::Symbol;
        hardware::AbstractDict{Symbol, Bool} = detect_gpu_hardware(),
        fallback::Function = _driver_library_probe,
    )
    backend === :JLArrays && return true
    haskey(BACKEND_SPECS, backend) || return false

    get(hardware, backend, false) && return true
    return fallback(backend)
end

"""Return direct host-level hints for native GPU backends.

This uses cheap hardware signals: `/dev` nodes and sysfs PCI vendor IDs on
Linux, the CPU architecture on macOS (Metal.jl supports Apple silicon only),
and OpenCL ICD paths. Windows has no such hints; there the driver-library
fallback of `default_backend_probe` decides. It intentionally avoids
importing GPU packages.
"""
function detect_gpu_hardware(
        ;
        os::Symbol = _host_os(),
        arch::Symbol = Sys.ARCH,
        ispath::Function = ispath,
        read_text::Function = _read_text,
        linux_vendor_paths::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    )
    detected = Dict(backend => false for backend in NATIVE_BACKENDS)

    if os === :linux
        vendor_paths = linux_vendor_paths === nothing ? _linux_vendor_paths() : collect(String.(linux_vendor_paths))
        _detect_linux_gpu_hardware!(detected; ispath, read_text, vendor_paths)
    elseif os === :macos
        _detect_macos_gpu_hardware!(detected; ispath, arch)
    end

    return detected
end

function _loaded_binding(backend::Symbol)
    spec = get(BACKEND_SPECS, backend, nothing)
    spec === nothing && return nothing
    pkg_id = Base.PkgId(Base.UUID(spec.uuid), spec.package)
    mod = get(Base.loaded_modules, pkg_id, nothing)
    mod === nothing && return nothing
    isdefined(mod, spec.array_type_name) || return nothing
    return (mod, getfield(mod, spec.array_type_name))
end


function _backend_allocate(backend::GpuBackend, method_name::Symbol, args...; fallback::Function)
    if !is_jlarray_backend(backend) && isdefined(backend.module_ref, method_name)
        method = getfield(backend.module_ref, method_name)
        try
            return Base.invokelatest(method, args...)
        catch err
            @debug "Backend allocation method failed; falling back to CPU transfer" backend = backend.name method = method_name exception = (err, catch_backtrace())
        end
    end

    return fallback()
end

const GPU_VENDOR_BACKENDS = Dict(
    "0x8086" => :oneAPI,
    "0x1002" => :AMDGPU,
    "0x10de" => :CUDA,
)
const OPENCL_VENDOR_IDS = Set(keys(GPU_VENDOR_BACKENDS))
const OPENCL_LINUX_HINT_PATHS = (
    "/etc/OpenCL/vendors",
    "/usr/share/OpenCL/vendors",
    "/dev/dri/renderD128",
)
const OPENCL_MACOS_HINT_PATHS = (
    "/System/Library/Frameworks/OpenCL.framework",
    "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL",
)

# Driver libraries each backend loads at runtime, as `(unix names, windows names)`.
# Finding one means the vendor driver is installed, without running any vendor
# tool. OpenCL uses the ICD loader. Metal is absent because the macOS
# architecture check is conclusive.
#
# The driver-library probing (this table and `_driver_library_probe`), the Apple
# silicon check for Metal and `report_backends` are adapted from GPUSelect.jl
# (`find_gpu_pkgid` and `test_backends`), https://github.com/SimonDanisch/GPUSelect.jl,
# Copyright (c) 2026 Simon Danisch, ReynKo, and contributors, MIT License.
# See THIRD_PARTY_NOTICES.md for the full notice.
const BACKEND_DRIVER_LIBRARIES = Dict(
    :CUDA => (["libcuda", "libcuda.so.1"], ["nvcuda"]),
    :AMDGPU => (["libamdhip64", "libhip_hcc"], ["amdhip64", "amdhip64_6", "amdhip64_7"]),
    :oneAPI => (["libze_loader", "libze_loader.so.1"], ["ze_loader"]),
    :OpenCL => (["libOpenCL", "libOpenCL.so.1"], ["OpenCL"]),
)

_read_text(path::AbstractString) = read(path, String)

function _host_os()
    if Sys.islinux()
        return :linux
    elseif Sys.iswindows()
        return :windows
    elseif Sys.isapple()
        return :macos
    end
    return :other
end

function _linux_vendor_paths()
    drm_root = "/sys/class/drm"
    isdir(drm_root) || return String[]

    vendor_paths = String[]
    for entry in readdir(drm_root)
        occursin(r"^card\d+$", entry) || continue
        vendor_path = joinpath(drm_root, entry, "device", "vendor")
        isfile(vendor_path) && push!(vendor_paths, vendor_path)
    end

    return sort!(unique(vendor_paths))
end

function _normalize_vendor_id(vendor_id::AbstractString)
    normalized = lowercase(strip(vendor_id))
    return startswith(normalized, "0x") ? normalized : "0x" * normalized
end

function _linux_vendor_ids(vendor_paths::AbstractVector{<:AbstractString}; read_text::Function)
    vendor_ids = String[]
    for vendor_path in vendor_paths
        try
            push!(vendor_ids, _normalize_vendor_id(read_text(vendor_path)))
        catch
        end
    end
    return unique(vendor_ids)
end

function _detect_linux_gpu_hardware!(detected::Dict{Symbol, Bool}; ispath::Function, read_text::Function, vendor_paths)
    detected[:CUDA] = ispath("/dev/nvidiactl") || ispath("/dev/nvidia0")
    detected[:AMDGPU] = ispath("/dev/kfd")
    detected[:OpenCL] = any(ispath, OPENCL_LINUX_HINT_PATHS)

    for vendor_id in _linux_vendor_ids(vendor_paths; read_text)
        backend = get(GPU_VENDOR_BACKENDS, vendor_id, nothing)
        backend === nothing || (detected[backend] = true)
        vendor_id in OPENCL_VENDOR_IDS && (detected[:OpenCL] = true)
    end

    return detected
end

# Metal.jl supports Apple silicon only (adapted from GPUSelect.jl, see BACKEND_DRIVER_LIBRARIES).
function _detect_macos_gpu_hardware!(detected::Dict{Symbol, Bool}; ispath::Function, arch::Symbol)
    detected[:Metal] = arch === :aarch64
    detected[:OpenCL] = any(ispath, OPENCL_MACOS_HINT_PATHS)
    return detected
end

# Return whether the driver library of `backend` can be found by the dynamic loader.
# Adapted from GPUSelect.jl's `find_gpu_pkgid`, see BACKEND_DRIVER_LIBRARIES.
function _driver_library_probe(
        backend::Symbol;
        os::Symbol = _host_os(),
        find_library::Function = Libdl.find_library,
    )
    libraries = get(BACKEND_DRIVER_LIBRARIES, backend, nothing)
    libraries === nothing && return false
    names = os === :windows ? libraries[2] : libraries[1]
    try
        return !isempty(find_library(names))
    catch err
        @debug "Driver library lookup failed" backend exception = (err, catch_backtrace())
        return false
    end
end

"""
    report_backends(io::IO = stdout; backends=collect(NATIVE_BACKENDS), kwargs...) -> Bool

Print what GPUEnv sees on this host and in the active environment, and return
whether every installed backend passed its functional check.

For JLArrays and each native backend the report shows:

- **hardware**: a direct host hint from `GPUEnv.detect_gpu_hardware`
  (device nodes, PCI vendor IDs, Apple silicon),
- **driver**: whether the backend's driver library (such as `libcuda`) is found,
- **installed**: whether the backend package can be loaded from the active
  environment (the load path),
- **functional**: the result of the functional check, run only for installed
  backends, which loads the package,
- **FFT**: whether the backend implements AbstractFFTs, as filtered by
  `gpu_backends(; supports_fftw = true)`.

A backend is predicted by [`predict_backends`](@ref) when it has a hardware hint
or a driver library. Nothing is installed. Call it from the REPL after
[`activate`](@ref) to check that the environment matches the hardware.

Adapted from `test_backends` of
[GPUSelect.jl](https://github.com/SimonDanisch/GPUSelect.jl) (MIT License).

# Arguments
- `io::IO=stdout`: where to print the report
- `backends::AbstractVector{Symbol}`: native backends to report (default: all supported)
- `hardware::AbstractDict{Symbol, Bool}=detect_gpu_hardware()`: direct hardware hints
- `driver_library::Function`: `backend -> Bool`, whether the driver library is found
- `installed::Function`: `backend -> Bool`, whether the package is in the active environment
- `checker::Function=default_backend_checker`: functional check for installed backends

# Examples
```julia
using GPUEnv
GPUEnv.activate()
report_backends()
```
"""
function report_backends(
        io::IO = stdout;
        backends::AbstractVector{Symbol} = collect(NATIVE_BACKENDS),
        hardware::AbstractDict{Symbol, Bool} = detect_gpu_hardware(),
        driver_library::Function = _driver_library_probe,
        installed::Function = _backend_installed,
        checker::Function = default_backend_checker,
    )
    yesno(x) = x === nothing ? "-" : x ? "yes" : "no"
    rows = Vector{String}[]
    ok = true
    for backend in vcat(:JLArrays, filter(!=(:JLArrays), collect(backends)))
        native = backend !== :JLArrays
        hint = native ? get(hardware, backend, false) : nothing
        driver = native ? driver_library(backend) : nothing
        is_installed = installed(backend)
        functional = is_installed ? checker(backend) : nothing
        ok &= functional !== false
        fft = if functional !== true
            nothing
        elseif !native
            _supports_fftw(backend, Main)
        else
            binding = _loaded_binding(backend)
            binding === nothing ? nothing : _supports_fftw(backend, binding[1])
        end
        push!(rows, [String(backend), yesno(hint), yesno(driver), yesno(is_installed), yesno(functional), yesno(fft)])
    end

    header = ["backend", "hardware", "driver", "installed", "functional", "FFT"]
    widths = [maximum(length(row[i]) for row in vcat([header], rows)) for i in eachindex(header)]
    line(row) = rstrip("  " * join(rpad.(row, widths), "  "))
    bar = "─"^max(60, length(line(header)))
    os_name = Sys.iswindows() ? "Windows" : Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : string(Sys.KERNEL)
    project = Base.active_project()

    println(io, bar)
    println(io, "  GPUEnv backend report")
    println(io, "  ", os_name, " / ", Sys.ARCH, ", Julia ", VERSION)
    println(io, "  Active project: ", project === nothing ? "none" : project)
    println(io, bar)
    println(io, line(header))
    foreach(row -> println(io, line(row)), rows)
    println(io, bar)
    return ok
end

# Whether the package of `backend` resolves to its known UUID from the load path.
function _backend_installed(backend::Symbol)
    spec = get(BACKEND_SPECS, backend, nothing)
    spec === nothing && return false
    pkg_id = Base.identify_package(spec.package)
    return pkg_id !== nothing && pkg_id.uuid == Base.UUID(spec.uuid)
end
