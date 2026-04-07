module GPUEnv

using Pkg
using TOML

export BackendSpec,
    GpuBackend,
    SyncResult,
    activate,
    backend_modules_and_array_types,
    backend_specs,
    gpu_backends,
    gpu_ones,
    gpu_randn,
    gpu_wrapper,
    gpu_zeros,
    is_jlarray_backend,
    predict_backends,
    resolve_backends,
    sync_test_env,
    supported_backends,
    synchronize_backend

include("backends.jl")
include("project.jl")
include("environment.jl")

end # module GPUEnv
