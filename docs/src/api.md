# API Reference

## Backend types

```@docs
GPUEnv.BackendSpec
GPUEnv.GpuBackend
GPUEnv.is_jlarray_backend
GPUEnv.SyncResult
```

## Backend metadata

```@docs
GPUEnv.supported_backends
GPUEnv.backend_specs
GPUEnv.predict_backends
GPUEnv.backend_modules_and_array_types
GPUEnv.resolve_backends
```

## Backend queries and helpers

```@docs
GPUEnv.gpu_backends
GPUEnv.gpu_zeros
GPUEnv.gpu_ones
GPUEnv.gpu_randn
GPUEnv.gpu_wrapper
GPUEnv.synchronize_backend
```

## Environment management

```@docs
GPUEnv.sync_test_env
GPUEnv.activate
```
