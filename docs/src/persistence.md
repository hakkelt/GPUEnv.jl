# Persistence

By default `activate` and `sync_test_env` create a **temporary** environment in a system
temp directory.  This is convenient for one-off runs but means Pkg must
resolve packages from scratch every time, which can add a minute or more
per run.

## Why persist?

A persisted environment stores a complete manifest alongside the generated
`Project.toml`. On Julia 1.11 and later, GPUEnv preserves the exact
manifest filename Julia would prefer, including `Manifest-v{major}.{minor}.toml`
when present. After the first successful sync it also records the source test
project plus source-manifest snapshot, so unchanged persisted environments are
reused directly instead of being re-created on every run. For a typical GPU
test environment this cuts setup from ~60–90 seconds down to 1–2 seconds.

Persistence is especially valuable in development workflows where you run
`julia --project=test test/runtests.jl` repeatedly: after the first run you
get near-instant startup on every subsequent run.

## Enabling persistence

Pass `persist = true` to place the environment at `gpu_env/` under the
overlay root:

```julia
GPUEnv.activate(; persist = true)
```

Override the location with `environment_path`:

```julia
GPUEnv.activate(; environment_path = "/tmp/my_gpu_overlay")
```

## Git ignore warning

When the persisted environment is inside a Git repository, GPUEnv checks
whether the environment directory is gitignored and emits a warning when it is
not:

```
┌ Warning: The persisted GPU test environment is inside a Git repository but is not ignored.
│   environment_path = "/path/to/project/gpu_env"
│   recommendation = "Add gpu_env/ to .gitignore …"
```

Add the generated path to `.gitignore` to silence the warning:

```
gpu_env/
```

Disable the check explicitly when needed:

```julia
GPUEnv.activate(; persist = true, warn_if_unignored = false)
```

## Refreshing the environment

GPUEnv automatically detects when the persisted environment is out of sync with
the parent project. Every time `activate` or `sync_test_env` runs it computes
a fingerprint of the current source project's resolved dependencies and compares it against the fingerprint
stored in the persisted environment. When the two differ — for example because
you added or updated a dependency — the overlay is rebuilt from scratch and the
new fingerprint is recorded.

## Backends that fail their functional check

Every predicted backend is installed and then checked; one that fails its
functional check (for example CUDA on a login node that has the NVIDIA driver but no
GPU) is removed from the overlay, and GPUEnv warns:

```
┌ Warning: Backend CUDA was installed but failed its functional check; it is dropped for now and retried on the next activation. Run with JULIA_DEBUG=GPUEnv to see the error.
```

A failure like this is often temporary, so the persisted overlay does not
remember it. When the overlay is reused, GPUEnv compares the predicted backends
with the ones in the overlay's `Project.toml`, adds back only the missing ones
(with the same `[compat]` bounds a fresh overlay has) and checks them again.
Nothing else is rebuilt. With `only_first = true` only the backends that come
before the installed one in the preference order are retried; if one of them
works now, it replaces the installed one.

Retrying a backend that keeps failing costs a package add, load and removal on
every run. Pass `exclude = [:CUDA]` (or the backend in question) on machines
where it can never work.

Changing `only_first` between runs rebuilds the overlay.
