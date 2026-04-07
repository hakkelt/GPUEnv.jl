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
