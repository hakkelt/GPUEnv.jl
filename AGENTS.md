# GPUEnv Agent Instructions

This repository contains the GPUEnv Julia package.

## Working rules

- Keep the package focused on GPU test-environment management.
- Prefer Pkg-based solutions over custom file rewriting when Pkg can do the job.
- Default to JLArrays, but keep the public API flexible enough to skip it.
- Treat backend prediction and backend verification as separate steps.
- Keep `sync_test_env` focused on synchronization; let `activate` be the function that changes the active environment.
- The `path` kwarg always points to the **exact directory** containing `Project.toml` to overlay (no auto-appending of `test/`).
- Preserve the persisted-environment warning when a project-local GPU env is not gitignored, unless the caller disables it explicitly.
- Keep tests deterministic and light; do not require real GPU hardware for the default test suite.
- Write or update docstrings for exported functions when behavior changes.
- Update README and docs together with code changes.

## Layout

- `src/GPUEnv.jl` contains the module declaration, exports, and `include` calls.
- `src/backends.jl` defines backend types (`BackendSpec`, `GpuBackend`, `SyncResult`),
  constants (`BACKEND_SPECS`, `NATIVE_BACKENDS`), and all backend query/helper functions.
- `src/project.jl` contains pure TOML/project-data manipulation helpers
  (`_base_test_environment`, `_merge_backend_entries`, etc.) — no Pkg calls.
- `src/environment.jl` contains Pkg operations and environment management
  (`sync_test_env`, `activate`, `_write_environment!`, etc.).
- `test/` uses TestItems and TestItemRunner.
- `docs/` contains the user-facing explanation and API reference.
- `.github/workflows/CI.yml` uses the TestItems reusable workflow for linting,
  tests, coverage reporting, and documentation deployment.

## Validation

- Run `julia --project=test test/runtests.jl` for the default suite.
- Keep any backend-install behavior behind a dry-run or dependency-injected path so CI remains fast.

## Definition Of Done

- Code, tests, README, and docs agree on `path` semantics and public API.
- Any changed workflow still matches the behavior implemented in `docs/make.jl` and the package entrypoints.
- Default validation passes.
- New persistence or backend-detection behavior is covered by deterministic tests.
