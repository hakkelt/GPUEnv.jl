---
applyTo: "test/**/*.jl"
description: "Test editing guidance for GPUEnv"
---

- Keep tests deterministic and backend-free by default.
- Use dependency injection for host probes and functional checks.
- Prefer small unit tests over integration tests that install GPU packages.
- Verify both the default JLArrays path and the opt-out path.
- Use TestItems, not ad hoc `@testset` includes.
- Cover both workspace-era and legacy test dependency handling.
- Keep doctests and Aqua checks passing when public APIs or docs change.
