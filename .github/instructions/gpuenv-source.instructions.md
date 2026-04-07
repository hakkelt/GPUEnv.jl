---
applyTo: "src/**/*.jl"
description: "Source editing guidance for GPUEnv"
---

- Keep the source small and explicit.
- Prefer pure helper functions for backend prediction and backend checking.
- Do not couple environment planning to test execution; `activate` may wrap `sync_test_env`, but sync should remain usable on its own.
- Preserve support for both `test/Project.toml` and legacy `[extras]` plus `[targets]` test dependency layouts.
- When writing persisted project-local environments, keep the gitignore warning behavior intact unless intentionally changed.
- Preserve the `include_jlarrays` default unless the user explicitly opts out.
- Keep docstrings on exported functions.
