const _SYNC_STATE_FILENAME = ".GPUEnv-state.toml"

"""
    sync_test_env(; path=nothing, include_jlarrays=nothing,
                  probe=default_backend_probe, checker=default_backend_checker,
                  backends_to_test=supported_backends(),
                  exclude=Symbol[], only_first=false,
                  persist=false, environment_path=nothing,
                  warn_if_unignored=true, dry_run=false, io=stderr)

Create or update a GPU overlay environment for the currently active project or
for a directory rooted at `path`.

When `path` is omitted the currently active Julia project is used as the source
environment directly, including unnamed and temporary projects. When `path` is
given it must point to a directory that contains a `Project.toml`; that project
file is used directly as the overlay base.

Unlike [`activate`](@ref), this function restores the previously active Julia
environment before returning. Use it when you want to inspect or manage the
synchronized environment yourself.

Set `dry_run = true` to inspect the planned backend list without installing
packages.

# Arguments
- `path::Union{Nothing, AbstractString}=nothing`: path to a directory containing `Project.toml`; defaults to active project
- `include_jlarrays::Union{Nothing,Bool}=nothing`: whether to include the JLArrays mock backend.
  When `nothing` (default), set to `true` automatically if `:JLArrays` appears in `backends_to_test`,
    otherwise `false`. Providing an explicit value that contradicts a custom `backends_to_test`
    or `exclude` throws an `ArgumentError`.
- `probe::Function=default_backend_probe`: function to predict which backends to install
- `checker::Function=default_backend_checker`: function to verify each backend's runtime availability
- `backends_to_test::AbstractVector{Symbol}`: which backends to attempt
- `exclude::AbstractVector{Symbol}=Symbol[]`: backends to skip even when predicted as available
- `only_first::Bool=false`: if true, stop after the first functional backend is installed
- `persist::Bool=false`: whether to cache the environment in `gpu_env/` under the overlay root
- `environment_path::Union{Nothing, AbstractString}=nothing`: exact path of the folder to use for the
  GPU overlay environment; when provided the folder is used directly (no subdirectory is appended).
  Set `persist=true` alongside `environment_path` to enable environment reuse.
- `warn_if_unignored::Bool=true`: whether to warn if a persisted env is not in .gitignore
- `dry_run::Bool=false`: if true, return the plan without installing packages
- `io::IO=stderr`: where to print warnings and informational messages
"""
function sync_test_env(
        ;
        path::Union{Nothing, AbstractString} = nothing,
        include_jlarrays::Union{Nothing, Bool} = nothing,
        probe::Function = default_backend_probe,
        checker::Function = default_backend_checker,
        backends_to_test::Union{AbstractVector{Symbol}, Tuple{Vararg{Symbol}}} = supported_backends(),
        exclude::AbstractVector{Symbol} = Symbol[],
        only_first::Bool = false,
        persist::Bool = false,
        environment_path::Union{Nothing, AbstractString} = nothing,
        warn_if_unignored::Bool = true,
        dry_run::Bool = false,
        io::IO = stderr,
    )
    requested_backends_to_test = collect(backends_to_test)
    if include_jlarrays === false && requested_backends_to_test == collect(supported_backends())
        filter!(!=(:JLArrays), requested_backends_to_test)
    end

    resolved_jlarrays, native_backends = _validate_and_resolve_include_jlarrays(
        include_jlarrays, requested_backends_to_test, exclude; default_jlarrays = false,
    )
    if path !== nothing
        return _sync_env_from_path_impl(
            abspath(path);
            include_jlarrays = resolved_jlarrays,
            probe = probe,
            checker = checker,
            backends_to_test = native_backends,
            exclude = collect(exclude),
            only_first = only_first,
            persist = persist,
            environment_path = environment_path,
            warn_if_unignored = warn_if_unignored,
            dry_run = dry_run,
            io = io,
            restore_previous_project = true,
        )
    else
        return _sync_active_project_env_impl(
            ;
            include_jlarrays = resolved_jlarrays,
            probe = probe,
            checker = checker,
            backends_to_test = native_backends,
            exclude = collect(exclude),
            only_first = only_first,
            persist = persist,
            environment_path = environment_path,
            warn_if_unignored = warn_if_unignored,
            dry_run = dry_run,
            io = io,
            restore_previous_project = true,
        )
    end
end

"""
    activate(; path=nothing, include_jlarrays=nothing,
             probe=default_backend_probe, checker=default_backend_checker,
             backends_to_test=supported_backends(),
             exclude=Symbol[], only_first=false,
             persist=false, environment_path=nothing,
             warn_if_unignored=true, dry_run=false, io=stderr)

Create, synchronize, and activate a GPU overlay environment.

When called without arguments, `activate` treats the currently active project
as the source. It creates a GPU-enabled overlay environment (temporary by
default), adds predicted functional backends, and activates that new
environment. This mode supports unnamed and temporary active projects.

When `path` is given it must point to a directory that contains a
`Project.toml`. That project is used directly as the overlay base.

All keyword arguments are forwarded to [`sync_test_env`](@ref), but unlike
`sync_test_env` the synchronized environment remains active on success.

# Arguments
- `path::Union{Nothing, AbstractString}=nothing`: path to a directory containing `Project.toml`; defaults to active project
- `include_jlarrays::Union{Nothing,Bool}=nothing`: whether to include the JLArrays mock backend.
  When `nothing` (default), set to `true` automatically if `:JLArrays` appears in `backends_to_test`,
    otherwise `false`. Providing an explicit value that contradicts a custom `backends_to_test`
    or `exclude` throws an `ArgumentError`.
- `probe::Function=default_backend_probe`: function to predict which backends to install
- `checker::Function=default_backend_checker`: function to verify each backend's runtime availability
- `backends_to_test::AbstractVector{Symbol}`: which backends to attempt
- `exclude::AbstractVector{Symbol}=Symbol[]`: backends to skip even when predicted as available
- `only_first::Bool=false`: if true, stop after the first functional backend is installed
- `persist::Bool=false`: whether to cache the environment in `gpu_env/` under the overlay root
- `environment_path::Union{Nothing, AbstractString}=nothing`: exact path of the folder to use for the
  GPU overlay environment; when provided the folder is used directly (no subdirectory is appended).
  Set `persist=true` alongside `environment_path` to enable environment reuse.
- `warn_if_unignored::Bool=true`: whether to warn if a persisted env is not in .gitignore
- `dry_run::Bool=false`: if true, return a SyncResult without installing packages or switching environments
- `io::IO=stderr`: where to print warnings and informational messages

## Example

```julia
# From a package's test environment (activated by Pkg.test):
using GPUEnv
GPUEnv.activate(; persist = true)
```
"""
function activate(
        ;
        path::Union{Nothing, AbstractString} = nothing,
        include_jlarrays::Union{Nothing, Bool} = nothing,
        probe::Function = default_backend_probe,
        checker::Function = default_backend_checker,
        backends_to_test::Union{AbstractVector{Symbol}, Tuple{Vararg{Symbol}}} = supported_backends(),
        exclude::AbstractVector{Symbol} = Symbol[],
        only_first::Bool = false,
        persist::Bool = false,
        environment_path::Union{Nothing, AbstractString} = nothing,
        warn_if_unignored::Bool = true,
        dry_run::Bool = false,
        io::IO = stderr,
    )
    requested_backends_to_test = collect(backends_to_test)
    if include_jlarrays === false && requested_backends_to_test == collect(supported_backends())
        filter!(!=(:JLArrays), requested_backends_to_test)
    end

    resolved_jlarrays, native_backends = _validate_and_resolve_include_jlarrays(
        include_jlarrays, requested_backends_to_test, exclude; default_jlarrays = false,
    )
    if path !== nothing
        return _sync_env_from_path_impl(
            abspath(path);
            include_jlarrays = resolved_jlarrays,
            probe = probe,
            checker = checker,
            backends_to_test = native_backends,
            exclude = collect(exclude),
            only_first = only_first,
            persist = persist,
            environment_path = environment_path,
            warn_if_unignored = warn_if_unignored,
            dry_run = dry_run,
            io = io,
            restore_previous_project = dry_run,
        )
    else
        return _sync_active_project_env_impl(
            ;
            include_jlarrays = resolved_jlarrays,
            probe = probe,
            checker = checker,
            backends_to_test = native_backends,
            exclude = collect(exclude),
            only_first = only_first,
            persist = persist,
            environment_path = environment_path,
            warn_if_unignored = warn_if_unignored,
            dry_run = dry_run,
            io = io,
            restore_previous_project = dry_run,
        )
    end
end

function _sync_active_project_env_impl(
        ;
        include_jlarrays::Bool,
        probe::Function,
        checker::Function,
        backends_to_test::AbstractVector{Symbol},
        exclude::AbstractVector{Symbol},
        only_first::Bool,
        persist::Bool,
        environment_path::Union{Nothing, AbstractString},
        warn_if_unignored::Bool,
        dry_run::Bool,
        io::IO,
        restore_previous_project::Bool,
    )
    active_project = Base.active_project()
    active_project === nothing && error("No active project is available.")

    source_root = dirname(active_project)
    source_project = _rewrite_sources(TOML.parsefile(active_project), source_root)
    # Prefer the workspace manifest for workspace members so copied manifests stay
    # consistent with sibling path dependencies inherited from the workspace.
    # Fall back to a local manifest or a parent manifest for non-workspace setups.
    source_manifest = _workspace_manifest_path(source_root)
    source_manifest === nothing && (source_manifest = _preferred_manifest_path(source_root))
    source_manifest === nothing && (source_manifest = _find_parent_manifest_path(source_root))
    source_project = _augment_source_project(source_project, source_root, source_manifest)
    source_project = _localize_running_package_source(source_project)

    requested = _filter_backends(
        predict_backends(
            ;
            include_jlarrays = include_jlarrays,
            probe = probe,
            backends_to_test = backends_to_test,
        ),
        exclude,
    )

    sync_project_data = _sanitize_environment_project(_merge_backend_entries(source_project, requested))

    env_dir, persisted = _environment_dir(
        source_root;
        persist = persist,
        environment_path = environment_path,
    )

    sync_state = _sync_state_data(sync_project_data, source_manifest, requested)
    reusable = _can_reuse_persisted_env(sync_state, env_dir; persisted = persisted)
    project_path = reusable ? joinpath(env_dir, "Project.toml") : _write_environment!(sync_project_data, source_manifest, env_dir)
    warned = _maybe_warn_about_persisted_env(source_root, env_dir; persisted = persisted, warn_if_unignored = warn_if_unignored)

    if dry_run
        return SyncResult(
            env_dir,
            project_path,
            active_project,
            requested,
            Symbol[],
            Symbol[],
            include_jlarrays,
            persisted,
            :active_project,
            warned,
            true,
        )
    end

    previous_project = Base.active_project()
    installed = Symbol[]
    success = false
    try
        Pkg.activate(env_dir; io = io)

        if reusable
            installed = copy(requested)
        else
            _develop_overlay_path_sources!(sync_project_data, io)
            installed, functional = _install_and_filter_backends!(requested, checker, only_first, io)
            _restore_overlay_sources!(project_path, sync_project_data)
            Pkg.instantiate(; io = io)
            Pkg.precompile(; io = io)
            sync_state["project_toml"] = read(project_path, String)
            persisted && _write_sync_state!(env_dir, sync_state)
            restore_previous_project || _load_functional_backends!(functional)

            success = true
            return SyncResult(
                env_dir,
                project_path,
                active_project,
                requested,
                installed,
                functional,
                include_jlarrays,
                persisted,
                :active_project,
                warned,
                false,
            )
        end

        functional = resolve_backends(installed; checker = checker)
        restore_previous_project || _load_functional_backends!(functional)
        success = true
        return SyncResult(
            env_dir,
            project_path,
            active_project,
            requested,
            installed,
            functional,
            include_jlarrays,
            persisted,
            :active_project,
            warned,
            false,
        )
    finally
        if previous_project !== nothing && (restore_previous_project || !success)
            Pkg.activate(dirname(previous_project); io = io)
        end
    end
end

function _sync_env_from_path_impl(
        root::AbstractString;
        include_jlarrays::Bool,
        probe::Function,
        checker::Function,
        backends_to_test::AbstractVector{Symbol},
        exclude::AbstractVector{Symbol},
        only_first::Bool,
        persist::Bool,
        environment_path::Union{Nothing, AbstractString},
        warn_if_unignored::Bool,
        dry_run::Bool,
        restore_previous_project::Bool,
        io::IO = stderr,
    )
    source_project_path = joinpath(root, "Project.toml")
    isfile(source_project_path) || error("No Project.toml found at path: $(source_project_path)")

    source_project = _rewrite_sources(TOML.parsefile(source_project_path), root)
    source_manifest = _workspace_manifest_path(root)
    source_manifest === nothing && (source_manifest = _preferred_manifest_path(root))
    source_manifest === nothing && (source_manifest = _find_parent_manifest_path(root))
    source_project = _augment_source_project(source_project, root, source_manifest)
    source_project = _localize_running_package_source(source_project)

    requested = _filter_backends(
        predict_backends(
            ;
            include_jlarrays = include_jlarrays,
            probe = probe,
            backends_to_test = backends_to_test,
        ),
        exclude,
    )

    sync_project_data = _sanitize_environment_project(_merge_backend_entries(source_project, requested))

    env_dir, persisted = _environment_dir(
        root;
        persist = persist,
        environment_path = environment_path,
    )

    sync_state = _sync_state_data(sync_project_data, source_manifest, requested)
    reusable = _can_reuse_persisted_env(sync_state, env_dir; persisted = persisted)
    project_path = reusable ? joinpath(env_dir, "Project.toml") : _write_environment!(sync_project_data, source_manifest, env_dir)
    warned = _maybe_warn_about_persisted_env(root, env_dir; persisted = persisted, warn_if_unignored = warn_if_unignored)

    if dry_run
        return SyncResult(
            env_dir,
            project_path,
            source_project_path,
            requested,
            Symbol[],
            Symbol[],
            include_jlarrays,
            persisted,
            :path_project,
            warned,
            true,
        )
    end

    previous_project = Base.active_project()
    installed = Symbol[]
    success = false
    try
        Pkg.activate(env_dir; io = io)

        if reusable
            installed = copy(requested)
        else
            _develop_overlay_path_sources!(sync_project_data, io)
            installed, functional = _install_and_filter_backends!(requested, checker, only_first, io)
            _restore_overlay_sources!(project_path, sync_project_data)
            Pkg.instantiate(; io = io)
            Pkg.precompile(; io = io)
            sync_state["project_toml"] = read(project_path, String)
            persisted && _write_sync_state!(env_dir, sync_state)
            restore_previous_project || _load_functional_backends!(functional)

            success = true
            return SyncResult(
                env_dir,
                project_path,
                source_project_path,
                requested,
                installed,
                functional,
                include_jlarrays,
                persisted,
                :path_project,
                warned,
                false,
            )
        end

        functional = resolve_backends(installed; checker = checker)
        restore_previous_project || _load_functional_backends!(functional)
        success = true
        return SyncResult(
            env_dir,
            project_path,
            source_project_path,
            requested,
            installed,
            functional,
            include_jlarrays,
            persisted,
            :path_project,
            warned,
            false,
        )
    finally
        if previous_project !== nothing && (restore_previous_project || !success)
            Pkg.activate(dirname(previous_project); io = io)
        end
    end
end

function _environment_dir(
        root::AbstractString;
        persist::Bool,
        environment_path::Union{Nothing, AbstractString},
    )
    if environment_path !== nothing
        # environment_path is used as-is; persisted follows the persist flag
        return abspath(environment_path), persist
    elseif persist
        return abspath(joinpath(root, "gpu_env")), true
    else
        return mktempdir(prefix = "GPUEnv-"), false
    end
end

function _load_functional_backends!(functional::AbstractVector{Symbol})
    for backend in functional
        spec = get(BACKEND_SPECS, backend, nothing)
        spec === nothing && continue
        pkg_id = Base.PkgId(Base.UUID(spec.uuid), spec.package)
        try
            Base.require(pkg_id)
        catch err
            @debug "Could not load functional backend after activation" backend exception = (err, catch_backtrace())
        end
    end
    return nothing
end

function _filter_backends(backends::AbstractVector{Symbol}, exclude::AbstractVector{Symbol})
    isempty(exclude) && return backends
    return filter(b -> b ∉ exclude, backends)
end

function _install_and_filter_backends!(
        requested::AbstractVector{Symbol},
        checker::Function,
        only_first::Bool,
        io::IO,
    )
    if only_first
        for backend in requested
            _install_backend!(backend, io) || continue
            # Instantiate after each add so extension dependencies are present before
            # we import the backend for a functional check.
            Pkg.instantiate(; io = io)
            if checker(backend)
                return [backend], [backend]
            else
                _remove_backend!(backend, io)
            end
        end
        return Symbol[], Symbol[]
    else
        installed = _install_backends!(requested, io)
        for backend in installed
            if !checker(backend)
                _remove_backend!(backend, io)
                filter!(b -> b != backend, installed)
            end
        end
        return installed, copy(installed)
    end
end

# Copy a manifest file to `dest`, rewriting any relative `path =` entries to
# absolute paths so the manifest is valid when placed in a different directory.
function _copy_manifest_with_absolute_paths(source::AbstractString, dest::AbstractString)
    source_dir = dirname(abspath(source))
    content = read(source, String)
    # Replace  path = "some/relative"  with  path = "/absolute/some/relative"
    # Only touches entries that use a relative (non-absolute) path value.
    content = replace(
        content, r"(path\s*=\s*)\"([^/\"][^\"]*)\""s => function (m)
            m_prefix = match(r"(path\s*=\s*)\"([^/\"][^\"]*)\""s, m)
            m_prefix === nothing && return m
            prefix, rel = m_prefix[1], m_prefix[2]
            abs_path = normpath(joinpath(source_dir, rel))
            return "$(prefix)$(repr(abs_path))"
        end
    )
    write(dest, content)
    return nothing
end

function _sanitize_environment_project(project_data::Dict{String, Any})
    sanitized = deepcopy(project_data)

    for key in ("name", "uuid", "version", "authors", "workspace", "weakdeps", "extensions", "extras", "targets")
        pop!(sanitized, key, nothing)
    end

    deps = get(sanitized, "deps", Dict{String, Any}())
    dep_names = deps isa Dict ? Set(String.(keys(deps))) : Set{String}()

    compat = get(sanitized, "compat", nothing)
    if compat isa Dict
        filtered_compat = Dict{String, Any}()
        for (name, value) in compat
            if name == "julia" || name in dep_names
                filtered_compat[name] = value
            end
        end
        isempty(filtered_compat) ? pop!(sanitized, "compat", nothing) : (sanitized["compat"] = filtered_compat)
    end

    sources = get(sanitized, "sources", nothing)
    if sources isa Dict
        filtered_sources = Dict{String, Any}()
        for (name, value) in sources
            name in dep_names && (filtered_sources[name] = value)
        end
        isempty(filtered_sources) ? pop!(sanitized, "sources", nothing) : (sanitized["sources"] = filtered_sources)
    end

    return sanitized
end

function _restore_overlay_sources!(project_path::AbstractString, baseline_project::Dict{String, Any})
    isfile(project_path) || return false

    baseline_sources = get(baseline_project, "sources", nothing)
    baseline_sources isa Dict || return false

    current_project = TOML.parsefile(project_path)
    current_deps = get(current_project, "deps", Dict{String, Any}())
    current_deps isa Dict || return false

    dep_names = Set(String.(keys(current_deps)))
    current_sources = get!(current_project, "sources", Dict{String, Any}())
    changed = false

    for (name, value) in baseline_sources
        name in dep_names || continue
        if get(current_sources, name, nothing) != value
            current_sources[name] = deepcopy(value)
            changed = true
        end
    end

    if changed
        open(project_path, "w") do io
            TOML.print(io, current_project)
        end
    end

    return changed
end

function _develop_overlay_path_sources!(baseline_project::Dict{String, Any}, io::IO)
    baseline_sources = get(baseline_project, "sources", nothing)
    baseline_sources isa Dict || return false

    baseline_deps = get(baseline_project, "deps", nothing)
    baseline_deps isa Dict || return false

    developed = false
    for (name, value) in baseline_sources
        name in keys(baseline_deps) || continue
        value isa Dict || continue
        path = get(value, "path", nothing)
        path isa AbstractString || continue
        isfile(joinpath(path, "Project.toml")) || continue
        Pkg.develop(Pkg.PackageSpec(path = path); io = io)
        developed = true
    end

    return developed
end

function _write_environment!(project_data::Dict{String, Any}, manifest_source::Union{Nothing, String}, env_dir::AbstractString)
    mkpath(env_dir)
    project_path = joinpath(env_dir, "Project.toml")

    open(project_path, "w") do io
        TOML.print(io, project_data)
    end

    for manifest_name in filter(name -> occursin(r"^Manifest(?:-v\d+\.\d+)?\.toml$", name), readdir(env_dir))
        rm(joinpath(env_dir, manifest_name); force = true)
    end
    rm(_sync_state_path(env_dir); force = true)

    if manifest_source !== nothing && isfile(manifest_source)
        dest = joinpath(env_dir, basename(manifest_source))
        _copy_manifest_with_absolute_paths(manifest_source, dest)
    end

    return project_path
end

# Extract a compact name → version/sha fingerprint from a manifest file.
# Equivalent to querying `Pkg.status(mode=PKGMODE_MANIFEST)` — captures the
# fully resolved transitive dependency set without storing the entire manifest.
function _manifest_deps_fingerprint(manifest_source::Union{Nothing, String})
    manifest_source === nothing && return Dict{String, Any}()
    isfile(manifest_source) || return Dict{String, Any}()
    manifest_data = TOML.parsefile(manifest_source)
    manifest_dir = dirname(abspath(manifest_source))
    deps = get(manifest_data, "deps", nothing)
    deps isa Dict || return Dict{String, Any}()
    manifest_meta = Dict{String, Any}()
    for key in ("manifest_format", "julia_version", "project_hash")
        value = get(manifest_data, key, nothing)
        value === nothing || (manifest_meta[key] = value)
    end
    deps_fingerprint = Dict{String, Any}()
    for (pkg_name, records) in deps
        record = records isa Vector ? first(records) : records
        record isa Dict || continue
        entry = Dict{String, Any}()
        version = get(record, "version", nothing)
        if version !== nothing
            entry["version"] = version
        end
        git_sha = get(record, "git-tree-sha1", nothing)
        git_sha isa AbstractString && (entry["git-tree-sha1"] = git_sha)
        path = get(record, "path", nothing)
        path isa AbstractString && (entry["path"] = normpath(joinpath(manifest_dir, path)))
        repo_rev = get(record, "repo-rev", nothing)
        repo_rev isa AbstractString && (entry["repo-rev"] = repo_rev)
        repo_url = get(record, "repo-url", nothing)
        repo_url isa AbstractString && (entry["repo-url"] = repo_url)
        isempty(entry) || (deps_fingerprint[pkg_name] = entry)
    end
    return Dict{String, Any}(
        "manifest_meta" => manifest_meta,
        "deps" => deps_fingerprint,
    )
end

function _sync_state_data(
        project_data::Dict{String, Any},
        manifest_source::Union{Nothing, String},
        requested_backends::AbstractVector{Symbol},
    )
    io = IOBuffer()
    TOML.print(io, project_data)
    project_toml = String(take!(io))
    return Dict{String, Any}(
        "source_project_toml" => project_toml,
        "project_toml" => project_toml,
        "requested_backends" => String.(requested_backends),
        "source_manifest_name" => manifest_source === nothing ? "" : basename(manifest_source),
        "source_manifest_deps" => _manifest_deps_fingerprint(manifest_source),
    )
end

_sync_state_path(env_dir::AbstractString) = joinpath(env_dir, _SYNC_STATE_FILENAME)

function _can_reuse_persisted_env(sync_state::Dict{String, Any}, env_dir::AbstractString; persisted::Bool)
    persisted || return false
    project_path = joinpath(env_dir, "Project.toml")
    state_path = _sync_state_path(env_dir)
    isfile(project_path) || return false
    isfile(state_path) || return false
    manifest_name = get(sync_state, "source_manifest_name", "")
    if !isempty(manifest_name)
        isfile(joinpath(env_dir, manifest_name)) || return false
    end
    stored_state = TOML.parsefile(state_path)
    get(stored_state, "source_project_toml", nothing) == sync_state["source_project_toml"] || return false
    get(stored_state, "requested_backends", nothing) == sync_state["requested_backends"] || return false
    get(stored_state, "source_manifest_name", nothing) == sync_state["source_manifest_name"] || return false
    get(stored_state, "source_manifest_deps", nothing) == sync_state["source_manifest_deps"] || return false
    read(project_path, String) == get(stored_state, "project_toml", "") || return false
    return true
end

function _write_sync_state!(env_dir::AbstractString, sync_state::Dict{String, Any})
    open(_sync_state_path(env_dir), "w") do io
        TOML.print(io, sync_state)
    end
    return nothing
end

function default_backend_checker(backend::Symbol)
    backend === :JLArrays && return true
    haskey(BACKEND_SPECS, backend) || return false

    try
        module_ref = _backend_module(backend)
        return _backend_functional(module_ref, backend)
    catch err
        @debug "Backend functional check failed" backend exception = (err, catch_backtrace())
        return false
    end
end

function default_backend_binding(backend::Symbol)
    haskey(BACKEND_SPECS, backend) || return nothing

    try
        module_ref = _backend_module(backend)
        spec = BACKEND_SPECS[backend]
        isdefined(module_ref, spec.array_type_name) || return nothing
        return module_ref, getfield(module_ref, spec.array_type_name)
    catch err
        @debug "Backend module binding failed" backend exception = (err, catch_backtrace())
        return nothing
    end
end

function _backend_functional(module_ref::Module, backend::Symbol)
    if backend === :OpenCL
        return _opencl_backend_functional(module_ref)
    end

    isdefined(module_ref, :functional) || return false
    return Base.invokelatest(getfield(module_ref, :functional))
end

function _opencl_backend_functional(module_ref::Module)
    isdefined(module_ref, :cl) || return false
    cl = getfield(module_ref, :cl)
    isdefined(cl, :platforms) || return false

    platforms = Base.invokelatest(getfield(cl, :platforms))
    isempty(platforms) && return false
    isdefined(cl, :devices) || return true

    devices = getfield(cl, :devices)
    for platform in platforms
        isempty(Base.invokelatest(devices, platform)) && continue

        # Some OpenCL stacks enumerate devices but fail the first kernel compile
        # (for example due to missing SPIR-V support). Treat those as non-functional.
        try
            sample = if isdefined(module_ref, :zeros)
                Base.invokelatest(getfield(module_ref, :zeros), Float32, 1)
            elseif isdefined(module_ref, :CLArray)
                Base.invokelatest(getfield(module_ref, :CLArray), Float32[1.0f0])
            else
                return false
            end
            Base.invokelatest(broadcast, +, sample, sample)
            return true
        catch err
            @debug "OpenCL smoke test failed" exception = (err, catch_backtrace())
        end
    end

    return false
end

function _backend_module(backend::Symbol)
    spec = BACKEND_SPECS[backend]
    pkg_id = Base.PkgId(Base.UUID(spec.uuid), spec.package)
    return Base.require(pkg_id)
end

function _install_backend!(backend::Symbol, io::IO)
    spec = get(BACKEND_SPECS, backend, nothing)
    spec === nothing && return false
    return _install_package!(spec.package, io)
end

function _install_backends!(backends::AbstractVector{Symbol}, io::IO)
    specs = [get(BACKEND_SPECS, backend, nothing) for backend in backends]
    if all(spec -> spec === nothing, specs)
        @warn "None of the requested backends have known packages to install." requested_backends = backends
        return Symbol[]
    end
    if any(spec -> spec === nothing, specs)
        unknown_backends = [backends[i] for i in eachindex(backends) if specs[i] === nothing]
        @warn "Some of the requested backends do not have known packages to install." unknown_backends = unknown_backends
        specs = filter(!isnothing, specs)
    end
    success = _install_packages!([spec.package for spec in specs], io)
    if success
        return [spec.name for spec in specs]
    else
        installed = Symbol[]
        for spec in specs
            success = _install_package!(spec.package, io)
            if success
                push!(installed, spec.name)
            else
                @warn "Failed to install backend package" package = spec.package
            end
        end
        return installed
    end
end

function _remove_backend!(backend::Symbol, io::IO)
    spec = get(BACKEND_SPECS, backend, nothing)
    spec === nothing && return false
    return _remove_package!(spec.package, io)
end

function _install_package!(package_name::AbstractString, io::IO)
    try
        Pkg.add(package_name; io = io)
        return true
    catch err
        @warn "Could not install backend package" package_name exception = (err, catch_backtrace())
        return false
    end
end

function _install_packages!(package_names::AbstractVector{<:AbstractString}, io::IO)
    try
        Pkg.add(package_names; io = io)
        return true
    catch err
        return false
    end
end

function _remove_package!(package_name::AbstractString, io::IO)
    try
        Pkg.rm(package_name; io = io)
        return true
    catch err
        @warn "Could not remove backend package" package_name exception = (err, catch_backtrace())
        return false
    end
end

function _maybe_warn_about_persisted_env(
        root::AbstractString,
        env_dir::AbstractString;
        persisted::Bool,
        warn_if_unignored::Bool,
    )
    if !persisted || !warn_if_unignored
        return false
    end

    repo_root = _git_repo_root(root)
    repo_root === nothing && return false
    _is_subpath(env_dir, repo_root) || return false

    git = Sys.which("git")
    git === nothing && return false

    relative_env = relpath(env_dir, repo_root)
    ignored = success(`$git -C $repo_root check-ignore -q -- $relative_env`)
    if !ignored
        @warn "The persisted GPU test environment is inside a Git repository but is not ignored." environment_path = env_dir recommendation = "Add $(relative_env)/ to .gitignore or call sync_test_env(...; warn_if_unignored = false)."
        return true
    end

    return false
end

function _git_repo_root(path::AbstractString)
    current = abspath(path)
    while true
        if isdir(joinpath(current, ".git")) || isfile(joinpath(current, ".git"))
            return current
        end
        parent = dirname(current)
        parent == current && return nothing
        current = parent
    end
    return nothing
end

function _is_subpath(path::AbstractString, root::AbstractString)
    normalized_path = normpath(abspath(path))
    normalized_root = normpath(abspath(root))
    return normalized_path == normalized_root || startswith(normalized_path, joinpath(normalized_root, ""))
end
