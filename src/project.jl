function _package_project_data()
    return TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
end

function _manifest_candidates(dir::AbstractString)
    candidates = String[]
    if VERSION >= v"1.11"
        push!(candidates, joinpath(dir, "Manifest-v$(VERSION.major).$(VERSION.minor).toml"))
    end
    push!(candidates, joinpath(dir, "Manifest.toml"))
    return candidates
end

function _preferred_manifest_path(dir::AbstractString)
    for candidate in _manifest_candidates(dir)
        isfile(candidate) && return candidate
    end
    return nothing
end

# Walk up the directory tree to find a workspace manifest (a Manifest.toml that
# lives alongside a Project.toml that has a [workspace] section).  Returns
# nothing when no workspace manifest can be found.
function _workspace_manifest_path(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        project_file = joinpath(dir, "Project.toml")
        if isfile(project_file)
            d = TOML.parsefile(project_file)
            if haskey(d, "workspace")
                manifest = _preferred_manifest_path(dir)
                manifest !== nothing && return manifest
            end
        end
        parent = dirname(dir)
        parent == dir && break  # reached filesystem root
        dir = parent
    end
    return nothing
end

function _workspace_project_path(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        project_file = joinpath(dir, "Project.toml")
        if isfile(project_file)
            d = TOML.parsefile(project_file)
            haskey(d, "workspace") && return project_file
        end
        parent = dirname(dir)
        parent == dir && break
        dir = parent
    end
    return nothing
end

# Walk up the directory tree to find any parent Manifest.toml file,
# useful for nested test/ or benchmark/ subdirectories that should inherit
# parent project dependencies. Returns nothing when no parent manifest can be found.
function _find_parent_manifest_path(start_dir::AbstractString)
    dir = abspath(start_dir)
    parent = dirname(dir)

    while parent != dir  # not at filesystem root
        manifest = _preferred_manifest_path(parent)
        if manifest !== nothing
            # Verify the parent directory has a Project.toml (is a real project)
            if isfile(joinpath(parent, "Project.toml"))
                return manifest
            end
        end
        dir = parent
        parent = dirname(dir)
    end
    return nothing
end

function _find_parent_project_path(start_dir::AbstractString)
    dir = abspath(start_dir)
    parent = dirname(dir)

    while parent != dir
        project_file = joinpath(parent, "Project.toml")
        isfile(project_file) && return project_file
        dir = parent
        parent = dirname(dir)
    end

    return nothing
end

function _rewrite_sources(data::Dict{String, Any}, source_root::AbstractString)
    rewritten = deepcopy(data)
    sources = get(rewritten, "sources", nothing)
    sources isa Dict || return rewritten

    for spec in values(sources)
        spec isa Dict || continue
        source_path = get(spec, "path", nothing)
        source_path isa AbstractString || continue
        if !isabspath(source_path)
            spec["path"] = abspath(joinpath(source_root, source_path))
        end
    end

    return rewritten
end

function _localize_running_package_source(data::Dict{String, Any})
    return _localize_source_path(data, String(_package_project_data()["name"]), dirname(@__DIR__))
end

function _localize_source_path(data::Dict{String, Any}, package_name::AbstractString, package_root::AbstractString)
    rewritten = deepcopy(data)
    deps = get(rewritten, "deps", nothing)
    deps isa Dict || return rewritten
    haskey(deps, package_name) || return rewritten

    sources = get!(rewritten, "sources", Dict{String, Any}())
    spec = get!(sources, package_name, Dict{String, Any}())
    spec["path"] = abspath(package_root)
    pop!(spec, "url", nothing)
    pop!(spec, "rev", nothing)
    pop!(spec, "subdir", nothing)
    return rewritten
end

function _merge_backend_entries(base_project::Dict{String, Any}, backends::AbstractVector{Symbol})
    project = deepcopy(base_project)
    deps = get!(project, "deps", Dict{String, Any}())
    compat = get!(project, "compat", Dict{String, Any}())
    package_compat = get(_package_project_data(), "compat", Dict{String, Any}())

    for backend in backends
        spec = BACKEND_SPECS[backend]
        if !haskey(deps, spec.package)
            deps[spec.package] = spec.uuid
        end
        if !haskey(compat, spec.package) && haskey(package_compat, spec.package)
            compat[spec.package] = package_compat[spec.package]
        end
    end

    return project
end

function _merge_project_tables(
        base_project::Dict{String, Any},
        extra_project::Dict{String, Any};
        exclude_packages::AbstractSet{String} = Set{String}(),
    )
    merged = deepcopy(base_project)

    for section in ("deps", "compat", "weakdeps", "extensions")
        extra_section = get(extra_project, section, nothing)
        extra_section isa Dict || continue
        merged_section = get!(merged, section, Dict{String, Any}())
        for (name, value) in extra_section
            name in exclude_packages && continue
            haskey(merged_section, name) || (merged_section[name] = deepcopy(value))
        end
    end

    extra_sources = get(extra_project, "sources", nothing)
    if extra_sources isa Dict
        merged_sources = get!(merged, "sources", Dict{String, Any}())
        for (name, value) in extra_sources
            name in exclude_packages && continue
            haskey(merged_sources, name) || (merged_sources[name] = deepcopy(value))
        end
    end

    return merged
end

function _path_project_records(project_data::Dict{String, Any}, manifest_source::Union{Nothing, String})
    manifest_source === nothing && return Pair{String, String}[]
    isfile(manifest_source) || return Pair{String, String}[]

    manifest_data = TOML.parsefile(manifest_source)
    deps = get(manifest_data, "deps", nothing)
    deps isa Dict || return Pair{String, String}[]

    direct_deps = get(project_data, "deps", Dict{String, Any}())
    direct_deps isa Dict || return Pair{String, String}[]

    manifest_dir = dirname(abspath(manifest_source))
    projects = Pair{String, String}[]
    for package_name in keys(direct_deps)
        records = get(deps, package_name, nothing)
        record = records isa Vector ? first(records) : records
        record isa Dict || continue
        path = get(record, "path", nothing)
        path isa AbstractString || continue
        abs_path = normpath(joinpath(manifest_dir, path))
        isfile(joinpath(abs_path, "Project.toml")) || continue
        push!(projects, package_name => abs_path)
    end

    return projects
end

function _path_project_records(
        project_data::Dict{String, Any},
        manifest_sources::AbstractVector{<:AbstractString},
    )
    projects = Dict{String, String}()

    for manifest_source in manifest_sources
        for (package_name, package_root) in _path_project_records(project_data, manifest_source)
            projects[package_name] = package_root
        end
    end

    return collect(pairs(projects))
end

function _augment_source_project(
        project_data::Dict{String, Any},
        source_root::AbstractString,
        manifest_source::Union{Nothing, String},
    )
    merged = deepcopy(project_data)
    excluded = Set([String(_package_project_data()["name"])])

    workspace_project = _workspace_project_path(source_root)
    if workspace_project !== nothing && dirname(workspace_project) != source_root
        workspace_root = dirname(workspace_project)
        workspace_data = _rewrite_sources(TOML.parsefile(workspace_project), workspace_root)
        merged = _merge_project_tables(merged, workspace_data; exclude_packages = excluded)
    else
        parent_project = _find_parent_project_path(source_root)
        if parent_project !== nothing && dirname(parent_project) != source_root
            parent_root = dirname(parent_project)
            parent_data = _rewrite_sources(TOML.parsefile(parent_project), parent_root)
            merged = _merge_project_tables(merged, parent_data; exclude_packages = excluded)
        end
    end

    manifest_sources = String[]
    manifest_source !== nothing && push!(manifest_sources, abspath(manifest_source))

    workspace_manifest = _workspace_manifest_path(source_root)
    if workspace_manifest !== nothing
        workspace_manifest = abspath(workspace_manifest)
        workspace_manifest in manifest_sources || push!(manifest_sources, workspace_manifest)
    end

    parent_manifest = _find_parent_manifest_path(source_root)
    if parent_manifest !== nothing
        parent_manifest = abspath(parent_manifest)
        parent_manifest in manifest_sources || push!(manifest_sources, parent_manifest)
    end

    for (package_name, package_root) in _path_project_records(project_data, manifest_sources)
        package_name in excluded && continue
        merged_sources = get!(merged, "sources", Dict{String, Any}())
        merged_sources[package_name] = Dict{String, Any}("path" => abspath(package_root))
        package_data = _rewrite_sources(TOML.parsefile(joinpath(package_root, "Project.toml")), package_root)
        merged = _merge_project_tables(merged, package_data; exclude_packages = excluded)
    end

    return merged
end
