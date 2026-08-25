using TestItems

@testitem "SyncResult Base.show" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        dry_run = true,
        include_jlarrays = false,
        probe = _ -> false,
    )
    io = IOBuffer()
    show(io, result)
    s = String(take!(io))
    @test startswith(s, "SyncResult(env=")
    @test contains(s, "dry_run=true")
    @test contains(s, "backends=")
end

@testitem "Non-JLArray GpuBackend callable uses array_type" begin
    using GPUEnv
    using Test

    backend = GpuBackend(:CUDA, @__MODULE__, Vector{Float32})
    x = Float32[1.0, 2.0, 3.0]
    result = to_gpu(backend, x)
    @test result isa Vector{Float32}
    @test result == x
end

@testitem "detect_gpu_hardware on Linux uses direct device and vendor hints" begin
    using GPUEnv
    using Test

    vendor_paths = ["/sys/class/drm/card0/device/vendor", "/sys/class/drm/card1/device/vendor"]
    vendor_ids = Dict(
        vendor_paths[1] => "0x10de\n",
        vendor_paths[2] => "0x8086\n",
    )
    present_paths = Set(["/dev/kfd", "/etc/OpenCL/vendors"])

    detected = GPUEnv.detect_gpu_hardware(
        ;
        os = :linux,
        ispath = path -> path in present_paths,
        read_text = path -> vendor_ids[path],
        linux_vendor_paths = vendor_paths,
    )

    @test detected[:CUDA]
    @test detected[:AMDGPU]
    @test detected[:oneAPI]
    @test detected[:OpenCL]
    @test !detected[:Metal]
end

@testitem "detect_gpu_hardware on Windows parses controller names" begin
    using GPUEnv
    using Test

    detected = GPUEnv.detect_gpu_hardware(
        ;
        os = :windows,
        windows_video_output = "NVIDIA RTX\nAMD Radeon\nIntel Arc\n",
    )

    @test detected[:CUDA]
    @test detected[:AMDGPU]
    @test detected[:oneAPI]
    @test detected[:OpenCL]
    @test !detected[:Metal]
end

@testitem "default_backend_probe uses direct detection before fallback" begin
    using GPUEnv
    using Test

    hardware = Dict(backend => false for backend in GPUEnv.NATIVE_BACKENDS)
    hardware[:CUDA] = true

    @test GPUEnv.default_backend_probe(:CUDA; hardware, fallback = _ -> error("fallback should not run"))
    @test GPUEnv.default_backend_probe(:OpenCL; hardware, fallback = backend -> backend === :OpenCL)
end

@testitem "_backend_functional handles OpenCL platform queries" begin
    using GPUEnv
    using Test

    module FakeOpenCLGood
    module cl
        platforms() = [:platform]
        devices(::Symbol) = [:device]
    end
    end

    module FakeOpenCLEmpty
    module cl
        platforms() = Symbol[]
        devices(::Symbol) = Symbol[]
    end
    end

    module FakeOpenCLCompileFail
    struct FakeArray
        data::Vector{Float32}
    end
    module cl
        platforms() = [:platform]
        devices(::Symbol) = [:device]
    end
    zeros(::Type{Float32}, ::Int) = FakeArray([1.0f0])
    Base.broadcast(::typeof(+), ::FakeArray, ::FakeArray) = error("SPIR-V unavailable")
    end

    module FakeOpenCLSmokePass
    struct FakeArray
        data::Vector{Float32}
    end
    module cl
        platforms() = [:platform]
        devices(::Symbol) = [:device]
    end
    zeros(::Type{Float32}, ::Int) = FakeArray([1.0f0])
    Base.broadcast(::typeof(+), a::FakeArray, b::FakeArray) = FakeArray(a.data .+ b.data)
    end

    @test GPUEnv._backend_functional(FakeOpenCLGood, :OpenCL)
    @test !GPUEnv._backend_functional(FakeOpenCLEmpty, :OpenCL)
    @test !GPUEnv._backend_functional(FakeOpenCLCompileFail, :OpenCL)
    @test GPUEnv._backend_functional(FakeOpenCLSmokePass, :OpenCL)
end

@testitem "_write_environment! copies manifest when source exists" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}("name" => "TestPkg")
    src_dir = mktempdir()
    manifest_src = joinpath(src_dir, "Manifest.toml")
    write(manifest_src, "# fake manifest\n")
    env_dir = mktempdir()

    GPUEnv._write_environment!(project_data, manifest_src, env_dir)

    @test isfile(joinpath(env_dir, "Manifest.toml"))
    @test read(joinpath(env_dir, "Manifest.toml"), String) == "# fake manifest\n"
end

@testitem "_write_environment! preserves versioned manifest filename" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}("name" => "TestPkg")
    src_dir = mktempdir()
    manifest_name = "Manifest-v$(VERSION.major).$(VERSION.minor).toml"
    manifest_src = joinpath(src_dir, manifest_name)
    write(manifest_src, "# versioned manifest\n")

    env_dir = mktempdir()
    write(joinpath(env_dir, "Manifest.toml"), "# stale plain\n")
    write(joinpath(env_dir, "Manifest-v0.0.toml"), "# stale versioned\n")

    GPUEnv._write_environment!(project_data, manifest_src, env_dir)

    @test isfile(joinpath(env_dir, manifest_name))
    @test read(joinpath(env_dir, manifest_name), String) == "# versioned manifest\n"
    @test !isfile(joinpath(env_dir, "Manifest.toml"))
    @test !isfile(joinpath(env_dir, "Manifest-v0.0.toml"))
end

@testitem "_write_environment! removes stale manifest when no source" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}("name" => "TestPkg")
    env_dir = mktempdir()
    write(joinpath(env_dir, "Manifest.toml"), "# stale\n")
    write(joinpath(env_dir, "Manifest-v0.0.toml"), "# stale versioned\n")

    GPUEnv._write_environment!(project_data, nothing, env_dir)

    @test !isfile(joinpath(env_dir, "Manifest.toml"))
    @test !isfile(joinpath(env_dir, "Manifest-v0.0.toml"))
end

@testitem "_manifest_deps_fingerprint captures manifest metadata and path deps" begin
    using GPUEnv
    using Test

    manifest_dir = mktempdir()
    local_dep = mkpath(joinpath(manifest_dir, "deps", "LocalPkg"))
    manifest_path = joinpath(manifest_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        julia_version = \"$(VERSION.major).$(VERSION.minor).$(VERSION.patch)\"
        manifest_format = \"2.0\"
        project_hash = \"abc123\"

        [[deps.LocalPkg]]
        path = \"deps/LocalPkg\"
        version = \"0.1.0\"

        [[deps.RemotePkg]]
        git-tree-sha1 = \"deadbeef\"
        repo-rev = \"main\"
        repo-url = \"https://example.invalid/RemotePkg.jl\"
        """,
    )

    fingerprint = GPUEnv._manifest_deps_fingerprint(manifest_path)

    @test fingerprint["manifest_meta"]["project_hash"] == "abc123"
    @test fingerprint["deps"]["LocalPkg"]["version"] == "0.1.0"
    @test fingerprint["deps"]["LocalPkg"]["path"] == normpath(local_dep)
    @test fingerprint["deps"]["RemotePkg"]["git-tree-sha1"] == "deadbeef"
    @test fingerprint["deps"]["RemotePkg"]["repo-rev"] == "main"
end

@testitem "_can_reuse_persisted_env requires matching manifest and sync state" begin
    using GPUEnv
    using Test

    source_dir = mktempdir()
    manifest_path = joinpath(source_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        julia_version = \"$(VERSION.major).$(VERSION.minor).$(VERSION.patch)\"
        manifest_format = \"2.0\"
        project_hash = \"reuse\"

        [[deps.Example]]
        version = \"1.0.0\"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict("Example" => "7876af07-990d-54b4-ab0e-23690620f79a"),
    )
    sync_state = GPUEnv._sync_state_data(project_data, manifest_path, [:CUDA])
    env_dir = mktempdir()

    GPUEnv._write_environment!(project_data, manifest_path, env_dir)
    GPUEnv._write_sync_state!(env_dir, sync_state)

    @test GPUEnv._can_reuse_persisted_env(sync_state, env_dir; persisted = true)

    rm(joinpath(env_dir, "Manifest.toml"); force = true)
    @test !GPUEnv._can_reuse_persisted_env(sync_state, env_dir; persisted = true)
end

@testitem "_can_reuse_persisted_env allows env-local manifest when no source manifest was tracked" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "deps" => Dict("Example" => "7876af07-990d-54b4-ab0e-23690620f79a"),
    )
    sync_state = GPUEnv._sync_state_data(project_data, nothing, Symbol[])
    env_dir = mktempdir()

    GPUEnv._write_environment!(project_data, nothing, env_dir)
    GPUEnv._write_sync_state!(env_dir, sync_state)

    @test GPUEnv._can_reuse_persisted_env(sync_state, env_dir; persisted = true)

    write(
        joinpath(env_dir, "Manifest.toml"),
        """
        julia_version = "$(VERSION.major).$(VERSION.minor).$(VERSION.patch)"
        manifest_format = "2.0"
        project_hash = "overlay"
        """,
    )

    @test GPUEnv._can_reuse_persisted_env(sync_state, env_dir; persisted = true)
end

@testitem "_restore_overlay_sources! rewrites changed sources to disk" begin
    using GPUEnv
    using TOML
    using Test

    env_dir = mktempdir()
    project_path = joinpath(env_dir, "Project.toml")
    write(
        project_path,
        """
        [deps]
        Foo = "00000000-0000-0000-0000-000000000011"

        [sources]
        Foo = { path = \"/tmp/old-foo\" }
        """,
    )

    baseline_project = Dict{String, Any}(
        "deps" => Dict{String, Any}("Foo" => "00000000-0000-0000-0000-000000000011"),
        "sources" => Dict{String, Any}("Foo" => Dict{String, Any}("path" => "/tmp/new-foo")),
    )

    @test GPUEnv._restore_overlay_sources!(project_path, baseline_project)

    restored = TOML.parsefile(project_path)
    @test restored["sources"]["Foo"]["path"] == "/tmp/new-foo"
end

@testitem "_restore_overlay_sources! returns false when nothing changed" begin
    using GPUEnv
    using TOML
    using Test

    env_dir = mktempdir()
    project_path = joinpath(env_dir, "Project.toml")
    write(
        project_path,
        """
        [deps]
        Foo = "00000000-0000-0000-0000-000000000011"

        [sources]
        Foo = { path = \"/tmp/same-foo\" }
        """,
    )

    baseline_project = Dict{String, Any}(
        "deps" => Dict{String, Any}("Foo" => "00000000-0000-0000-0000-000000000011"),
        "sources" => Dict{String, Any}("Foo" => Dict{String, Any}("path" => "/tmp/same-foo")),
    )

    before = read(project_path, String)
    @test GPUEnv._restore_overlay_sources!(project_path, baseline_project) == false
    @test read(project_path, String) == before
end

@testitem "_install_backend! returns false for unknown backend" begin
    using GPUEnv
    using Test

    @test GPUEnv._install_backend!(:NoSuchBackend, devnull) == false
end

@testitem "_remove_backend! returns false for unknown backend" begin
    using GPUEnv
    using Test

    @test GPUEnv._remove_backend!(:NoSuchBackend, devnull) == false
end

@testitem "_install_package! returns false for invalid package" begin
    using GPUEnv
    using Pkg
    using Test

    env_dir = mktempdir()
    write(joinpath(env_dir, "Project.toml"), "name = \"PkgInstallFailure\"\nuuid = \"00000000-0000-0000-0000-000000000097\"\n")
    prev = Base.active_project()
    try
        Pkg.activate(env_dir)
        @test_logs (:warn, r"Could not install backend package") GPUEnv._install_package!("DefinitelyNotARegisteredPackageXYZ", devnull) == false
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "_install_packages! returns false when batch install fails" begin
    using GPUEnv
    using Pkg
    using Test

    env_dir = mktempdir()
    write(joinpath(env_dir, "Project.toml"), "name = \"PkgBatchInstallFailure\"\nuuid = \"00000000-0000-0000-0000-000000000095\"\n")
    prev = Base.active_project()
    try
        Pkg.activate(env_dir)
        @test GPUEnv._install_packages!(["DefinitelyNotARegisteredPackageXYZ"], devnull) == false
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "_install_backends! warns for unknown backends and falls back to individual installs" begin
    using GPUEnv
    using Test

    mutable struct FallbackInstallIO <: IO
        batch_attempts::Vector{Vector{String}}
        individual_attempts::Vector{String}
    end
    FallbackInstallIO() = FallbackInstallIO(Vector{Vector{String}}(), String[])

    @eval GPUEnv begin
        function _install_packages!(package_names::AbstractVector{<:AbstractString}, io::$FallbackInstallIO)
            push!(io.batch_attempts, collect(String.(package_names)))
            return false
        end

        function _install_package!(package_name::AbstractString, io::$FallbackInstallIO)
            push!(io.individual_attempts, package_name)
            return package_name == "JLArrays"
        end
    end

    io = FallbackInstallIO()
    @test_logs (:warn, r"Some of the requested backends do not have known packages to install") (
        :warn,
        r"Failed to install backend package",
    ) begin
        @test GPUEnv._install_backends!([:JLArrays, :NoSuchBackend, :CUDA], io) == [:JLArrays]
    end

    @test io.batch_attempts == [["JLArrays", "CUDA"]]
    @test io.individual_attempts == ["JLArrays", "CUDA"]
end

@testitem "_remove_package! returns false for missing package" begin
    using GPUEnv
    using Pkg
    using Test

    env_dir = mktempdir()
    write(joinpath(env_dir, "Project.toml"), "name = \"PkgRemoveFailure\"\nuuid = \"00000000-0000-0000-0000-000000000096\"\n")
    prev = Base.active_project()
    try
        Pkg.activate(env_dir)
        @test_logs (:warn, r"Could not remove backend package") GPUEnv._remove_package!("DefinitelyNotInstalledPackageXYZ", devnull) == false
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "_install_and_filter_backends! only_first removes non-functional backends" begin
    using GPUEnv
    using Pkg
    using Test

    env_dir = mktempdir()
    write(joinpath(env_dir, "Project.toml"), "name = \"OnlyFirstRemove\"\nuuid = \"00000000-0000-0000-0000-000000000132\"\n")

    previous_project = Base.active_project()
    try
        Pkg.activate(env_dir)
        installed, functional = GPUEnv._install_and_filter_backends!([:JLArrays], _ -> false, true, devnull)
        @test installed == Symbol[]
        @test functional == Symbol[]
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "_install_and_filter_backends! only_first stops at first functional backend" begin
    using GPUEnv
    using Pkg
    using Test

    env_dir = mktempdir()
    write(joinpath(env_dir, "Project.toml"), "name = \"OnlyFirstKeep\"\nuuid = \"00000000-0000-0000-0000-000000000133\"\n")

    previous_project = Base.active_project()
    try
        Pkg.activate(env_dir)
        installed, functional = GPUEnv._install_and_filter_backends!([:JLArrays], _ -> true, true, devnull)
        @test installed == [:JLArrays]
        @test functional == [:JLArrays]
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "_preferred_manifest_path ignores non-matching versioned manifests" begin
    using GPUEnv
    using Test

    dir = mktempdir()
    write(joinpath(dir, "Manifest-v0.0.toml"), "# unrelated\n")
    plain_manifest = joinpath(dir, "Manifest.toml")
    write(plain_manifest, "# plain\n")

    @test GPUEnv._preferred_manifest_path(dir) == plain_manifest
end

@testitem "_workspace_manifest_path finds workspace root manifest" begin
    using GPUEnv
    using Test

    root = mktempdir()
    write(joinpath(root, "Project.toml"), "[workspace]\nprojects = [\"test\"]\n")
    manifest = joinpath(root, "Manifest.toml")
    write(manifest, "# workspace\n")
    nested = mkpath(joinpath(root, "test"))

    @test GPUEnv._workspace_manifest_path(nested) == manifest
end

@testitem "_workspace_manifest_path returns nothing without workspace" begin
    using GPUEnv
    using Test

    root = mktempdir()
    write(joinpath(root, "Project.toml"), "name = \"NoWorkspace\"\n")
    nested = mkpath(joinpath(root, "a", "b"))

    @test GPUEnv._workspace_manifest_path(nested) === nothing
end

@testitem "_find_parent_manifest_path requires parent project" begin
    using GPUEnv
    using Test

    root = mktempdir()
    parent = mkpath(joinpath(root, "parent"))
    child = mkpath(joinpath(parent, "child"))
    manifest = joinpath(parent, "Manifest.toml")
    write(manifest, "# parent manifest\n")

    @test GPUEnv._find_parent_manifest_path(child) === nothing

    write(joinpath(parent, "Project.toml"), "name = \"Parent\"\n")
    @test GPUEnv._find_parent_manifest_path(child) == manifest
end

@testitem "_merge_project_tables merges sources without overwriting" begin
    using GPUEnv
    using Test

    base_project = Dict{String, Any}(
        "sources" => Dict{String, Any}(
            "Existing" => Dict{String, Any}("path" => "/existing"),
        ),
    )
    extra_project = Dict{String, Any}(
        "sources" => Dict{String, Any}(
            "Existing" => Dict{String, Any}("path" => "/ignored"),
            "Added" => Dict{String, Any}("path" => "/added"),
            "Skipped" => Dict{String, Any}("path" => "/skipped"),
        ),
    )

    merged = GPUEnv._merge_project_tables(base_project, extra_project; exclude_packages = Set(["Skipped"]))

    @test merged["sources"]["Existing"]["path"] == "/existing"
    @test merged["sources"]["Added"]["path"] == "/added"
    @test !haskey(merged["sources"], "Skipped")
end

@testitem "_augment_source_project merges path dependency projects" begin
    using GPUEnv
    using Test

    root = mktempdir()
    dep_root = mkpath(joinpath(root, "deps", "LocalDep"))
    write(
        joinpath(dep_root, "Project.toml"),
        """
        name = "LocalDep"
        uuid = "00000000-0000-0000-0000-000000000130"

        [deps]
        ExtraDep = "00000000-0000-0000-0000-000000000131"

        [sources]
        ExtraDep = { path = "../ExtraDep" }
        """,
    )
    extra_root = mkpath(joinpath(root, "deps", "ExtraDep"))
    write(joinpath(extra_root, "Project.toml"), "name = \"ExtraDep\"\n")

    manifest_path = joinpath(root, "Manifest.toml")
    write(
        manifest_path,
        """
        [[deps.LocalDep]]
        path = "deps/LocalDep"
        version = "0.1.0"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("LocalDep" => "00000000-0000-0000-0000-000000000130"),
    )

    augmented = GPUEnv._augment_source_project(project_data, root, manifest_path)

    @test augmented["sources"]["LocalDep"]["path"] == abspath(dep_root)
    @test augmented["deps"]["ExtraDep"] == "00000000-0000-0000-0000-000000000131"
    @test augmented["sources"]["ExtraDep"]["path"] == abspath(extra_root)
end

@testitem "_augment_source_project falls back to workspace manifest paths" begin
    using GPUEnv
    using Test

    root = mktempdir()
    write(
        joinpath(root, "Project.toml"),
        """
        name = "WorkspaceRoot"
        uuid = "00000000-0000-0000-0000-000000000140"

        [workspace]
        projects = ["benchmark", "LocalDep"]
        """,
    )

    dep_root = mkpath(joinpath(root, "LocalDep"))
    write(
        joinpath(dep_root, "Project.toml"),
        """
        name = "LocalDep"
        uuid = "00000000-0000-0000-0000-000000000141"
        version = "0.1.0"
        """,
    )

    benchmark_root = mkpath(joinpath(root, "benchmark"))
    write(
        joinpath(benchmark_root, "Project.toml"),
        """
        name = "BenchmarkEnv"
        uuid = "00000000-0000-0000-0000-000000000142"

        [deps]
        LocalDep = "00000000-0000-0000-0000-000000000141"
        """,
    )
    write(joinpath(benchmark_root, "Manifest.toml"), "manifest_format = \"2.0\"\n")

    write(
        joinpath(root, "Manifest.toml"),
        """
        manifest_format = "2.0"

        [[deps.LocalDep]]
        path = "LocalDep"
        uuid = "00000000-0000-0000-0000-000000000141"
        version = "0.1.0"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("LocalDep" => "00000000-0000-0000-0000-000000000141"),
    )

    augmented = GPUEnv._augment_source_project(project_data, benchmark_root, joinpath(benchmark_root, "Manifest.toml"))

    @test augmented["sources"]["LocalDep"]["path"] == abspath(dep_root)
end

@testitem "_git_repo_root finds .git directory at parent" begin
    using GPUEnv
    using Test

    root = mktempdir()
    mkpath(joinpath(root, ".git"))
    subdir = mkpath(joinpath(root, "sub", "dir"))

    found = GPUEnv._git_repo_root(subdir)
    @test found == root
end

@testitem "_git_repo_root finds .git file (worktree)" begin
    using GPUEnv
    using Test

    root = mktempdir()
    write(joinpath(root, ".git"), "gitdir: /some/other/path\n")
    subdir = mkpath(joinpath(root, "a"))

    found = GPUEnv._git_repo_root(subdir)
    @test found == root
end

@testitem "_maybe_warn_about_persisted_env warns when env is not gitignored" begin
    using GPUEnv
    using Pkg
    using Test

    root = mktempdir()
    run(`git -C $root init -q`)
    env_dir = mkpath(joinpath(root, "gpu_env"))

    @test_logs (:warn, r"not ignored") begin
        @test GPUEnv._maybe_warn_about_persisted_env(root, env_dir; persisted = true, warn_if_unignored = true)
    end

    project_root = mkpath(joinpath(root, "pkg"))
    write(
        joinpath(project_root, "Project.toml"),
        """
        name = "WarnPkg"
        uuid = "00000000-0000-0000-0000-000000000134"
        version = "0.1.0"
        """,
    )

    previous_project = Base.active_project()
    try
        Pkg.activate(project_root)
        @test_logs (:warn, r"not ignored") begin
            result = GPUEnv.sync_test_env(
                ;
                persist = true,
                dry_run = true,
                include_jlarrays = false,
                probe = _ -> false,
            )
            @test result.warned_about_gitignore
        end
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "_is_subpath" begin
    using GPUEnv
    using Test

    root = mktempdir()
    sub = mkpath(joinpath(root, "a", "b"))

    @test GPUEnv._is_subpath(root, root)
    @test GPUEnv._is_subpath(sub, root)
    @test !GPUEnv._is_subpath(root, sub)
    @test !GPUEnv._is_subpath(mktempdir(), root)
end

@testitem "Full sync installs JLArrays" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        include_jlarrays = true,
        probe = backend -> backend === :JLArrays,
        checker = _ -> true,
    )
    @test :JLArrays in result.installed_backends
    @test :JLArrays in result.functional_backends
    @test isfile(result.project_path)
    @test !result.dry_run
end

@testitem "Full sync removes non-functional backends" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        include_jlarrays = true,
        probe = backend -> backend === :JLArrays,
        checker = _ -> false,
    )
    @test isempty(result.installed_backends)
    @test isempty(result.functional_backends)
end

@testitem "_sanitize_environment_project strips package-only metadata" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000001",
        "version" => "1.0.0",
        "authors" => ["Alice"],
        "workspace" => Dict("projects" => ["test"]),
        "weakdeps" => Dict("CUDA" => "052768ef-5323-5732-b1bb-66c8b64840ba"),
        "extensions" => Dict("CUDAExt" => ["CUDA"]),
        "extras" => Dict("Test" => "8dfed614-e22c-5e08-85e1-65c5234f0b40"),
        "targets" => Dict("test" => ["Test"]),
        "deps" => Dict("Foo" => "00000000-0000-0000-0000-000000000002"),
    )

    sanitized = GPUEnv._sanitize_environment_project(project_data)

    @test !haskey(sanitized, "name")
    @test !haskey(sanitized, "uuid")
    @test !haskey(sanitized, "version")
    @test !haskey(sanitized, "authors")
    @test !haskey(sanitized, "workspace")
    @test !haskey(sanitized, "weakdeps")
    @test !haskey(sanitized, "extensions")
    @test !haskey(sanitized, "extras")
    @test !haskey(sanitized, "targets")
    @test sanitized["deps"]["Foo"] == "00000000-0000-0000-0000-000000000002"
end

@testitem "_sanitize_environment_project filters compat to deps only" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000003",
        "deps" => Dict{String, Any}(
            "Foo" => "00000000-0000-0000-0000-000000000004",
        ),
        "compat" => Dict{String, Any}(
            "julia" => "1.10",
            "Foo" => "1",
            "Bar" => "2",  # not a dep — should be removed
        ),
    )

    sanitized = GPUEnv._sanitize_environment_project(project_data)

    @test sanitized["compat"]["julia"] == "1.10"
    @test sanitized["compat"]["Foo"] == "1"
    @test !haskey(sanitized["compat"], "Bar")
end

@testitem "_sanitize_environment_project removes empty compat after filtering" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000005",
        "deps" => Dict{String, Any}(),
        "compat" => Dict{String, Any}(
            "SomeWeakDep" => "1",  # not a dep — should be removed
        ),
    )

    sanitized = GPUEnv._sanitize_environment_project(project_data)

    @test !haskey(sanitized, "compat")
end

@testitem "_sanitize_environment_project filters sources to deps only" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000006",
        "deps" => Dict{String, Any}(
            "Foo" => "00000000-0000-0000-0000-000000000007",
        ),
        "sources" => Dict{String, Any}(
            "Foo" => Dict{String, Any}("path" => "/path/to/Foo"),
            "Bar" => Dict{String, Any}("path" => "/path/to/Bar"),  # not a dep
        ),
    )

    sanitized = GPUEnv._sanitize_environment_project(project_data)

    @test haskey(sanitized["sources"], "Foo")
    @test !haskey(sanitized["sources"], "Bar")
end

@testitem "_sanitize_environment_project removes empty sources after filtering" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000008",
        "deps" => Dict{String, Any}(),
        "sources" => Dict{String, Any}(
            "SomeWeakDep" => Dict{String, Any}("path" => "/path/to/SomeWeakDep"),
        ),
    )

    sanitized = GPUEnv._sanitize_environment_project(project_data)

    @test !haskey(sanitized, "sources")
end

@testitem "_sanitize_environment_project does not mutate input" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "name" => "MyPkg",
        "uuid" => "00000000-0000-0000-0000-000000000009",
        "weakdeps" => Dict("CUDA" => "052768ef-5323-5732-b1bb-66c8b64840ba"),
    )

    GPUEnv._sanitize_environment_project(project_data)

    @test haskey(project_data, "name")
    @test haskey(project_data, "uuid")
    @test haskey(project_data, "weakdeps")
end

@testitem "_develop_overlay_path_sources! catches Pkg.develop failure and returns false" begin
    using GPUEnv
    using Pkg
    using Test

    # A Project.toml with a malformed uuid causes Pkg.develop to throw an
    # ArgumentError on any Julia version, driving the catch block at line 657.
    pkg_dir = mktempdir()
    write(
        joinpath(pkg_dir, "Project.toml"),
        "name = \"BadUUIDPkg\"\nuuid = \"not-a-valid-uuid\"\n",
    )

    overlay_dir = mktempdir()
    write(
        joinpath(overlay_dir, "Project.toml"),
        "name = \"Overlay\"\nuuid = \"00000000-0000-0000-0000-000000000070\"\n",
    )

    baseline_project = Dict{String, Any}(
        "deps" => Dict{String, Any}("BadUUIDPkg" => "00000000-0000-0000-0000-000000000071"),
        "sources" => Dict{String, Any}(
            "BadUUIDPkg" => Dict{String, Any}("path" => pkg_dir),
        ),
    )

    prev = Base.active_project()
    try
        Pkg.activate(overlay_dir)
        developed = GPUEnv._develop_overlay_path_sources!(baseline_project, devnull)
        @test developed == false
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "_develop_overlay_path_sources! develops multiple path sources in one batched call" begin
    using GPUEnv
    using Pkg
    using TOML
    using Test

    function make_pkg_dir(name, uuid)
        dir = mktempdir()
        write(joinpath(dir, "Project.toml"), "name = \"$name\"\nuuid = \"$uuid\"\nversion = \"0.1.0\"\n")
        mkpath(joinpath(dir, "src"))
        write(joinpath(dir, "src", "$name.jl"), "module $name\nend\n")
        return dir
    end

    pkg_dir_a = make_pkg_dir("BatchPkgA", "00000000-0000-0000-0000-000000000072")
    pkg_dir_b = make_pkg_dir("BatchPkgB", "00000000-0000-0000-0000-000000000073")

    overlay_dir = mktempdir()
    write(
        joinpath(overlay_dir, "Project.toml"),
        "name = \"Overlay\"\nuuid = \"00000000-0000-0000-0000-000000000074\"\n",
    )

    baseline_project = Dict{String, Any}(
        "deps" => Dict{String, Any}(
            "BatchPkgA" => "00000000-0000-0000-0000-000000000072",
            "BatchPkgB" => "00000000-0000-0000-0000-000000000073",
        ),
        "sources" => Dict{String, Any}(
            "BatchPkgA" => Dict{String, Any}("path" => pkg_dir_a),
            "BatchPkgB" => Dict{String, Any}("path" => pkg_dir_b),
        ),
    )

    prev = Base.active_project()
    try
        Pkg.activate(overlay_dir)
        developed = GPUEnv._develop_overlay_path_sources!(baseline_project, devnull)
        @test developed == true

        overlay_project = TOML.parsefile(joinpath(overlay_dir, "Project.toml"))
        @test haskey(get(overlay_project, "deps", Dict()), "BatchPkgA")
        @test haskey(get(overlay_project, "deps", Dict()), "BatchPkgB")
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "_develop_overlay_path_sources! falls back per-source when the batch fails" begin
    using GPUEnv
    using Pkg
    using TOML
    using Test

    good_dir = mktempdir()
    write(
        joinpath(good_dir, "Project.toml"),
        "name = \"GoodPkg\"\nuuid = \"00000000-0000-0000-0000-000000000075\"\nversion = \"0.1.0\"\n",
    )
    mkpath(joinpath(good_dir, "src"))
    write(joinpath(good_dir, "src", "GoodPkg.jl"), "module GoodPkg\nend\n")

    bad_dir = mktempdir()
    write(
        joinpath(bad_dir, "Project.toml"),
        "name = \"BadUUIDPkg2\"\nuuid = \"not-a-valid-uuid\"\n",
    )

    overlay_dir = mktempdir()
    write(
        joinpath(overlay_dir, "Project.toml"),
        "name = \"Overlay\"\nuuid = \"00000000-0000-0000-0000-000000000076\"\n",
    )

    baseline_project = Dict{String, Any}(
        "deps" => Dict{String, Any}(
            "GoodPkg" => "00000000-0000-0000-0000-000000000075",
            "BadUUIDPkg2" => "00000000-0000-0000-0000-000000000077",
        ),
        "sources" => Dict{String, Any}(
            "GoodPkg" => Dict{String, Any}("path" => good_dir),
            "BadUUIDPkg2" => Dict{String, Any}("path" => bad_dir),
        ),
    )

    prev = Base.active_project()
    try
        Pkg.activate(overlay_dir)
        developed = GPUEnv._develop_overlay_path_sources!(baseline_project, devnull)
        # The batched call fails because of BadUUIDPkg2, so the fallback loop
        # runs and still develops GoodPkg individually.
        @test developed == true

        overlay_project = TOML.parsefile(joinpath(overlay_dir, "Project.toml"))
        @test haskey(get(overlay_project, "deps", Dict()), "GoodPkg")
        @test !haskey(get(overlay_project, "deps", Dict()), "BadUUIDPkg2")
    finally
        prev === nothing || Pkg.activate(dirname(prev))
    end
end

@testitem "activate pushes stripped local-pkg root to LOAD_PATH on Julia < 1.11 (path variant)" begin
    # Lines 429-431 in environment.jl are only reachable on Julia < 1.11.
    # The `if` gates the whole body so this is a no-op on Julia 1.11+.
    if VERSION < v"1.11"
        using GPUEnv
        using Pkg
        using Test

        root = mktempdir()
        mkpath(joinpath(root, "src"))
        write(joinpath(root, "src", "FakePkg.jl"), "module FakePkg\nend\n")

        local_pkg = mkpath(joinpath(root, "LocalDep"))
        write(
            joinpath(local_pkg, "Project.toml"),
            "name = \"LocalDep\"\nuuid = \"00000000-0000-0000-0000-000000000072\"\nversion = \"0.1.0\"\n",
        )

        write(
            joinpath(root, "Project.toml"),
            """
            name = "FakePkg"
            uuid = "00000000-0000-0000-0000-000000000073"
            version = "0.1.0"

            [deps]
            LocalDep = "00000000-0000-0000-0000-000000000072"

            [compat]
            julia = "1.10"
            """,
        )
        write(
            joinpath(root, "Manifest.toml"),
            """
            manifest_format = "2.0"

            [[deps.LocalDep]]
            path = "LocalDep"
            uuid = "00000000-0000-0000-0000-000000000072"
            version = "0.1.0"
            """,
        )

        original_load_path = copy(Base.LOAD_PATH)
        prev = Base.active_project()
        try
            # activate passes restore_previous_project = dry_run = false, enabling
            # the LOAD_PATH push at lines 429-431 of environment.jl.
            GPUEnv.activate(
                ;
                path = root,
                include_jlarrays = false,
                probe = _ -> false,
            )
            @test abspath(local_pkg) in Base.LOAD_PATH
        finally
            prev === nothing || Pkg.activate(dirname(prev))
            empty!(Base.LOAD_PATH)
            append!(Base.LOAD_PATH, original_load_path)
        end
    end
end

@testitem "activate pushes stripped local-pkg root to LOAD_PATH on Julia < 1.11 (active project / workspace variant)" begin
    # Lines 258 and 302-304 in environment.jl are only reachable on Julia < 1.11.
    # Line 258 executes when the workspace manifest differs from the local (test/)
    # manifest: the workspace manifest is used first, then the local manifest is
    # checked separately for additional path-only deps.
    # The `if` gates the whole body so this is a no-op on Julia 1.11+.
    if VERSION < v"1.11"
        using GPUEnv
        using Pkg
        using Test

        # Workspace root — Manifest intentionally has no path deps so the first strip
        # call finds nothing.
        root = mktempdir()
        write(
            joinpath(root, "Project.toml"),
            "[workspace]\nprojects = [\"test\"]\n",
        )
        write(joinpath(root, "Manifest.toml"), "manifest_format = \"2.0\"\n")

        # Local sub-package referenced only from the test/ Manifest.
        local_pkg = mkpath(joinpath(root, "LocalDep"))
        write(
            joinpath(local_pkg, "Project.toml"),
            "name = \"LocalDep\"\nuuid = \"00000000-0000-0000-0000-000000000074\"\nversion = \"0.1.0\"\n",
        )

        # Test project — lists LocalDep as a dep, and its Manifest marks it as path-only.
        test_dir = mkpath(joinpath(root, "test"))
        write(
            joinpath(test_dir, "Project.toml"),
            """
            [deps]
            LocalDep = "00000000-0000-0000-0000-000000000074"
            """,
        )
        write(
            joinpath(test_dir, "Manifest.toml"),
            """
            manifest_format = "2.0"

            [[deps.LocalDep]]
            path = "../LocalDep"
            uuid = "00000000-0000-0000-0000-000000000074"
            version = "0.1.0"
            """,
        )

        original_load_path = copy(Base.LOAD_PATH)
        prev = Base.active_project()
        try
            Pkg.activate(test_dir)
            # activate passes restore_previous_project = false, so after activation:
            # line 258 runs the second _strip_unregistered_path_deps! call (for the
            # local test Manifest that differs from the workspace Manifest), and
            # lines 302-304 push the stripped package root onto Base.LOAD_PATH.
            GPUEnv.activate(
                ;
                include_jlarrays = false,
                probe = _ -> false,
            )
            @test abspath(local_pkg) in Base.LOAD_PATH
        finally
            prev === nothing || Pkg.activate(dirname(prev))
            empty!(Base.LOAD_PATH)
            append!(Base.LOAD_PATH, original_load_path)
        end
    end
end

@testitem "_strip_unregistered_path_deps! removes path-only dep and returns its path" begin
    using GPUEnv
    using Test

    manifest_dir = mktempdir()
    pkg_dir = mkpath(joinpath(manifest_dir, "LocalPkg"))
    manifest_path = joinpath(manifest_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        manifest_format = "2.0"

        [[deps.LocalPkg]]
        path = "LocalPkg"
        uuid = "00000000-0000-0000-0000-000000000050"
        version = "0.1.0"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}(
            "LocalPkg" => "00000000-0000-0000-0000-000000000050",
            "RegisteredPkg" => "00000000-0000-0000-0000-000000000051",
        ),
        "compat" => Dict{String, Any}("LocalPkg" => "0.1", "RegisteredPkg" => "1"),
    )

    stripped = GPUEnv._strip_unregistered_path_deps!(project_data, manifest_path)

    @test !haskey(project_data["deps"], "LocalPkg")
    @test haskey(project_data["deps"], "RegisteredPkg")
    @test !haskey(project_data["compat"], "LocalPkg")
    @test haskey(project_data["compat"], "RegisteredPkg")
    @test stripped == [abspath(pkg_dir)]
end

@testitem "_strip_unregistered_path_deps! also cleans sources entry" begin
    using GPUEnv
    using Test

    manifest_dir = mktempdir()
    pkg_dir = mkpath(joinpath(manifest_dir, "LocalPkg"))
    manifest_path = joinpath(manifest_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        manifest_format = "2.0"

        [[deps.LocalPkg]]
        path = "LocalPkg"
        uuid = "00000000-0000-0000-0000-000000000052"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("LocalPkg" => "00000000-0000-0000-0000-000000000052"),
        "sources" => Dict{String, Any}(
            "LocalPkg" => Dict{String, Any}("path" => abspath(pkg_dir)),
        ),
    )

    GPUEnv._strip_unregistered_path_deps!(project_data, manifest_path)

    @test !haskey(project_data["deps"], "LocalPkg")
    @test !haskey(project_data["sources"], "LocalPkg")
end

@testitem "_strip_unregistered_path_deps! keeps deps that have a git-tree-sha1" begin
    using GPUEnv
    using Test

    manifest_dir = mktempdir()
    manifest_path = joinpath(manifest_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        manifest_format = "2.0"

        [[deps.PinnedPkg]]
        path = "PinnedPkg"
        git-tree-sha1 = "aabbccdd"
        uuid = "00000000-0000-0000-0000-000000000053"
        version = "1.0.0"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("PinnedPkg" => "00000000-0000-0000-0000-000000000053"),
    )

    stripped = GPUEnv._strip_unregistered_path_deps!(project_data, manifest_path)

    @test haskey(project_data["deps"], "PinnedPkg")
    @test isempty(stripped)
end

@testitem "_strip_unregistered_path_deps! is a no-op for missing manifest" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("Foo" => "00000000-0000-0000-0000-000000000054"),
    )

    stripped_nothing = GPUEnv._strip_unregistered_path_deps!(project_data, nothing)
    stripped_missing = GPUEnv._strip_unregistered_path_deps!(project_data, "/nonexistent/Manifest.toml")

    @test isempty(stripped_nothing)
    @test isempty(stripped_missing)
    @test haskey(project_data["deps"], "Foo")
end

@testitem "_strip_unregistered_path_deps! ignores deps not in project" begin
    using GPUEnv
    using Test

    manifest_dir = mktempdir()
    manifest_path = joinpath(manifest_dir, "Manifest.toml")
    write(
        manifest_path,
        """
        manifest_format = "2.0"

        [[deps.TransitivePkg]]
        path = "TransitivePkg"
        uuid = "00000000-0000-0000-0000-000000000055"
        """,
    )

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("DirectPkg" => "00000000-0000-0000-0000-000000000056"),
    )

    stripped = GPUEnv._strip_unregistered_path_deps!(project_data, manifest_path)

    @test haskey(project_data["deps"], "DirectPkg")
    @test isempty(stripped)
end

@testitem "_write_environment! copies manifest even when local path sources exist" begin
    using GPUEnv
    using Test

    project_data = Dict{String, Any}(
        "deps" => Dict{String, Any}("Foo" => "00000000-0000-0000-0000-000000000020"),
        "sources" => Dict{String, Any}(
            "Foo" => Dict{String, Any}("path" => "/absolute/path/to/Foo"),
        ),
    )
    src_dir = mktempdir()
    manifest_src = joinpath(src_dir, "Manifest.toml")
    write(manifest_src, "# fake manifest\n")
    env_dir = mktempdir()

    GPUEnv._write_environment!(project_data, manifest_src, env_dir)

    # Manifest must be copied regardless of whether the project has local path sources
    @test isfile(joinpath(env_dir, "Manifest.toml"))
end
