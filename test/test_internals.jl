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
    result = backend(x)
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

@testitem "_can_reuse_persisted_env allows manifest-free state when no source manifest was tracked" begin
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

    @test augmented["deps"]["ExtraDep"] == "00000000-0000-0000-0000-000000000131"
    @test augmented["sources"]["ExtraDep"]["path"] == abspath(extra_root)
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
    @test :JLArrays in result.installed_backends
    @test isempty(result.functional_backends)
end
