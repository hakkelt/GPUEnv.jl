using TestItems

@testitem "Relative sources become absolute" setup = [SyncTestHelpers] begin
    using GPUEnv
    using TOML
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = joinpath(root, "test"),
        dry_run = true,
        include_jlarrays = false,
        probe = _ -> false,
    )
    data = TOML.parsefile(result.project_path)
    @test data["sources"]["Foo"]["path"] == abspath(joinpath(root, "Foo"))
    @test result.base_environment_kind == :path_project
end

@testitem "Path-only sync works from unnamed active project" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Pkg
    using Test

    root = make_fake_package()
    bootstrap = mktempdir()
    write(
        joinpath(bootstrap, "Project.toml"),
        "uuid = \"00000000-0000-0000-0000-000000000012\"\nversion = \"0.1.0\"\n",
    )

    previous_project = Base.active_project()
    try
        Pkg.activate(bootstrap)
        result = GPUEnv.sync_test_env(
            ;
            path = joinpath(root, "test"),
            dry_run = true,
            include_jlarrays = false,
            probe = _ -> false,
        )
        @test result.source_project_path == joinpath(root, "test", "Project.toml")
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "GPUEnv source becomes a local path" setup = [SyncTestHelpers] begin
    using GPUEnv
    using TOML
    using Test

    root = make_fake_package(; GPUEnv_source = :url)
    result = GPUEnv.sync_test_env(
        ;
        path = joinpath(root, "test"),
        dry_run = true,
        include_jlarrays = false,
        probe = _ -> false,
    )
    data = TOML.parsefile(result.project_path)
    source = data["sources"]["GPUEnv"]

    @test normpath(source["path"]) == dirname(dirname(pathof(GPUEnv)))
    @test !haskey(source, "url")
    @test !haskey(source, "rev")
end

@testitem "Target-based project can be overlay source" setup = [SyncTestHelpers] begin
    using GPUEnv
    using TOML
    using Test

    root = make_fake_package(; with_test_project = false, with_legacy_target = true)
    result = GPUEnv.sync_test_env(; path = root, dry_run = true, include_jlarrays = true, probe = _ -> false)
    data = TOML.parsefile(result.project_path)

    @test result.base_environment_kind == :path_project
    @test haskey(data["deps"], "JLArrays")
end

@testitem "Explicit environment_path is used as-is" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    env_dir = mktempdir()
    result = GPUEnv.sync_test_env(
        ;
        path = joinpath(root, "test"),
        environment_path = env_dir,
        dry_run = true,
        include_jlarrays = false,
        probe = _ -> false,
    )

    @test result.environment_path == abspath(env_dir)
    @test result.persisted == false
end

@testitem "Explicit environment_path with persist=true is persisted" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    env_dir = mktempdir()
    result = GPUEnv.sync_test_env(
        ;
        path = joinpath(root, "test"),
        environment_path = env_dir,
        persist = true,
        dry_run = true,
        include_jlarrays = false,
        probe = _ -> false,
    )

    @test result.environment_path == abspath(env_dir)
    @test result.persisted == true
end

@testitem "include_jlarrays auto-resolves from backends_to_test" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        backends_to_test = [:JLArrays],
        dry_run = true,
        probe = _ -> true,
    )
    @test :JLArrays in result.requested_backends
end

@testitem "include_jlarrays=false conflicts with :JLArrays in backends_to_test" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    @test_throws ArgumentError GPUEnv.sync_test_env(
        ;
        path = root,
        include_jlarrays = false,
        backends_to_test = [:JLArrays],
        dry_run = true,
    )
end

@testitem "include_jlarrays=true conflicts with :JLArrays in exclude" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    @test_throws ArgumentError GPUEnv.sync_test_env(
        ;
        path = root,
        include_jlarrays = true,
        exclude = [:JLArrays],
        dry_run = true,
    )
end

@testitem "Non-existent backend throws" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package()
    @test_throws ArgumentError GPUEnv.sync_test_env(
        ;
        path = root,
        backends_to_test = [:ASDF],
        dry_run = true,
    )
end

@testitem "Persistent env warns when not ignored" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package(; git = true)
    @test_logs (:warn, r"not ignored") begin
        result = GPUEnv.sync_test_env(; path = root, persist = true, dry_run = true, probe = _ -> false)
        @test result.persisted
        @test result.warned_about_gitignore
        @test endswith(result.environment_path, "gpu_env")
    end
end

@testitem "Persistent env warning can be suppressed" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package(; git = true)
    result = GPUEnv.sync_test_env(
        ;
        path = root,
        persist = true,
        warn_if_unignored = false,
        dry_run = true,
        probe = _ -> false,
    )
    @test !result.warned_about_gitignore
end

@testitem "Ignored persisted env does not warn" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Test

    root = make_fake_package(; git = true, ignored_gpu_env = true)
    result = GPUEnv.sync_test_env(; path = root, persist = true, dry_run = true, probe = _ -> false)
    @test !result.warned_about_gitignore
end

@testitem "activate dry_run returns SyncResult without switching project" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Pkg
    using Test

    root = make_fake_package()
    previous_project = Base.active_project()
    try
        result = GPUEnv.activate(
            ;
            path = root,
            dry_run = true,
            include_jlarrays = false,
            probe = _ -> false,
        )
        @test result isa GPUEnv.SyncResult
        @test result.dry_run
        @test Base.active_project() == previous_project
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "activate without args uses active unnamed project" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Pkg
    using TOML
    using Test

    unnamed = mktempdir()
    write(
        joinpath(unnamed, "Project.toml"),
        "uuid = \"00000000-0000-0000-0000-000000000123\"\nversion = \"0.1.0\"\n",
    )

    previous_project = Base.active_project()
    try
        Pkg.activate(unnamed)

        result = GPUEnv.activate(
            ;
            include_jlarrays = false,
            probe = _ -> false,
            checker = _ -> false,
        )

        @test result isa GPUEnv.SyncResult
        @test result.base_environment_kind == :active_project
        @test result.source_project_path == joinpath(unnamed, "Project.toml")
        @test Base.active_project() == joinpath(result.environment_path, "Project.toml")

        data = TOML.parsefile(result.project_path)
        @test !haskey(data, "name")
        @test !haskey(get(data, "deps", Dict{String, Any}()), "JLArrays")
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "activate without args dry_run restores previous project" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Pkg
    using Test

    unnamed = mktempdir()
    write(
        joinpath(unnamed, "Project.toml"),
        "uuid = \"00000000-0000-0000-0000-000000000124\"\nversion = \"0.1.0\"\n",
    )

    previous_project = Base.active_project()
    try
        Pkg.activate(unnamed)
        active_before = Base.active_project()

        result = GPUEnv.activate(
            ;
            dry_run = true,
            include_jlarrays = false,
            probe = _ -> false,
            checker = _ -> false,
        )

        @test result isa GPUEnv.SyncResult
        @test result.base_environment_kind == :active_project
        @test result.dry_run
        @test Base.active_project() == active_before
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end

@testitem "activate switches active environment" setup = [SyncTestHelpers] begin
    using GPUEnv
    using Pkg
    using Test

    root = make_fake_package()
    previous_project = Base.active_project()
    try
        result = GPUEnv.activate(
            ;
            path = root,
            include_jlarrays = false,
            probe = _ -> false,
            checker = _ -> false,
        )
        @test result isa GPUEnv.SyncResult
        @test !result.dry_run
        @test result.base_environment_kind == :path_project
        @test Base.active_project() == joinpath(result.environment_path, "Project.toml")
    finally
        previous_project === nothing || Pkg.activate(dirname(previous_project))
    end
end
