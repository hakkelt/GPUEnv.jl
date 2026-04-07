using TestItems

@testmodule SyncTestHelpers begin
    export make_fake_package

    function make_fake_package(
            ;
            with_test_project::Bool = true,
            with_legacy_target::Bool = false,
            git::Bool = false,
            ignored_gpu_env::Bool = false,
            GPUEnv_source::Union{Nothing, Symbol} = nothing,
        )
        root = mktempdir()
        mkpath(joinpath(root, "src"))

        write(
            joinpath(root, "src", "FakePkg.jl"),
            """module FakePkg
            end
            """,
        )

        if with_legacy_target
            write(
                joinpath(root, "Project.toml"),
                """name = "FakePkg"
                uuid = "00000000-0000-0000-0000-000000000010"
                version = "0.1.0"

                [extras]
                Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

                [targets]
                test = ["Test"]

                [compat]
                julia = "1.10"
                """,
            )
        else
            write(
                joinpath(root, "Project.toml"),
                """name = "FakePkg"
                uuid = "00000000-0000-0000-0000-000000000010"
                version = "0.1.0"

                [compat]
                julia = "1.10"
                """,
            )
        end

        if with_test_project
            mkpath(joinpath(root, "Foo", "src"))
            write(
                joinpath(root, "Foo", "Project.toml"),
                """name = "Foo"
                uuid = "00000000-0000-0000-0000-000000000011"
                version = "0.1.0"
                """,
            )
            write(
                joinpath(root, "Foo", "src", "Foo.jl"),
                """module Foo
                end
                """,
            )

            mkpath(joinpath(root, "test"))
            test_project = """name = \"FakePkgTests\"
            version = \"0.1.0\"

            [deps]
            Test = \"8dfed614-e22c-5e08-85e1-65c5234f0b40\"
            Foo = \"00000000-0000-0000-0000-000000000011\"
            """

            if GPUEnv_source !== nothing
                test_project *= "GPUEnv = \"78a0b619-6146-4252-b244-0f81c54be577\"\n"
            end

            test_project *= """

            [sources]
            Foo = { path = \"../Foo\" }
            """

            if GPUEnv_source === :url
                test_project *= "GPUEnv = { url = \"https://github.com/example/GPUEnv.jl.git\", rev = \"main\" }\n"
            elseif GPUEnv_source === :path
                test_project *= "GPUEnv = { path = \"../GPUEnv\" }\n"
            end

            write(
                joinpath(root, "test", "Project.toml"),
                test_project,
            )
        end

        if git
            run(`git -C $root init -q`)
            if ignored_gpu_env
                write(joinpath(root, ".gitignore"), "gpu_env/\n")
            end
        end

        return root
    end
end
