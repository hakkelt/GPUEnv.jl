using Pkg

if VERSION < v"1.12.0-"
    package_root = normpath(joinpath(@__DIR__, ".."))
    temp_env = mktempdir(prefix = "GPUEnv-tests-")
    cp(joinpath(@__DIR__, "Project.toml"), joinpath(temp_env, "Project.toml"); force = true)
    Pkg.activate(temp_env)
    Pkg.develop(Pkg.PackageSpec(path = package_root))
    Pkg.instantiate()
end

using TestItemRunner

@run_package_tests
