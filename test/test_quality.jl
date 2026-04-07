using TestItems

@testitem "Documentation doctests" tags = [:quality] begin
    using Documenter, GPUEnv

    Documenter.DocMeta.setdocmeta!(GPUEnv, :DocTestSetup, :(using GPUEnv); recursive = true)
    doctest(GPUEnv; fix = false)
end

@testitem "Aqua" tags = [:quality] begin
    using Aqua, GPUEnv

    Aqua.test_all(GPUEnv)
end
