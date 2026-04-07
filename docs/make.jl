using Documenter
using GPUEnv

Documenter.DocMeta.setdocmeta!(GPUEnv, :DocTestSetup, :(using GPUEnv); recursive = true)

makedocs(
    sitename = "GPUEnv",
    modules = [GPUEnv],
    format = Documenter.HTML(repolink = nothing, edit_link = nothing),
    checkdocs = :exports,
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "Persistence" => "persistence.md",
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(repo = "github.com/hakkelt/GPUEnv.jl.git", devbranch = "master")
