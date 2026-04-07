module ThirdLevel

export probe_overlay_packages

_package_available(name::AbstractString) = Base.find_package(name) !== nothing

function probe_overlay_packages()
    return Dict(
        "JSON" => _package_available("JSON"),
        "YAML" => _package_available("YAML"),
    )
end

end # module ThirdLevel
