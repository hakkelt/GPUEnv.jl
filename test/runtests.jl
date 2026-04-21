using Pkg
using TestItemRunner

const FILTER_PARTS = if isempty(ARGS)
    String[]
else
    @assert length(ARGS) == 1
    split(ARGS[1], ",")
end
const FILTER_TAGS = map(part -> Symbol(part[2:end]), filter(part -> startswith(part, ":"), FILTER_PARTS))
const FILTER_NAMES = filter(part -> !startswith(part, ":"), FILTER_PARTS)
const FILTER = isempty(FILTER_PARTS) ? ti -> true : ti -> any(tag -> tag in ti.tags, FILTER_TAGS) || any(name -> name == ti.name, FILTER_NAMES)

@run_package_tests filter = FILTER
