# Absolute path of the authors' release. `include` resolves relative paths against the
# including file, not the working directory, so a relative argument is made absolute here.
# An empty argument ("") means $QCTN_EXTERNAL/authors, or external/authors in the main checkout.
function authors_path(arg::AbstractString)
    isempty(arg) || return abspath(expanduser(arg))
    haskey(ENV, "QCTN_EXTERNAL") && return abspath(ENV["QCTN_EXTERNAL"], "authors")
    common = readchomp(`git -C $(@__DIR__) rev-parse --path-format=absolute --git-common-dir`)
    return joinpath(dirname(common), "external", "authors")
end
