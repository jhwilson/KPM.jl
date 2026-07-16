# Docs build script for Documenter.jl
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using KPM
using Documenter

makedocs(
    sitename = "KPM",
    modules  = [KPM],
    pages    = ["Home" => "index.md"],
    build    = joinpath("build", "dev"),
    # single-page docs: the BdG/pairing-channel API pushed index.html past
    # Documenter's 200 KiB default
    format   = Documenter.HTML(size_threshold = 400 * 2^10,
                               size_threshold_warn = 300 * 2^10),
)
