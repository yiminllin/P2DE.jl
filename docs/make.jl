push!(LOAD_PATH,"../src/")

using P2DE
using Documenter

makedocs(
    sitename = "P2DE.jl",
    modules  = [P2DE],
    pages = [
        "Home" => "index.md"
    ]
)
        
deploydocs(;
    repo = "github.com/yiminllin/P2DE.jl",
    devbranch = "main"
)

