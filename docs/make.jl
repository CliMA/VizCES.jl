using VizCES
using Documenter

makedocs(;
    modules=[VizCES],
    authors="Ignacio <ilopezgo@caltech.edu> and contributors",
    repo="https://github.com/ilopezgp/VizCES.jl/blob/{commit}{path}#L{line}",
    sitename="VizCES.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ilopezgp.github.io/VizCES.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ilopezgp/VizCES.jl",
)
