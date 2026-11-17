using Documenter
using ForecastBaselines

DocMeta.setdocmeta!(ForecastBaselines, :DocTestSetup, :(using ForecastBaselines); recursive = true)

makedocs(;
    modules = [ForecastBaselines],
    sitename = "ForecastBaselines.jl",
    authors = "Manuel Stapper and contributors",
    format = Documenter.HTML(;
        canonical = "https://epiforecasts.github.io/ForecastBaselines.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Models" => "models.md",
        "API reference" => [
            "Forecasts and fitting" => "api/forecasts.md",
            "Model specifications" => "api/models.md",
            "Prediction intervals" => "api/intervals.md",
            "Transformations" => "api/transformations.md",
            "Evaluation" => "api/evaluation.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/epiforecasts/ForecastBaselines.jl",
    devbranch = "main",
    push_preview = false,
)
