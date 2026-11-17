# ForecastBaselines.jl

ForecastBaselines.jl fits baseline forecasting models to univariate time series, produces point forecasts and prediction intervals from them, and scores the forecasts against what was observed. Every model, from carrying the last value forward to exponential smoothing, is fitted with [`fit_baseline`](@ref) and forecast with [`forecast`](@ref), so models can be swapped without changing the code around them.

The package provides:

- ten models: constant, marginal, kernel density, last similar dates, polynomial trend (OLS), increase–decrease–stable, ARMA, INARCH for counts, STL decomposition and all 30 ETS variants (see [Models](@ref));
- prediction intervals from past forecast errors ([`EmpiricalInterval`](@ref)), closed-form formulae ([`ParametricInterval`](@ref)) or simulated trajectories ([`ModelTrajectoryInterval`](@ref));
- data transformations and seasonal-trend preprocessing that wrap any model ([`transform`](@ref), [`STTransform`](@ref));
- point and interval scores, and calibration measures for evaluating forecasts ([`score`](@ref), [`CvM_divergence`](@ref)).

## Installation

ForecastBaselines.jl is in the Julia General registry:

```julia
using Pkg
Pkg.add("ForecastBaselines")
```

To use the development version instead:

```julia
Pkg.add(url = "https://github.com/epiforecasts/ForecastBaselines.jl")
```

## Where to go next

[Getting started](@ref) fits a model, forecasts with each interval method and compares models on held-out data. [Models](@ref) describes each model's formulation, parameters and settings, and the API reference documents every exported function and type.
