# Getting started

This page fits a model to a simulated monthly series, forecasts from it with each interval method, and compares three models on forecasts made from earlier stretches of the same series.

## Fitting and forecasting

The series has a slow upward trend, a yearly cycle and noise.

```@example guide
using ForecastBaselines, Random
Random.seed!(1)
n = 120
y = [20 + 0.1t + 5sin(2π * t / 12) + 2randn() for t in 1:n]
nothing # hide
```

A model specification holds the model's structure but no data. [`fit_baseline`](@ref) estimates its parameters, and [`forecast`](@ref) projects the fitted model forward over the requested horizons, with prediction intervals at the requested levels.

```@example guide
fitted = fit_baseline(y, ARMAModel(p = 1, q = 1))
fc = forecast(fitted; interval_method = ParametricInterval(), horizon = 1:6, levels = [0.5, 0.9])
```

The forecast stores the point forecasts in `fc.mean` and one [`ForecastInterval`](@ref) per horizon in `fc.intervals`:

```@example guide
fc.intervals[1]
```

Models fitted by numerical optimisation (ARMA, INARCH and ETS) record whether the optimiser converged. A fit that stopped at the iteration limit gives a warning once per model type, and [`has_converged`](@ref) reports it for each fit:

```@example guide
has_converged(fitted)
```

## Prediction intervals

Three methods produce prediction intervals, and [`NoInterval`](@ref) gives point forecasts only. [`ParametricInterval`](@ref), used above, uses the model's closed-form forecast distribution where one exists. [`ModelTrajectoryInterval`](@ref) simulates future paths from the fitted model, and [`EmpiricalInterval`](@ref) refits the model on shorter stretches of the series and resamples the errors those fits made. Empirical intervals work for every model, but take longer, as they refit the model many times. Here are the 90% intervals one step ahead from each of the three methods:

```@example guide
parametric = forecast(fitted; interval_method = ParametricInterval(), horizon = 1:6, levels = [0.9])
simulated = forecast(fitted; interval_method = ModelTrajectoryInterval(n_trajectories = 1000, seed = 1), horizon = 1:6, levels = [0.9])
empirical = forecast(fitted; interval_method = EmpiricalInterval(seed = 1), horizon = 1:6, levels = [0.9])
[(f.intervals[1].lower[1], f.intervals[1].upper[1]) for f in (parametric, simulated, empirical)]
```

[Models](@ref) lists which interval methods each model supports.

## Transformations and seasonality

[`transform`](@ref) wraps a model so that it is fitted to transformed data and its forecasts are transformed back. A log transformation suits series that are positive and whose variation grows with their level:

```@example guide
log_arma = ForecastBaselines.transform(ARMAModel(p = 1, q = 1); transformation = LogTransform())
forecast(fit_baseline(y, log_arma); horizon = 1:6).mean
```

[`STTransform`](@ref) removes a seasonal cycle, and optionally a trend, before fitting, and adds them back to the forecasts. This lets a model without seasonality of its own, here [`ConstantModel`](@ref), forecast a seasonal series:

```@example guide
seasonal_constant = ForecastBaselines.transform(ConstantModel(); season_trend = STTransform(s = 12, trend = true))
forecast(fit_baseline(y, seasonal_constant); horizon = 1:6).mean
```

Calling it as `ForecastBaselines.transform` avoids a clash with `transform` from DataFrames.jl when both packages are loaded.

## Comparing models

To compare models, forecast from several points in the past and score each forecast against what was observed afterwards. [`add_truth`](@ref) attaches the observed values to a forecast, and [`score`](@ref) averages a scoring rule over a collection of forecasts.

```@example guide
models = [
    "Constant" => ConstantModel(),
    "ARMA(1,1)" => ARMAModel(p = 1, q = 1),
    "ETS(A,A,A)" => ETSModel(error = "A", trend = "A", season = "A", s = 12),
]
forecasts = Dict(
    name => [
            add_truth(
                forecast(
                    fit_baseline(y[1:origin], model);
                    interval_method = EmpiricalInterval(seed = 1), horizon = 1:6,
                    levels = [0.5, 0.8, 0.9]
                ),
                y[(origin + 1):(origin + 6)]
            )
            for origin in 72:6:(n - 6)
        ]
        for (name, model) in models
)
[(name, round(score(forecasts[name], MAE()); digits = 2), round(score(forecasts[name], WIS()); digits = 2)) for (name, _) in models]
```

The mean absolute error ([`MAE`](@ref)) scores the point forecasts and the weighted interval score ([`WIS`](@ref)) the intervals; lower is better for both. ETS, the only one of the three models with a seasonal component, does best on both.

[`CvM_divergence`](@ref) measures how far the forecasts are from being calibrated, that is how far the observations are from being spread across the forecast distributions as often as the distributions imply. It is zero for perfectly calibrated forecasts.

```@example guide
[(name, round(CvM_divergence(forecasts[name]); digits = 3)) for (name, _) in models]
```
