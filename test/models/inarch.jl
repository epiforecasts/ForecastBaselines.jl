@testitem "INARCH Model Tests" setup = [TestData] tags = [:models, :inarch] begin
    using Random, Distributions, Dates
    # Test INARCH(1) without seasonality
    model = INARCHModel(p = 1, s = 0, nb = false)
    fitted = fit_baseline(count_data, model)

    @test fitted isa INARCHFitted
    @test length(fitted.par.α) == 1
    @test fitted.par.β0 > 0

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3
    @test all(fc .>= 0)  # Count data should be non-negative

    # Test trajectory intervals (only method available for INARCH)
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true), 1:2, [0.8]
    )
    @test size(fc_traj) == (50, 2)
    @test all(fc_traj .>= 0)  # All trajectories should be non-negative

    # Test INARCH with negative binomial
    model_nb = INARCHModel(p = 1, s = 0, nb = true)
    fitted_nb = fit_baseline(count_data, model_nb)
    @test fitted_nb.par.ϕ > 0

    # Test INARCH with seasonality
    model_seasonal = INARCHModel(p = 1, s = 12, k = 1, nb = false)
    fitted_seasonal = fit_baseline(count_data, model_seasonal)
    @test length(fitted_seasonal.par.γ) == 2  # sin + cos coefficients
end

@testitem "INARCH forecasts evaluate each harmonic at its own frequency" setup = [TestData] tags = [:models, :inarch] begin
    using Random, Distributions, Dates
    # Without autoregression the forecast is the intercept times the
    # seasonal factor, which repeats every s/2 steps under the second
    # harmonic alone.
    model = INARCHModel(p = 1, s = 12, k = 2)
    par = INARCHParameter(5.0, [0.0], 0.0, [0.0, 0.0, 0.0, 0.7])
    fitted = INARCHFitted(
        fill(5, 48), model, par,
        INARCHEstimationSetting(), TemporalInfo()
    )
    expected = 5 .* exp.(0.7 .* cos.(2 .* 2π .* (49:60) ./ 12))
    @test point_forecast(fitted, 1:12) ≈ expected
    fc_point, _, _, _ = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 10), 1:12, [0.8]
    )
    @test fc_point ≈ expected
end

@testitem "Seasonal INARCH point forecasts follow the conditional mean recursion" tags = [:models, :inarch] begin
    using ForecastBaselines: INARCHParameter, INARCHEstimationSetting
    # λ_t = μ_t (β₀ + α z_{t-1}), where z_t is the observation divided by μ_t up to T
    # and the forecast divided by μ_t after it.
    p, s = 1, 12
    β0, α, γ = 5.0, [0.6], [0.8, 0.3]
    x = fill(20, 48)
    T = length(x)
    μ(t) = exp(γ[1] * sin(2π * t / s) + γ[2] * cos(2π * t / s))
    model = INARCHModel(p = p, s = s, k = 1)
    fitted = INARCHFitted(x, model, INARCHParameter(β0, α, 1.0, γ), INARCHEstimationSetting(), TemporalInfo())

    z = Dict(t => x[t] / μ(t) for t in 1:T)
    expected = Float64[]
    for h in 1:6
        t = T + h
        λ = μ(t) * (β0 + α[1] * z[t - 1])
        z[t] = λ / μ(t)
        push!(expected, λ)
    end
    @test point_forecast(fitted, 1:6) ≈ expected
end
