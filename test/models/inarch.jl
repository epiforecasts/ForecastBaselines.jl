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
