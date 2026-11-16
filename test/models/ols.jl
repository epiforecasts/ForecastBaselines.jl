@testitem "OLS Model Tests" setup = [TestData] tags = [:models, :ols] begin
    using Random, Distributions, Dates
    # Test linear trend
    model = OLSModel(p = 5, d = 1)
    fitted = fit_baseline(trend_data, model)

    @test fitted isa OLSFitted
    @test length(fitted.par.β) == 2  # intercept + linear

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3
    @test fc[2] > fc[1]  # Should show trend

    # Test parametric intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ParametricInterval(), 1:2, [0.9]
    )
    @test length(fc_intervals) == 2

    # Test trajectory intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true), 1:2, [0.8]
    )
    @test size(fc_traj) == (50, 2)

    # Test quadratic trend
    model_quad = OLSModel(p = 10, d = 2)
    fitted_quad = fit_baseline(trend_data, model_quad)
    @test length(fitted_quad.par.β) == 3  # intercept + linear + quadratic
end
