@testitem "LSD Model Tests" setup = [TestData] tags = [:models, :lsd] begin
    using Random, Distributions, Dates
    model = LSDModel(s = 12, w = 2)  # 12 period seasonality, window ±2
    fitted = fit_baseline(seasonal_data, model)

    @test fitted isa LSDFitted
    @test length(fitted.par.μ) == 12

    # Test point forecasts
    fc = point_forecast(fitted, 1:12)
    @test length(fc) == 12

    # Test parametric intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ParametricInterval(), 1:3, [0.9]
    )
    @test length(fc_intervals) == 3

    # Test trajectory intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true), 1:2, [0.8]
    )
    @test size(fc_traj) == (50, 2)
end
