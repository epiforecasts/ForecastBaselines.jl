@testitem "KDE Model Tests" setup = [TestData] tags = [:models, :kde] begin
    using Random, Distributions, Dates
    model = KDEModel()
    fitted = fit_baseline(stationary_data, model)

    @test fitted isa KDEFitted
    @test length(fitted.par.x_seq) == length(fitted.par.density)

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3
    @test all(fc .≈ fc[1])  # KDE gives constant forecast (mean)

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
end
