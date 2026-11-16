@testitem "IDS Model Tests" setup = [TestData] tags = [:models, :ids] begin
    using Random, Distributions, Dates
    model = IDSModel(p = 3)
    fitted = fit_baseline(trend_data, model)

    @test fitted isa IDSFitted
    @test fitted.par.a isa Float64
    @test fitted.par.b isa Float64

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3

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
