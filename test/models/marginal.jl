@testitem "Marginal Model Tests" setup = [TestData] tags = [:models, :marginal] begin
    using Random, Distributions, Dates
    model = MarginalModel(p = 10)
    fitted = fit_baseline(stationary_data, model)

    @test fitted isa MarginalFitted
    @test fitted.par.μ ≈ mean(stationary_data[(end - 9):end])

    # Test point forecasts
    fc = point_forecast(fitted, 5)
    @test length(fc) == 5
    @test all(fc .≈ fitted.par.μ)

    # Test parametric intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ParametricInterval(), 1:3, [0.8, 0.9]
    )
    @test length(fc_intervals) == 3
    @test fc_traj === nothing

    # Test trajectory intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 100, return_trajectories = true), 1:3, [0.8]
    )
    @test size(fc_traj) == (100, 3)
end
