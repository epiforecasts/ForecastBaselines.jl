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

@testitem "LSD parametric intervals use each seasonal position's own data" tags = [:models, :lsd] begin
    using Distributions: quantile
    # Three seasonal positions at very different levels, so bounds taken from the
    # wrong position are far off.
    x = Float64[mod(t, 3) == 1 ? 100 + t % 5 : mod(t, 3) == 2 ? 10 + t % 5 : 1 + t % 5 for t in 1:60]
    T = length(x)
    fitted = fit_baseline(x, LSDModel(s = 3, w = 0))
    fc = forecast(fitted; interval_method = ParametricInterval(), horizon = 1:3, levels = [0.9])
    for h in 1:3
        same_position = x[mod1.(1:T, 3) .== mod1(T + h, 3)]
        @test fc.intervals[h].lower ≈ [quantile(same_position, 0.05)]
        @test fc.intervals[h].upper ≈ [quantile(same_position, 0.95)]
    end
end
