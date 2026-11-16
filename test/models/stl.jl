@testitem "STL Model Tests" setup = [TestData] tags = [:models, :stl] begin
    using Random, Distributions, Dates
    model = STLModel(s = 12)  # 12-period seasonality
    fitted = fit_baseline(seasonal_data, model)

    @test fitted isa STLFitted
    @test length(fitted.par.S) == length(seasonal_data)
    @test length(fitted.par.T) == length(seasonal_data)
    @test length(fitted.par.R) == length(seasonal_data)

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3

    # Test trajectory intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true), 1:2, [0.8]
    )
    @test size(fc_traj) == (50, 2)
end

@testitem "STL minimum length is ns * s + 1" setup = [TestData] tags = [:models, :stl] begin
    using Random, Distributions, Dates
    Random.seed!(1)
    x = 10 .+ sin.(2π .* (1:200) ./ 12) .+ 0.1 .* randn(200)
    for ns in (7, 13)
        setting = STLEstimationSetting(s = 12, ns = ns)
        n = ns * 12 + 1
        @test fit_baseline(x[1:n], STLModel(s = 12); setting = setting) isa STLFitted
        err = try
            fit_baseline(x[1:(n - 1)], STLModel(s = 12); setting = setting)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(n), err.msg)
    end
end
