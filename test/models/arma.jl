@testitem "ARMA Model Tests" setup = [TestData] tags = [:models, :arma] begin
    using Random, Distributions, Dates
    # Test ARMA(1,1) with constant mean
    model = ARMAModel(p = 1, q = 1)
    fitted = fit_baseline(stationary_data, model)

    @test fitted isa ARMAFitted
    @test length(fitted.par.α) == 1
    @test length(fitted.par.β) == 1
    @test fitted.par.σ² > 0

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

    # Test ARMA with seasonality
    model_seasonal = ARMAModel(p = 1, q = 1, s = 12, trend = false)
    fitted_seasonal = fit_baseline(seasonal_data, model_seasonal)
    @test fitted_seasonal isa ARMAFitted
    @test length(fitted_seasonal.par.μ) == 3  # β₀ + sin + cos
end

@testitem "ARMA mean function fits a series far from zero" setup = [TestData] tags = [:models, :arma] begin
    using Random, Distributions, Dates
    s = 12
    mu = (θ, t) -> θ[1] + θ[2] * sin(2π * t / s) + θ[3] * cos(2π * t / s) +
        θ[4] * sin(2 * 2π * t / s) + θ[5] * cos(2 * 2π * t / s)
    t = 1:200
    Random.seed!(11)
    signal = 100 .+ 3 .* sin.(2π .* t ./ s) .+ 1.5 .* cos.(2 .* 2π .* t ./ s)
    y = collect(signal .+ 0.5 .* randn(200))
    fitted = fit_baseline(y, ARMAModel(1, 0, mu, 5))
    fitted_mean = [mu(fitted.par.μ, tt) for tt in t]
    @test sqrt(mean((fitted_mean .- signal) .^ 2)) < 1
end
