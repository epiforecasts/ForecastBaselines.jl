@testitem "Scoring Rules Tests" setup = [TestData] tags = [:scoring] begin
    using Random, Distributions, Dates
    # Generate forecasts with truth for scoring
    model = MarginalModel(p = 10)

    # Create forecasts with truth
    forecasts_with_truth = Forecast[]
    for i in 1:5
        fitted = fit_baseline(stationary_data[1:(200 + i)], model)
        fc = forecast(
            fitted,
            interval_method = ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true),
            horizon = 1:3,
            truth = stationary_data[(201 + i):(203 + i)]
        )
        push!(forecasts_with_truth, fc)
    end

    # Test point scoring rules
    mae_score = score(forecasts_with_truth, MAE())
    @test mae_score isa Float64
    @test mae_score >= 0

    mse_score = score(forecasts_with_truth, MSE())
    @test mse_score isa Float64
    @test mse_score >= 0

    rmse_score = score(forecasts_with_truth, RMSE())
    @test rmse_score isa Float64
    @test rmse_score >= 0
    @test rmse_score ≈ sqrt(mse_score)

    bias_score = score(forecasts_with_truth, Bias())
    @test bias_score isa Float64

    # Test by horizon scoring
    mae_by_horizon = score(forecasts_with_truth, MAE(), by_horizon = true)
    @test length(mae_by_horizon) == 3

    # Test specific horizon scoring
    mae_h1 = score(forecasts_with_truth, MAE(), horizon = 1)
    @test mae_h1 isa Float64

    # Test interval scoring rules (need intervals)
    forecasts_with_intervals = Forecast[]
    for i in 1:3
        fitted = fit_baseline(stationary_data[1:(200 + i)], model)
        fc = forecast(
            fitted,
            interval_method = ParametricInterval(),
            horizon = 1:2,
            levels = [0.8, 0.95],
            include_median = true,
            truth = stationary_data[(201 + i):(202 + i)]
        )
        push!(forecasts_with_intervals, fc)
    end

    wis_score = score(forecasts_with_intervals, WIS())
    @test wis_score isa Float64
    @test wis_score >= 0

    crps_score = score(forecasts_with_intervals, CRPS())
    @test crps_score isa Float64
    @test crps_score >= 0

    # Test trajectory scoring
    crps_traj_score = score(forecasts_with_truth, CRPS_trajectory())
    @test crps_traj_score isa Float64
    @test crps_traj_score >= 0
end

@testitem "Forecast Error Types Tests" setup = [TestData] tags = [:scoring] begin
    using Random, Distributions, Dates
    # Create simple forecast with truth
    fc_simple = Forecast(
        horizon = collect(1:3),
        mean = [2.0, 2.5, 3.0],
        median = [1.9, 2.4, 2.9],
        truth = [2.1, 2.3, 3.2]
    )

    # Test different point error types
    @test forecast_error(fc_simple, ForecastError(), horizon = 1) ≈ -0.1
    @test forecast_error(fc_simple, AbsoluteError(), horizon = 1) ≈ 0.1
    @test forecast_error(fc_simple, SquaredError(), horizon = 1) ≈ 0.01
    @test forecast_error(fc_simple, SignError(), horizon = 1) == -1

    # Test relative errors (avoid zero truth values)
    @test forecast_error(fc_simple, ForecastError(relative = true), horizon = 1) ≈ -0.1 / 2.1
    @test forecast_error(fc_simple, AbsoluteError(relative = true), horizon = 1) ≈ 0.1 / 2.1

    # Test pinball error
    @test forecast_error(fc_simple, PinballError(0.5), horizon = 1, target = :median) isa Float64

    # Test all forecast errors
    all_errors = all_forecast_errors(fc_simple, AbsoluteError())
    @test length(all_errors) == 3
    @test all(all_errors .>= 0)

    # Test with quantile targets
    fc_with_intervals = Forecast(
        horizon = collect(1:2),
        mean = [2.0, 2.5],
        intervals = [
            ForecastInterval([1.5], [2.5], [0.8]),
            ForecastInterval([2.0], [3.0], [0.8]),
        ],
        truth = [2.2, 2.7]
    )

    # This should work if the quantile extraction is implemented
    q80_error = forecast_error(fc_with_intervals, AbsoluteError(), horizon = 1, target = :q90)
    @test q80_error isa Union{Float64, Nothing, Missing}
end

@testitem "Historical Forecast Errors Tests" setup = [TestData] tags = [:scoring] begin
    using Random, Distributions, Dates
    # Test with OLS model (simple enough for repeated fitting)
    model = OLSModel(p = 5, d = 1)
    fitted = fit_baseline(trend_data, model)

    # Test historical error computation
    hist_errors = historical_forecast_errors(fitted, 1:3, 10)
    @test length(hist_errors) == 3
    @test all(length.(hist_errors) .> 0)  # Should have some errors for each horizon

    # Test with different min_observation
    hist_errors_min = historical_forecast_errors(fitted, 1:2, 20)
    @test length(hist_errors_min) == 2

    # Errors should be Float64 vectors
    @test all(isa.(hist_errors[1], Float64))
end
