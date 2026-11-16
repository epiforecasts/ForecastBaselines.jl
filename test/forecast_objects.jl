@testitem "Temporal Info Tests" setup = [TestData] tags = [:forecast] begin
    using Random, Distributions, Dates
    # Test with Date temporal info
    ti = TemporalInfo(Date("2020-01-01"), Day(1))
    model = ConstantModel()
    fitted = fit_baseline(stationary_data, model, temporal_info = ti)

    forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:3, model_name = "Temporal Test")
    @test has_temporal_info(forecast_result)
    @test forecast_result.reference_date isa Date
    @test length(forecast_result.target_date) == 3

    # Test with integer temporal info
    ti_int = TemporalInfo(1, 1)
    fitted_int = fit_baseline(stationary_data, model, temporal_info = ti_int)
    forecast_result_int = forecast(fitted_int, interval_method = NoInterval(), horizon = 1:3)
    @test forecast_result_int.reference_date isa Int
end

@testitem "Forecast Object Tests" setup = [TestData] tags = [:forecast] begin
    using Random, Distributions, Dates
    model = ConstantModel()
    fitted = fit_baseline(stationary_data, model)

    # Test basic forecast
    forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:5, model_name = "Test Model")
    @test forecast_length(forecast_result) == 5
    @test max_horizon(forecast_result) == 5
    @test min_horizon(forecast_result) == 1
    @test forecast_result.model_name == "Test Model"

    # Test adding truth
    truth_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    forecast_with_truth = add_truth(forecast_result, truth_values)
    @test has_truth(forecast_with_truth)
    @test forecast_with_truth.truth == truth_values

    # Test truncating horizon
    truncated_forecast = truncate_horizon(forecast_result, 3)
    @test forecast_length(truncated_forecast) == 3
    @test max_horizon(truncated_forecast) == 3

    # Test filtering horizons
    filtered = filter_horizons(forecast_result, [1, 3, 5])
    @test forecast_length(filtered) == 3
    @test filtered.horizon == [1, 3, 5]

    # Test updating model name
    renamed = update_model_name(forecast_result, "New Name")
    @test renamed.model_name == "New Name"
end

@testitem "Forecast Utilities Extended Tests" setup = [TestData] tags = [:forecast] begin
    using Random, Distributions, Dates
    # Create complex forecast for testing utilities
    intervals = [ForecastInterval([1.0, 1.5], [3.0, 2.5], [0.5, 0.8]) for _ in 1:5]
    trajectories = randn(100, 5)

    complex_fc = Forecast(
        horizon = collect(1:5),
        mean = [2.0, 2.1, 2.2, 2.3, 2.4],
        median = [1.9, 2.0, 2.1, 2.2, 2.3],
        intervals = intervals,
        trajectories = trajectories,
        truth = [2.1, 2.0, 2.3, 2.2, 2.5],
        reference_date = Date("2023-12-31"),
        target_date = [
            Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03"),
            Date("2024-01-04"), Date("2024-01-05"),
        ],
        model_name = "Complex Test"
    )

    # Test all has_* functions
    @test has_horizon(complex_fc)
    @test has_mean(complex_fc)
    @test has_median(complex_fc)
    @test has_intervals(complex_fc)
    @test has_truth(complex_fc)
    @test has_trajectories(complex_fc)
    @test has_reference_date(complex_fc)
    @test has_target_date(complex_fc)
    @test has_temporal_info(complex_fc)

    # Test trajectory properties
    @test num_trajectories(complex_fc) == 100

    # Test temporal span
    span = get_temporal_span(complex_fc)
    @test span == Day(4)

    # Test horizon range
    range = get_horizon_range(complex_fc)
    @test range == 1:5

    # Test filter_levels (need to implement if not already)
    filtered_fc = filter_levels(complex_fc, [0.8])
    @test has_intervals(filtered_fc)

    # Test forecast extension
    fc1 = Forecast(horizon = [1, 2], mean = [1.0, 1.1], model_name = "Model1")
    fc2 = Forecast(horizon = [3, 4], mean = [1.2, 1.3], model_name = "Model2")
    extended = extend_forecast(fc1, fc2)
    @test extended.horizon == [1, 2, 3, 4]
    @test extended.mean == [1.0, 1.1, 1.2, 1.3]
    @test extended.model_name == "Model1 + Model2"
end
