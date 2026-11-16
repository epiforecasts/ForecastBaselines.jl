@testitem "Transformations Tests" setup = [TestData] tags = [:transformations] begin
    using Random, Distributions, Dates
    # Test log transformation
    positive_data = abs.(stationary_data) .+ 1
    log_transform = LogTransform()
    transformed_data = transform(positive_data, log_transform)
    recovered_data = inverse_transform(transformed_data, log_transform)
    @test positive_data ≈ recovered_data

    # Test log-plus-one transformation
    log_plus_one = LogPlusOneTransform(1.0)
    transformed_data = transform(positive_data, log_plus_one)
    recovered_data = inverse_transform(transformed_data, log_plus_one)
    @test positive_data ≈ recovered_data

    # Test power transformation
    power_transform = PowerTransform(0.5)
    transformed_data = transform(positive_data, power_transform)
    recovered_data = inverse_transform(transformed_data, power_transform)
    @test positive_data ≈ recovered_data rtol = 1.0e-10

    # Test transformed model
    base_model = ConstantModel()
    transformed_model = transform(base_model, transformation = log_transform)
    fitted = fit_baseline(positive_data, transformed_model)
    @test fitted isa TransformedFitted

    # Test forecasting with transformation
    forecast_result = forecast(fitted, interval_method = NoInterval(), horizon = 1:3)
    @test length(forecast_result.mean) == 3
end

@testitem "Seasonality Transformation Tests" setup = [TestData] tags = [:transformations] begin
    using Random, Distributions, Dates
    # Test STTransform
    st_transform = STTransform(s = 12, k = 2, trend = true, additive = true)

    # Test pre-filtering
    filtered_data, st_params = preFilter(seasonal_data, st_transform)
    @test length(filtered_data) == length(seasonal_data)
    @test st_params isa STParameter
    @test length(st_params.θ) == 2
    @test length(st_params.κ) == 2

    # Fit model to filtered data
    model = ConstantModel()
    fitted = fit_baseline(filtered_data, model)

    # Generate forecast on filtered scale
    filtered_forecast = forecast(fitted, interval_method = NoInterval(), horizon = 1:12)

    # Test post-filtering
    final_forecast = postFilter(seasonal_data, filtered_forecast, st_transform, st_params)
    @test has_mean(final_forecast)
    @test length(final_forecast.mean) == 12

    # Test multiplicative decomposition (with positive data)
    positive_seasonal = abs.(seasonal_data) .+ 10
    st_mult = STTransform(s = 12, k = 1, trend = false, additive = false)

    filtered_mult, params_mult = preFilter(positive_seasonal, st_mult)
    @test all(filtered_mult .> 0)  # Should remain positive after division

    fitted_mult = fit_baseline(filtered_mult, model)
    forecast_mult = forecast(fitted_mult, interval_method = NoInterval(), horizon = 1:6)
    final_mult = postFilter(positive_seasonal, forecast_mult, st_mult, params_mult)
    @test all(final_mult.mean .> 0)
end
