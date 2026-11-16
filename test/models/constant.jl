@testitem "Constant Model Tests" setup = [TestData] tags = [:models, :constant] begin
    using Random, Distributions, Dates
    model = ConstantModel()
    fitted = fit_baseline(stationary_data, model)

    @test fitted isa ConstantFitted
    @test fitted.par.μ ≈ stationary_data[end]

    # Test point forecasts
    for horizon in test_horizons
        fc = point_forecast(fitted, horizon)
        expected_length = horizon isa Int ? horizon : length(horizon)
        @test length(fc) == expected_length
        @test all(fc .≈ stationary_data[end])
    end

    # Test with forecast wrapper
    forecast_result = forecast(fitted, model_name = "Constant Test")
    @test forecast_result.model_name == "Constant Test"
    @test has_mean(forecast_result)
end
