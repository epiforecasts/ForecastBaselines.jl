@testitem "Error Handling Tests" setup = [TestData] tags = [:edge_cases] begin
    using Random, Distributions, Dates
    model = ConstantModel()

    # Test with insufficient data
    tiny_data = [1.0, 2.0]
    fitted = fit_baseline(tiny_data, model)
    @test fitted isa ConstantFitted

    # Test with negative horizons
    fitted = fit_baseline(stationary_data, model)
    @test_throws ArgumentError point_forecast(fitted, [-1, 2])
    @test_throws ArgumentError point_forecast(fitted, 0)

    # Test with invalid levels
    model = MarginalModel(p = 10)
    fitted = fit_baseline(stationary_data, model)
    @test_throws ArgumentError interval_forecast(fitted, ParametricInterval(), 1:2, [0.0, 1.0])
    @test_throws ArgumentError interval_forecast(fitted, ParametricInterval(), 1:2, [1.5])

    # Test ARMA stability checking
    arma_model = ARMAModel(p = 1, q = 1)
    unstable_data = cumsum(randn(20))  # Random walk (non-stationary)
    # Should still fit but may warn about stability
    fitted_arma = fit_baseline(unstable_data, arma_model)
    @test fitted_arma isa ARMAFitted
end

@testitem "Edge Cases Tests" setup = [TestData] tags = [:edge_cases] begin
    using Random, Distributions, Dates
    # Test with constant data
    constant_data = fill(5.0, 20)

    # Constant model
    model = ConstantModel()
    fitted = fit_baseline(constant_data, model)
    fc = point_forecast(fitted, 1:3)
    @test all(fc .== 5.0)

    # Marginal model
    marginal_model = MarginalModel()
    fitted_marginal = fit_baseline(constant_data, marginal_model)
    fc_marginal = point_forecast(fitted_marginal, 1:3)
    @test all(fc_marginal .== 5.0)

    # Test with single observation
    single_obs = [10.0]
    fitted_single = fit_baseline(single_obs, model)
    fc_single = point_forecast(fitted_single, 1)
    @test fc_single == [10.0]
end

@testitem "Performance Tests" setup = [TestData] tags = [:edge_cases] begin
    using Random, Distributions, Dates
    # Test that models can handle moderately large datasets
    large_data = randn(500)

    models_to_test = [
        ConstantModel(),
        MarginalModel(p = 50),
        LSDModel(s = 7, w = 1),
        OLSModel(p = 10, d = 1),
    ]

    for model in models_to_test
        fitted = fit_baseline(large_data, model)
        @test fitted isa AbstractFittedModel
        fc = point_forecast(fitted, 1:10)
        @test fc isa Vector{Float64}
        @test length(fc) == 10
    end
end

@testitem "Edge Cases and Error Handling Extended" setup = [TestData] tags = [:edge_cases] begin
    using Random, Distributions, Dates
    # Test various error conditions
    model = ConstantModel()
    fitted = fit_baseline([1.0, 2.0, 3.0], model)

    # Test empty horizon
    @test_throws ArgumentError point_forecast(fitted, Int[])

    # Test invalid forecast construction
    @test_throws ArgumentError Forecast(horizon = [1, 2], mean = [1.0])  # Length mismatch

    # Test invalid interval construction
    @test_throws ArgumentError ForecastInterval([2.0], [1.0], [0.95])  # Lower > upper
    @test_throws ArgumentError ForecastInterval([1.0], [2.0], [1.5])   # Invalid level

    # Test transformation with invalid data
    log_transform = LogTransform()
    @test_throws ArgumentError transform([-1.0, 2.0, 3.0], log_transform)  # Negative values

    # Test ARMA with insufficient data for degree
    arma_model = ARMAModel(p = 5, q = 3)
    tiny_data = [1.0, 2.0]
    # Should handle gracefully or throw appropriate error
    @test_throws Exception fit_baseline(tiny_data, arma_model)
end

@testitem "Memory and Performance Edge Cases" setup = [TestData] tags = [:edge_cases] begin
    using Random, Distributions, Dates
    # Test with moderately large trajectory matrices
    model = MarginalModel(p = 50)
    fitted = fit_baseline(randn(100), model)

    # Large trajectory test
    large_traj_forecast = forecast(
        fitted,
        interval_method = ModelTrajectoryInterval(n_trajectories = 1000, return_trajectories = true),
        horizon = 1:20
    )

    @test size(large_traj_forecast.trajectories) == (1000, 20)
    @test has_trajectories(large_traj_forecast)

    # Test forecast without returning trajectories (memory efficient)
    efficient_forecast = forecast(
        fitted,
        interval_method = ModelTrajectoryInterval(n_trajectories = 1000, return_trajectories = false),
        horizon = 1:20
    )

    @test !has_trajectories(efficient_forecast)
    @test has_intervals(efficient_forecast)
end
