@testitem "All Combinations" setup = [TestData] tags = [:integration] begin
    using Random, Distributions, Dates
    all_models = [
        MarginalModel(p = 10),
        ConstantModel(),
        KDEModel(),
        LSDModel(s = 10, w = 1),
        OLSModel(p = 10, d = 1),
        IDSModel(p = 3),
        ARMAModel(p = 1, q = 1),
        INARCHModel(p = 1),
        STLModel(),
        ETSModel(trend = "A"),
    ]
    #
    i_methods = [
        NoInterval(),
        EmpiricalInterval(n_trajectories = 50, positivity_correction = :none),
        EmpiricalInterval(n_trajectories = 50, positivity_correction = :zero_floor),
        EmpiricalInterval(n_trajectories = 50, positivity_correction = :truncate),
        EmpiricalInterval(n_trajectories = 50, positivity_correction = :post_clip),
        ParametricInterval(positivity_correction = :none),
        ParametricInterval(positivity_correction = :post_clip),
        ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :none),
        ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :zero_floor),
        ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :truncate),
        ModelTrajectoryInterval(n_trajectories = 50, positivity_correction = :post_clip),
    ]
    #
    hasPI = [true; false; fill(true, 5); fill(false, 3)]
    hasMTI = [true; false; fill(true, 8)]

    for i in 1:10
        if i == 8
            fitted = fit_baseline(count_data, all_models[i])
        else
            fitted = fit_baseline(stationary_data, all_models[i])
        end

        fc = forecast(fitted, interval_method = i_methods[1], horizon = 3)
        @test length(fc.mean) == 3
        fc = forecast(fitted, interval_method = i_methods[2], horizon = 3)
        @test length(fc.mean) == 3
        fc = forecast(fitted, interval_method = i_methods[3], horizon = 3)
        @test length(fc.mean) == 3
        fc = forecast(fitted, interval_method = i_methods[4], horizon = 3)
        @test length(fc.mean) == 3
        fc = forecast(fitted, interval_method = i_methods[5], horizon = 3)
        @test length(fc.mean) == 3
        if hasPI[i]
            fc = forecast(fitted, interval_method = i_methods[6], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[7], horizon = 3)
            @test length(fc.mean) == 3
        end
        if hasMTI[i]
            fc = forecast(fitted, interval_method = i_methods[8], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[9], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[10], horizon = 3)
            @test length(fc.mean) == 3
            fc = forecast(fitted, interval_method = i_methods[11], horizon = 3)
            @test length(fc.mean) == 3
        end
    end
end

@testitem "Complex Integration Tests" setup = [TestData] tags = [:integration] begin
    using Random, Distributions, Dates
    # Test model with both data transformation and seasonal preprocessing
    positive_seasonal = abs.(seasonal_data) .+ 1

    # Create complex transformed model
    base_model = MarginalModel(p = 20)
    complex_model = transform(
        base_model,
        transformation = LogTransform(),
        season_trend = STTransform(s = 12, k = 1, trend = true, additive = false)
    )

    fitted_complex = fit_baseline(positive_seasonal, complex_model)
    @test fitted_complex isa TransformedFitted

    # Generate forecast with complex model
    complex_forecast = forecast(
        fitted_complex,
        interval_method = EmpiricalInterval(n_trajectories = 50, return_trajectories = true),
        horizon = 1:6,
        levels = [0.8, 0.95]
    )

    @test has_mean(complex_forecast)
    @test has_intervals(complex_forecast)
    @test has_trajectories(complex_forecast)
    @test all(complex_forecast.mean .> 0)  # Should be positive after inverse transform

    # Test scoring on transformed forecasts
    truth_vals = positive_seasonal[(end - 5):end]
    complex_forecast_with_truth = add_truth(complex_forecast, truth_vals)

    mae_complex = score([complex_forecast_with_truth], MAE())
    @test mae_complex isa Float64
    @test mae_complex >= 0
end
