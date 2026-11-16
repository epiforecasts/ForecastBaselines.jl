@testitem "Calibration Tests" setup = [TestData] tags = [:calibration] begin
    using Random, Distributions, Dates
    # Generate multiple forecasts for calibration testing
    model = MarginalModel(p = 20)
    fitted = fit_baseline(stationary_data, model)

    calibration_forecasts = Forecast[]
    for i in 1:10
        # Create synthetic truth values for testing
        synthetic_truth = [fitted.par.μ + 0.5 * randn(), fitted.par.μ + 0.3 * randn()]

        fc = forecast(
            fitted,
            interval_method = ParametricInterval(),
            horizon = 1:2,
            levels = [0.5, 0.8, 0.95],
            truth = synthetic_truth
        )
        push!(calibration_forecasts, fc)
    end

    # Test PIT function creation
    pit_func = PIT_function(calibration_forecasts, horizon = 1)
    @test pit_func isa Function

    # Test PIT function evaluation
    pit_values = pit_func.([0.1, 0.5, 0.9])
    @test length(pit_values) == 3
    @test all(0 .<= pit_values .<= 1)

    # Test CvM divergence
    cvm_score = CvM_divergence(calibration_forecasts, horizon = 1)
    @test cvm_score isa Float64
    @test cvm_score >= 0

    # Test step function creation
    step = create_step(calibration_forecasts[1], horizon = 1)
    @test step isa OneStepFunction
    @test 0 <= step.l <= 1
    @test 0 <= step.u <= 1
end
