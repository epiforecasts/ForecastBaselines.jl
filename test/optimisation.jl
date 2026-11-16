@testitem "Fits report whether the optimiser converged" setup = [TestData] tags = [:optimisation] begin
    using ForecastBaselines: OptimisationInfo
    # Nelder-Mead reaches its iteration limit on this Holt-Winters fit but converges on
    # simple exponential smoothing, ARMA(1,1) and INARCH(1).
    holt_winters = fit_baseline(seasonal_data, ETSModel(error = "A", trend = "A", season = "A", s = 12))
    @test has_converged(holt_winters) === false
    @test holt_winters.optimisation.iterations == 1000

    for fitted in (
            fit_baseline(seasonal_data, ETSModel()),
            fit_baseline(stationary_data, ARMAModel(p = 1, q = 1)),
            fit_baseline(count_data, INARCHModel(p = 1)),
        )
        @test has_converged(fitted) === true
        @test 0 < fitted.optimisation.iterations < 1000
    end

    # A transformed fit reports the fit it wraps.
    transformed = fit_baseline(seasonal_data, ForecastBaselines.transform(ETSModel(), transformation = LogTransform()))
    @test has_converged(transformed) === true

    # Models that are not fitted by optimisation, or fits built directly, report nothing.
    @test isnothing(has_converged(fit_baseline(stationary_data, ConstantModel())))
    @test isnothing(has_converged(ETSFitted(seasonal_data, ETSModel(), holt_winters.par, ETSEstimationSetting())))
end

@testitem "A fit that stops without converging warns" setup = [TestData] tags = [:optimisation] begin
    model = ETSModel(error = "A", trend = "A", season = "A", s = 12)
    @test_logs (:warn, r"without converging") fit_baseline(seasonal_data, model)
    @test_logs fit_baseline(seasonal_data, ETSModel())
end
