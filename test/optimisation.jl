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

@testitem "Refits behind empirical intervals do not warn" setup = [TestData] tags = [:optimisation] begin
    using Logging, Random
    # The fit itself converges, but some of the refits on shorter stretches of the
    # series that empirical intervals use do not.
    Random.seed!(1)
    x = [10 + 5 * sin(2π * t / 12) + randn() for t in 1:60]
    fitted = fit_baseline(x, ETSModel(error = "A", trend = "A"))
    @test has_converged(fitted)
    @test_logs min_level = Logging.Warn forecast(fitted; interval_method = EmpiricalInterval(), horizon = 1:4)
end

@testitem "Each model type warns about its own non-converged fits" setup = [TestData] tags = [:optimisation] begin
    holt_winters = ETSModel(error = "A", trend = "A", season = "A", s = 12)
    @test_logs (:warn, r"ETS fit .* without converging") (:warn, r"ARMA fit .* without converging") begin
        fit_baseline(seasonal_data, holt_winters)
        fit_baseline(stationary_data, ARMAModel(p = 3, q = 3))
    end
end

@testitem "ETS refits can start from an earlier fit" setup = [TestData] tags = [:optimisation] begin
    model = ETSModel(error = "A", trend = "A")
    full = fit_baseline(trend_data, model)
    shorter = trend_data[1:(end - 5)]
    refit = fit_baseline(shorter, model, start = full.par)
    @test has_converged(refit)
    @test refit.par.θ.α ≈ fit_baseline(shorter, model).par.θ.α atol = 0.05
end

@testitem "Historical ETS refits start from the full fit" setup = [TestData] tags = [:optimisation] begin
    model = ETSModel(error = "A", trend = "A")
    full = fit_baseline(trend_data, model)
    shorter = trend_data[1:(end - 5)]
    refit = ForecastBaselines.refit_baseline(full, shorter)
    started = fit_baseline(shorter, model, start = full.par)
    @test refit.par.θ.α == started.par.θ.α
    @test refit.par.θ.β == started.par.θ.β
end
