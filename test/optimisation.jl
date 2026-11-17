@testitem "Fits report whether the optimiser converged" setup = [TestData] tags = [:optimisation] begin
    using ForecastBaselines: OptimisationInfo
    # Nelder-Mead reaches its iteration limit on this damped-trend seasonal fit but
    # converges for simple exponential smoothing, ARMA(1,1) and INARCH(1).
    holt_winters = fit_baseline(seasonal_data, ETSModel(error = "A", trend = "Ad", season = "A", s = 12))
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
    model = ETSModel(error = "A", trend = "Ad", season = "A", s = 12)
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
    holt_winters = ETSModel(error = "A", trend = "Ad", season = "A", s = 12)
    @test_logs (:warn, r"ETS fit .* without converging") (:warn, r"ARMA fit .* without converging") begin
        fit_baseline(seasonal_data, holt_winters)
        fit_baseline(stationary_data, ARMAModel(p = 3, q = 3))
    end
end

@testitem "ETS smoothing parameters map to and from an unconstrained scale" tags = [:optimisation] begin
    using ForecastBaselines: smoothing_to_unconstrained, smoothing_from_unconstrained
    for model in (
            ETSModel(), ETSModel(error = "A", trend = "A"), ETSModel(error = "M", trend = "Ad"),
            ETSModel(season = "A", s = 4), ETSModel(error = "M", trend = "Md", season = "M", s = 4),
        )
        k = 1 + !(model.trend isa ForecastBaselines.NTrend) + !(model.season isa ForecastBaselines.NSeason) +
            (model.trend isa Union{ForecastBaselines.AdTrend, ForecastBaselines.MdTrend})
        # Values with beta < alpha, gamma < 1 - alpha and phi < 1 survive a round trip.
        θ = [[0.4, 0.2, 0.3, 0.8][1:k]; 5.0; 1.0]
        back = smoothing_from_unconstrained(smoothing_to_unconstrained(θ, model), model)
        @test back ≈ θ
        # Any point on the unconstrained scale is admissible.
        for u in ([-30.0, 30.0, -5.0, 8.0], [30.0, 30.0, 30.0, 30.0])
            v = smoothing_from_unconstrained([u[1:k]; zeros(2)], model)
            α = v[1]
            @test 0 <= α <= 1
            if !(model.trend isa ForecastBaselines.NTrend)
                @test 0 <= v[2] <= α
            end
            if !(model.season isa ForecastBaselines.NSeason)
                γ = v[model.trend isa ForecastBaselines.NTrend ? 2 : 3]
                @test 0 <= γ <= 1 - α
            end
        end
    end
end
