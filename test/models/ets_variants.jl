@testmodule ETSReference begin
    # One-step predictions and point forecasts for all 30 ETS variants, written in the
    # smoothing form of Hyndman et al. (2008, table 2.1) rather than the error-correction
    # form the package uses. For given parameters and initial state the two forms give the
    # same predictions, whether the error is additive or multiplicative.
    #
    # Seasonal states follow the package's ordering: s[end] is the state used for the next
    # observation, and each new seasonal state is put in front.

    export trend_term, predict, reference_filter, reference_forecast

    function trend_term(trend, l, b, ϕ)
        trend == :N && return l
        trend == :A && return l + b
        trend == :Ad && return l + ϕ * b
        trend == :M && return l * b
        return l * b^ϕ
    end

    function predict(season, T, s)
        season == :N && return T
        season == :A && return T + s[end]
        return T * s[end]
    end

    function reference_filter(y, trend, season, α, β, γ, ϕ, l, b, s)
        s = copy(s)
        βstar = β / α
        yhat = zeros(length(y))
        for t in eachindex(y)
            T = trend_term(trend, l, b, ϕ)
            yhat[t] = predict(season, T, s)
            deseasonalised = season == :N ? y[t] : season == :A ? y[t] - s[end] : y[t] / s[end]
            l_new = α * deseasonalised + (1 - α) * T
            if trend == :A
                b = βstar * (l_new - l) + (1 - βstar) * b
            elseif trend == :Ad
                b = βstar * (l_new - l) + (1 - βstar) * ϕ * b
            elseif trend == :M
                b = βstar * (l_new / l) + (1 - βstar) * b
            elseif trend == :Md
                b = βstar * (l_new / l) + (1 - βstar) * b^ϕ
            end
            if season == :A
                s = [γ * (y[t] - T) + (1 - γ) * s[end]; s[1:(end - 1)]]
            elseif season == :M
                s = [γ * y[t] / T + (1 - γ) * s[end]; s[1:(end - 1)]]
            end
            l = l_new
        end
        return yhat, l, b, s
    end

    function reference_forecast(trend, season, ϕ, l, b, s, h)
        damping = sum(ϕ^i for i in 1:h)
        T = trend == :N ? l :
            trend == :A ? l + h * b :
            trend == :Ad ? l + damping * b :
            trend == :M ? l * b^h :
            l * b^damping
        season == :N && return T
        m = length(s)
        seasonal = s[end - mod(h - 1, m)]
        return season == :A ? T + seasonal : T * seasonal
    end
end

@testitem "ETS filtering matches the smoothing equations for every variant" setup = [ETSReference] tags = [:models, :ets] begin
    using ForecastBaselines: ETSFilter, ETSParameter, ETSTuning, ETSSpace

    m = 4
    y = [10 + 0.2 * t + 1.5 * sin(2π * t / m) + 0.3 * cos(1.7 * t) for t in 1:40]
    α, β, γ, ϕ = 0.3, 0.1, 0.2, 0.9

    for error in ("A", "M"), trend in ("N", "A", "Ad", "M", "Md"), season in ("N", "A", "M")
        model = ETSModel(error = error, trend = trend, season = season, s = m)
        b0 = trend in ("M", "Md") ? 1.02 : trend == "N" ? 0.0 : 0.2
        s0 = season == "A" ? [0.5, -1.0, -0.5, 1.0] :
            season == "M" ? [1.05, 0.9, 0.95, 1.1] : [0.0]
        par = ETSParameter(ETSTuning(α, β, γ, ϕ), ETSSpace(10.0, b0, s0))

        yhat, = reference_filter(y, Symbol(trend), Symbol(season), α, β, γ, ϕ, 10.0, b0, s0)
        @testset "$error,$trend,$season" begin
            @test ETSFilter(y, model, par).xHat ≈ yhat
        end
    end
end

@testitem "ETS point forecasts follow the closed-form forecast function" setup = [ETSReference] tags = [:models, :ets] begin
    using ForecastBaselines: ETSParameter, ETSTuning, ETSSpace, ETSEstimationSetting

    m = 4
    y = [10 + 0.2 * t + 1.5 * sin(2π * t / m) + 0.3 * cos(1.7 * t) for t in 1:40]
    α, β, γ, ϕ = 0.3, 0.1, 0.2, 0.9
    horizons = 1:9

    for error in ("A", "M"), trend in ("N", "A", "Ad", "M", "Md"), season in ("N", "A", "M")
        model = ETSModel(error = error, trend = trend, season = season, s = m)
        b0 = trend in ("M", "Md") ? 1.02 : trend == "N" ? 0.0 : 0.2
        s0 = season == "A" ? [0.5, -1.0, -0.5, 1.0] :
            season == "M" ? [1.05, 0.9, 0.95, 1.1] : [0.0]
        par = ETSParameter(ETSTuning(α, β, γ, ϕ), ETSSpace(10.0, b0, s0))
        fitted = ETSFitted(y, model, par, ETSEstimationSetting(), TemporalInfo())

        _, l, b, s = reference_filter(y, Symbol(trend), Symbol(season), α, β, γ, ϕ, 10.0, b0, s0)
        expected = [reference_forecast(Symbol(trend), Symbol(season), ϕ, l, b, s, h) for h in horizons]
        @testset "$error,$trend,$season" begin
            @test point_forecast(fitted, horizons) ≈ expected
        end
    end
end

@testitem "ETS fits and simulates every variant" setup = [TestData] tags = [:models, :ets] begin
    using Distributions: std
    for error in ("A", "M"), trend in ("N", "A", "Ad", "M", "Md"), season in ("N", "A", "M")
        model = ETSModel(error = error, trend = trend, season = season, s = 12)
        fitted = fit_baseline(seasonal_data, model)
        fc = forecast(
            fitted;
            interval_method = ModelTrajectoryInterval(n_trajectories = 200, seed = 1),
            horizon = 1:6, levels = [0.5, 0.9]
        )
        @testset "$error,$trend,$season" begin
            @test all(isfinite, fc.mean)
            @test all(i -> all(i.lower .<= i.upper), fc.intervals)
            @test all(i -> i.lower[2] <= i.lower[1] && i.upper[1] <= i.upper[2], fc.intervals)
            # One step ahead a trajectory is the point forecast plus noise on the scale
            # of the in-sample residuals: additive residuals, or relative ones times
            # the point forecast for multiplicative errors. The 90% interval is then
            # about 2 * 1.645 times that wide, and half of it is a floor that
            # simulation without noise, or with noise on the wrong scale, cannot reach.
            filtered = ForecastBaselines.ETSFilter(seasonal_data, model, fitted.par)
            scale = error == "M" ? abs(fc.mean[1]) : 1.0
            width(i) = i.upper[2] - i.lower[2]
            @test width(fc.intervals[1]) > 0.5 * 2 * 1.645 * std(filtered.ϵ) * scale
            # Uncertainty grows with the horizon.
            @test width(fc.intervals[6]) > width(fc.intervals[1])
        end
    end
end
