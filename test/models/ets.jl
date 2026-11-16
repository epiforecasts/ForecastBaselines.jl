@testitem "ETS Model Tests" setup = [TestData] tags = [:models, :ets] begin
    using Random, Distributions, Dates
    # Test simple ETS(A,N,N)
    model = ETSModel(error = "A", trend = "N", season = "N")
    fitted = fit_baseline(stationary_data, model)

    @test fitted isa ETSFitted
    @test fitted.model.error isa AError
    @test fitted.model.trend isa NTrend
    @test fitted.model.season isa NSeason

    # Test point forecasts
    fc = point_forecast(fitted, 1:3)
    @test length(fc) == 3

    # Test trajectory intervals
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ModelTrajectoryInterval(n_trajectories = 50, return_trajectories = true), 1:2, [0.8]
    )
    @test size(fc_traj) == (50, 2)

    # Test ETS(A,A,N) with trend
    model_trend = ETSModel(error = "A", trend = "A", season = "N")
    fitted_trend = fit_baseline(trend_data, model_trend)
    @test fitted_trend.model.trend isa ATrend

    # Test ETS(A,N,A) with seasonality
    model_seasonal = ETSModel(error = "A", trend = "N", season = "A", s = 12)
    fitted_seasonal = fit_baseline(seasonal_data, model_seasonal)
    @test fitted_seasonal.model.season isa ASeason
    @test fitted_seasonal.model.season.s == 12
end

@testitem "ETSModel error argument selects the error type" setup = [TestData] tags = [:models, :ets] begin
    using Random, Distributions, Dates
    @test ETSModel(error = "A").error isa AError
    @test ETSModel(error = "M").error isa MError
    @test_throws ArgumentError ETSModel(error = "N")
end

@testitem "ETS estimates the parameters of damped trends" setup = [TestData] tags = [:models, :ets] begin
    using Random, Distributions, Dates
    Random.seed!(3)
    n = 300
    y = collect((100.0 .+ 0.4 .* (1:n)) .* (1 .+ 0.05 .* randn(n)))
    model = ETSModel(error = "M", trend = "Ad")
    start = ForecastBaselines.getInitial(y, model)
    fitted = fit_baseline(y, model)
    moved = [
        !(fitted.par.θ.α ≈ start.θ.α), !(fitted.par.θ.β ≈ start.θ.β),
        !(fitted.par.θ.ϕ ≈ start.θ.ϕ),
    ]
    @test any(moved)
    @test 0 <= fitted.par.θ.ϕ <= 1
end

@testitem "Damped trends start from the fitted slope" setup = [TestData] tags = [:models, :ets] begin
    using Random, Distributions, Dates
    Random.seed!(7)
    x = collect(300.0 .- 2.25 .* (1:60) .+ randn(60))
    slope(trend) = ForecastBaselines.getInitial(
        x, ETSModel(error = "A", trend = trend)
    ).z0.B
    @test slope("Ad") ≈ slope("A")
    @test slope("Md") ≈ slope("M")
    @test slope("A") < -2
end

@testitem "Damped multiplicative trends fit when the search turns the trend negative" setup = [TestData] tags = [:models, :ets] begin
    # On this series the simplex passes through a negative trend state, which the damped
    # multiplicative trend raises to the power ϕ.
    for error in ("A", "M")
        fitted = fit_baseline(seasonal_data, ETSModel(error = error, trend = "Md", season = "A", s = 12))
        @test all(isfinite, point_forecast(fitted, 1:6))
    end
end

@testitem "A multiplicative trend fit either forecasts or fails at fitting" setup = [TestData] tags = [:models, :ets] begin
    using Random
    # The regression getInitial uses has a negative intercept here, so the trend state
    # starts negative.
    Random.seed!(1)
    x = 2.0 .* (1:40) .- 1 .+ 0.1 .* randn(40)
    @test ForecastBaselines.getInitial(x, ETSModel(trend = "Md")).z0.B < 0
    for error in ("A", "M"), trend in ("M", "Md")
        model = ETSModel(error = error, trend = trend)
        fitted = try
            fit_baseline(x, model)
        catch err
            err
        end
        @testset "$error,$trend" begin
            if fitted isa Exception
                @test fitted isa ArgumentError
            else
                @test fitted.par.z0.B > 0
                @test all(isfinite, point_forecast(fitted, 1:3))
            end
        end
    end
end
