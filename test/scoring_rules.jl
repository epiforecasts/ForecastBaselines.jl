@testmodule ScoringFixtures begin
    using ForecastBaselines

    export point_forecasts, interval_forecast_fixture

    # Two forecasts whose errors are easy to work out by hand: forecast minus truth is
    # (1, -2, 3) for the first and (-4, 0.5) for the second.
    function point_forecasts()
        return [
            Forecast(horizon = [1, 2, 3], mean = [11.0, 18.0, 33.0], truth = [10.0, 20.0, 30.0]),
            Forecast(horizon = [1, 2], mean = [6.0, 4.5], truth = [10.0, 4.0]),
        ]
    end

    function interval_forecast_fixture()
        return Forecast(
            horizon = [1, 2],
            mean = [10.0, 10.0],
            median = [10.0, 11.0],
            intervals = [
                ForecastInterval([8.0, 5.0], [12.0, 15.0], [0.5, 0.9]),
                ForecastInterval([9.0, 6.0], [13.0, 16.0], [0.5, 0.9]),
            ],
            truth = [11.0, 17.0]
        )
    end
end

@testitem "Point scoring rules pool errors over forecasts and horizons" setup = [ScoringFixtures] tags = [:scoring] begin
    using Distributions: mean, median
    fcs = point_forecasts()
    e = [1.0, -2.0, 3.0, -4.0, 0.5]
    truth = [10.0, 20.0, 30.0, 10.0, 4.0]

    @test score(fcs, MAE()) ≈ mean(abs.(e))
    @test score(fcs, MdAE()) ≈ median(abs.(e))
    @test score(fcs, MSE()) ≈ mean(e .^ 2)
    @test score(fcs, RMSE()) ≈ sqrt(mean(e .^ 2))
    @test score(fcs, Bias()) ≈ mean(e)
    @test score(fcs, MAPE()) ≈ mean(abs.(e ./ truth))
    # MSPE squares the error and divides by the truth once, rather than squaring the
    # relative error.
    @test_broken score(fcs, MSPE()) ≈ mean((e ./ truth) .^ 2)
    @test score(fcs, RelativeBias()) ≈ mean(e ./ truth)

    @test score(fcs[1], MAE()) ≈ mean(abs.(e[1:3]))
end

@testitem "Point scores by horizon and for selected horizons" setup = [ScoringFixtures] tags = [:scoring] begin
    using Distributions: mean
    fcs = point_forecasts()

    @test score(fcs, MAE(); horizon = 1) ≈ mean([1.0, 4.0])
    @test score(fcs, MAE(); horizon = 3) ≈ 3.0
    # A forecast that has only some of the selected horizons makes score throw.
    @test_broken score(fcs, MAE(); horizon = [2, 3]) ≈ mean([2.0, 3.0, 0.5])
    @test_broken score(fcs, MAE(); horizon = 2:3) ≈ mean([2.0, 3.0, 0.5])
    @test score(fcs[1:1], MAE(); horizon = [2, 3]) ≈ mean([2.0, 3.0])
    @test score(fcs, MAE(); by_horizon = true) ≈ [mean([1.0, 4.0]), mean([2.0, 0.5]), 3.0]
    @test score(fcs, MAE(); horizon = [1, 3], by_horizon = true) ≈ [mean([1.0, 4.0]), 3.0]
end

@testitem "Point scores handle missing truth by the rule's strategy" setup = [ScoringFixtures] tags = [:scoring] begin
    fc = Forecast(horizon = [1, 2], mean = [1.0, 5.0], truth = [2.0, 2.0])
    no_truth = Forecast(horizon = [1, 2], mean = [1.0, 5.0])

    @test score([fc, no_truth], MAE()) ≈ 2.0
    @test_throws ArgumentError score([fc, no_truth], MAE(missing_strategy = :error))
    @test ismissing(score([fc, no_truth], MAE(missing_strategy = :propagate)))
end

@testitem "WIS combines the median error with the interval scores" setup = [ScoringFixtures] tags = [:scoring] begin
    using Distributions: mean
    fc = interval_forecast_fixture()

    # Interval score: width plus 2 / alpha times the distance by which the truth falls
    # outside, for alpha = 1 - level.
    is(lower, upper, level, y) = (upper - lower) + 2 / (1 - level) * (max(lower - y, 0) + max(y - upper, 0))
    expected = [
        0.5 * abs(10 - 11) + 0.25 * is(8, 12, 0.5, 11) + 0.05 * is(5, 15, 0.9, 11),
        0.5 * abs(11 - 17) + 0.25 * is(9, 13, 0.5, 17) + 0.05 * is(6, 16, 0.9, 17),
    ]
    @test score(fc, WIS()) ≈ mean(expected)
    @test score(fc, WIS(); by_horizon = true) ≈ expected
    @test score(fc, WIS(levels = [0.9])) ≈ mean([0.5 * 1 + 0.05 * is(5, 15, 0.9, 11), 0.5 * 6 + 0.05 * is(6, 16, 0.9, 17)])
end
