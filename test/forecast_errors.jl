@testitem "Point errors" tags = [:scoring] begin
    @test compute_error(12.0, 10.0, ForecastError()) == 2.0
    @test compute_error(12.0, 10.0, ForecastError(relative = true)) ≈ 0.2
    @test compute_error(8.0, 10.0, AbsoluteError()) == 2.0
    # By default the absolute value is taken of the error alone, before dividing.
    @test compute_error(8.0, -10.0, AbsoluteError(relative = true)) ≈ -1.8
    @test compute_error(8.0, -10.0, AbsoluteError(relative = true, inner = false)) ≈ 1.8
    @test compute_error(8.0, 10.0, SquaredError()) == 4.0
    @test compute_error(8.0, 10.0, SquaredError(relative = true, inner = false)) ≈ 0.04
    @test compute_error(8.0, 10.0, SignError()) == -1
    @test compute_error(10.0, 10.0, SignError()) == 0

    # Twice the quantile (pinball) loss at level alpha.
    @test compute_error(12.0, 10.0, PinballError(0.25)) ≈ 2 * 0.75 * 2
    @test compute_error(8.0, 10.0, PinballError(0.25)) ≈ 2 * 0.25 * 2

    for error in (ForecastError(relative = true), AbsoluteError(relative = true), SquaredError(relative = true))
        @test_throws ArgumentError compute_error(1.0, 0.0, error)
    end
end

@testitem "Missing values follow the missing strategy" tags = [:scoring] begin
    target = [1.0, missing, 3.0]
    truth = [2.0, 2.0, missing]
    @test compute_error(target, truth, AbsoluteError(), :skip) == [1.0]
    @test isequal(compute_error(target, truth, AbsoluteError(), :propagate), [1.0, missing, missing])
    @test_throws ArgumentError compute_error(target, truth, AbsoluteError(), :error)
    @test compute_error([missing], [1.0], AbsoluteError(), :skip) == Float64[]
    # An unknown strategy falls back to skipping.
    @test compute_error(target, truth, AbsoluteError(), :unknown) == [1.0]

    @test isnothing(compute_error(missing, 1.0, AbsoluteError(), :skip))
    @test ismissing(compute_error(missing, 1.0, AbsoluteError(), :propagate))
    @test_throws ArgumentError compute_error(missing, 1.0, AbsoluteError(), :error)
    @test_throws ArgumentError compute_error([1.0, 2.0], [1.0], AbsoluteError())
end

@testitem "Targets are read from the matching part of the forecast" tags = [:scoring] begin
    fc = Forecast(
        horizon = [1, 2],
        mean = [10.0, 20.0],
        median = [11.0, 21.0],
        intervals = [
            ForecastInterval([8.0, 5.0], [12.0, 15.0], [0.5, 0.9]),
            ForecastInterval([18.0], [22.0], [0.5]),
        ]
    )
    @test get_target_from_forecast(fc, :mean) == [10.0, 20.0]
    @test get_target_from_forecast(fc, :median) == [11.0, 21.0]
    @test get_target_from_forecast(fc, :q50) == [11.0, 21.0]
    @test get_target_from_forecast(fc, :q25) == [8.0, 18.0]
    @test get_target_from_forecast(fc, :q75) == [12.0, 22.0]
    # The second horizon has no 90% interval.
    @test isequal(get_target_from_forecast(fc, :q05), [5.0, missing])
    @test get_target_from_forecast(fc, :unknown) == [10.0, 20.0]
    @test_throws ArgumentError get_target_from_forecast(Forecast(horizon = [1], mean = [1.0]), :median)
    @test_throws ArgumentError get_target_from_forecast(Forecast(horizon = [1], mean = [1.0], median = [1.0]), :q25)
end

@testitem "Point errors of a forecast by horizon" tags = [:scoring] begin
    fc = Forecast(horizon = [1, 2, 3], mean = [10.0, 20.0, 30.0], median = [9.0, 19.0, 29.0], truth = [12.0, 17.0, 30.0])
    @test forecast_error(fc, AbsoluteError(); horizon = 2) == 3.0
    @test forecast_error(fc, AbsoluteError(); horizon = 2, target = :median) == 2.0
    @test forecast_error(fc, AbsoluteError(); horizon = 1, truth = [0.0, 0.0, 0.0]) == 10.0
    @test all_forecast_errors(fc, ForecastError()) == [-2.0, 3.0, 0.0]
    @test all_forecast_errors(fc, ForecastError(); target = :median) == [-3.0, 2.0, -1.0]
    @test_throws ArgumentError forecast_error(fc, AbsoluteError(); horizon = 4)
    @test_throws ArgumentError forecast_error(fc, AbsoluteError(); horizon = 0)
    @test_throws ArgumentError all_forecast_errors(fc, ForecastError(); truth = [1.0])

    no_truth = Forecast(horizon = [1, 2], mean = [1.0, 2.0])
    @test all_forecast_errors(no_truth, ForecastError()) == Float64[]
    @test_throws ArgumentError all_forecast_errors(no_truth, ForecastError(); missing_strategy = :error)

    @test forecast_error([fc, fc], AbsoluteError(); horizon = 2) == [3.0, 3.0]
    @test all_forecast_errors([fc, fc], ForecastError()) == [[-2.0, 3.0, 0.0], [-2.0, 3.0, 0.0]]
end

@testitem "Interval errors" tags = [:scoring] begin
    interval = ForecastInterval([8.0, 5.0], [12.0, 15.0], [0.5, 0.9])

    @test compute_error(interval, 13.0, CoverageError(correct_level = false)) == [true, false]
    @test compute_error(interval, 13.0, CoverageError()) ≈ [1 - 0.5, 0 - 0.1]
    @test compute_error(interval, 13.0, IntervalWidth()) == [4.0, 10.0]
    # Width plus 2 / alpha times the distance outside the interval, alpha = 1 - level.
    @test compute_error(interval, 13.0, IntervalScore()) ≈ [4 + 2 / 0.5 * 1, 10.0]
    @test compute_error(interval, 4.0, IntervalScore()) ≈ [4 + 2 / 0.5 * 4, 10 + 2 / 0.1 * 1]

    fc = Forecast(
        horizon = [1, 2], mean = [10.0, 10.0],
        intervals = [interval, ForecastInterval([9.0, 6.0], [13.0, 16.0], [0.5, 0.9])],
        truth = [13.0, 20.0]
    )
    @test forecast_error(fc, IntervalWidth(); horizon = 2) == [4.0, 10.0]
    @test all_forecast_errors(fc, IntervalWidth()) == [[4.0, 10.0], [4.0, 10.0]]
    @test all_forecast_errors(fc, IntervalScore()) ≈ [[8.0, 10.0], [4 + 2 / 0.5 * 7, 10 + 2 / 0.1 * 4]]

    no_truth = Forecast(horizon = fc.horizon, mean = fc.mean, intervals = fc.intervals)
    # Without truth, interval errors come back as missing whatever the strategy, where
    # point errors are skipped or raise an error.
    @test_broken isempty(all_forecast_errors(no_truth, IntervalScore()))
    @test_broken try
        all_forecast_errors(no_truth, IntervalScore(); missing_strategy = :error)
        false
    catch err
        err isa ArgumentError
    end
end

@testitem "Trajectory errors" tags = [:scoring] begin
    trajectories = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]
    # LogLoss is minus the log of the share of samples equal to the truth, plus an offset
    # (0.01 by default). The columns hold 1 to 4 and 2 to 5, so 1.0 is one of four samples
    # in the first column and none of the second, and 5.0 the other way round. Scoring a
    # truth against the wrong horizon's samples changes the result.
    fc = Forecast(horizon = [1, 2], mean = [2.5, 3.5], trajectories = trajectories, truth = [1.0, 5.0])
    @test all_forecast_errors(fc, LogLoss()) ≈ [-log(1 / 4 + 0.01), -log(1 / 4 + 0.01)]

    fc = Forecast(horizon = [1, 2], mean = [2.5, 3.5], trajectories = trajectories, truth = [1.0, 1.0])
    @test all_forecast_errors(fc, LogLoss()) ≈ [-log(1 / 4 + 0.01), -log(0 + 0.01)]
    @test all_forecast_errors(fc, LogLoss(offset = 0.1)) ≈ [-log(1 / 4 + 0.1), -log(0 + 0.1)]

    no_truth = Forecast(horizon = fc.horizon, mean = fc.mean, trajectories = trajectories)
    @test isempty(all_forecast_errors(no_truth, LogLoss()))
    @test_throws ArgumentError all_forecast_errors(no_truth, LogLoss(); missing_strategy = :error)
end
