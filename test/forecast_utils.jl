@testmodule ForecastFixtures begin
    using ForecastBaselines, Dates

    export example_forecast, unchanged_except

    # With `dated = true` the forecast also carries a reference date, weekly target
    # dates and a resolution, so that every field of a Forecast is set.
    function example_forecast(; horizon = [1, 2, 3], model_name = "Example", dated = false)
        n = length(horizon)
        fc = Forecast(
            horizon = horizon,
            mean = Float64.(10 .* horizon),
            median = Float64.(10 .* horizon .+ 1),
            intervals = [
                ForecastInterval([10h - 2.0, 10h - 5.0], [10h + 2.0, 10h + 5.0], [0.5, 0.9])
                    for h in horizon
            ],
            truth = Float64.(10 .* horizon .+ 3),
            # Each horizon's samples differ, so a misplaced column shows up.
            trajectories = reduce(hcat, [100.0 * h .+ (1:4) for h in horizon]),
            model_name = model_name
        )
        return dated ? add_temporal_info(fc, Date(2024, 1, 1); resolution = Week(1)) : fc
    end

    # Whether every field of `b` other than `field` is the one in `a`.
    function unchanged_except(a, b, field)
        return all(f -> f == field || isequal(getfield(a, f), getfield(b, f)), fieldnames(Forecast))
    end
end

@testitem "add_* functions set one field and keep the others" setup = [ForecastFixtures] tags = [:forecast] begin
    # Every field is set, so dropping or changing any other field shows up.
    fc = example_forecast(dated = true)

    with_truth = add_truth(fc, [3, 4, 5])
    @test with_truth.truth == [3.0, 4.0, 5.0]
    @test unchanged_except(fc, with_truth, :truth)
    @test_throws ArgumentError add_truth(fc, [1.0, 2.0])
    @test add_truth(Forecast(horizon = [1], mean = [1.0]), 5).truth == [5.0]

    with_median = add_median(fc, [1.5, 2.5, 3.5])
    @test with_median.median == [1.5, 2.5, 3.5]
    @test unchanged_except(fc, with_median, :median)
    @test_throws ArgumentError add_median(fc, [1.0])

    intervals = [ForecastInterval([0.0], [2.0], [0.9]) for _ in 1:3]
    with_intervals = add_intervals(fc, intervals)
    @test with_intervals.intervals == intervals
    @test unchanged_except(fc, with_intervals, :intervals)
    @test_throws ArgumentError add_intervals(fc, intervals[1:1])

    with_trajectories = add_trajectories(fc, [1.0 2.0 3.0; 4.0 5.0 6.0])
    @test num_trajectories(with_trajectories) == 2
    @test unchanged_except(fc, with_trajectories, :trajectories)
    @test_throws ArgumentError add_trajectories(fc, ones(3, 2))
    single = add_trajectories(Forecast(horizon = [1], mean = [1.0]), [1.0, 2.0, 3.0])
    @test size(single.trajectories) == (3, 1)

    removed = remove_trajectories(fc)
    @test !has_trajectories(removed)
    @test num_trajectories(removed) == 0
    @test unchanged_except(fc, removed, :trajectories)

    renamed = update_model_name(fc, "renamed")
    @test renamed.model_name == "renamed"
    @test unchanged_except(fc, renamed, :model_name)
end

@testitem "Horizon accessors" setup = [ForecastFixtures] tags = [:forecast] begin
    fc = example_forecast(horizon = [2, 5, 3])
    @test forecast_length(fc) == 3
    @test max_horizon(fc) == 5
    @test min_horizon(fc) == 2
    @test get_horizon_range(fc) == 2:5

    empty = Forecast()
    @test forecast_length(empty) == 0
    @test isnothing(max_horizon(empty))
    @test isnothing(min_horizon(empty))
    @test isnothing(get_horizon_range(empty))
end

@testitem "filter_horizons keeps the requested horizons in every field" setup = [ForecastFixtures] tags = [:forecast] begin
    fc = example_forecast(horizon = [1, 2, 3])
    kept = filter_horizons(fc, [3, 1])
    @test kept.horizon == [3, 1]
    @test kept.mean == [30.0, 10.0]
    @test kept.median == [31.0, 11.0]
    @test kept.truth == [33.0, 13.0]
    @test kept.intervals == fc.intervals[[3, 1]]
    @test kept.trajectories == fc.trajectories[:, [3, 1]]
    @test filter_horizons(fc) === fc
    @test_throws ArgumentError filter_horizons(fc, [4])
end

@testitem "filter_levels keeps the requested interval levels" setup = [ForecastFixtures] tags = [:forecast] begin
    fc = example_forecast()
    kept = filter_levels(fc, [0.9])
    @test all(i -> i.levels == [0.9], kept.intervals)
    @test kept.intervals[1].lower == [5.0]
    @test kept.intervals[1].upper == [15.0]
    @test filter_levels(fc) === fc
    @test filter_levels(Forecast(horizon = [1], mean = [1.0]), [0.9]).intervals === nothing
end

@testitem "extend_forecast merges forecasts with disjoint horizons in horizon order" setup = [ForecastFixtures] tags = [:forecast] begin
    early = example_forecast(horizon = [1, 3], model_name = "a")
    late = example_forecast(horizon = [2, 4], model_name = "b")
    combined = extend_forecast(early, late)
    @test combined.horizon == [1, 2, 3, 4]
    @test combined.mean == [10.0, 20.0, 30.0, 40.0]
    @test combined.truth == [13.0, 23.0, 33.0, 43.0]
    @test combined.intervals == [early.intervals[1], late.intervals[1], early.intervals[2], late.intervals[2]]
    @test combined.trajectories == hcat(early.trajectories, late.trajectories)[:, [1, 3, 2, 4]]
    @test combined.model_name == "a + b"

    @test_throws ArgumentError extend_forecast(early, example_forecast(horizon = [3]))
    @test_throws ArgumentError extend_forecast(
        early, Forecast(horizon = [2], mean = [1.0], resolution = 2)
    )
end

@testitem "Temporal helpers" setup = [ForecastFixtures] tags = [:forecast] begin
    using Dates
    fc = add_temporal_info(example_forecast(), Date(2024, 1, 1); resolution = Week(1))
    @test fc.target_date == [Date(2024, 1, 8), Date(2024, 1, 15), Date(2024, 1, 22)]
    @test get_temporal_span(fc) == Day(14)
    @test isnothing(get_temporal_span(example_forecast()))
    @test isnothing(get_temporal_span(add_temporal_info(example_forecast(horizon = [1]), Date(2024, 1, 1))))
end

@testitem "Forecasts display their contents" setup = [ForecastFixtures] tags = [:forecast] begin
    using Dates
    fc = add_temporal_info(example_forecast(model_name = "Shown"), Date(2024, 1, 1))

    compact = sprint(show, fc)
    @test occursin("Shown", compact)

    full = sprint(show, MIME("text/plain"), fc)
    @test occursin("Shown", full)
    @test occursin("2024-01-08", full)
    @test occursin("10.00", full)
    @test occursin("trajectories (4 samples)", full)

    bare = sprint(show, MIME("text/plain"), Forecast(horizon = [1], mean = [1.5]))
    @test occursin("1.50", bare)
end
