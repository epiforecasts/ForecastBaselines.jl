@testitem "Interval Method Tests" setup = [TestData] tags = [:intervals] begin
    using Random, Distributions, Dates
    # Test different interval methods with Marginal model
    model = MarginalModel(p = 10)
    fitted = fit_baseline(stationary_data, model)

    # Test NoInterval
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, NoInterval(), 1:3
    )
    @test fc_intervals === nothing
    @test fc_traj === nothing
    @test fc_median == fc_point

    # Test EmpiricalInterval
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, EmpiricalInterval(n_trajectories = 50, min_observation = 5, return_trajectories = true), 1:2, [0.8]
    )
    @test length(fc_intervals) == 2
    @test size(fc_traj) == (50, 2)

    # Test ParametricInterval with positivity correction
    fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
        fitted, ParametricInterval(positivity_correction = :post_clip), 1:2, [0.8]
    )
    @test length(fc_intervals) == 2

    # Test ModelTrajectoryInterval with different positivity corrections
    for correction in [:none, :post_clip, :zero_floor, :truncate]
        fc_point, fc_median, fc_intervals, fc_traj = interval_forecast(
            fitted, ModelTrajectoryInterval(n_trajectories = 20, positivity_correction = correction, return_trajectories = true), 1:2, [0.8]
        )
        @test size(fc_traj) == (20, 2)
    end
end

@testitem "More comprehensive interval method test" setup = [TestData] tags = [:intervals] begin
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

            c = forecast(fitted, interval_method = i_methods[9], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[10], horizon = 3)
            @test length(fc.mean) == 3

            fc = forecast(fitted, interval_method = i_methods[11], horizon = 3)
            @test length(fc.mean) == 3
        end
    end
end
