@testitem "A seasonal ETS fit converges" setup = [TestData] tags = [:models, :ets] begin
    using ForecastBaselines
    model = ETSModel(error = "A", trend = "A", season = "A", s = 12)
    fitted = fit_baseline(seasonal_data, model)
    @test has_converged(fitted)
    @test ForecastBaselines.L_fun(seasonal_data, model, ForecastBaselines.ETSpar2θ(fitted.par, model)) < 1015.5
end

@testitem "ETS keeps a better result from the original search" tags = [:models, :ets] begin
    using ForecastBaselines, Optim, Random
    Random.seed!(123)
    n = 250
    randn(n)
    randn(n)
    randn(n)
    x = (1:n) .* 0.05 .+ 20 .+ 3 .* sin.(2π .* (1:n) ./ 12) .+ 0.5 .* randn(n)
    model = ETSModel(error = "A", trend = "Ad")
    initial = ForecastBaselines.ETSpar2θ(ForecastBaselines.getInitial(x, model), model)
    original = Optim.optimize(v -> ForecastBaselines.L_fun(x, model, v), initial)
    fitted = fit_baseline(x, model)
    value = ForecastBaselines.L_fun(x, model, ForecastBaselines.ETSpar2θ(fitted.par, model))
    @test value <= Optim.minimum(original) + 1.0e-6
end

@testitem "Seasonal parameters survive conversion without a trend" tags = [:models, :ets] begin
    using ForecastBaselines
    model = ETSModel(season = "A", s = 4)
    par = ForecastBaselines.ETSParameter(
        ForecastBaselines.ETSTuning(0.3, 0.0, 0.2, 0.0),
        ForecastBaselines.ETSSpace(10.0, 0.0, [1.0, -1.0, 2.0, -2.0])
    )
    @test ForecastBaselines.ETSθ2par(ForecastBaselines.ETSpar2θ(par, model), model).θ.γ == 0.2
end
