@testmodule TestData begin
    using Random, Distributions

    export trend_data, seasonal_data, stationary_data, count_data, test_horizons

    Random.seed!(123)

    const n = 250
    const trend_data = collect(1:n) + 0.1 * randn(n)
    const seasonal_data = 10 .+ 2 * sin.(2π * (1:n) / 12) + 0.5 * randn(n)
    const stationary_data = 5 .+ 0.3 * randn(n)
    const count_data = rand(Poisson(5), n)

    const test_horizons = [1, 3, [1, 2, 5], 1:6]
end
