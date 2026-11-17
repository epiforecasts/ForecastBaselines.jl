@testitem "Aqua" tags = [:quality] begin
    using Aqua
    Aqua.test_all(ForecastBaselines)
end

@testitem "ExplicitImports" tags = [:quality] begin
    using ExplicitImports
    @test check_no_implicit_imports(ForecastBaselines) === nothing
    @test check_no_stale_explicit_imports(ForecastBaselines) === nothing
    # Julia only records which names are public from 1.11 on.
    if VERSION >= v"1.11"
        # KDEModel reuses KernelDensity's bandwidth, boundary and weight defaults so
        # that its fits match kde(), and fits read Optim's documented `converged` and
        # `iterations` accessors and keep the better of two fits using `minimum`. Neither package marks these names as public.
        @test check_all_qualified_accesses_are_public(
            ForecastBaselines;
            ignore = (:default_bandwidth, :default_weights, :kde_boundary, :converged, :iterations, :minimum)
        ) === nothing
    end
end
