using Aqua, ExplicitImports

@testset "Code quality" begin
    @testset "Aqua" begin
        Aqua.test_all(ForecastBaselines)
    end

    @testset "ExplicitImports" begin
        @test check_no_implicit_imports(ForecastBaselines) === nothing
        @test check_no_stale_explicit_imports(ForecastBaselines) === nothing
        # KDEModel reuses KernelDensity's bandwidth, boundary and weight defaults so
        # that its fits match kde(); KernelDensity does not mark these as public.
        @test check_all_qualified_accesses_are_public(
            ForecastBaselines;
            ignore = (:default_bandwidth, :default_weights, :kde_boundary)
        ) === nothing
    end
end
