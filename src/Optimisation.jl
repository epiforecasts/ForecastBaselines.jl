"""
    OptimisationInfo(converged::Bool, iterations::Int)

Whether the optimiser that fitted a model met its convergence criterion, and how many
iterations it took. Models fitted by numerical optimisation (ARMA, INARCH, ETS) record
it, and `has_converged` reports it.
"""
struct OptimisationInfo
    converged::Bool
    iterations::Int
end

# Record how the optimisation went, and warn when it stopped without converging. The
# warning is shown once per session, as fitting many series would otherwise repeat it.
function optimisation_info(result, model_name::AbstractString)
    info = OptimisationInfo(Optim.converged(result), Optim.iterations(result))
    if !info.converged
        @warn "The $model_name fit stopped after $(info.iterations) iterations without " *
            "converging, so its parameters may not be the best fit. Use " *
            "`has_converged(fitted)` to check each fit; this warning is shown once." maxlog = 1
    end
    return info
end

"""
    has_converged(fitted::AbstractFittedModel) -> Union{Bool, Nothing}

Whether the optimiser that fitted `fitted` converged. Returns `nothing` for models that
are not fitted by numerical optimisation, or that were built without one.
"""
has_converged(fitted::AbstractFittedModel) = nothing
