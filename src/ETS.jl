################################
### Error-Trend-Season model ###
################################

abstract type ErrorType end
abstract type TrendType end
abstract type SeasonType end

"""
    AError()

Additive error component of an [`ETSModel`](@ref).
"""
struct AError <: ErrorType end

"""
    MError()

Multiplicative error component of an [`ETSModel`](@ref).
"""
struct MError <: ErrorType end

"""
    NTrend()

No trend in an [`ETSModel`](@ref).
"""
struct NTrend <: TrendType end

"""
    ATrend()

Additive trend component of an [`ETSModel`](@ref).
"""
struct ATrend <: TrendType end

"""
    AdTrend()

Additive damped trend component of an [`ETSModel`](@ref).
"""
struct AdTrend <: TrendType end

"""
    MTrend()

Multiplicative trend component of an [`ETSModel`](@ref).
"""
struct MTrend <: TrendType end

"""
    MdTrend()

Multiplicative damped trend component of an [`ETSModel`](@ref).
"""
struct MdTrend <: TrendType end

"""
    NSeason()

No seasonality in an [`ETSModel`](@ref).
"""
struct NSeason <: SeasonType end

"""
    ASeason(s::Int)

Additive seasonal component of an [`ETSModel`](@ref), with period `s`.
"""
struct ASeason <: SeasonType
    s::Int
end

"""
    Mseason(s::Int)

Multiplicative seasonal component of an [`ETSModel`](@ref), with period `s`.
"""
struct Mseason <: SeasonType
    s::Int
end

"""
    ETSModel{T1, T2, T3}(error::T1, trend::T2, season::T3)
    ETSModel(;error::String = "A", trend::String = "N", season::String = "N", s::Int = 1)

Error-Trend-Season exponential smoothing model.

Implements all 30 variants of ETS models with different combinations of:
- **Error**: Additive ("A") or Multiplicative ("M")
- **Trend**: None ("N"), Additive ("A"), Multiplicative ("M"), or Damped versions ("Ad", "Md")
- **Season**: None ("N"), Additive ("A"), or Multiplicative ("M")

# Mathematical Framework
The general ETS formulation:
- x_t = w(z_{t-1}) + r(z_{t-1})ϵ_t
- z_t = f(z_{t-1}) + g(z_{t-1})ϵ_t

where z_t contains level, trend, and seasonal components.

# Arguments
- `error::String`: Error type ("A" or "M")
- `trend::String`: Trend type ("N", "A", "Ad", "M", "Md")  
- `season::String`: Seasonal type ("N", "A", "M")
- `s::Int`: Seasonal period (ignored if season="N")

# Common Model Examples
```julia
# Simple exponential smoothing (A,N,N)
model = ETSModel(error="A", trend="N", season="N")

# Holt's linear trend (A,A,N)  
model = ETSModel(error="A", trend="A", season="N")

# Holt-Winters additive (A,A,A)
model = ETSModel(error="A", trend="A", season="A", s=12)

# Damped trend multiplicative seasonal (A,Ad,M)
model = ETSModel(error="A", trend="Ad", season="M", s=4)
```

# Model Variants
All 30 ETS combinations are supported with automatic function dispatch
based on the error/trend/season type combination.
"""
struct ETSModel{T1, T2, T3} <: AbstractBaselineModel where {T1 <: ErrorType, T2 <: TrendType, T3 <: SeasonType}
    error::T1
    trend::T2
    season::T3
end

function ETSModel(;
        error::String = "A",
        trend::String = "N",
        season::String = "N",
        s::Int = 1
    )
    if error == "A"
        et = AError()
    elseif error == "M"
        et = MError()
    else
        throw(ArgumentError("error must be \"A\" or \"M\", got \"$error\""))
    end

    if trend == "A"
        tt = ATrend()
    elseif trend == "Ad"
        tt = AdTrend()
    elseif trend == "M"
        tt = MTrend()
    elseif trend == "Md"
        tt = MdTrend()
    else
        tt = NTrend()
    end

    if season == "A"
        st = ASeason(s)
    elseif season == "M"
        st = Mseason(s)
    else
        st = NSeason()
    end
    return ETSModel(et, tt, st)
end

"""
    ETSSpace(L::Float64, B::Float64, S::Vector{Float64})

State space representation for ETS models.

# Fields  
- `L::Float64`: Level component
- `B::Float64`: Trend component (0 if no trend)
- `S::Vector{Float64}`: Seasonal components (length s for seasonal models, [0] otherwise)

Supports arithmetic operations (+, *) for state vector updates during filtering.
"""
struct ETSSpace
    L::Float64
    B::Float64
    S::Vector{Float64}
end

"""
    ETSTuning(α::Float64, β::Float64, γ::Float64, ϕ::Float64)

Smoothing parameters for ETS models.

# Fields
- `α::Float64`: Level smoothing parameter (0 ≤ α ≤ 1)
- `β::Float64`: Trend smoothing parameter (0 ≤ β ≤ α)  
- `γ::Float64`: Seasonal smoothing parameter (0 ≤ γ ≤ 1-α)
- `ϕ::Float64`: Damping parameter (0 ≤ ϕ ≤ 1, used only for damped trends)

Parameters control how quickly the model adapts to new information.
"""
struct ETSTuning
    α::Float64
    β::Float64
    γ::Float64
    ϕ::Float64
end

"""
    ETSParameter(θ::ETSTuning, z0::ETSSpace)

Complete parameter set for fitted ETS model.

# Fields
- `θ::ETSTuning`: Smoothing parameters
- `z0::ETSSpace`: Initial state values

Contains all estimated parameters needed for forecasting and filtering.
"""
struct ETSParameter <: AbstractModelParameters
    θ::ETSTuning
    z0::ETSSpace
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, NTrend, NSeason}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], 0, 0, 0), ETSSpace(θ[2], 0, [0]))
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, NTrend, <:Union{ASeason, Mseason}}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], 0, θ[2], 0), ETSSpace(θ[3], 0, θ[4:(3 + model.season.s)]))
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, <:Union{ATrend, MTrend}, NSeason}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], θ[2], 0, 0), ETSSpace(θ[3], θ[4], [0]))
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, <:Union{ATrend, MTrend}, <:Union{ASeason, Mseason}}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], θ[2], θ[3], 0), ETSSpace(θ[4], θ[5], θ[6:(5 + model.season.s)]))
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, <:Union{AdTrend, MdTrend}, NSeason}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], θ[2], 0, θ[3]), ETSSpace(θ[4], θ[5], [0]))
end

function ETSθ2par(θ::Vector{T}, model::ETSModel{<:ErrorType, <:Union{AdTrend, MdTrend}, <:Union{ASeason, Mseason}}) where {T <: Real}
    return ETSParameter(ETSTuning(θ[1], θ[2], θ[3], θ[4]), ETSSpace(θ[5], θ[6], θ[7:(6 + model.season.s)]))
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, NTrend, NSeason})
    return [par.θ.α; par.z0.L]
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, NTrend, <:Union{ASeason, Mseason}})
    return [par.θ.α; par.θ.β; par.z0.L; par.z0.S]
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, <:Union{ATrend, MTrend}, NSeason})
    return [par.θ.α; par.θ.β; par.z0.L; par.z0.B]
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, <:Union{ATrend, MTrend}, <:Union{ASeason, Mseason}})
    return [par.θ.α; par.θ.β; par.θ.γ; par.z0.L; par.z0.B; par.z0.S]
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, <:Union{AdTrend, MdTrend}, NSeason})
    return [par.θ.α; par.θ.β; par.θ.ϕ; par.z0.L; par.z0.B]
end

function ETSpar2θ(par::ETSParameter, model::ETSModel{<:ErrorType, <:Union{AdTrend, MdTrend}, <:Union{ASeason, Mseason}})
    return [par.θ.α; par.θ.β; par.θ.γ; par.θ.ϕ; par.z0.L; par.z0.B; par.z0.S]
end

# A-N-N
function w_function(z::ETSSpace, model::ETSModel{AError, NTrend, NSeason}, coef::ETSTuning)
    return z.L
end

function f_function(z::ETSSpace, model::ETSModel{AError, NTrend, NSeason}, coef::ETSTuning)
    return z
end

function g_function(z::ETSSpace, model::ETSModel{AError, NTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α, 0.0, [0.0])
end

# A-N-A
function w_function(z::ETSSpace, model::ETSModel{AError, NTrend, ASeason}, coef::ETSTuning)
    return z.L + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, NTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L, 0.0, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, NTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α, 0.0, [coef.γ; zeros(length(z.S) - 1)])
end

# A-N-M
function w_function(z::ETSSpace, model::ETSModel{AError, NTrend, Mseason}, coef::ETSTuning)
    return z.L * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, NTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L, 0.0, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, NTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α / z.S[end], 0.0, [coef.γ / z.L; zeros(length(z.S) - 1)])
end

# A-A-N
function w_function(z::ETSSpace, model::ETSModel{AError, ATrend, NSeason}, coef::ETSTuning)
    return z.L + z.B
end


function f_function(z::ETSSpace, model::ETSModel{AError, ATrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{AError, ATrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β, [0.0])
end

# A-A-A
function w_function(z::ETSSpace, model::ETSModel{AError, ATrend, ASeason}, coef::ETSTuning)
    return z.L + z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, ATrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, ATrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β, [coef.γ; zeros(length(z.S) - 1)])
end

# A-A-M
function w_function(z::ETSSpace, model::ETSModel{AError, ATrend, Mseason}, coef::ETSTuning)
    return (z.L + z.B) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, ATrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, ATrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α / z.S[end], coef.β / z.S[end], [coef.γ / (z.L + z.B); zeros(length(z.S) - 1)])
end

# A-Ad-N
function w_function(z::ETSSpace, model::ETSModel{AError, AdTrend, NSeason}, coef::ETSTuning)
    return z.L + coef.ϕ * z.B
end

function f_function(z::ETSSpace, model::ETSModel{AError, AdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{AError, AdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β, [0.0])
end

# A-Ad-A
function w_function(z::ETSSpace, model::ETSModel{AError, AdTrend, ASeason}, coef::ETSTuning)
    return z.L + coef.ϕ * z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, AdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, AdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β, [coef.γ; zeros(length(z.S) - 1)])
end

# A-Ad-M
function w_function(z::ETSSpace, model::ETSModel{AError, AdTrend, Mseason}, coef::ETSTuning)
    return (z.L + coef.ϕ * z.B) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, AdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, AdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α / z.S[end], coef.β / z.S[end], [coef.γ / (z.L + coef.ϕ * z.B); zeros(length(z.S) - 1)])
end

# A-M-N
function w_function(z::ETSSpace, model::ETSModel{AError, MTrend, NSeason}, coef::ETSTuning)
    return z.L * z.B
end

function f_function(z::ETSSpace, model::ETSModel{AError, MTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β / z.L, [0.0])
end

# A-M-A
function w_function(z::ETSSpace, model::ETSModel{AError, MTrend, ASeason}, coef::ETSTuning)
    return z.L * z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, MTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β / z.L, [coef.γ; zeros(length(z.S) - 1)])
end

# A-M-M
function w_function(z::ETSSpace, model::ETSModel{AError, MTrend, Mseason}, coef::ETSTuning)
    return z.L * z.B * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, MTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α / z.S[end], coef.β / (z.S[end] * z.L), [coef.γ / (z.L * z.B); zeros(length(z.S) - 1)])
end

# A-Md-N
function w_function(z::ETSSpace, model::ETSModel{AError, MdTrend, NSeason}, coef::ETSTuning)
    return z.L * z.B^coef.ϕ
end

function f_function(z::ETSSpace, model::ETSModel{AError, MdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β / z.L, [0.0])
end

# A-Md-A
function w_function(z::ETSSpace, model::ETSModel{AError, MdTrend, ASeason}, coef::ETSTuning)
    return z.L * z.B^coef.ϕ + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, MdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α, coef.β / z.L, [coef.γ; zeros(length(z.S) - 1)])
end

# A-Md-M
function w_function(z::ETSSpace, model::ETSModel{AError, MdTrend, Mseason}, coef::ETSTuning)
    return (z.L * z.B^coef.ϕ) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{AError, MdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{AError, MdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α / z.S[end], coef.β / (z.S[end] * z.L), [coef.γ / (z.L * z.B^coef.ϕ); zeros(length(z.S) - 1)])
end

### Multiplicative errors

# M-N-N
function w_function(z::ETSSpace, model::ETSModel{MError, NTrend, NSeason}, coef::ETSTuning)
    return z.L
end

function f_function(z::ETSSpace, model::ETSModel{MError, NTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L, 0.0, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{MError, NTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α * z.L, 0.0, [0.0])
end

# M-N-A
function w_function(z::ETSSpace, model::ETSModel{MError, NTrend, ASeason}, coef::ETSTuning)
    return z.L + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, NTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L, 0.0, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, NTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(coef.α * (z.L + z.S[end]), 0.0, [coef.γ * (z.L + z.S[end]); zeros(length(z.S) - 1)])
end

# M-N-M
function w_function(z::ETSSpace, model::ETSModel{MError, NTrend, Mseason}, coef::ETSTuning)
    return z.L * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, NTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L, 0.0, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, NTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L * coef.α, 0.0, [coef.γ * z.S[end]; zeros(length(z.S) - 1)])
end

# M-A-N
function w_function(z::ETSSpace, model::ETSModel{MError, ATrend, NSeason}, coef::ETSTuning)
    return z.L + z.B
end

function f_function(z::ETSSpace, model::ETSModel{MError, ATrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{MError, ATrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α * (z.L + z.B), coef.β * (z.L + z.B), [0.0])
end

# M-A-A
function w_function(z::ETSSpace, model::ETSModel{MError, ATrend, ASeason}, coef::ETSTuning)
    return z.L + z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, ATrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, ATrend, ASeason}, coef::ETSTuning)
    scl = (z.L + z.B + z.S[end])
    return ETSSpace(coef.α * scl, coef.β * scl, [coef.γ * scl; zeros(length(z.S) - 1)])
end

# M-A-M
function w_function(z::ETSSpace, model::ETSModel{MError, ATrend, Mseason}, coef::ETSTuning)
    return (z.L + z.B) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, ATrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L + z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, ATrend, Mseason}, coef::ETSTuning)
    scl = z.L + z.B
    return ETSSpace(coef.α * scl, coef.β * scl, [coef.γ * z.S[end]; zeros(length(z.S) - 1)])
end

# M-Ad-N
function w_function(z::ETSSpace, model::ETSModel{MError, AdTrend, NSeason}, coef::ETSTuning)
    return z.L + coef.ϕ * z.B
end

function f_function(z::ETSSpace, model::ETSModel{MError, AdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{MError, AdTrend, NSeason}, coef::ETSTuning)
    scl = z.L + coef.ϕ * z.B
    return ETSSpace(coef.α * scl, coef.β * scl, [0.0])
end

# M-Ad-A
function w_function(z::ETSSpace, model::ETSModel{MError, AdTrend, ASeason}, coef::ETSTuning)
    return z.L + coef.ϕ * z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, AdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, AdTrend, ASeason}, coef::ETSTuning)
    scl = z.L + coef.ϕ * z.B + z.S[end]
    return ETSSpace(coef.α * scl, coef.β * scl, [coef.γ * scl; zeros(length(z.S) - 1)])
end

# M-Ad-M
function w_function(z::ETSSpace, model::ETSModel{MError, AdTrend, Mseason}, coef::ETSTuning)
    return (z.L + coef.ϕ * z.B) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, AdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L + coef.ϕ * z.B, coef.ϕ * z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, AdTrend, Mseason}, coef::ETSTuning)
    scl = z.L + coef.ϕ * z.B
    return ETSSpace(coef.α * scl, coef.β * scl, [coef.γ * z.S[end]; zeros(length(z.S) - 1)])
end

# M-M-N
function w_function(z::ETSSpace, model::ETSModel{MError, MTrend, NSeason}, coef::ETSTuning)
    return z.L * z.B
end

function f_function(z::ETSSpace, model::ETSModel{MError, MTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α * z.L * z.B, coef.β * z.B, [0.0])
end

# M-M-A
function w_function(z::ETSSpace, model::ETSModel{MError, MTrend, ASeason}, coef::ETSTuning)
    return z.L * z.B + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, MTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MTrend, ASeason}, coef::ETSTuning)
    scl = z.L * z.B + z.S[end]
    return ETSSpace(coef.α * scl, coef.β * scl / z.L, [coef.γ * scl; zeros(length(z.S) - 1)])
end

# M-M-M
function w_function(z::ETSSpace, model::ETSModel{MError, MTrend, Mseason}, coef::ETSTuning)
    return z.L * z.B * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, MTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B, z.B, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α * z.L * z.B, coef.β * z.B, [coef.γ * z.S[end]; zeros(length(z.S) - 1)])
end

# M-Md-N
function w_function(z::ETSSpace, model::ETSModel{MError, MdTrend, NSeason}, coef::ETSTuning)
    return z.L * z.B^coef.ϕ
end

function f_function(z::ETSSpace, model::ETSModel{MError, MdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [0.0])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MdTrend, NSeason}, coef::ETSTuning)
    return ETSSpace(coef.α * z.L * z.B^coef.ϕ, coef.β * z.B^coef.ϕ, [0.0])
end

# M-Md-A
function w_function(z::ETSSpace, model::ETSModel{MError, MdTrend, ASeason}, coef::ETSTuning)
    return z.L * z.B^coef.ϕ + z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, MdTrend, ASeason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MdTrend, ASeason}, coef::ETSTuning)
    scl = z.L * z.B^coef.ϕ + z.S[end]
    return ETSSpace(coef.α * scl, coef.β * scl / z.L, [coef.γ * scl; zeros(length(z.S) - 1)])
end

# M-Md-M
function w_function(z::ETSSpace, model::ETSModel{MError, MdTrend, Mseason}, coef::ETSTuning)
    return (z.L * z.B^coef.ϕ) * z.S[end]
end

function f_function(z::ETSSpace, model::ETSModel{MError, MdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(z.L * z.B^coef.ϕ, z.B^coef.ϕ, [z.S[end]; z.S[1:(end - 1)]])
end

function g_function(z::ETSSpace, model::ETSModel{MError, MdTrend, Mseason}, coef::ETSTuning)
    return ETSSpace(coef.α * z.L * z.B^coef.ϕ, coef.β * z.B^coef.ϕ, [coef.γ * z.S[end]; zeros(length(z.S) - 1)])
end

function r_function(z::ETSSpace, model::ETSModel{AError, <:Union{NTrend, ATrend, MTrend, AdTrend, MdTrend}, <:Union{NSeason, ASeason, Mseason}}, coef::ETSTuning)
    return 1.0
end

function r_function(z::ETSSpace, model::ETSModel{MError, <:Union{NTrend, ATrend, MTrend, AdTrend, MdTrend}, <:Union{NSeason, ASeason, Mseason}}, coef::ETSTuning)
    return w_function(z, model, coef)
end


# Arithmetics for "ETSSpace" objects
import Base.+
+(s1::ETSSpace, s2::ETSSpace) = begin
    ETSSpace(s1.L + s2.L, s1.B + s2.B, s1.S .+ s2.S)
end

import Base.*
*(s1::ETSSpace, s2::ETSSpace) = begin
    ETSSpace(s1.L * s2.L, s1.B * s2.B, s1.S .* s2.S)
end

*(s1::ETSSpace, s2::T) where {T <: Real} = begin
    ETSSpace(s1.L * s2, s1.B * s2, s1.S .* s2)
end

"""
    ETSFiltered(z0::ETSSpace, z::Vector{ETSSpace}, xHat::Vector{Float64}, ϵ::Vector{Float64})

Results from ETS filtering process.

# Fields
- `z0::ETSSpace`: Initial state
- `z::Vector{ETSSpace}`: State evolution over time
- `xHat::Vector{Float64}`: One-step-ahead fitted values
- `ϵ::Vector{Float64}`: Standardised residuals

Contains complete filtering history for analysis and forecasting.
"""
struct ETSFiltered
    z0::ETSSpace
    z::Vector{ETSSpace}
    xHat::Vector{Float64}
    ϵ::Vector{Float64}
    function ETSFiltered(z0::ETSSpace, z::Vector{ETSSpace}, xHat, ϵ)
        return new(z0, z, Float64.(xHat), Float64.(ϵ))
    end
end

# Function to carry out filtering
"""
    ETSFilter(x::Vector{Float64}, model::ETSModel, par::ETSParameter) -> ETSFiltered

Apply ETS filtering to time series.

Performs forward pass through the data computing:
1. One-step-ahead predictions using w_function
2. Forecast errors (additive or relative)
3. State updates using f_function and g_function

# Arguments
- `x::Vector{Float64}`: Time series data
- `model::ETSModel`: Model specification
- `par::ETSParameter`: Model parameters

# Returns
- `ETSFiltered`: Complete filtering results

Used internally during estimation and for generating residual diagnostics.
"""
function ETSFilter(x::Vector{T1}, model::ETSModel, par::ETSParameter) where {T1 <: Real}
    T = length(x)
    xHat = zeros(T)
    ϵ = zeros(T)
    z0 = par.z0
    z = Vector{typeof(z0)}(undef, T)

    xHat[1] = w_function(par.z0, model, par.θ)
    if typeof(model.error) == AError
        ϵ[1] = x[1] - xHat[1]
    else
        ϵ[1] = (x[1] - xHat[1]) / xHat[1]
    end
    z[1] = f_function(z0, model, par.θ) + g_function(z0, model, par.θ) * ϵ[1]

    for t in 2:T
        xHat[t] = w_function(z[t - 1], model, par.θ)
        if typeof(model.error) == AError
            ϵ[t] = x[t] - xHat[t]
        else
            ϵ[t] = (x[t] - xHat[t]) / xHat[t]
        end
        z[t] = f_function(z[t - 1], model, par.θ) + g_function(z[t - 1], model, par.θ) * ϵ[t]
    end
    return ETSFiltered(z0, z, xHat, ϵ)
end

"""
    getInitial(x::Vector{Float64}, model::ETSModel) -> ETSParameter

Generate initial parameter values for ETS estimation.

Computes reasonable starting values for optimisation:
- Uses classical decomposition for seasonal initialisation
- Linear regression for trend initialisation  
- Default smoothing parameters (α=0.1, β=0.01, γ=0.01, ϕ=0.99)

# Arguments
- `x::Vector{Float64}`: Time series data
- `model::ETSModel`: Model specification

# Returns
- `ETSParameter`: Initial parameter guess

Automatically handles different model types and seasonal periods.
"""
function getInitial(x::Vector{T1}, model::ETSModel) where {T1 <: Real}
    coef = ETSTuning(0.1, 0.01, 0.01, 0.99)
    T = length(x)
    if !(model.season isa NSeason)
        s = model.season.s
        # The moving average over two seasons leaves T - 2s + 1 deviations, and each
        # season needs at least one of them.
        if length(x) - 2 * s + 1 >= s
            ff = (t -> mean(x[t:(t + 2 * s - 1)])).(1:(T - 2 * s + 1))
            tSeq = (1:(T - 2 * s + 1)) .+ s
            if model.season isa ASeason
                xx = x[tSeq] .- ff
                sRaw = (ss -> mean(xx[ss:s:length(xx)])).(1:s)
                # Additive seasonal effects sum to zero.
                sNorm = sRaw .- mean(sRaw)
            else
                xx = x[tSeq] ./ ff
                sRaw = (ss -> mean(xx[ss:s:length(xx)])).(1:s)
                # Multiplicative seasonal factors average one.
                sNorm = sRaw ./ mean(sRaw)
            end
        else
            # Too short to estimate a seasonal pattern: start from none.
            xx = x
            sNorm = model.season isa ASeason ? zeros(s) : ones(s)
        end
        # The filter uses S[end] for the first observation, S[end - 1] for the second
        # and so on, so the seasons go in back to front.
        sNorm = reverse(sNorm)
    else
        xx = x
        sNorm = [0.0]
    end

    # The level and trend describe the series with the season taken out, not the
    # deviations from the moving average that the seasonal states come from.
    if model.season isa NSeason
        adjusted = x
    else
        season_of_t = reverse(sNorm)[mod1.(1:T, model.season.s)]
        adjusted = model.season isa ASeason ? x .- season_of_t : x ./ season_of_t
    end
    nxx = minimum([10, length(adjusted)])
    yy = adjusted[1:nxx]
    X = [ones(nxx) collect(1:nxx)]
    beta = inv(X'X) * X'yy
    l = beta[1]
    if model.trend isa ATrend || model.trend isa AdTrend
        b = beta[2]
    elseif model.trend isa MTrend || model.trend isa MdTrend
        if all(yy .> 0)
            # A growth factor is positive, so fit the line on the log scale; the
            # level and factor then start positive.
            log_beta = inv(X'X) * X' * log.(yy)
            l = exp(log_beta[1])
            b = exp(log_beta[2])
        else
            b = 1.0
        end
    else
        b = 0.0
    end
    return ETSParameter(coef, ETSSpace(l, b, sNorm))
end

function L_fun(x::Vector{T1}, model::ETSModel, parVec::Vector{Float64}) where {T1 <: Real}
    par = ETSθ2par(parVec, model)

    # Nelder-Mead stops once the objective values over the simplex are close
    # together. A single infinite vertex makes their spread NaN, and the search
    # then stops at its starting point. The initial simplex moves phi = 0.99
    # past 1, which puts such a vertex on every damped-trend fit. A finite
    # penalty that grows with the violation keeps the spread finite and pushes
    # the simplex back inside the constraints.
    violation = max(0, -par.θ.α) + max(0, par.θ.α - 1) +
        max(0, -par.θ.β) + max(0, par.θ.β - par.θ.α) +
        max(0, -par.θ.γ) + max(0, par.θ.γ - (1 - par.θ.α)) +
        max(0, -par.θ.ϕ) + max(0, par.θ.ϕ - 1)
    if violation > 0
        return 1.0e8 * (1 + violation)
    end

    filtered = try
        ETSFilter(x, model, par)
    catch err
        err isa DomainError || rethrow()
        # The trend state turned negative partway through the series.
        return 1.0e8
    end
    # The filter only raises the states before the last to the power phi, but
    # point_forecast starts from the last one.
    if model.trend isa MdTrend && filtered.z[end].B < 0
        return 1.0e8 * (1 - filtered.z[end].B)
    end
    ϵ = filtered.ϵ
    rr = [r_function(filtered.z0, model, par.θ); (zz -> r_function(zz, model, par.θ)).(filtered.z[1:(end - 1)])]
    return length(x) * log(sum(ϵ .^ 2)) + 2 * sum(log.(abs.(rr)))
end

"""
    ETSEstimationSetting(; extended_search = false)

Settings for the ETS fit. With `extended_search = true` the smoothing parameters are
estimated on a logistic scale with Nelder-Mead, followed by L-BFGS and a second starting
point if Nelder-Mead has not converged (see `fit_baseline`). The default is a single
Nelder-Mead run on the original scale.
"""
struct ETSEstimationSetting <: AbstractEstimationSetting
    extended_search::Bool
end
ETSEstimationSetting(; extended_search::Bool = false) = ETSEstimationSetting(extended_search)

logit(p) = log(p / (1 - p))
logistic(u) = 1 / (1 + exp(-u))

# The smoothing parameters come first in the parameter vector, in the order alpha,
# beta (trend), gamma (season), phi (damped trend). They are admissible when
# 0 < alpha < 1, 0 < beta < alpha, 0 < gamma < 1 - alpha and 0 < phi < 1. Each is
# therefore a logistic fraction of the room left by the earlier ones. On this scale the
# optimiser never meets a boundary and the objective is smooth everywhere.
is_damped(model::ETSModel) = model.trend isa Union{AdTrend, MdTrend}

# A starting value on or outside a bound is moved just inside it.
function smoothing_to_unconstrained(θ::Vector{Float64}, model::ETSModel)
    inside(p) = clamp(p, 1.0e-3, 1 - 1.0e-3)
    u = copy(θ)
    α = inside(θ[1])
    u[1] = logit(α)
    i = 2
    if !(model.trend isa NTrend)
        u[i] = logit(inside(θ[i] / α))
        i += 1
    end
    if !(model.season isa NSeason)
        u[i] = logit(inside(θ[i] / (1 - α)))
        i += 1
    end
    if is_damped(model)
        u[i] = logit(inside(θ[i]))
    end
    return u
end

# Nelder-Mead needs no gradients, which the penalised regions of the objective (states
# that leave their domain) would corrupt, and it gets near an optimum. Where it has
# converged the search stops and the fit costs what it always did. Otherwise L-BFGS
# continues from there, where the objective is smooth.
function optimise_ets(objective, u0::Vector{Float64})
    start = optimize(objective, u0, NelderMead())
    Optim.converged(start) && return start
    result = try
        optimize(objective, start.minimizer, LBFGS())
    catch err
        # The line search can step across a penalised region and then fail.
        err isa AssertionError || rethrow()
        start
    end
    return best_of(result, start)
end

best_of(a, b) = Optim.minimum(a) <= Optim.minimum(b) ? a : b

function smoothing_from_unconstrained(u::Vector{Float64}, model::ETSModel)
    θ = copy(u)
    α = logistic(u[1])
    θ[1] = α
    i = 2
    if !(model.trend isa NTrend)
        θ[i] = α * logistic(u[i])
        i += 1
    end
    if !(model.season isa NSeason)
        θ[i] = (1 - α) * logistic(u[i])
        i += 1
    end
    if is_damped(model)
        θ[i] = logistic(u[i])
    end
    return θ
end

"""
    ETSFitted

Container for fitted ETS model.

# Fields
- `x::Vector{Float64}`: Original time series
- `model::ETSModel`: Model specification  
- `par::ETSParameter`: Estimated parameters
- `estimation_setting::ETSEstimationSetting`: Settings used
- `temporal_info::TemporalInfo`: Temporal metadata
"""
struct ETSFitted <: AbstractFittedModel
    x::Vector{Float64}
    model::ETSModel
    par::ETSParameter
    estimation_setting::ETSEstimationSetting
    temporal_info::TemporalInfo
    optimisation::Union{OptimisationInfo, Nothing}
    function ETSFitted(
            x,
            model::ETSModel,
            par::ETSParameter,
            estimation_setting::ETSEstimationSetting,
            temporal_info::TemporalInfo = TemporalInfo(),
            optimisation::Union{OptimisationInfo, Nothing} = nothing
        )
        if !(x isa Vector{Float64})
            x = Float64.(x)
        end
        return new(x, model, par, estimation_setting, temporal_info, optimisation)
    end
end

"""
    fit_baseline(x::Vector{Float64}, model::ETSModel;
                setting::Union{ETSEstimationSetting, Nothing} = ETSEstimationSetting(),
                temporal_info::TemporalInfo = TemporalInfo(),
                start::Union{ETSParameter, Nothing} = nothing) -> ETSFitted

Fit ETS model using maximum likelihood estimation.

# Arguments
- `x::Vector{Float64}`: Time series data
- `model::ETSModel`: ETS model specification
- `setting`: Estimation settings (optional); see `ETSEstimationSetting`
- `temporal_info`: Temporal metadata (optional)
- `start`: Parameters to start the optimisation from (optional)

# Returns
- `ETSFitted`: Fitted model with estimated parameters

# Estimation Method
By default the fit is a single Nelder-Mead run on the original parameter scale from the initial values of `getInitial`. With `ETSEstimationSetting(extended_search = true)`:

1. Generate initial parameter values using `getInitial`
2. Map the smoothing parameters to an unconstrained scale with logistic transforms
3. Optimise the concentrated log-likelihood with Nelder-Mead. If it has not converged, continue with L-BFGS (finite-difference gradients), falling back to the Nelder-Mead result if L-BFGS fails
4. If the fit has still not converged, repeat step 3 from a second starting point with alpha = 0.9, because the likelihood is often bimodal in alpha. Keep the fit with the lower objective

`start` replaces the initial values of step 1 with the parameters of an earlier fit, for example of the same model to a longer series. `has_converged` and the iteration count describe only the final stage of the winning run.

# Example
```julia
# Fit Holt-Winters model to monthly data
model = ETSModel(error="A", trend="A", season="A", s=12)
fitted = fit_baseline(monthly_data, model)

# Access fitted parameters
println("Level smoothing: ", fitted.par.θ.α)
println("Trend smoothing: ", fitted.par.θ.β)  
println("Seasonal smoothing: ", fitted.par.θ.γ)
```
"""
function fit_baseline(
        x::Vector{T1},
        model::ETSModel;
        setting::Union{ETSEstimationSetting, Nothing} = ETSEstimationSetting(),
        temporal_info::TemporalInfo = TemporalInfo(),
        start::Union{ETSParameter, Nothing} = nothing
    ) where {T1 <: Real}
    if isnothing(setting)
        setting = ETSEstimationSetting()
    end
    init = isnothing(start) ? getInitial(x, model) : start
    initVec = ETSpar2θ(init, model)
    if !setting.extended_search
        result = optimize(vars -> L_fun(x, model, vars), initVec)
        estVec = result.minimizer
    else
        objective = u -> L_fun(x, model, smoothing_from_unconstrained(u, model))
        u0 = smoothing_to_unconstrained(initVec, model)
        result = optimise_ets(objective, u0)
        if !Optim.converged(result)
            # The likelihood is often multimodal in the smoothing parameters: one local optimum
            # lies at small values and another near a random walk (alpha close to one). An
            # unconverged fit therefore also starts from a large alpha and keeps the better result.
            u1 = copy(u0)
            u1[1] = logit(0.9)
            result = best_of(result, optimise_ets(objective, u1))
        end
        estVec = smoothing_from_unconstrained(result.minimizer, model)
    end
    return ETSFitted(
        x, model, ETSθ2par(estVec, model), setting, temporal_info, optimisation_info(result, "ETS")
    )
end

# With the extended search, the parameters of the fit to the full series lie close to the optimum for a
# shorter series. The default search still starts from getInitial and gives the same results as before.
function refit_baseline(fitted::ETSFitted, x)
    start = fitted.estimation_setting.extended_search ? fitted.par : nothing
    return fit_baseline(x, fitted.model, setting = fitted.estimation_setting, start = start)
end

has_converged(fitted::ETSFitted) = isnothing(fitted.optimisation) ? nothing : fitted.optimisation.converged

"""
    point_forecast(fitted::ETSFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}}) -> Vector{Float64}

Generate point forecasts from fitted ETS model.

# Arguments
- `fitted::ETSFitted`: Fitted ETS model
- `horizon`: Forecast horizons

# Returns  
- `Vector{Float64}`: Point forecasts

# Method
Applies the deterministic part of the model recursively:
1. Start from final filtered state z_T
2. Apply f_function for each future period
3. Compute forecasts using w_function

# Example
```julia
# Generate 12-month forecasts
forecasts = point_forecast(fitted, 1:12)

# Single period forecast
next_value = point_forecast(fitted, 1)[1]
```
"""
function point_forecast(fitted::ETSFitted, horizon::Union{Vector{Int}, Int, UnitRange{Int}})
    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    all(horizon .> 0) || throw(ArgumentError("Horizons must be non-negative."))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    hMax = maximum(horizon)
    x = fitted.x
    T = length(x)
    fc_point = zeros(hMax)
    filtered = ETSFilter(x, fitted.model, fitted.par)

    σ = std(filtered.ϵ) + 0.01
    zCurr = filtered.z[end]
    for hh in 1:hMax
        fc_point[hh] = w_function(zCurr, fitted.model, fitted.par.θ)
        zCurr = f_function(zCurr, fitted.model, fitted.par.θ)
    end
    return fc_point[horizon]
end

# Prametric interval might be possible, see
# https://otexts.com/fpp3/ets-forecasting.html table 8.8

"""
    interval_forecast(fitted::ETSFitted, method::ModelTrajectoryInterval,
                     horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
                     levels::Vector{Float64} = [0.95];
                     alpha_precision::Int = 10,
                     include_median::Bool = true) -> (Vector{Float64}, Union{Vector{Float64}, Nothing}, Vector{ForecastInterval}, Union{Matrix{Float64}, Nothing})

Generate prediction intervals using ETS model trajectories.

# Arguments
- `fitted::ETSFitted`: Fitted ETS model
- `method::ModelTrajectoryInterval`: Trajectory simulation method
- `horizon`: Forecast horizons
- `levels`: Confidence levels
- Additional standard arguments

# Returns
- Standard interval forecast tuple

# Simulation Method
1. **Initialise**: Start from final filtered state z_T
2. **Generate innovations**: Sample ϵ_t ~ N(0, σ̂²) 
3. **State evolution**: z_{t+1} = f(z_t) + g(z_t)ϵ_{t+1}
4. **Observations**: x_{t+1} = w(z_t) + r(z_t)ϵ_{t+1}
5. **Repeat**: Generate multiple trajectory samples

# Error Structure
- **Additive errors**: x_t = w(z_{t-1}) + ϵ_t
- **Multiplicative errors**: x_t = w(z_{t-1})(1 + ϵ_t)

# Example
```julia
# Generate prediction intervals with 2000 trajectories
method = ModelTrajectoryInterval(n_trajectories=2000, return_trajectories=true)
fc_point, fc_median, fc_intervals, trajectories = interval_forecast(
    fitted, method, 1:24, [0.8, 0.95]
)

# Analyse trajectory distribution
seasonal_pattern = mean(trajectories[:, 1:12], dims=1)
```
"""
function interval_forecast(
        fitted::ETSFitted,
        method::ModelTrajectoryInterval,
        horizon::Union{Vector{Int}, Int, UnitRange{Int}} = [1],
        levels::Vector{Float64} = [0.95];
        alpha_precision::Int = 10,
        include_median::Bool = true
    )

    if horizon isa Int
        horizon = collect(1:horizon)
    end
    if horizon isa UnitRange{Int}
        horizon = collect(horizon)
    end

    # Validate input:
    all(0 .< levels .< 1.0) || throw(ArgumentError("Levels must be between 0 and 1"))
    all(horizon .> 0) || throw(ArgumentError("Horizons must be positive"))
    length(horizon) .> 0 || throw(ArgumentError("Valid forecast horizons must be provided."))

    # Setting seed if specified
    Random.seed!(method.seed)

    # Create point forecasts
    fc_point = point_forecast(fitted, horizon)

    # Structure level and probabilities
    if include_median
        levels = [levels; 0.0]
    end
    levels = sort(levels)
    alpha = sort(unique(round.(vcat((l -> 0.5 .+ [-1, 1] .* l * 0.5).(levels)...), digits = alpha_precision)))
    hMax = maximum(horizon)

    # Creating trajectories
    x = fitted.x
    T = length(x)
    filtered = ETSFilter(x, fitted.model, fitted.par)

    # Use filter to obtain new state values
    newFiltered = zeros(hMax)
    σ = std(filtered.ϵ) + 0.01
    zCurr = filtered.z[end]
    for hh in 1:hMax
        newFiltered[hh] = w_function(zCurr, fitted.model, fitted.par.θ)
        zCurr = f_function(zCurr, fitted.model, fitted.par.θ)
    end

    trajectories = zeros(method.n_trajectories, hMax)
    for i in 1:method.n_trajectories
        ϵ = zeros(hMax)
        zCurr = filtered.z[end]
        for hh in 1:hMax
            if method.positivity_correction == :truncate
                wCurr = w_function(zCurr, fitted.model, fitted.par.θ)
                rCurr = r_function(zCurr, fitted.model, fitted.par.θ)
                if rCurr > 0
                    ϵ[hh] = rand(truncated(Normal(0, σ), -wCurr / rCurr, Inf))
                else # Might not happen?
                    ϵ[hh] = rand(truncated(Normal(0, σ), -Inf, -wCurr / rCurr))
                end
            else
                ϵ[hh] = rand(Normal(0, σ))
            end
            trajectories[i, hh] = w_function(zCurr, fitted.model, fitted.par.θ) + r_function(zCurr, fitted.model, fitted.par.θ) * ϵ[hh]
            zCurr = f_function(zCurr, fitted.model, fitted.par.θ) + g_function(zCurr, fitted.model, fitted.par.θ) * ϵ[hh]

            if method.positivity_correction == :zero_floor
                trajectories[i, hh] = max(trajectories[i, hh], 0.0)
            end
        end
    end

    # Removing the seed
    Random.seed!(nothing)

    trajectories = trajectories[:, horizon]

    if method.positivity_correction == :post_clip
        trajectories[trajectories .< 0] .= 0.0
    end

    # Compute intervals from trajectories
    fc_intervals = Vector{ForecastInterval}(undef, length(horizon))
    fc_median = include_median ? zeros(length(horizon)) : nothing

    if include_median
        i_median = findfirst(alpha .== 0.5)
    end

    for i in 1:length(horizon)
        all_quantiles = [quantile(trajectories[:, i], q) for q in alpha]
        ls = reverse(all_quantiles[alpha .< 0.5])
        us = all_quantiles[alpha .> 0.5]
        fc_intervals[i] = ForecastInterval(ls, us, levels[levels .> 0])

        if include_median
            fc_median[i] = all_quantiles[i_median]
        end
    end

    if !method.return_trajectories
        trajectories = nothing
    end

    return fc_point, fc_median, fc_intervals, trajectories
end
