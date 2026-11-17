# Models

Every model is fitted with [`fit_baseline`](@ref) and forecast with [`forecast`](@ref). This page describes each model's formulation, its parameters and settings, and which interval methods it supports.

## `ConstantModel`

Naïve forecasts using the last observed value.

### Formulation

The model simply predicts the last observed value into the future.

### Architecture

No settings required

### Parameters

| Parameter | Description |
|-----------|-------------|
| `μ` | Last observation |

### Estimation settings

No settings required

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ❌ `ParametricInterval()`
- ❌ `ModelTrajectoryInterval()`

## `MarginalModel`

Forecasts based on empirical marginal distribution.

### Formulation

The model takes the most recent p observations to estimate the marginal mean.

### Architecture

- `p`: Number of observations

### Parameters

| Parameter | Description |
|-----------|-------------|
| `μ` | Marginal mean |

### Estimation settings

- `estimation_function`: Function that estimates the marginal mean from data (default: mean)

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ✅ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `KDEModel`

Kernel density estimation for non-parametric forecasting.

### Formulation

The density of the marginal distribution is estimated as

```math
\hat{f}(y) = \frac{1}{Th} \sum_{t = 1}^T K\left(\frac{y - y_t}{h}\right)
```

where K is a kernel and h is the bandwidth.

### Architecture

No settings required

### Parameters

| Parameter | Description |
|-----------|-------------|
| `x_seq` | sequence of nodes |
| `density` | density at nodes |

### Estimation settings

- `bandwidth_selection`: Bandwidth selection method (function) or fixed value
- `kernel`: Kernel distribution type (default: Normal())
- `npoints::Int`: Number of grid points for density evaluation (default: 2048)
- `boundary`: Boundary handling method or fixed boundaries
- `weights`: Observation weights method or fixed weights

For details, see [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl)

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ✅ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `LSDModel`

Last Similar Dates method for seasonal patterns.

### Formulation

The model estimates the means of periods for a seasonal time series. For estimation, the model takes not only observations of corresponding periods but also neighbouring observations (``\pm w``).

### Architecture

- `s`: Periodicity
- `w`: Window width

### Parameters

| Parameter | Description |
|-----------|-------------|
| `μ` | Vector of means |

### Estimation settings

- `estimation_function`: Function that estimates the marginal mean from data (default: mean)

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ❌ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `OLSModel`

Ordinary least squares with polynomial trends.

### Formulation

Linear regression model with polynomial time trend of order d fitted to p most recent obervations

```math
y_t = \beta_0 + \beta_1 t + ... + \beta_d t^d + \epsilon_t
```

### Architecture

- `p`: Number of observations
- `d`: Order of time trend polynomial

### Parameters

| Parameter | Description |
|-----------|-------------|
| `β` | Parameter vector |

### Estimation settings

No settings required

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ✅ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `IDSModel`

Increase-Decrease-Stable trend detection.

### Formulation

Fits an OLS model to the p most recent observations. If all observations go into the same direction (increase/decrease), it contains a linear trend and reduces to an intercept model otherwise (stable).

### Architecture

- `p`: Number of observations

### Parameters

| Parameter | Description |
|-----------|-------------|
| `a` | Intercept |
| `b` | Slope |

### Estimation settings

No settings required

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ✅ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `ARMAModel`

Autoregressive moving average processes.

### Formulation

The ARMA(p, q) model follows the specification:

```math
X_t - \mu_t = \epsilon_t + \sum_{i=1}^p \alpha_i (X_{t-i} - \mu_{t-i}) + \sum_{i=1}^q \beta_i \epsilon_{t-i}
```

where ``\mu_t = \mu(\theta, t)`` is a deterministic trend/seasonal function.

### Architecture

- **`p`, `q`**: Model orders (autoregressive and moving average)
- **`μ`**: Mean function `μ(θ, t)` where `θ` are parameters and `t` is time
- **`μDim`**: Number of parameters in the mean function

Users can specify `s` (periodicity) and `trend` (Boolean) instead of custom mean functions for convenience.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `α` | Autoregressive coefficients [α₁, α₂, ..., αₚ] |
| `β` | Moving average coefficients [β₁, β₂, ..., βₑ] |
| `μ` | Mean function parameters |
| `σ²` | Innovation variance (≥ 0) |

### Estimation settings

- `ensure_stability` - constrains for stationarity/invertibility

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ✅ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `INARCHModel`

Integer-valued ARCH for count time series.

### Formulation

INARCH(p) model with optional seasonality, either Poisson or negative binomial (conditional) distribution with mean

```math
\lambda_t = \mu_t(\beta_0 + \sum_{i = 1}^p (y_{t-i}/\mu_{t-i}))
```

where ``\mu_t`` is the seasonality component, ``\log(\mu_t)`` is a harmonic wave of order k.

### Architecture

- `p`: Autoregressive order (default: 1)
- `s`: Periodicity (default: 0, i.e. no seasonality)
- `k`: Order of harmonic waves (default: 1)
- `nb`: Negative Binomial distribution? (default: false)

### Parameters

| Parameter | Description |
|-----------|-------------|
| `β0` | Intercept |
| `α` | Autoregressive parameters |
| `ϕ` | Overdispersion parameter |
| `γ` | Seasonality parameters |

### Estimation settings

No settings required

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ❌ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `ETSModel`

Error-Trend-Season exponential smoothing (all 30 variants).

### Formulation

Exponential smoothing, decomposition into error, trend and seasonality. Errors can be additive or multiplicative. Seasonality can additionally be none. Trend component can be damped, if included in the model. All model variants can be summarised in state-space form

```math
x_t = w(z_{t-1}) + r(z_{t-1})\epsilon_t
```

```math
z_t = f(z_{t-1}) + g(z_{t-1})\epsilon_t
```

where ``z_t`` is the state vector.

### Architecture

- `error`: Error type
- `season`: Seasonality type (including periodicity s)
- `trend`: Trend type

For convenience, a constructor is implemented that takes

- `error`: `"A"` or `"M"` for additive or multiplicative errors respectively
- `season`: `"A"`, `"M"` or `"N"` (where `"N"` = no seasonality)
- `s`: Periodicity
- `trend`: `"A"`, `"Ad"`, `"M"`, `"Md"` or `"N"` (where `"Ad"` and `"Md"` are damped additive/multiplicative trends)

### Parameters

| Parameter | Description |
|-----------|-------------|
| `θ` | Smoothing coefficients |
| `z0` | Initial state |

### Estimation settings

No settings required

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ❌ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`

## `STLModel`

Seasonal-Trend decomposition using Loess.

### Formulation

For details, see [Paper](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf)

### Architecture

- `s`: Periodicity

### Parameters

| Parameter | Description |
|-----------|-------------|
| `S` | Seasonality terms |
| `T` | Trend terms |
| `R` | Remainder terms |

### Estimation settings

- `ni`: Number of inner loops
- `no`: Number of outer loops
- `ns`: Smoothing coefficient for seasonality term
- `nt`: Smoothing coefficient for trend term
- `nl`: Smoothing coefficient for Loess
- `s`: Periodicity

### Supported intervals

- ✅ `NoInterval()`
- ✅ `EmpiricalInterval()`
- ❌ `ParametricInterval()`
- ✅ `ModelTrajectoryInterval()`
