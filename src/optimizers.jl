
"""
    AbstractOptimizer{T}
Abstract optimizer class, subtypes has to implement `step!(param, optimizer, gradient)`
"""
abstract type AbstractOptimizer{T} end
reset!(optimizer::AbstractOptimizer{T}) where {T} = optimizer
step_size(optimizer::AbstractOptimizer) = optimizer.α

"""
    AdamOptimizer{T} <: AbstractOptimizer{T}
An Adam optimizer with default `α=0.001`, `β1=0.9`, `β2=0.999`, and `ϵ=1e-8`
Supports initialization with `@kwdef`, by default utilizes `DynamicDimensionArray{T}`
"""
@kwdef mutable struct AdamOptimizer{T,A<:AbstractDynamicDimensionArray} <:
                      AbstractOptimizer{T}
    α::T = 0.001  # step size
    β1::T = 0.9   # decay parameter [0, 1)
    β2::T = 0.999 # decay parameter [0, 1)

    β1_t::T = 1.0  # keep track of bias correcting term β1^t
    β2_t::T = 1.0  # keep track of bias correcting term β2^t

    m::A = DynamicDimensionArray2to4()   # 1st order momentum
    v::A = DynamicDimensionArray2to4()   # 2nd order momentum
    ϵ::T = 1e-8  # smooth factor
end
function AdamOptimizer(step_size::T; kwargs...) where {T}
    return AdamOptimizer{T,DynamicDimensionArray2to4{T}}(; α=step_size, kwargs...)
end

"""
    reset(optimizer)
Reset the Adam optimzier to `t=0` and clear all 1st and 2nd momentum
"""
function reset!(optimizer::AdamOptimizer{T}) where {T}
    optimizer.β1_t = one(T)
    optimizer.β2_t = one(T)
    optimizer.m = empty(optimizer.m)
    optimizer.v = empty(optimizer.v)

    return optimizer
end

"""
    step!(param, adam, grad; perturbation, rng)
Update parameter in gradient ascend manner with Adam optimizer using a pre-computed gradient

# Arguments
- `param::A`: parameter to be updated
- `adam::AdamOptimizer{T}`: adam optimizer
- `grad::A`: gradient
"""
function step!(
    param::A, adam::AdamOptimizer{T}, grad::A
) where {T,A<:AbstractDynamicDimensionArray{T}}
    adam.β1_t *= adam.β1
    adam.β2_t *= adam.β2

    # Iterate a DynamicDimensionArray only returns non-default values
    for (idx, grad_val) in grad
        adam.m[idx...] = adam.β1 * adam.m[idx...] + (1 - adam.β1) * grad_val
        adam.v[idx...] = adam.β2 * adam.v[idx...] + (1 - adam.β2) * grad_val^2

        corrected_m = adam.m[idx...] / (1 - adam.β1_t)
        corrected_v = adam.v[idx...] / (1 - adam.β2_t)

        # plus sign because of gradient ascend
        update_val = param[idx...] + adam.α * corrected_m / (√corrected_v + adam.ϵ)

        if update_val <= zero(T)
            param[idx...] = zero(T)
        else
            param[idx...] = update_val
        end
    end

    return param
end

"""
    SimpleGradientOptimizer{T} <: AbstractOptimizer{T}
An basic gradient optimizer with default step size `α=0.001`.
Supports initialization with `@kwdef`
"""
@kwdef mutable struct SimpleGradientOptimizer{T} <: AbstractOptimizer{T}
    α::T = 0.001  # step size
end

"""
    step!(param, opt, grad; perturbation, rng)
Update parameter in gradient ascend manner with simple gradient optimizer using a pre-computed gradient

# Arguments
- `param::A`: parameter to be updated
- `opt::SimpleGradientOptimizer{T}`: simple gradient optimizer
- `grad::A`: gradient
"""
function step!(
    param::A, opt::SimpleGradientOptimizer{T}, grad::A
) where {T,A<:AbstractDynamicDimensionArray{T}}
    for (idx, grad_val) in grad
        update_val = param[idx...] + opt.α * grad_val

        if update_val <= zero(T)
            param[idx...] = zero(T)
        else
            param[idx...] = update_val
        end
    end
    return param
end

"""
    DecayGradientOptimizer{T} <: AbstractOptimizer{T}
An basic gradient optimizer with default starting step size `α=1.0`
Supports initialization with `@kwdef`
"""
@kwdef mutable struct DecayGradientOptimizer{T} <: AbstractOptimizer{T}
    α::T = 1.0  # step size
    decay_rate::T = 0.999  # decay rate
    min_step_size::T = 1e-4  # minimum step size
end
function DecayGradientOptimizer(step_size::T; kwargs...) where {T}
    return DecayGradientOptimizer{T}(; α=step_size, kwargs...)
end
"""
    step!(param, opt, grad; perturbation, rng)
Update parameter in gradient ascend manner with step size decaying gradient optimizer
using a pre-computed gradient

# Arguments
- `param::A`: parameter to be updated
- `opt::DecayGradientOptimizer{T}`: gradient optimizer with decaying step size
- `grad::A`: gradient
"""
function step!(
    param::A, opt::DecayGradientOptimizer{T}, grad::A
) where {T,A<:AbstractDynamicDimensionArray{T}}
    for (idx, grad_val) in grad
        update_val = max(zero(T), param[idx...] + opt.α * grad_val)
        if update_val <= zero(T)
            param[idx...] = zero(T)
        else
            param[idx...] = update_val
        end
    end
    opt.α = max(opt.min_step_size, opt.α * opt.decay_rate)
    return param
end
