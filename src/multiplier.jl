"""
    rand_perturbation(val[, perturbation]; rng)
Return a modified value with `±perturbation` ratio of random perturbation.
For example, `rand_perturbation(1.0, 0.01)` returns a random number in `[0.99, 1.01]`
"""
rand_perturbation(val::T, perturbation::T=1e-3; rng=default_rng()) where {T} =
    val * (one(T) + perturbation * 2.0 * (rand(rng) - 0.5))

"""
    AbstractOptimizer{T}
Abstract optimizer class, subtypes has to implement `step!(param, optimizer, gradient)`
"""
abstract type AbstractOptimizer{T} end
reset!(optimizer::AbstractOptimizer{T}) where {T} = optimizer

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
function AdamOptimizer(step_size::T) where {T}
    return AdamOptimizer{T,DynamicDimensionArray2to4{T}}(; α=step_size)
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

# Keyword arguments
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator, by default `default_rng()`
"""
function step!(
    param::A, adam::AdamOptimizer{T}, grad::A; perturbation::T=1e-3, rng=default_rng()
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
        update_val = max(
            zero(T), param[idx...] + adam.α * corrected_m / (√corrected_v + adam.ϵ)
        )
        param[idx...] = rand_perturbation(update_val, perturbation; rng)
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

# Keyword arguments
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator, by default `default_rng()`
"""
function step!(
    param::A,
    opt::SimpleGradientOptimizer{T},
    grad::A;
    perturbation::T=1e-3,
    rng=default_rng(),
) where {T,A<:AbstractDynamicDimensionArray{T}}
    for (idx, grad_val) in grad
        update_val = max(zero(T), param[idx...] + opt.α * grad_val)
        param[idx...] = rand_perturbation(update_val, perturbation; rng)
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
function DecayGradientOptimizer(step_size::T) where {T}
    return DecayGradientOptimizer{T}(; α=step_size)
end
"""
    step!(param, opt, grad; perturbation, rng)
Update parameter in gradient ascend manner with step size decaying gradient optimizer
using a pre-computed gradient

# Arguments
- `param::A`: parameter to be updated
- `opt::DecayGradientOptimizer{T}`: gradient optimizer with decaying step size
- `grad::A`: gradient

# Keyword arguments
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator, by default `default_rng()`
"""
function step!(
    param::A,
    opt::DecayGradientOptimizer{T},
    grad::A;
    perturbation::T=1e-3,
    rng=default_rng(),
) where {T,A<:AbstractDynamicDimensionArray{T}}
    for (idx, grad_val) in grad
        update_val = max(zero(T), param[idx...] + opt.α * grad_val)
        param[idx...] = rand_perturbation(update_val, perturbation; rng)
    end
    opt.α = max(opt.min_step_size, opt.α * opt.decay_rate)
    return param
end

"""
    compute_gradient(multiplier, vertex_conflict, edge_conflict)
Compute gradient given the current conflict table of agents on the network.
Only vertex and edge conflicts (potential swapping conflicts) are considered.

# Arguments
- `multiplier::AbstractDynamicDimensionArray{C}`: Lagrange multiplier with type of cost `C`
- `vertex_conflict::VertexConflicts{T,V,A}`: conflict table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_conflict::EdgeConflicts{T,V,A}`: conflict table of edges with type of time `T`,
vertex `V`, agent `A`
"""
function compute_gradient(
    multiplier::AbstractDynamicDimensionArray{C},
    vertex_conflicts::VertexConflicts{T,V,A},
    edge_conflicts::EdgeConflicts{T,V,A},
    num_agents::Int,
) where {C,T,V,A}
    grad = empty(multiplier)
    # vertex conflicts
    for ((timestamp, vertex), agents) in vertex_conflicts
        # Multiple agents occupies, contains conflict
        violation = (length(agents) - one(C)) / (num_agents - one(C))
        for (ag, from_v) in agents
            grad[timestamp, ag, from_v, vertex] = violation
        end
    end

    # edge conflicts
    for ((timestamp, from_v, to_v), agents) in edge_conflicts
        # Multiple agents occupies, contains conflict
        violation = (length(agents) - one(C)) / (num_agents - one(C))
        for (ag, is_flip) in agents
            v1, v2 = is_flip ? (to_v, from_v) : (from_v, to_v)
            grad[timestamp, ag, v1, v2] += violation
        end
    end

    return grad
end

"""
"""
function polyak_step!(
    grad::AbstractDynamicDimensionArray{C}, upper_bound::Vector{C}, current_val::Vector{C}
) where {C}
    for (index, grad_val) in grad
        ag = index[2]
        if upper_bound[ag] < current_val[ag]
            continue
        end
        grad[index...] *= (upper_bound[ag] - current_val[ag])
    end
    return grad
end

"""
    update_multiplier!(
        multiplier, optimizer, vertex_conflicts, edge_conflicts; perturbation, rng
    )
Update the Lagrange multipleir based on the current conflicts table

# Arguments
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier with type of cost `C`
- `optimizer::Optimizer{C}`: optimizer for gradient ascend on the Lagrange multiplier
- `vertex_conflicts::VertexConflicts{T,V,A}`: conflicts table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_conflicts::EdgeConflicts{T,V,A}`: conflicts table of edges with type of time `T`,
vertex `V`, agent `A`

# Keyword arguments
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator, by default `default_rng()`
"""
function update_multiplier!(
    multiplier::AbstractDynamicDimensionArray{C},
    optimizer::AbstractOptimizer{C},
    vertex_conflicts::VertexConflicts{T,V,A},
    edge_conflicts::EdgeConflicts{T,V,A},
    num_agents::Int;
    perturbation::C=1e-3,
    rng=default_rng(),
    upper_bound::Vector{C}=Vector{C}(),
    current_val::Vector{C}=Vector{C}(),
) where {C,T,V,A}
    grad = compute_gradient(multiplier, vertex_conflicts, edge_conflicts, num_agents)
    # if !isempty(upper_bound) && !isempty(current_val)
    #     polyak_step!(grad, upper_bound, current_val)
    # end
    step!(multiplier, optimizer, grad; perturbation, rng)

    return grad
end
