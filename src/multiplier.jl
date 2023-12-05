"""
    rand_perturbation(val, err=1e-3; rng)
Return a modified value with ±err ratio of random perturbation.
For example, `rand_perturbation(1.0, 0.01)` returns a random number in `[0.99, 1.01]`
"""
rand_perturbation(val::T, perturbation::T=1e-3; rng=default_rng()) where {T} =
    val * (one(T) + perturbation * 2.0 * (rand(rng) - 0.5))

"""
    Optimizer{T}
Abstract optimizer class, subtypes has to implement `step!(param, optimizer, gradient)`
"""
abstract type Optimizer{T} end
reset!(optimizer::Optimizer{T}) where {T} = optimizer

"""
    AdamOptimizer{T} <: Optimizer{T}
An Adam optimizer with default `α=0.001`, `β1=0.9`, `β2=0.999`, and `ϵ=1e-8`
Supports initialization with `@kwdef`, by default utilizes `DynamicDimensionArray{T}`
"""
@kwdef mutable struct AdamOptimizer{T} <: Optimizer{T}
    α::T = 0.001  # step size
    β1::T = 0.9   # decay parameter [0, 1)
    β2::T = 0.999 # decay parameter [0, 1)

    β1_t::T = 1.0  # keep track of bias correcting term β1^t
    β2_t::T = 1.0  # keep track of bias correcting term β2^t

    m::DynamicDimensionArray{T} = DynamicDimensionArray()   # 1st order momentum
    v::DynamicDimensionArray{T} = DynamicDimensionArray()   # 2nd order momentum
    ϵ::T = 1e-8  # smooth factor
end

"""
    reset(optimizer)
Reset the Adam optimzier to `t=0` and clear all 1st and 2nd momentum
"""
function reset!(optimizer::AdamOptimizer{T}) where {T}
    optimizer.β1_t = one(T)
    optimizer.β2_t = one(T)
    optimizer.m = DynamicDimensionArray(zero(T))
    optimizer.v = DynamicDimensionArray(zero(T))

    return optimizer
end

"""
    step!(param, adam, grad)
Update parameter in gradient ascend manner with Adam optimizer using a pre-computed gradient

# Arguments
- `param::DynamicDimensionArray{T}`: parameter to be updated
- `adam::AdamOptimizer{T}`: adam optimizer
- `grad::DynamicDimensionArray{T}`: gradient
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator
"""
function step!(
    param::DynamicDimensionArray{T},
    adam::AdamOptimizer{T},
    grad::DynamicDimensionArray{T};
    perturbation::T=1e-3,
    rng=default_rng(),
) where {T}
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
    SimpleGradientOptimizer{T} <: Optimizer{T}
An basic gradient optimizer with default `α=0.001`, `β1=0.9`, `β2=0.999`, and `ϵ=1e-8`
Supports initialization with `@kwdef`, by default utilizes `DynamicDimensionArray{T}`
"""
@kwdef mutable struct SimpleGradientOptimizer{T} <: Optimizer{T}
    α::T = 0.001  # step size
end

"""
    step!(param, opt, grad)
Update parameter in gradient ascend manner with simple gradient optimizer using a pre-computed gradient

# Arguments
- `param::DynamicDimensionArray{T}`: parameter to be updated
- `opt::SimpleGradientOptimizer{T}`: simple gradient optimizer
- `grad::DynamicDimensionArray{T}`: gradient
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator
"""
function step!(
    param::DynamicDimensionArray{T},
    opt::SimpleGradientOptimizer{T},
    grad::DynamicDimensionArray{T};
    perturbation::T=1e-3,
    rng=default_rng(),
) where {T}
    for (idx, grad_val) in grad
        update_val = max(zero(T), param[idx...] + opt.α * grad_val)
        param[idx...] = rand_perturbation(update_val, perturbation; rng)
    end
    return param
end

"""
    compute_gradient(multiplier, vertex_conflict, edge_conflict)
Compute gradient given the current conflict table of agents on the network.
Only vertex and edge conflicts (potential swapping conflicts) are considered.

# Arguments
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier with type of cost `C`
- `vertex_conflict::VertexConflicts{T,V,A}`: conflict table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_conflict::EdgeConflicts{T,V,A}`: conflict table of edges with type of time `T`,
vertex `V`, agent `A`
"""
function compute_gradient(
    multiplier::DynamicDimensionArray{C},
    vertex_conflicts::VertexConflicts{T,V,A},
    edge_conflicts::EdgeConflicts{T,V,A},
) where {C,T,V,A}
    grad = DynamicDimensionArray(zero(C))
    # vertex conflicts
    for ((timestamp, vertex), agents) in vertex_conflicts
        # Multiple agents occupies, contains conflict
        violation = length(agents) - one(C)
        for (ag, from_v) in agents
            grad[timestamp, ag, from_v, vertex] = violation
            push!(visited_instances, (timestamp, ag, from_v, vertex))
        end
    end

    # edge conflicts
    for ((timestamp, from_v, to_v), agents) in edge_conflicts
        # Multiple agents occupies, contains conflict
        violation = length(agents) - one(C)
        for (ag, is_flip) in agents
            v1, v2 = is_flip ? (to_v, from_v) : (from_v, to_v)
            grad[timestamp, ag, v1, v2] += violation
            push!(visited_instances, (timestamp, ag, v1, v2))
        end
    end

    return grad
end

"""
    update_multiplier!(
        multiplier, optimizer, vertex_occupancy, edge_occupancy
    )
Update the Lagrange multipleir based on the current occupancy table

# Arguments
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier with type of cost `C`
- `optimizer::Optimizer{C}`: optimizer for gradient ascend on the Lagrange multiplier
- `vertex_occupancy::VertexConflicts{T,V,A}`: occupancy table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_occupancy::EdgeConflicts{T,V,A}`: occupancy table of edges with type of time `T`,
vertex `V`, agent `A`
- `perturbation::T`: ratio of perturbation, by default `1e-3`, (update value in ratio 1±0.001)
- `rng`: random generator
"""
function update_multiplier!(
    multiplier::DynamicDimensionArray{C},
    optimizer::Optimizer{C},
    vertex_occupancy::VertexConflicts{T,V,A},
    edge_occupancy::EdgeConflicts{T,V,A};
    perturbation::C=1e-3,
    rng=default_rng(),
) where {C,T,V,A}
    grad = compute_gradient(multiplier, vertex_occupancy, edge_occupancy)
    step!(multiplier, optimizer, grad; perturbation, rng)

    return grad
end
