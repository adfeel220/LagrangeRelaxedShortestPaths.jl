"""
    rand_perturbation(val[, perturbation]; rng)
Return a modified value with `±perturbation` ratio of random perturbation.
For example, `rand_perturbation(1.0, 0.01)` returns a random number in `[0.99, 1.01]`
"""
rand_perturbation(val::T, perturbation::T=1e-3; rng=default_rng()) where {T} =
    val * (one(T) + perturbation * 2.0 * (rand(rng) - 0.5))

"""
    compute_gradient(
        vertex_multiplier, edge_multiplier, vertex_occupancy, edge_occupancy, num_agents;
        perturbation, rng
    )
Compute gradient given the current occupancy table of agents on the network.
Return gradients with respect to vertex and edge conflicts.

# Arguments
- `vertex_multiplier::AbstractDynamicDimensionArray{C}`: Lagrange multiplier for vertex conflicts
with type of cost `C`, indexed by (time, vertex)
- `edge_multiplier::AbstractDynamicDimensionArray{C}`: Lagrange multiplier for edge conflicts
with type of cost `C`, indexed by (time, vertex)
- `vertex_occupancy::VertexConflicts{T,V,A}`: occupancy table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_occupancy::EdgeConflicts{T,V,A}`: occupancy table of edges with type of time `T`,
vertex `V`, agent `A`
- `num_agents::Int`: Number of agents in the network, for normalizing gradient size

# Keyword arguments
- `perturbation`: ratio of perturbation (± percentage), by default `0.0`
- `rng`: random generator, by default `default_rng()`
"""
function compute_gradient(
    vertex_multiplier::AbstractDynamicDimensionArray{C},
    edge_multiplier::AbstractDynamicDimensionArray{C},
    vertex_occupancy::VertexConflicts{T,V,A},
    edge_occupancy::EdgeConflicts{T,V,A},
    num_agents::Int;
    perturbation=0.0,
    rng=default_rng(),
) where {C,T,V,A}
    vertex_grad = empty(vertex_multiplier; default=zero(C))
    edge_grad = empty(edge_multiplier; default=zero(C))

    vertex_visited_instances = Set{Tuple{T,V}}()
    edge_visited_instances = Set{Tuple{T,V,V}}()

    # vertex conflicts
    for ((timestamp, vertex), agents) in vertex_occupancy

        # Only one agent occupies, maintain multiplier value
        if length(agents) == 1
            push!(vertex_visited_instances, (timestamp, vertex))
            continue
        end

        # Multiple agents occupies this vertex, update 
        violation = (length(agents) - one(C)) / (num_agents - one(C))
        vertex_grad[timestamp, vertex] = rand_perturbation(violation, perturbation; rng)
        push!(vertex_visited_instances, (timestamp, vertex))
    end

    # Decrease the multipliers that no agent touches
    for (idx, val) in vertex_multiplier
        (idx in vertex_visited_instances) && continue
        vertex_grad[idx] =
            -rand_perturbation(one(C) / (num_agents - one(C)), perturbation; rng)
    end

    # edge conflicts
    for ((timestamp, from_v, to_v), agents) in edge_occupancy

        # Only one agent occupies, maintain multiplier value
        if length(agents) == 1
            push!(edge_visited_instances, (timestamp, from_v, to_v))
            continue
        end

        violation = (length(agents) - one(C)) / (num_agents - one(C))
        edge_grad[timestamp, from_v, to_v] = rand_perturbation(violation, perturbation; rng)
        push!(edge_visited_instances, (timestamp, from_v, to_v))
    end

    # Decrease the multipliers that no agent touches
    for (idx, val) in edge_multiplier
        (idx in edge_visited_instances) && continue
        edge_grad[idx] =
            -rand_perturbation(one(C) / (num_agents - one(C)), perturbation; rng)
    end

    return vertex_grad, edge_grad
end

"""
    update_multiplier!(
        vertex_multiplier, edge_multiplier, vertex_optimizer, edge_optimizer,
        vertex_occupancy, edge_occupancy; perturbation, rng
    )
Update the Lagrange multipleir based on the current conflicts table

# Arguments
- `vertex_multiplier::AbstractDynamicDimensionArray{C}`: Lagrange multiplier for vertex conflicts
with type of cost `C`, indexed by (time, vertex)
- `edge_multiplier::AbstractDynamicDimensionArray{C}`: Lagrange multiplier for edge conflicts
with type of cost `C`, indexed by (time, vertex)
- `vertex_optimizer::Optimizer{C}`: optimizer for gradient ascend on the vertex multiplier
- `edge_optimizer::Optimizer{C}`: optimizer for gradient ascend on the edge multiplier
- `vertex_occupancy::VertexConflicts{T,V,A}`: occupancy table of vertices with type of time `T`,
vertex `V`, agent `A`
- `edge_occupancy::EdgeConflicts{T,V,A}`: occupancy table of edges with type of time `T`,
vertex `V`, agent `A`
- `num_agents::Int`: Number of agents in the network, for normalizing gradient size

# Keyword arguments
- `perturbation`: ratio of perturbation (± percentage), by default `0.0`
- `rng`: random generator, by default `default_rng()`
"""
function update_multiplier!(
    vertex_multiplier::AbstractDynamicDimensionArray{C},
    edge_multiplier::AbstractDynamicDimensionArray{C},
    vertex_optimizer::AbstractOptimizer{C},
    edge_optimizer::AbstractOptimizer{C},
    vertex_occupancy::VertexConflicts{T,V,A},
    edge_occupancy::EdgeConflicts{T,V,A},
    num_agents::Int;
    perturbation=0.0,
    rng=default_rng(),
) where {C,T,V,A}
    vertex_grad, edge_grad = compute_gradient(
        vertex_multiplier,
        edge_multiplier,
        vertex_occupancy,
        edge_occupancy,
        num_agents;
        perturbation,
        rng,
    )
    step!(vertex_multiplier, vertex_optimizer, vertex_grad)
    step!(edge_multiplier, edge_optimizer, edge_grad)

    return vertex_grad, edge_grad
end
