
"""
    compute_relaxed_cost(origin_cost, multiplier)
Compute the modified cost function based on the Lagrange multiplier

# Arguments
- `origin_cost::DynamicDimensionArray{C}`: original network edge cost, indexed by (time, agent, from-v, to-v)
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier
"""
function compute_relaxed_cost(
    origin_cost::DynamicDimensionArray{C}, multiplier::DynamicDimensionArray{C}
) where {C}
    @assert multiplier.default ≈ zero(multiplier.default) "The default value of multiplier has to be zero but get $(multiplier.default) as default value"
    cost = deepcopy(origin_cost)
    for (idx, val) in multiplier
        cost[idx...] += val
    end
    return cost
end

function update_multiplier!(
    multiplier::DynamicDimensionArray{C},
    conflicts::VertexConflicts{T,V,A},
    current_score::C,
    upper_bound::C;
    step_size=1e-2,
) where {C,T,V,A}
    coeff = step_size * (upper_bound - current_score) / length(conflicts)

    for ((timestamp, vertex), agents) in conflicts
        update = coeff * (length(agents) - 1)
        for (ag, from_v) in agents
            multiplier[timestamp, ag, from_v, vertex] += update
        end
    end
end

function update_multiplier!(
    multiplier::DynamicDimensionArray{C},
    conflicts::EdgeConflicts{T,V,A},
    current_score::C,
    upper_bound::C;
    step_size=1e-2,
) where {C,T,V,A}
    coeff = step_size * (upper_bound - current_score) / length(conflicts)
    for ((timestamp, from_v, to_v), agents) in conflicts, (ag, is_flip) in agents
        v1, v2 = is_flip ? (to_v, from_v) : (from_v, to_v)
        multiplier[timestamp, ag, v1, v2] += coeff * (length(agents) - 1)
    end
end

"""
    lagrange_relaxed_shortest_path(
        network, edge_costs, sources, targets, departure_times;
        heuristic, max_iter, multi_threads, step_size, lagrange_multiplier
    )
Solve MAPF problem by iteratively resolve the shortest path problems with edge cost
modified based on Lagrange relaxation.

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `sources::Vector{V}`: starting vertices of agents
- `targets::Vector{V}`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling

# Keyword arguments
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports
    - `:lazy`: always return 0
    - `:dijkstra`: dijkstra on the static graph from target vertex as estimation
- `astar_max_iter::UInt`: maximum iteration of individual A*, by default `typemax(UInt)`
- `multi_threads::Bool`: whether to apply multi threading, by default `true`
- `lagrange_max_iter::UInt`: maximum iteration number of Lagrange optimization step
- `step_size`: base step size of lagrange optimization step
"""
function lagrange_relaxed_shortest_path(
    network::AbstractGraph,
    source_vertices,
    target_vertices,
    edge_cost::DynamicDimensionArray{C},
    departure_times=zeros(Int, length(source_vertices));
    swap_conflict::Bool=false,
    heuristic::Union{Symbol,Function}=:dijkstra,
    astar_max_iter::UInt=typemax(UInt),
    multi_threads::Bool=true,
    lagrange_max_iter::UInt=typemax(UInt),
    step_size=1e-2,
) where {C}
    iter = 0
    start_time = -1.0

    multiplier = DynamicDimensionArray(zero(C))

    while true
        if time() - start_time > 0.2
            print("Iter = $iter \r")
            start_time = time()
        end
        iter += 1

        cost = compute_relaxed_cost(edge_cost, multiplier)

        paths, scores = shortest_paths(
            network,
            cost,
            source_vertices,
            target_vertices,
            departure_times;
            heuristic,
            max_iter=astar_max_iter,
            multi_threads,
        )

        if iter >= lagrange_max_iter
            @info "Timeout after $iter iterations"
            return paths
        end

        vertex_conflicts = detect_vertex_conflict(paths)
        edge_conflicts = detect_edge_conflict(paths; swap=swap_conflict)

        if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
            @info "Iter $iter"
            return paths
        end

        total_score = sum(scores)

        if !is_conflict_free(vertex_conflicts)
            update_multiplier!(
                multiplier,
                vertex_conflicts,
                total_score,
                length(source_vertices) * total_score;
                step_size=step_size,
            )
        end
        if !is_conflict_free(edge_conflicts)
            update_multiplier!(
                multiplier,
                edge_conflicts,
                total_score,
                length(source_vertices) * total_score;
                step_size=step_size,
            )
        end
    end
end

"""
    vertex_path_split(vertex_only_result)
Split the result from a sequence of time-expanded vertices into vertex and edge entering time
"""
function vertex_path_split(vertex_only_result::Vector{Vector{Tuple{T,V}}}) where {T,V}
    vertex_paths = [
        [
            (t, v) for (idx, (t, v)) in enumerate(agent_path) if
            idx == 1 || v != agent_path[idx - 1][2]
        ] for agent_path in vertex_only_result
    ]
    edge_paths = [
        [
            (t1, (a1, a2)) for ((t1, a1), (t2, a2)) in
            zip(agent_path[begin:(end - 1)], agent_path[(begin + 1):end]) if a1 != a2
        ] for agent_path in vertex_only_result
    ]
    return vertex_paths, edge_paths
end