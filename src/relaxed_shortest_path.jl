
"""
    compute_relaxed_cost(origin_cost, multiplier)
Compute the modified cost function based on the Lagrange multiplier

# Arguments
- `origin_cost::DynamicDimensionArray{C}`: original network edge cost, indexed by (time, agent, from-v, to-v)
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier
"""
function update_cost!(
    origin_cost::DynamicDimensionArray{C},
    cost::DynamicDimensionArray{C},
    multiplier::DynamicDimensionArray{C},
) where {C}
    for (idx, val) in multiplier
        cost[idx...] = origin_cost[idx...] + val
    end
    return cost
end

"""
    compute_scores(paths, edge_costs)
Compute the cost of each path with a reference edge cost table

# Arguments
- `paths::Vector{TimedPath{T,V}}`: path as a sequence of time-expanded vertices of every agent.
`T` is type of time and `V` is type of vertex
- `edge_costs::DynamicDimensionArray{C}`: cost to traverse an edge indexed by (time, agent, from-v, to-v),
where the time of edge traversal is aligned with the arriving vertex
"""
function compute_scores(
    paths::Vector{TimedPath{T,V}}, edge_costs::DynamicDimensionArray{C}
) where {T,V,C}
    return [
        sum(
            edge_costs[t2, ag, v1, v2] for ((t1, v1), (t2, v2)) in
            zip(agent_path[begin:(end - 1)], agent_path[(begin + 1):end])
        ) for (ag, agent_path) in enumerate(paths)
    ]
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
- `sources_vertices`: starting vertices of agents
- `targets_vertices`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling

# Keyword arguments
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports `:lazy` always return 0;
`:dijkstra`: Dijkstra on the static graph from target vertex as estimation
- `astar_max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
- `multi_threads::Bool`: whether to apply multi threading, by default `true`
- `lagrange_max_iter::Int`: maximum iteration number of Lagrange optimization step, by default `typemax(Int)`
- `optimizer::Optimizer{C}`: optimizer for gradient ascend of Lagrange multiplier, by default is the
`AdamOptimizer` with step size equals to 1% of minimum edge cost
- `perturbation::C`: 
- `silent::Bool`: disable printing status on the console, by default `true`
"""
function lagrange_relaxed_shortest_path(
    network::AbstractGraph,
    edge_costs::DynamicDimensionArray{C},
    source_vertices,
    target_vertices,
    departure_times=zeros(Int, length(source_vertices)),
    priority=Base.OneTo(length(source_vertices));
    swap_conflict::Bool=false,
    heuristic::Union{Symbol,Function}=:dijkstra,
    astar_max_iter::Int=typemax(Int),
    multi_threads::Bool=true,
    lagrange_max_iter::Int=typemax(Int),
    optimizer::Optimizer{C}=AdamOptimizer{C}(; α=1e-2 * minimum(x -> x.second, edge_costs; init=edge_costs.default)),  # default step size as 1% of minimum cost
    perturbation::C=1e-3,
    rng_seed=nothing,
    silent::Bool=true,
) where {C}
    multiplier = DynamicDimensionArray(zero(C))
    cost = deepcopy(edge_costs) #  modified cost
    reset!(optimizer)
    rng = isnothing(rng_seed) ? Xoshiro() : Xoshiro(rng_seed)

    start_time = -1.0
    # main loop for lagrange relaxed problem
    for iter in zero(Int):lagrange_max_iter
        # Show status
        if !silent && (time() - start_time > 0.2)
            print("Iter = $iter \r")
            start_time = time()
        end

        # Update the modified cost from original cost
        update_cost!(edge_costs, cost, multiplier)

        paths, _ = shortest_paths(
            network,
            cost,
            source_vertices,
            target_vertices,
            departure_times;
            heuristic,
            max_iter=astar_max_iter,
            multi_threads,
        )

        vertex_occupancy = detect_vertex_occupancy(paths)
        edge_occupancy = detect_edge_occupancy(paths)

        vertex_conflicts = detect_conflict(vertex_occupancy)
        edge_conflicts = detect_conflict(edge_occupancy)

        if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
            !silent && @info "Find solution after $iter iterations"
            return paths, compute_scores(paths, edge_costs)
        end

        update_multiplier!(
            multiplier, optimizer, vertex_occupancy, edge_occupancy; perturbation, rng
        )
    end

    if !silent
        @info "Timeout after $lagrange_max_iter iterations, return result from prioritized planning"
    end
    # Guarantee a feasible solution by prioritized planning
    return prioritized_planning(
        network,
        edge_costs,
        source_vertices,
        target_vertices,
        departure_times;
        priority,
        swap_conflict,
        heuristic,
        max_iter=astar_max_iter,
    )
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
