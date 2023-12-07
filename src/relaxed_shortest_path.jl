
"""
    update_cost!(origin_cost, cost, multiplier)
Compute the modified cost function based on the Lagrange multiplier

# Arguments
- `origin_cost::DynamicDimensionArray{C}`: original network edge cost, indexed by (time, agent, from-v, to-v)
- `cost::DynamicDimensionArray{C}`: modified cost, indexed by (time, agent, from-v, to-v)
- `multiplier::DynamicDimensionArray{C}`: Lagrange multiplier, indexed by (time, agent, from-v, to-v)
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
            zip(agent_path[begin:(end - 1)], agent_path[(begin + 1):end]);
            init=typemax(C),
        ) for (ag, agent_path) in enumerate(paths)
    ]
end

"""
    lagrange_relaxed_shortest_path(
        network, edge_costs, sources, targets, departure_times, priority;
        heuristic, astar_max_iter, lagrange_max_iter, hard_timeout,
        optimizer, perturbation, rng_seed,
        multi_threads, silent,
    )
Solve MAPF problem by iteratively resolve the shortest path problems with edge cost
modified based on Lagrange relaxation.

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `sources_vertices`: starting vertices of agents
- `targets_vertices`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling
- `priority`: sequence to do shortest path, is a permutation of agents. By default `OneTo(#agents)`

# Keyword arguments
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports `:lazy` always return 0;
`:dijkstra`: Dijkstra on the static graph from target vertex as estimation
- `astar_max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
- `lagrange_max_iter::Int`: maximum iteration number of Lagrange optimization step, by default `typemax(Int)`
- `hard_timeout::Float64`: maximum physical time duration (in seconds) allowed for the program, by default `Inf`
- `optimizer::Optimizer{C}`: optimizer for gradient ascend of Lagrange multiplier, by default is the
`AdamOptimizer` with step size equals to 1% of minimum edge cost
- `perturbation::T`: ratio of perturbation for lagrange multiplier update,
by default `1e-3` (update value in ratio 1±0.001)
- `rng_seed`: random seed for the program
- `multi_threads::Bool`: whether to apply multi threading, by default `true`
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
    lagrange_max_iter::Int=typemax(Int),
    hard_timeout::Float64=Inf,
    optimizer::Optimizer{C}=AdamOptimizer{C}(;
        α=1e-2 * minimum(x -> x.second, edge_costs; init=edge_costs.default)
    ),  # default step size as 1% of minimum cost
    perturbation::C=1e-3,
    rng_seed=nothing,
    multi_threads::Bool=true,
    silent::Bool=true,
) where {C}
    global_timer = time()

    multiplier = DynamicDimensionArray(zero(C))
    cost = deepcopy(edge_costs)  # modified cost
    reset!(optimizer)
    rng = isnothing(rng_seed) ? Xoshiro() : Xoshiro(rng_seed)

    start_time = -1.0
    iter = zero(Int)
    # main loop for lagrange relaxed problem
    while true
        (time() - global_timer) > hard_timeout && break
        (iter >= lagrange_max_iter) && break

        # Show status
        if !silent && (time() - start_time > 0.25)
            print("\rIter = $iter")
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

        vertex_conflicts = detect_vertex_conflict(paths)
        edge_conflicts = detect_edge_conflict(paths; swap=swap_conflict)

        if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
            if !silent
                print("\r")
                @info "Find solution after $iter iterations"
            end
            return paths, compute_scores(paths, edge_costs)
        end

        update_multiplier!(
            multiplier, optimizer, vertex_conflicts, edge_conflicts; perturbation, rng
        )

        iter += 1
    end

    # Guarantee a feasible solution by prioritized planning
    origin_pp_path, origin_pp_scores = prioritized_planning(
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

    modified_pp_path, _ = prioritized_planning(
        network,
        cost,
        source_vertices,
        target_vertices,
        departure_times;
        priority,
        swap_conflict,
        heuristic,
        max_iter=astar_max_iter,
    )
    modified_pp_scores = compute_scores(modified_pp_path, edge_costs)

    if sum(origin_pp_scores) <= sum(modified_pp_scores)
        if !silent
            print("\r")
            @info "Timeout after $iter iterations, return result from prioritized planning with origin cost"
        end
        return origin_pp_path, origin_pp_scores
    else
        if !silent
            print("\r")
            @info "Timeout after $iter iterations, return result from prioritized planning with modified cost"
        end
        return modified_pp_path, modified_pp_scores
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
