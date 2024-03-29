
"""
    update_cost!(
        cost, origin_cost, vertex_multiplier, edge_multiplier, network, num_agents;
        perturbation, rng
    )
Compute the modified cost based on the Lagrange multiplier

# Arguments
- `cost::AbstractDynamicDimensionArray`: modified cost, indexed by (agent, time, from-v, to-v)
- `origin_cost::AbstractDynamicDimensionArray`: original network edge cost, indexed by (agent, time, from-v, to-v)
- `vertex_multiplier::AbstractDynamicDimensionArray`: Lagrange multiplier about vertex conflicts, indexed by (time, to-v)
- `edge_multiplier::AbstractDynamicDimensionArray`: Lagrange multiplier about edge conflicts, indexed by (time, from-v, to-v)
- `network::AbstractGraph`: network where the agent travels on, used to retrieve `inneighbors`
- `num_agents::Int`: number of agents

# Keyword arguments
- `perturbation`: ratio of perturbation for each agent's cost update, by default `0.0`
- `rng`: random generator, by default `default_rng()`
"""
function update_cost!(
    cost::CA,
    origin_cost::CA,
    vertex_multiplier::CA,
    edge_multiplier::CA,
    network::AbstractGraph,
    num_agents::Int;
    perturbation=0.0,
    rng=default_rng(),
) where {CA<:AbstractDynamicDimensionArray}
    # Store all the values need to be updated
    update_vals = Dict{NTuple{3,Int},Float64}()

    # multiplier by vertex conflicts
    zero_vertex_multipliers = Set{NTuple{2,Int}}()
    for (idx, mul_val) in vertex_multiplier
        t, v2 = idx
        for v1 in inneighbors(network, v2)
            if !haskey(update_vals, (t, v1, v2))
                update_vals[t, v1, v2] = mul_val
            else
                update_vals[t, v1, v2] += mul_val
            end
        end

        if mul_val ≈ 0.0
            push!(zero_vertex_multipliers, idx)
        end
    end
    for idx in zero_vertex_multipliers
        delete!(vertex_multiplier, idx)
    end

    # multiplier by edge conflicts
    zero_edge_multipliers = Set{NTuple{3,Int}}()
    for (idx, mul_val) in edge_multiplier
        t, v1, v2 = idx
        if !haskey(update_vals, (t, v1, v2))
            update_vals[t, v1, v2] = mul_val
        else
            update_vals[t, v1, v2] += mul_val
        end

        if mul_val ≈ 0.0
            push!(zero_edge_multipliers, idx)
        end
    end
    for idx in zero_edge_multipliers
        delete!(edge_multiplier, idx)
    end

    # update cost value
    for (idx, val) in update_vals
        # delete if value is empty, reduce to original cost
        # without using extra memory
        if val ≈ 0.0
            # If no perturbation, degenerate into 3 index
            if perturbation ≈ 0.0
                delete!(cost, idx)
            else
                # Has perturbation, delete each agent differently
                for a in 1:num_agents
                    t, v1, v2 = idx
                    delete!(cost, (a, t, v1, v2))
                end
            end
        else
            # If no perturbation, degenerate into 3 index
            if perturbation ≈ 0.0
                cost[idx] = origin_cost[idx] + val
            else
                # Has perturbation, distribute to each agent differently
                for a in 1:num_agents
                    t, v1, v2 = idx
                    cost[a, t, v1, v2] =
                        origin_cost[idx] + rand_perturbation(val, perturbation; rng)
                end
            end
        end
    end

    return cost
end

"""
    compute_scores(paths, edge_costs)
Compute the cost of each path with a reference edge cost table

# Arguments
- `paths::Vector{TimedPath{T,V}}`: path as a sequence of time-expanded vertices of every agent.
`T` is type of time and `V` is type of vertex
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost to traverse an edge indexed by (agent, time, from-v, to-v),
where the time of edge traversal is aligned with the arriving vertex
"""
function compute_scores(
    paths::Vector{TimedPath{T,V}}, edge_costs::AbstractDynamicDimensionArray{C}
) where {T,V,C}
    return [
        if isempty(agent_path)
            typemax(C)
        elseif length(agent_path) == 1  # starting point == end point
            zero(C)
        else
            sum(
                edge_costs[ag, t2, v1, v2] for ((t1, v1), (t2, v2)) in
                zip(agent_path[begin:(end - 1)], agent_path[(begin + 1):end]);
            )
        end for (ag, agent_path) in enumerate(paths)
    ]
end

"""
    compute_relaxed_score(paths, costs, multiplier, vertex_occupancy, edge_occupancy)
Calculate the score of lagrange relaxed problem cx + μ(Dx-q)
"""
function compute_relaxed_score(
    paths::Vector{TimedPath{T,V}},
    costs::AbstractDynamicDimensionArray{C},
    vertex_multiplier::AbstractDynamicDimensionArray{C},
    edge_multiplier::AbstractDynamicDimensionArray{C},
    vertex_occupancy::VertexConflicts{T,V,A}=detect_vertex_occupancy(paths),
    edge_occupancy::EdgeConflicts{T,V,A}=detect_edge_occupancy(paths),
) where {T,V,A,C}
    # Path score with original cost
    score = sum(compute_scores(paths, costs))

    # add penalty terms related to vertex multiplier
    for (idx, val) in vertex_multiplier
        t, v2 = idx
        if haskey(vertex_occupancy, (t, v2))
            violation = length(vertex_occupancy[(t, v2)]) - one(C)
        else
            violation = -one(C)
        end
        score += val * violation
    end

    # add penalty terms related to edge multiplier
    for (idx, val) in edge_multiplier
        t, v1, v2 = idx
        if haskey(edge_occupancy, (t, v1, v2))
            violation = length(edge_occupancy[(t, v1, v2)]) - one(C)
        else
            violation = -one(C)
        end
        score += val * violation
    end

    return score
end

"""
    lagrange_relaxed_shortest_path(
        network, edge_costs, sources_vertices, target_vertices, departure_times, priority;
        swap_conflict, heuristic, multi_threads, pp_frequency, random_shuffle_priority,
        optimizer, perturbation, rng_seed,
        astar_max_iter, lagrange_max_iter, hard_timeout, optimality_threshold, max_exploration_time,
        refresh_rate, silent,
    )
Solve MAPF problem by iteratively resolve the shortest path problems with edge cost
modified based on Lagrange relaxation.

# Arguments
- `network::AbstractGraph`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray`: cost indexed by (agent, time, from-v, to-v)
- `sources_vertices`: starting vertices of agents
- `targets_vertices`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling, by default all zero
- `priority`: sequence to do shortest path, is a permutation of agents. By default `OneTo(#agents)`

# Keyword arguments
## Low-level Algorithm Parameters
- `swap_conflict::Bool`: whether to detect swapping conflicts, by default `false`
- `heuristic::Union{Symbol,Function}`: heuristic function for A* algorithm, it returns an estimation given a vertex `v`.
    Can also be some predefined methods passed as a `Symbol`, supports the following symbols with default `:dijkstra`.

    `:lazy`: always return 0;

    `:dijkstra`: Dijkstra on the static graph from target vertex as estimation;

    `:euclidean`: Euclidean distance between two vertices, only works with grid edge costs

- `multi_threads::Bool`: whether to apply multi threading for parallel A*, by default `true`
- `pp_frequency::Union{Integer,AbstractFloat}`: how often should the program run prioritized planning to estimate upper bound, by default `1`.
    Can be a `Integer`, then PP is run every `pp_frequency` iterations;

    Can be a `Float`, then PP is run approximately every `pp_frequency` seconds.

- `random_shuffle_priority::Bool`: whether to try a random permutation of priority every time trying PP, by default `false`

## Optimization Parameters
- `optimizer::AbstractOptimizer`: optimizer for gradient ascend of Lagrange multiplier, by default is `AdamOptimizer()`
- `gradient_perturbation`: ratio of random perturbation for Lagrange multiplier update, by default `0`
- `cost_perturbation`: ratio of random perturbation for modified cost across each agent, by default `0`
- `rng_seed`: random seed for all randomness of program, generate random numbers by `Xoshiro` generator, by default `nothing`

## Termination Criteria / Computation Budget
- `astar_max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
- `lagrange_max_iter::Int`: maximum iteration number of Lagrange optimization step, by default `typemax(Int)`
- `hard_timeout::Float64`: maximum physical time duration (in seconds) allowed for the program, by default `Inf`
- `absolute_optimality_threshold`: absolute suboptimality tolerance, program terminates when (UB - LB) < threshold, by default `0`
- `relative_optimality_threshold`: relative suboptimality tolerance, program terminates when (UB - LB) / LB ≤ threshold, by default `0`
- `max_exploration_time::Union{Integer,AbstractFloat}`: maximum budget for exploration without any improvement in optimality, by default `Inf`
    Can be a `Integer`, then the program terminates if no improvement of optimality in `max_exploration_time` iterations;

    Can be a `Float`, then the program terminates if no improvement of optimality in `max_exploration_time` seconds.

## Status Display
- `refresh_rate::Union{Integer,AbstractFloat}`: how often should we print status, by default `0.25` (seconds).
    Can be a `Integer`, then the status is printed every `refresh_rate` iterations;

    Can be a `Float`, then the status is printed every `refresh_rate` seconds.

- `silent::Bool`: disable printing status on the console, by default `true`
- `record_vars`: returns the value of local variables stored in this collection of symbols upon returning.
For example, `[:lower_bound]` records the lower bound at the point of return, by default an empty symbol vector.
- `track_vars`: tracks the value of local variables stored in this collection of symbols at each iteration.
For example, `[:suboptimality]` records the suboptimality at each iteration, by default an empty symbol vector.
- `track_time_by_suboptimality`: track the number of iteration and time used to reach a certain level of optimality
"""
function lagrange_relaxed_shortest_path(
    network::AbstractGraph,
    edge_costs::A,
    source_vertices,
    target_vertices,
    departure_times=zeros(Int, length(source_vertices)),
    priority=Base.OneTo(length(source_vertices));
    # Low-level Algorithm Parameters
    swap_conflict::Bool=false,
    heuristic::Union{Symbol,Function}=:dijkstra,
    multi_threads::Bool=true,
    pp_frequency::Union{Integer,AbstractFloat}=1,
    random_shuffle_priority::Bool=false,
    # Optimization Parameters
    optimizer::AbstractOptimizer{C}=AdamOptimizer(),
    gradient_perturbation::C=zero(C),
    cost_perturbation::C=zero(C),
    rng_seed=nothing,
    gradient_bias::C=1.0,
    # Termination Criteria / Computation Budget
    astar_max_iter::Int=typemax(Int),
    lagrange_max_iter::Int=typemax(Int),
    hard_timeout::Float64=Inf,
    absolute_optimality_threshold::C=zero(C),
    relative_optimality_threshold::C=zero(C),
    max_exploration_time::Union{Integer,AbstractFloat}=Inf,
    # Status Display
    refresh_rate::Union{Integer,AbstractFloat}=0.25,
    silent::Bool=true,
    record_vars=Symbol[],
    track_vars=Symbol[],
    track_time_by_suboptimality=Float64[],
) where {C,A<:AbstractDynamicDimensionArray{C}}
    @assert length(source_vertices) ==
        length(target_vertices) ==
        length(departure_times) ==
        length(priority) "Number of agents should agree between source, target, departure time, and priority"

    global_timer = time()

    ##############
    # Initialize #
    ##############
    if !silent
        @info "Initializing program"
    end

    vertex_multiplier::A = empty(edge_costs; default=zero(C))
    edge_multiplier::A = empty(edge_costs; default=zero(C))

    reset!(optimizer)
    vertex_optimizer = optimizer
    edge_optimizer = deepcopy(optimizer)

    cost::A = deepcopy(edge_costs)  # modified cost

    rng = isnothing(rng_seed) ? Xoshiro() : Xoshiro(rng_seed)

    # Tracking program status
    exploration_status = init_status(max_exploration_time)
    pp_run_status = init_status(pp_frequency)
    last_status_printed_time = time()  # time where last time status is printed
    iter = zero(Int)  # number of iteration for the entire program
    previous_printing_length = 0  # for clean printing status

    # Information gathering
    tracker = Dict([(var, Vector{Any}()) for var in track_vars])
    # Track time and iteration to reach a certain level of optimality
    suboptimality_timing = [(0, Inf) for subopt in track_time_by_suboptimality]

    # Prepare heuristic for every agent
    !silent && @info "Resolving A* heuristics"

    heuristics = [
        resolve_heuristic(heuristic, network, target_v, edge_costs) for
        target_v in target_vertices
    ]

    # Simple lower bound as parallel shortest path with the original cost
    !silent && @info "Precompute parallel A* for lower bound estimation"

    paths, scores = shortest_paths(
        network,
        edge_costs,
        source_vertices,
        target_vertices,
        departure_times;
        heuristics=heuristics,
        max_iter=astar_max_iter,
        multi_threads,
    )
    initial_astar_score = sum(scores)

    !silent && @info "Obtain initial lower bound = $(sum(scores))"

    vertex_occupancy = detect_vertex_occupancy(paths)
    edge_occupancy = detect_edge_occupancy(paths; swap=swap_conflict)
    vertex_conflicts = detect_conflict(vertex_occupancy)
    edge_conflicts = detect_conflict(edge_occupancy)

    # Early termination if precomputation is already the best
    if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
        if !silent
            println("")
            @info "Optimal solution found without any conflict, " *
                "return total score of $(sum(scores)) with parallel A*"
        end

        # Able to return necessary information even if it finds answer early
        lower_bound = initial_astar_score
        upper_bound = lower_bound
        initial_pp_score = lower_bound
        relaxed_score = lower_bound
        num_conflicts = 0
        suboptimality = 0.0

        # Record return information
        return_info = Dict(k => v for (k, v) in Base.@locals() if k in record_vars)
        time_used = time() - global_timer
        for (idx, subopt) in enumerate(track_time_by_suboptimality)
            suboptimality_timing[idx] = (0, time_used)
        end

        return paths, scores, return_info, tracker, suboptimality_timing
    end

    # Simple upper bound as plain prioritized planning score
    !silent && @info "Precompute prioritized planning for upper bound estimation"

    best_pp_path, best_pp_scores = prioritized_planning(
        network,
        edge_costs,
        source_vertices,
        target_vertices,
        departure_times;
        priority,
        swap_conflict,
        heuristic,
        max_iter=astar_max_iter,
        timeout=hard_timeout,
    )

    initial_pp_score = sum(best_pp_scores)
    lower_bound = initial_astar_score
    upper_bound = initial_pp_score
    relaxed_score = lower_bound
    num_conflicts = n_conflicts(vertex_conflicts) + n_conflicts(edge_conflicts)
    suboptimality = (upper_bound - lower_bound) / lower_bound

    # Update tracker
    time_used = time() - global_timer
    for (idx, subopt) in enumerate(track_time_by_suboptimality)
        if time_used < suboptimality_timing[idx][2] && suboptimality <= subopt
            suboptimality_timing[idx] = (iter, time_used)
        end
    end

    !silent && @info "Obtain initial upper bound = $upper_bound"
    !silent && @info "Start searching with $num_conflicts conflicts"

    #############
    # Main Loop #
    #############
    # Catch error to return answer even being forced to quit
    try
        while true
            # Early termination if iteration or timing threshold is met
            (time() - global_timer) > hard_timeout && break
            (iter >= lagrange_max_iter) && break

            # Show status
            if !silent &&
                is_time_for_next_event(refresh_rate, iter, last_status_printed_time)
                print("\r" * " "^previous_printing_length * "\r")
                print_info =
                    "Iter = $iter ($(time_with_unit(time() - global_timer; digits=2))): " *
                    "≤$(round(suboptimality*1e2; digits=3))% suboptimal " *
                    "in $(round(lower_bound; digits=5))($relaxed_score) - $(round(upper_bound; digits=3)) " *
                    "with $num_conflicts conflicts"
                print(print_info)
                previous_printing_length = length(print_info)
                last_status_printed_time = time()
            end

            # Test all termination criteria
            is_terminate, terminate_message = ready_to_terminate(
                num_conflicts,
                upper_bound,
                lower_bound,
                absolute_optimality_threshold,
                exploration_status;
                relative_optimality_threshold,
                max_exploration_time,
            )

            if is_terminate
                astar_scores = compute_scores(paths, edge_costs)
                a_star_total_score = sum(astar_scores)

                if !silent
                    print("\r" * " "^previous_printing_length * "\r")
                    @info terminate_message
                    @info "Comparing A* score $a_star_total_score and PP score $upper_bound"
                end

                # Return result from prioritized planning if
                # 1. parallel A* does not have a feasible solution yet
                # 2. parallel A* has a feasible solution but worst than PP
                if num_conflicts > 0 || upper_bound < a_star_total_score
                    if !silent
                        println(
                            "Iter = $iter ($(time_with_unit(time() - global_timer; digits=2))): " *
                            "≤$(round(suboptimality*1e2; digits=3))% suboptimal " *
                            "in $(round(lower_bound; digits=3)) - $(round(upper_bound; digits=3))",
                        )
                        @info "Return with total score of $(upper_bound) using prioritized planning"
                    end

                    # Record return information
                    return_info = Dict(
                        k => v for (k, v) in Base.@locals() if k in record_vars
                    )

                    return best_pp_path,
                    best_pp_scores, return_info, tracker,
                    suboptimality_timing

                    # Return parallel A* solution
                else
                    suboptimality = max(a_star_total_score - lower_bound) / lower_bound
                    if !silent
                        println(
                            "Iter = $iter ($(time_with_unit(time() - global_timer; digits=2))): " *
                            "≤$(round(suboptimality*1e2; digits=3))% suboptimal " *
                            "in $(round(lower_bound; digits=3)) - $(round(upper_bound; digits=3))",
                        )
                        @info "Return with total score of $(sum(a_star_total_score)) using relaxed A*"
                    end

                    # Record return information
                    return_info = Dict(
                        k => v for (k, v) in Base.@locals() if k in record_vars
                    )

                    return paths, astar_scores, return_info, tracker, suboptimality_timing
                end
            end

            # Update multipliers to compute modified costs
            update_multiplier!(
                vertex_multiplier,
                edge_multiplier,
                vertex_optimizer,
                edge_optimizer,
                vertex_occupancy,
                edge_occupancy;
                gradient_bias,
                perturbation=gradient_perturbation,
                rng,
            )

            # Update the modified cost from original cost
            update_cost!(
                cost,
                edge_costs,
                vertex_multiplier,
                edge_multiplier,
                network,
                length(source_vertices);
                perturbation=cost_perturbation,
                rng,
            )

            # Run prioritized planning for upper bound every once in a while
            if is_time_for_next_event(pp_frequency, pp_run_status)
                # Test different priority permutation every time
                test_priority = random_shuffle_priority ? shuffle(rng, priority) : priority

                pp_paths, _ = prioritized_planning(
                    network,
                    cost,
                    source_vertices,
                    target_vertices,
                    departure_times;
                    priority=test_priority,
                    swap_conflict,
                    heuristic,
                    max_iter=astar_max_iter,
                )
                pp_scores = compute_scores(pp_paths, edge_costs)
                total_pp_score = sum(pp_scores)

                # Update current best if get a better solution
                if total_pp_score < upper_bound
                    upper_bound = total_pp_score
                    best_pp_path = pp_paths
                    best_pp_scores = pp_scores
                    # refresh exploration status since new estimation is found
                    exploration_status = init_status(exploration_status)
                end
            end

            # Parallel A* to obtain the current Lagrange relaxation result
            paths, scores = shortest_paths(
                network,
                cost,
                source_vertices,
                target_vertices,
                departure_times;
                heuristics=heuristics,
                max_iter=astar_max_iter,
                multi_threads,
            )

            # Register occupancy and conflicts
            vertex_occupancy = detect_vertex_occupancy(paths)
            edge_occupancy = detect_edge_occupancy(paths; swap=swap_conflict)
            vertex_conflicts = detect_conflict(vertex_occupancy)
            edge_conflicts = detect_conflict(edge_occupancy)

            # Estimate lower bound
            relaxed_score = compute_relaxed_score(
                paths,
                edge_costs,
                vertex_multiplier,
                edge_multiplier,
                vertex_occupancy,
                edge_occupancy,
            )
            if relaxed_score > lower_bound
                lower_bound = relaxed_score
                # refresh exploration status since new estimation is found
                exploration_status = init_status(exploration_status)
            end

            # Record info in tracker
            if length(track_vars) > 0
                for (k, v) in Base.@locals
                    if k in track_vars
                        push!(tracker[k], v)
                    end
                end
            end

            # Update status
            num_conflicts = n_conflicts(vertex_conflicts) + n_conflicts(edge_conflicts)
            suboptimality = (upper_bound - lower_bound) / lower_bound

            time_used = time() - global_timer
            for (idx, subopt) in enumerate(track_time_by_suboptimality)
                if time_used < suboptimality_timing[idx][2] && suboptimality <= subopt
                    suboptimality_timing[idx] = (iter, time_used)
                end
            end

            iter += 1
            exploration_status = next_status(exploration_status)
            pp_run_status = next_status(pp_run_status)
        end
    catch e
        isa(e, InterruptException) || throw(e)
        @warn "Stop by user interruption, return best solution found so far."
    end

    # Meaningful solution hasn't been reach during the main loop, return the best known result at the moment
    if !silent
        println("")
        @info "Timeout after $iter iterations ($(time_with_unit(time() - global_timer; digits=2))) " *
            "with ≤$(round(suboptimality*1e2; digits=3))% suboptimal solution, " *
            "return result from prioritized planning with score $(sum(best_pp_scores))"
    end

    # Record return information
    return_info = Dict(k => v for (k, v) in Base.@locals() if k in record_vars)

    return best_pp_path, best_pp_scores, return_info, tracker, suboptimality_timing
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
