
"""
    vertex_path_to_edge_reservation(
        vertex_path; swap
    )
Return the edge reservation from an agent's vertex path. i.e. given sequence of
time-expanded vertices.

# Arguments
- `vertex_path::Vector{Tuple{T,V}}`: agent's path stored as a sequence of time-expanded
vertices. `T` is the type of time and `V` is the type of vertex.

# Keyword arguments
- `swap::Bool`: whether to reserve edges for swapping conflict, by default `false`
"""
function vertex_path_to_edge_reservation(
    vertex_path::Vector{Tuple{T,V}}; swap::Bool=false
) where {T<:Integer,V}
    edge_reserve = [
        (t2, a1, a2) for ((t1, a1), (t2, a2)) in
        zip(vertex_path[begin:(end - 1)], vertex_path[(begin + 1):end])
    ]

    if swap
        append!(
            edge_reserve,
            [
                (t2, a2, a1) for ((t1, a1), (t2, a2)) in
                zip(vertex_path[begin:(end - 1)], vertex_path[(begin + 1):end])
            ],
        )
    end

    return edge_reserve
end

"""
    prioritized_planning(
        network, edge_costs, sources, targets, departure_times, priority;
        swap_conflict, heuristic, max_iter
    )
Run prioritized planning for all the agents

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `sources`: starting vertices of agents
- `targets`: target vertices for the agents to go to
- `departure_times::Vector{T}`: time when agents start traveling

# Keyword arguments
- `priority`: sequence to do shortest path, is a permutation of agents
- `swap_conflict::Bool`: whether to include swapping conflict in the algorithm
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports `:lazy` always return 0;
`:dijkstra`: Dijkstra on the static graph from target vertex as estimation
- `max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
"""
function prioritized_planning(
    network::AbstractGraph{V},
    edge_costs::AbstractDynamicDimensionArray{C},
    sources,
    targets,
    departure_times::Vector{T}=zeros(Int, length(sources));
    priority=Base.OneTo(length(sources)),
    swap_conflict::Bool=false,
    heuristic::Union{Symbol,Function}=:dijkstra,
    max_iter::Int=typemax(Int),
    timeout::Float64=Inf,
    silent::Bool=true,
) where {V,C,T<:Integer}
    @assert length(sources) ==
        length(targets) ==
        length(departure_times) ==
        length(priority) "Number of agents must be consistent on sources, targets, and departure_times"
    @assert length(unique(priority)) == length(priority) "No duplicated element is allowed in prioritized planning, got $priority"

    paths = Vector{Vector{Tuple{T,V}}}(undef, length(sources))
    costs = fill(typemax(C), length(sources))
    reserved_vertices = Set{Tuple{T,V}}()
    reserved_edges = Set{Tuple{T,V,V}}()

    start_time = time()
    previous_printing_length = 0
    # Run shortest paths one by one
    for (ag_idx, ag) in enumerate(priority)
        (time() - start_time) > timeout && break

        if !silent
            print("\r" * " "^previous_printing_length * "\r")
            print_info = "Computing shortest path of agent $ag of $ag_idx-th priority"
            print(print_info)
            previous_printing_length = length(print_info)
        end

        sv = sources[ag]
        tv = targets[ag]
        dep_time = departure_times[ag]

        paths[ag], costs[ag] = temporal_astar(
            network,
            edge_costs,
            ag,
            sv,
            tv,
            dep_time;
            reserved_vertices,
            reserved_edges,
            heuristic,
            max_iter,
        )

        # reserve vertex and edge occupancies
        if is_planning_failed(paths[ag])
            continue
        end
        for v in paths[ag]
            push!(reserved_vertices, v)
        end
        edge_reserve = vertex_path_to_edge_reservation(paths[ag]; swap=swap_conflict)
        for e in edge_reserve
            push!(reserved_edges, e)
        end
    end

    silent || print("\n")

    return paths, costs
end
