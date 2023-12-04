
"""
    VertexConflicts{T,V,A}
Type alias for `Dict{Tuple{T,V},Vector{Tuple{A,V}}}` to store the vertex conflicts within a time-expanded graph.
`T` is the type of time; `V` is the type of vertex; and `A` is the type of agent

For each time-expanded vertex (as key), stores the vector of agents and the previous vertex before entering
this conflicting vertex.
"""
const VertexConflicts{T,V,A} = Dict{Tuple{T,V},Vector{Tuple{A,V}}}

"""
    EdgeConflicts{T,V,A}
Type alias for `Dict{Tuple{T,V,V},Vector{Tuple{A,Bool}}}` to store the edge conflicts within a
time-expanded graph. A time-expanded edge is stored as a `Tuple{T,V,V}` as (time, from-v, to-v).
The conflicting agent is stored alongside with a boolean value to indecate whether it's flipped
to detect swapping conflict, as the format of (agent, is_flip)
`T` is the type of time; `V` is the type of vertex; and `A` is the type of agent
"""
const EdgeConflicts{T,V,A} = Dict{Tuple{T,V,V},Vector{Tuple{A,Bool}}}

"""
    detect_vertex_occupancy(timed_paths; return_first)
Return a vertex occupancy table

# Arguments
- `timed_paths::Vector{TimedPath{T,V}}`: Paths as sequence of time-expanded vertices for every agent

# Keyword arguments
- `return_first::Bool`: return only the first conflict if `true`, otherwise return all conflicts. By default `false`
"""
function detect_vertex_occupancy(
    timed_paths::Vector{TimedPath{T,V}}; return_first::Bool=false
)::VertexConflicts{T,V,Int} where {T,V}
    # for each time and vertex, the occupancy is recorded as agent IDs
    vertex_occupancy = VertexConflicts{T,V,Int}()

    # Register occupancy
    for (agent, itinerary) in enumerate(timed_paths)
        for (step_id, timed_step) in enumerate(itinerary)
            prev_v = get(itinerary, step_id - 1, (0, zero(V)))[2]
            # Record the vertex occupancy
            if haskey(vertex_occupancy, timed_step)
                push!(vertex_occupancy[timed_step], (agent, prev_v))

                # Early termination for the first-found conflict
                if return_first
                    return Dict(timed_step => vertex_occupancy[timed_step])
                end
            else
                vertex_occupancy[timed_step] = [(agent, prev_v)]
            end
        end
    end

    return vertex_occupancy
end

"""
    detect_vertex_conflict(timed_paths; return_first)
Detect vertex conflicts given the timed paths of all agents

# Arguments
- `timed_paths::Vector{TimedPath{T,V}}`: Paths as sequence of time-expanded vertices for every agent
- `capacity::Int`: capacity of vertices, by default 1

# Keyword arguments
- `return_first::Bool`: return only the first conflict if `true`, otherwise return all conflicts. By default `false`
"""
function detect_vertex_conflict(
    timed_paths::Vector{TimedPath{T,V}}, capacity::Int=1; return_first::Bool=false
)::VertexConflicts{T,V,Int} where {T,V}
    vertex_occupancy = detect_vertex_occupancy(timed_paths; return_first)

    # Check overlaps and returns those having overlaps
    return Dict(
        timed_vertex => agents for
        (timed_vertex, agents) in vertex_occupancy if length(agents) > capacity
    )
end

"""
    detect_edge_occupancy(timed_paths; return_first)
Return edge occupancy table

# Arguments
- `timed_paths::Vector{TimedPath{T,V}}`: Paths as sequence of time-expanded vertices for every agent

# Keyword arguments
- `swap::Bool`: whether to detect swapping confliccts, by default `false`
- `return_first::Bool`: return only the first conflict if `true`, otherwise return all conflicts. By default `false`
"""
function detect_edge_occupancy(
    timed_paths::Vector{TimedPath{T,V}}; swap::Bool=false, return_first::Bool=false
)::EdgeConflicts{T,V,Int} where {T,V}
    # for each edge as time and from/to vertices, the occupancy is recorded as agent IDs and whether they flipped
    edge_occupancy = EdgeConflicts{T,V,Int}()

    # Register occupancy
    for (agent, itinerary) in enumerate(timed_paths)
        for (step_id, (timestamp, to_vertex)) in enumerate(itinerary[(begin + 1):end])
            from_vertex = last(itinerary[step_id])

            # If we need to detect swap conflicts, we make sure from_v always <= to_v
            is_flip = swap && (from_vertex > to_vertex)
            if is_flip
                from_vertex, to_vertex = to_vertex, from_vertex
            end

            timed_step = (timestamp, from_vertex, to_vertex)

            # Record occupancy along with flip information
            if haskey(edge_occupancy, timed_step)
                push!(edge_occupancy[timed_step], (agent, is_flip))

                # Early termination for the first-found conflict
                if return_first
                    return Dict(timed_step => edge_occupancy[timed_step])
                end
            else
                edge_occupancy[timed_step] = [(agent, is_flip)]
            end
        end
    end

    return edge_occupancy
end

"""
    detect_edge_conflict(timed_paths; return_first)
Detect edge conflicts given the timed paths of all agents

# Arguments
- `timed_paths::Vector{TimedPath{T,V}}`: Paths as sequence of time-expanded vertices for every agent
- `capacity::Int`: capacity of vertices, by default 1

# Keyword arguments
- `swap::Bool`: whether to detect swapping confliccts, by default `false`
- `return_first::Bool`: return only the first conflict if `true`, otherwise return all conflicts. By default `false`
"""
function detect_edge_conflict(
    timed_paths::Vector{TimedPath{T,V}},
    capacity::Int=1;
    swap::Bool=false,
    return_first::Bool=false,
)::EdgeConflicts{T,V,Int} where {T,V}
    edge_occupancy = detect_edge_occupancy(timed_paths; swap, return_first)

    # Check overlaps and returns those having overlaps
    return Dict(
        timed_edge => agents for
        (timed_edge, agents) in edge_occupancy if length(agents) > capacity
    )
end

"""
    is_conflict_free(conflict::Dict)
Return whether the conflict is conflict free
"""
is_conflict_free(conflict::Dict)::Bool = length(conflict) == 0
