
"""
    init_status(<:Integer)
Return zero of the same type as input, treating as number of iterations
"""
init_status(::T) where {T<:Integer} = zero(T)

"""
    init_status(<:AbstractFloat)
Return current time, treating as physical running time
"""
init_status(::T) where {T<:AbstractFloat} = time()

"""
    next_status(<:Integer)
Return input + 1, as treating input as iteration, next iteration is +1
"""
next_status(status::T) where {T<:Integer} = status + 1

"""
    next_status(<:AbstractFloat)
Return current time, as treating input as physical running time
"""
next_status(::T) where {T<:AbstractFloat} = time()

"""
    is_time_for_next_event(interval[, iter][, lastEventTime])
Determine whether the next event is hit. `interval` is interpreted as
1. number of iterations if it's an integer
2. time duration if it's a float
"""
function is_time_for_next_event(interval::Integer, iter::Integer)
    return iter % interval == interval - one(interval)
end
function is_time_for_next_event(time_duration::AbstractFloat, last_event_time::AbstractFloat)
    return (time() - last_event_time) > time_duration
end
function is_time_for_next_event(interval::Integer, iter::Integer, last_event_time)
    return is_time_for_next_event(interval, iter)
end
function is_time_for_next_event(time_duration::AbstractFloat, iter, last_event_time::AbstractFloat)
    return is_time_for_next_event(time_duration, last_event_time)
end

"""
    ready_to_terminate(
        vertex_conflict, edge_conflicts, upper_bound, lower_bound, min_edge_cost, exploration_status;
        optimality_threshold, max_exploration_time, silent
    )
Test termination criteria and return `true` if one of the condition is met.
"""
function ready_to_terminate(
    vertex_conflicts::VertexConflicts{T,V,A},
    edge_conflicts::EdgeConflicts{T,V,A},
    upper_bound::C,
    lower_bound::C,
    min_edge_cost::C,
    exploration_status::Union{Integer,AbstractFloat};
    optimality_threshold::C=0.0,
    max_exploration_time::Union{Integer,AbstractFloat},
    silent::Bool=true,
) where {T,V,A,C}
    # Criteria 1: conflict free
    if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
        if !silent
            println("")
            @info "Terminate upon finding a conflict free solution"
        end
        return true
    end

    # Criteria 2: absolute optimality gap smaller than minimum edge cost
    if (upper_bound - lower_bound) < min_edge_cost
        if !silent
            println("")
            @info "Terminate upon optimality gap smaller than minimum edge cost $min_edge_cost"
        end
        return true
    end

    suboptimality = (upper_bound - lower_bound) / lower_bound
    # Criteria 3: relative optimaligy gap smaller than a threshold
    if suboptimality <= optimality_threshold
        if !silent
            println("")
            @info "Terminate upon reaching optimality gap ≤ $optimality_threshold"
        end
        return true
    end

    # Criteria 4: optimality have not improved for a long time
    if is_time_for_next_event(max_exploration_time, exploration_status)
        if !silent
            println("")
            unit = isa(max_exploration_time, Integer) ? "iterations" : "seconds"
            @info "Terminate since no improvement has been made over $max_exploration_time $unit"
        end
        return true
    end

    return false
end