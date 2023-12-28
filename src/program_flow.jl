
"""
    init_status(<:Integer)
Return zero of the same type as input, treating as number of iterations
"""
init_status(status::T) where {T<:Integer} = zero(T)

"""
    init_status(<:AbstractFloat)
Return current time, treating as physical running time
"""
init_status(status::T) where {T<:AbstractFloat} = convert(T, time())

"""
    next_status(<:Integer)
Return input + 1, as treating input as iteration, next iteration is +1
"""
next_status(status::T) where {T<:Integer} = status + one(T)

"""
    next_status(<:AbstractFloat)
Return current time, as treating input as physical running time
"""
next_status(status::T) where {T<:AbstractFloat} = convert(T, time())

"""
    is_time_for_next_event(interval[, iter][, lastEventTime])
Determine whether the next event is hit. `interval` is interpreted as
1. number of iterations if it's an integer
2. time duration if it's a float
"""
function is_time_for_next_event(interval::Integer, iter)
    return iter % interval == interval - one(interval)
end
function is_time_for_next_event(time_duration::AbstractFloat, last_event_time)
    return (time() - last_event_time) > time_duration
end
function is_time_for_next_event(interval::Integer, iter, last_event_time)
    return is_time_for_next_event(interval, iter)
end
function is_time_for_next_event(time_duration::AbstractFloat, iter, last_event_time)
    return is_time_for_next_event(time_duration, last_event_time)
end

"""
    ready_to_terminate(
        vertex_conflict, edge_conflicts, upper_bound, lower_bound, min_edge_cost, exploration_status;
        optimality_threshold, max_exploration_time
    )
Test termination criteria and return `true` and message if one of the condition is met.
"""
function ready_to_terminate(
    vertex_conflicts::VertexConflicts{T,V,A},
    edge_conflicts::EdgeConflicts{T,V,A},
    upper_bound::C,
    lower_bound::C,
    absolute_optimality_threshold::C,
    exploration_status::S;
    relative_optimality_threshold::C=0.0,
    max_exploration_time::S,
) where {T,V,A,C,S<:Union{Integer,AbstractFloat}}
    # Criteria 1: conflict free
    if is_conflict_free(vertex_conflicts) && is_conflict_free(edge_conflicts)
        return true, "Terminate upon finding a conflict free solution"
    end

    # Criteria 2: absolute optimality gap smaller than threshold
    if (upper_bound - lower_bound) < absolute_optimality_threshold
        return true,
        "Terminate upon optimality gap smaller than absolute gap threshold $absolute_optimality_threshold"
    end

    suboptimality = (upper_bound - lower_bound) / lower_bound
    # Criteria 3: relative optimaligy gap smaller than a threshold
    if suboptimality <= relative_optimality_threshold
        return true,
        "Terminate upon reaching optimality gap ≤ $relative_optimality_threshold"
    end

    # Criteria 4: optimality have not improved for a long time
    if is_time_for_next_event(max_exploration_time, exploration_status)
        unit = isa(max_exploration_time, Integer) ? "iterations" : "seconds"
        return true,
        "Terminate since no improvement has been made over $max_exploration_time $unit"
    end

    return false, ""
end

"""
    time_with_unit(time_s; digits)
Return a `String` with human friendly readable format with input time in the unit of seconds.
If a number is larger than 1.0, returns `day-hour-minute-second` format;
If a number is smaller than 1.0, returns `ms`, `μs`, `ns`, etc.
"""
function time_with_unit(time_s::AbstractFloat; digits=3)::String
    if time_s >= 1.0
        units = ["d" => 86400.0, "h" => 3600.0, "m" => 60.0]

        unallocated_time = time_s
        time_with_unit = ""
        for (u, mul) in units
            if unallocated_time >= mul
                high_unit_number = round(Int, div(unallocated_time, mul))
                unallocated_time %= mul
                time_with_unit *= "$high_unit_number$u"
            end
        end

        time_with_unit *= "$(round(unallocated_time; digits=digits))s"

        return time_with_unit

    else
        units = [
            "m" => 1e-3, "μ" => 1e-6, "n" => 1e-9, "p" => 1e-12, "f" => 1e-15, "a" => 1e-18
        ]

        for (u, mul) in units
            if time_s >= mul
                return "$(round(time_s / mul; digits=digits))$(u)s"
            end
        end

        return "$(round(time_s * 1e18; digits=digits))as"
    end
end
