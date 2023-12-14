"""
    TimedPath{T,V}
Type alias of `Vector{Tuple{T,V}}` to represent a sequence of time expanded vertices
"""
const TimedPath{T,V} = Vector{Tuple{T,V}}

"""
    backtrace_path(parents, node)
Starting from the vertex and arrival time, back-tracing by the parent nodes

# Arguments
- `parents::Dict{T,T}`: dictionary keep track of parent relation. (node => from-node)
- `node::T`: node to start backtracing
"""
function backtrace_path(parents::Dict{T,T}, node::T)::Vector{T} where {T}
    current = node
    path = [current]
    while haskey(parents, current)
        current = parents[current]
        pushfirst!(path, current)
    end
    return path
end

"""
    dijkstra(
        network, edge_costs, agent, source, target; backwards
    )
Run Dijkstra's algorithm and return the total costs and parents of the entire network.
Parent node of 

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `agent`: agent index
- `source::V`: starting vertex of agent
where `V` is the type of vertex and `C` is the type of cost

# Keyword arguments
- `backwards::Bool`: Whether to apply Dijkstra in a backward fashion (on reversed network),
by default `false`
"""
function dijkstra(
    network::AbstractGraph{V},
    edge_costs::AbstractDynamicDimensionArray{C},
    agent,
    source::V;
    backwards::Bool=false,
) where {V,C}
    # customize direction of exploration
    get_neighbors = backwards ? inneighbors : outneighbors

    # set of candidate nodes to be explored, use heap to retrieve minimum cost
    # node in constant time
    open_set = BinaryHeap(Base.By(last), [Pair(source, zero(C))])

    # parents store the traversing relationship between the time-expanded vertices
    parents = zeros(V, nv(network))
    # scores store the cost from source to a specific node
    scores = fill(typemax(C), nv(network))

    # Initialize score for starting point
    scores[source] = zero(C)

    while !isempty(open_set)
        # Retrieve the smallest cost node
        vertex, score = pop!(open_set)

        # Explore the neighbors of current vertex
        for v in get_neighbors(network, vertex)
            travel_cost =
                backwards ? edge_costs[agent, v, vertex] : edge_costs[agent, vertex, v]
            tentative_score = score + travel_cost
            neighbor_score = scores[v]

            # Record a neighbor as a good node to move forward if we find a lower
            # cost path compared to the previous exploration on this node
            if tentative_score < neighbor_score
                parents[v] = vertex
                scores[v] = tentative_score
                push!(open_set, v => tentative_score)
            end
        end
    end

    return scores, parents
end

"""
    astar(
        network, edge_costs, agent, source, target;
        heuristic, max_iter
    )
Run A* algorithm on the network.
Returns the path as a vector of vertices, return nothing if fail to find a solution

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `agent`: agent index
- `source::V`: starting vertex of agent
- `target::V`: target vertex for the agent to go to
where `V` is the type of vertex and `C` is the type of cost

# Keyword arguments
- `heuristic::Function`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. By default always returns 0
"""
function astar(
    network::AbstractGraph{V},
    edge_costs::AbstractDynamicDimensionArray{C},
    agent,
    source::V,
    target::V;
    heuristic::Function=n -> zero(C),
) where {V,C}
    # set of candidate nodes to be explored, use heap to retrieve minimum cost
    # node in constant time
    open_set = BinaryHeap(Base.By(last), [Pair(source, heuristic(source))])

    # parents store the traversing relationship between the time-expanded vertices
    parents = Dict{V,V}()
    # g_score store the cost from source to a specific node
    g_score = Dict{V,C}(source => zero(C))

    while !isempty(open_set)
        # Retrieve the smallest cost node
        vertex, _ = pop!(open_set)

        # Arrive at target, retrieve path and cost
        if vertex == target
            return backtrace_path(parents, vertex), g_score[vertex]
        end

        # Explore the neighbors of current vertex
        for v in outneighbors(network, vertex)
            tentative_g_score = g_score[vertex] + edge_costs[agent, vertex, v]
            neighbor_g_score::C = get(g_score, v, typemax(C))

            # Record a neighbor as a good node to move forward if
            # we find a lower cost path compared to the previous exploration on this node
            if tentative_g_score < neighbor_g_score
                parents[v] = vertex
                g_score[v] = tentative_g_score
                f_score = tentative_g_score + heuristic(v)
                push!(open_set, v => f_score)
            end
        end
    end

    return nothing
end

"""
    resolve_heuristic(heuristic, network, agent, target, edge_costs)
Resolve the heuristic so that the output is a function which maps a vertex `v` to a
heuristic value.
"""
function resolve_heuristic(
    heuristic::Union{Symbol,Function,Number},
    network::AbstractGraph{V},
    agent,
    target::V,
    edge_costs::AbstractDynamicDimensionArray,
) where {V}
    # Direct return if heuristic is already a function
    if isa(heuristic, Function)
        return heuristic
    end

    # Lazy heuristic of always return a humber
    if isa(heuristic, Number)
        return (v -> heuristic)
    end

    # Symbol for special cases
    if heuristic == :dijkstra
        dijkstra_scores, _ = dijkstra(network, edge_costs, agent, target; backwards=true)
        heuristic = (v -> dijkstra_scores[v])

    elseif heuristic == :euclidean
        heuristic = (v -> euclidean_distance(edge_costs, v, target))

    elseif heuristic == :lazy
        heuristic = (v -> zero(C))

    else
        error("Unrecognized heuristic symbol $heuristic")
    end

    return heuristic
end

"""
    temporal_astar(
        network, edge_costs, agent, source, target, departure_time;
        heuristic, max_iter
    )
Run A* algorithm on the network with discrete-time records on graph traversal.
Returns the path as a vector of time-expanded vertices, return nothing if fail to find a solution

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `agent`: agent index
- `source::V`: starting vertex of agent
- `target::V`: target vertex for the agent to go to
- `departure_time::T`: time when agent start traveling
where `V` is the type of vertex; `T` is the type of time; and `C` is the type of cost

# Keyword arguments
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports `:lazy` always return 0;
`:dijkstra`: Dijkstra on the static graph from target vertex as estimation
- `max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
"""
function temporal_astar(
    network::AbstractGraph{V},
    edge_costs::AbstractDynamicDimensionArray{C},
    agent,
    source::V,
    target::V,
    departure_time::T=0;
    reserved_vertices::Set{Tuple{T,V}}=Set{Tuple{T,V}}(),
    reserved_edges::Set{Tuple{T,V,V}}=Set{Tuple{T,V,V}}(),
    heuristic::Union{Symbol,Function}=:dijkstra,
    max_iter::Int=typemax(Int),
) where {V,T<:Integer,C}
    # Resolve heuristic, after resolving heuristic can only be a function
    heuristic = resolve_heuristic(heuristic, network, agent, target, edge_costs)

    # set of candidate nodes to be explored, use heap to retrieve minimum cost
    # node in constant time
    open_set = BinaryHeap{Pair{Tuple{T,V},C}}(
        Base.Order.By{typeof(last),FasterForward}(last, FasterForward()),
        [Pair((departure_time, source), heuristic(source))],
    )

    # parents store the traversing relationship between the time-expanded vertices
    parents = Dict{Tuple{T,V},Tuple{T,V}}()
    # g_score store the cost from source to a specific node
    g_score = Dict{Tuple{T,V},C}((departure_time, source) => zero(C))

    for itr in zero(Int):max_iter
        if isempty(open_set)
            break
        end

        # Retrieve the smallest cost node
        node, _ = pop!(open_set)
        t, vertex = node

        # Arrive at target, retrieve path and cost
        if vertex == target
            return backtrace_path(parents, node), g_score[node]
        end

        # Explore the neighbors of current vertex
        for v in outneighbors(network, vertex)
            next_time = t + one(T)
            # Skip exploration if the vertex is already reserved
            if (next_time, v) in reserved_vertices ||
                (next_time, vertex, v) in reserved_edges
                continue
            end

            # Compute tentative score for potential update
            tentative_g_score = g_score[t, vertex] + edge_costs[next_time, agent, vertex, v]
            neighbor_g_score::C = get(g_score, (next_time, v), typemax(C))

            # Record a neighbor as a good node to move forward if
            # we find a lower cost path compared to the previous exploration on this node
            if tentative_g_score < neighbor_g_score
                parents[next_time, v] = node
                g_score[next_time, v] = tentative_g_score
                f_score = tentative_g_score + heuristic(v)
                push!(open_set, (next_time, v) => f_score)
            end
        end
    end

    return Vector{Tuple{T,V}}(), typemax(C)
end

"""
    shortest_paths(
        network, edge_costs, sources, targets, departure_times;
        heuristic, max_iter, multi_threads
    )
Apply A* for all the agents in parallel. Returns the paths and costs of individual agent.

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::AbstractDynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `sources::Vector{V}`: starting vertices of agents
- `targets::Vector{V}`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling

# Keyword arguments
- `heuristic::Union{Symbol,Function}`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. Can also be some predefined methods, supports `:lazy` always return 0;
`:dijkstra`: Dijkstra on the static graph from target vertex as estimation
- `max_iter::Int`: maximum iteration of individual A*, by default `typemax(Int)`
- `multi_threads::Bool`: whether to apply multi threading, by default `true`
"""
function shortest_paths(
    network::AbstractGraph{V},
    edge_costs::AbstractDynamicDimensionArray{C},
    sources::Vector{V},
    targets::Vector{V},
    departure_times::Vector{T}=zeros(Int, length(sources));
    heuristic::Union{Symbol,Function}=:dijkstra,
    max_iter::Int=typemax(Int),
    multi_threads::Bool=true,
) where {V,C,T<:Integer}
    @assert length(sources) == length(targets) == length(departure_times) "Number of agents must be consistent on sources, targets, and departure_times"

    # Multi-threaded solution
    if multi_threads
        paths = Vector{Vector{Tuple{T,V}}}(undef, length(sources))
        costs = zeros(C, length(sources))

        Base.Threads.@threads for ag in 1:length(sources)
            sv = sources[ag]
            tv = targets[ag]
            dep_time = departure_times[ag]

            paths[ag], costs[ag] = temporal_astar(
                network, edge_costs, ag, sv, tv, dep_time; heuristic, max_iter
            )
        end
        return paths, costs
    end

    # Applying with generator instead of multi-threading
    results = [
        temporal_astar(network, edge_costs, ag, sv, tv, dep_time; heuristic, max_iter) for
        (ag, (sv, tv, dep_time)) in enumerate(zip(sources, targets, departure_times))
    ]
    # unzip results into two separated vectors for paths and costs
    return [first(res) for res in results], [last(res) for res in results]
end

"""
    is_planning_failed(path[, cost])
Return whether the planning is failed by path

# Arguments
- `path::Vector{Tuple{T,V}}`: sequence of time-expanded vertices, planning failed if it's empty
- `cost`: dummy, only checks path if path is given
"""
function is_planning_failed(path::Vector{Tuple{T,V}}, cost=0) where {T,V}
    return length(path) == 0
end
"""
    is_planning_failed(cost)
Return whether the planning is failed by cost value

# Arguments
- `cost<:Number`: cost of the proposed path, planning failed if it's `typemax`
"""
function is_planning_failed(cost::C) where {C<:Number}
    return cost == typemax(C)
end
