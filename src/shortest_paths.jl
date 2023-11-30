
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
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `agent`: agent index
- `source::V`: starting vertex of agent
where `V` is the type of vertex and `C` is the type of cost

# Keyword arguments
- `backwards::Bool`: Whether to apply Dijkstra in a backward fashion (on reversed network),
by default `false`
"""
function dijkstra(
    network::AbstractGraph{V},
    edge_costs::DynamicDimensionArray{C},
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
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
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
    edge_costs::DynamicDimensionArray{C},
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
    temporal_astar(
        network, edge_costs, agent, source, target, departure_time;
        heuristic, max_iter
    )
Run A* algorithm on the network with discrete-time records on graph traversal.
Returns the path as a vector of time-expanded vertices, return nothing if fail to find a solution

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `agent`: agent index
- `source::V`: starting vertex of agent
- `target::V`: target vertex for the agent to go to
- `departure_time::T`: time when agent start traveling
where `V` is the type of vertex; `T` is the type of time; and `C` is the type of cost

# Keyword arguments
- `heuristic::Function`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. By default always returns 0
- `max_iter::UInt`: maximum iteration of individual A*, by default `typemax(UInt)`
"""
function temporal_astar(
    network::AbstractGraph{V},
    edge_costs::DynamicDimensionArray{C},
    agent,
    source::V,
    target::V,
    departure_time::T=0;
    heuristic::Union{Symbol,Function}=:dijkstra,
    max_iter::UInt=typemax(UInt),
) where {V,T,C}
    # Resolve heuristic
    if isa(heuristic, Symbol)
        if heuristic == :dijkstra
            dijkstra_scores, _ = dijkstra(
                network, edge_costs, agent, target; backwards=true
            )
            heuristic = (v -> dijkstra_scores[v])
        elseif heuristic == :lazy
            heuristic = (v -> zero(C))
        else
            error("Unrecognized heuristic symbol $heuristic")
        end
    end

    # set of candidate nodes to be explored, use heap to retrieve minimum cost
    # node in constant time
    open_set = BinaryHeap(
        Base.By(last), [Pair((departure_time, source), heuristic(source))]
    )

    # parents store the traversing relationship between the time-expanded vertices
    parents = Dict{Tuple{T,V},Tuple{T,V}}()
    # g_score store the cost from source to a specific node
    g_score = Dict{Tuple{T,V},C}((departure_time, source) => zero(C))

    for itr in zero(UInt):max_iter
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
            tentative_g_score =
                g_score[t, vertex] + edge_costs[t + one(T), agent, vertex, v]
            neighbor_g_score::C = get(g_score, (t + one(T), v), typemax(C))

            # Record a neighbor as a good node to move forward if
            # we find a lower cost path compared to the previous exploration on this node
            if tentative_g_score < neighbor_g_score
                parents[t + one(T), v] = node
                g_score[t + one(T), v] = tentative_g_score
                f_score = tentative_g_score + heuristic(v)
                push!(open_set, (t + one(T), v) => f_score)
            end
        end
    end

    return nothing
end

"""
    shortest_paths(
        network, edge_costs, sources, targets, departure_times;
        heuristic, max_iter, multi_threads
    )
Apply A* for all the agents in parallel. Returns the paths and costs of individual agent.

# Arguments
- `network::AbstractGraph{V}`: network for the agent to travel on
- `edge_costs::DynamicDimensionArray{C}`: cost indexed by (time, agent, from-v, to-v)
- `sources::Vector{V}`: starting vertices of agents
- `targets::Vector{V}`: target vertices for the agents to go to
- `departure_times`: time when agents start traveling

# Keyword arguments
- `heuristic::Function`: given a vertex as input, returns the estimated cost from this vertex
to target. This estimation has to always underestimate the cost to guarantee optimal result.
i.e. h(n) ≤ d(n) always true for all n. By default always returns 0
- `max_iter::UInt`: maximum iteration of individual A*, by default `typemax(UInt)`
- `multi_threads::Bool`: whether to apply multi threading, by default `true`
"""
function shortest_paths(
    network::AbstractGraph{V},
    edge_costs::DynamicDimensionArray{C},
    sources::Vector{V},
    targets::Vector{V},
    departure_times::Vector{T}=zeros(T, length(sources));
    heuristic::Union{Symbol,Function}=:dijkstra,
    max_iter::UInt=typemax(UInt),
    multi_threads::Bool=true,
) where {V,C,T}
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