"""
    break_segments(network, edge_break)
Break every edge into multiple segments. Suppose the original network has V0 vertices and E0 edges,
the new network will have `V0 + edge_break*(E0-1)` vertices and `E0 * edge_break` edges.

# Arguments
- `network::AbstractGraph`: original graph as reference (will not be modified)
- `edge_break::Int`: number of segments to break into (original network = 1 segment)
"""
function break_segments(network::AbstractGraph, edge_break::Int=1)
    new_network = deepcopy(network)

    if edge_break <= 1
        return new_network
    end

    for ed in edges(network)
        u, v = src(ed), dst(ed)
        rem_edge!(new_network, u, v)

        prev_v = u
        original_network_size = nv(new_network)
        for s in 1:(edge_break - 1)
            add_vertex!(new_network)
            add_edge!(new_network, prev_v, original_network_size + s)
            prev_v = original_network_size + s
        end
        add_edge!(new_network, prev_v, v)
    end

    return new_network
end

"""
    parallel_lines(a, edge_break)
Create a scenario where all agents can go to their destination with a dedicated path without any conflict

The generated graph has `|V| = 2*a`

# Arguments
- `a::Int`: number of agents

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
"""
function parallel_lines(a::Int; edge_break::Int=1)
    source_vertices = Vector{Int}(1:a)
    target_vertices = Vector{Int}((a + 1):(2 * a))

    base_network = DiGraph(a * 2)
    for (sv, tv) in zip(source_vertices, target_vertices)
        add_edge!(base_network, sv, tv)
    end

    experiment = MapfConfig(;
        network=base_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = DiGraph(a * (edge_break + 1))

    for (sv, tv) in zip(source_vertices, target_vertices)
        prev_v = sv
        for b in 1:(edge_break - 1)
            add_edge!(discrete_network, prev_v, tv + a * b)
        end
        add_edge!(discrete_network, prev_v, tv)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end

"""
    directional_star(a, edge_break)
Create a scenario where all agents goes via a core vertex and then spread out to their destinations.
It's equivalent to a star topology where agents start from half of the tips, go through the center vertex,
and then arrive at the other half of the tips.
There will be (`a` choose 2) vertex conflicts in this scenario.

The generated base graph has `|V| = 2*a + 1` and `|E| = 2*a`

# Arguments
- `a::Int`: number of agents

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
"""
function directional_star(a::Int; edge_break::Int=1)
    source_vertices = Vector{Int}(2:(a + 1))
    target_vertices = Vector{Int}((a + 2):(2 * a + 1))

    # vertex 1 is the central vertex
    base_network = DiGraph(a * 2 + 1)
    for (sv, tv) in zip(source_vertices, target_vertices)
        add_edge!(base_network, sv, 1)
        add_edge!(base_network, 1, tv)
    end

    exp_network = deepcopy(base_network)
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
    end
    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end

"""
    grid_cross(row, column, edge_break)
Create a scenario where the network is a `row * column` grid and all agents go from one side to the other.
i.e. `row` agents go from left to right and `column` agents go from top to buttom. There are
`row * column` potential vertex conflicts and `2*row*column - row - column` potential edge conflicts.

There will be `row + column` agents, `2*(row+column) + row*column` vertices, and `2*row*column + row + column` edges.

# Arguments
- `row::Int`: number of rows
- `column::Int`: number of columns

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
"""
function grid_cross(row::Int, column::Int; edge_break::Int=1)
    num_agent = row + column
    source_vertices = Vector{Int}(1:num_agent)
    target_vertices = Vector{Int}((num_agent + 1):(2 * num_agent))

    base_network = DiGraph(2 * num_agent + row * column)

    # horizontal lines
    for r in 1:row
        prev_v = r
        for c in 1:column
            next_v = (r - 1) * column + c + 2 * num_agent
            add_edge!(base_network, prev_v, next_v)
            prev_v = next_v
        end
        add_edge!(base_network, prev_v, num_agent + r)
    end

    # vertical lines
    for c in 1:column
        prev_v = row + c
        for r in 1:row
            next_v = (r - 1) * column + c + 2 * num_agent
            add_edge!(base_network, prev_v, next_v)
            prev_v = next_v
        end
        add_edge!(base_network, prev_v, num_agent + row + c)
    end

    exp_network = deepcopy(base_network)
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
    end
    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end

"""
    line_overlap(n, a, edge_break; delayed_departure)
Create a scenario where all agents goes on the same line but overlap each other. i.e. the vertices form a
line of `1` to `n+a-1`. Agent 1 goes from `1` to `n`, 2 goes from `2` to `n+1`, ...
There are `O(n * a)` potential vertex and edge conflicts depending on the departure time.

The generated base graph has `|V| = n + a - 1` and `|E| = n + a - 2`

# Arguments
- `n::Int`: distance each agent must travel to their destination
- `a::Int`: number of agents

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
- `delayed_departure::Bool`: agents in the front (larger indices) will depart later than
agents at the back if this is `true`. All agents depart at the same time otherwise (default behavior)
"""
function line_overlap(n::Int, a::Int; edge_break::Int=1, delayed_departure::Bool=false)
    source_vertices = Vector{Int}(1:a)
    target_vertices = Vector{Int}((n - 1) .+ (1:a))

    base_network = DiGraph(n + a - 1)
    for v in 1:(n + a - 2)
        add_edge!(base_network, v, v + 1)
    end

    departure_time = delayed_departure ? (edge_break) .* collect(0:(a - 1)) : zeros(Int, a)

    exp_network = deepcopy(base_network)
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
    end
    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
        departure_time=departure_time,
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
        departure_time=departure_time,
    )

    return experiment, discrete_experiment
end

"""
    wheel_pass(a, edge_break)
Create a network forming a wheel graph where the outer vertices form a unidirectional cycle
and their connections with the center vertex are bidirectional. Agents start from all the outer vertices
and tries to travel to their destination that is a circular shift from the starting vertices. There are
`O(a^2)` potential vertex and edge conflicts in worst case and `O(1)` conflicts in best case depending on
the shift number.

The generated base graph has `|V| = a + 1` and `|E| = 3*a`

# Arguments
- `a::Int`: number of agents

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
- `shift::Int`: the counter-directional shift with respect to the starting and ending vertices of the agents
"""
function wheel_pass(a::Int; edge_break::Int=1, shift::Int=1)
    num_vertices = a + 1
    source_vertices = Vector{Int}(2:(a + 1))
    target_vertices = circshift(source_vertices, shift)

    base_network = wheel_digraph(num_vertices)
    for v in 2:num_vertices
        add_edge!(base_network, v, 1)
    end

    exp_network = deepcopy(base_network)
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
    end
    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end

"""
    circular_ladder_pass(a, edge_break)
Create a network forming a circular ladder graph where the outer vertices form a unidirectional cycle, the
inner vertices form another unidirectional cycle with the opposite direction compared to the outer circle.
The connections between the outer and inner cycles are bidirectional. Agents start from all the outer vertices
and tries to travel to their destination that is a circular shift from the starting vertices. There are
`O(a)` potential vertex and edge conflicts in worst case and `O(1)` conflicts in best case depending on
the shift number.

The generated base graph has `|V| = 2*a` and `|E| = 4*a`

# Arguments
- `a::Int`: number of agents

# Keyword arguments
- `edge_break::Int`: number of segments to break down in discrete case, by default `1` (no break)
- `shift::Int`: the counter-directional shift with respect to the starting and ending vertices of the agents
"""
function circular_ladder_pass(a::Int; edge_break::Int=1, shift::Int=1)
    base_network = DiGraph(2 * a)
    for v in 1:(a - 1)
        add_edge!(base_network, v, v + 1)
    end
    add_edge!(base_network, a, 1)

    for v in (2 * a):(-1):(a + 2)
        add_edge!(base_network, v, v - 1)
    end
    add_edge!(base_network, a + 1, 2 * a)

    for v in 1:a
        add_edge!(base_network, v, v + a)
        add_edge!(base_network, v + a, v)
    end

    source_vertices = Vector{Int}(1:a)
    target_vertices = circshift(source_vertices, shift)

    exp_network = deepcopy(base_network)
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
    end
    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end

"""
    pp_infeasible_case(a; edge_break)
Create a network with a star-like topology. Each branch has exactly two vertices.
There is also a buffer vertex attached only to the middle vertex.
Every agent starts from the vertex closer to the center and tries to reach
the vertex at the back of another agent. Prioritized planning can not find any
feasible solution regardless of the order. Only even number of agents is meaningful.
Odd number of agents leads to one redundant path.

The generated network has |V| = 2*(a+1) and |E| = 6*a + 4

# Arguments
- `a::Int`: Number of agents

# Keyword arguments
- `edge_break::Int`: break every edge into this number of sub-segments, self-loops
are not effected, by default 1 (no break)
"""
function pp_infeasible_case(a::Int; edge_break::Int=1)
    base_network = DiGraph(2 * (a + 1))
    n = nv(base_network)
    for v in 1:a
        add_edge!(base_network, v, v + a)
        add_edge!(base_network, v + a, v)

        add_edge!(base_network, v, n - 1)
        add_edge!(base_network, n - 1, v)
    end

    add_edge!(base_network, n - 1, n)
    add_edge!(base_network, n, n - 1)

    source_vertices = Vector{Int}(1:a)
    target_vertices = Vector{Int}((2 * a):-1:(a + 1))

    edge_costs = DynamicDimensionArray(1.0)

    exp_network = deepcopy(base_network)

    # Self loops
    for v in vertices(base_network)
        add_edge!(exp_network, v, v)
        edge_costs[v, v] = 0.5
    end

    experiment = MapfConfig(;
        network=exp_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    discrete_network = break_segments(base_network, edge_break)
    for v in vertices(discrete_network)
        add_edge!(discrete_network, v, v)
    end

    discrete_experiment = MapfConfig(;
        network=discrete_network,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        edge_costs=DynamicDimensionArray(1.0),
    )

    return experiment, discrete_experiment
end
