
"""
    MapfConfig{V, G<:AbstractGraph{V}, T, C}
A structure to hold all necessary information for executing an MAPF solver
`G`: type for graph, `T`: type for time; `V`: type for vertex; and `C`: type of cost
"""
@kwdef mutable struct MapfConfig{V,G<:AbstractGraph{V},T,C}
    network::G
    source_vertices::Vector{V}
    target_vertices::Vector{V}
    edge_costs::DynamicDimensionArray{C} = ones(Float64, (nv(network), nv(network)))
    departure_time::Vector{T} = zeros(Int, length(source_vertices))
end
nagents(config::MapfConfig) = length(config.source_vertices)
function Base.show(io::IO, config::MapfConfig)
    return show(
        io,
        "MAPF configuration of $(nagents(config)) agents on a {$(nv(config.network)),$(ne(config.network))} network",
    )
end

"""
    lagrange_relaxed_shortest_path(config)
Execute a Lagrange relaxed shortest paths with configuration file

# Arguments
- `config::MapfConfig`: Configuration struct containing all necessary information
"""
function lagrange_relaxed_shortest_path(config::MapfConfig; kwargs...)
    return lagrange_relaxed_shortest_path(
        config.network,
        config.edge_costs,
        config.source_vertices,
        config.target_vertices,
        config.departure_time;
        kwargs...,
    )
end
