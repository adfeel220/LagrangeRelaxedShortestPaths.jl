module LagrangeRelaxedShortestPaths

using DataStructures: AVLTree, BinaryHeap
using DataStructures: push!, delete!

import DataStructures: sorted_rank

export DimensionFreeData, DynamicDimensionArray
export shortest_paths, astar, temporal_astar
export detect_vertex_conflict, detect_edge_conflict

include("arrays.jl")
include("shortest_paths.jl")
include("conflict.jl")

end
