module LagrangeRelaxedShortestPaths

using DataStructures: AVLTree, BinaryHeap
using DataStructures: push!, delete!

import DataStructures: sorted_rank

export DimensionFreeData, DynamicDimensionArray
export shortest_paths, astar, temporal_astar

include("arrays.jl")
include("shortest_paths.jl")

end
