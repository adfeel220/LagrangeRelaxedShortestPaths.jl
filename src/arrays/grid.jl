
"""
    DynamicDimensionGridArray{T}
A dimension free array in which overindexing is allowed. No out of bound error for this array.
An special optimized version that is based on a grid and has index only 2-4.

Implemented for fast searching and single entity modification, not suitable for vectorized computation
The indexing of this array follows the rules below:
1. If exact match of index exists, return the index
2. No exact match exists, try to find a shorter index. e.g. if right alignment, try to index `arr[1,2,3]`
while `arr[1,2,3]` doesn't exist but `arr[2,3]` exists, return `arr[2,3]`
3. If non of the degenerated indices exists, return the default value.
"""
mutable struct DynamicDimensionGridArray{T} <: AbstractDynamicDimensionArray{T}
    d2::AVLTree{DimensionFreeData{T,2}}
    d3::AVLTree{DimensionFreeData{T,3}}
    d4::AVLTree{DimensionFreeData{T,4}}

    default::T
    grid_size::NTuple{2,Int}
    min_val::T
end

function Base.length(arr::DynamicDimensionGridArray)
    return length(arr.d2) + length(arr.d3) + length(arr.d4)
end
Base.size(arr::DynamicDimensionGridArray) = arr.grid_size

function Base.show(io::IO, arr::DynamicDimensionGridArray)
    return show(
        io,
        """DynamicDimensionGridArray{$(typeof(arr.default))} on grid of size $(arr.grid_size): $(length(arr)) registered entries with default = $(arr.default).""",
    )
end

"""
    DynamicDimensionGridArray(size[, default=1.0])
Create an empty `DynamicDimensionGridArray` with a default value (`{Float64}(1.0)` if not specified).
"""
function DynamicDimensionGridArray(size::NTuple{2,Int}; default::T=one(Float64), min_val::T=0.1*default) where {T}
    t2 = AVLTree{DimensionFreeData{T,2}}()
    t3 = AVLTree{DimensionFreeData{T,3}}()
    t4 = AVLTree{DimensionFreeData{T,4}}()
    return DynamicDimensionGridArray{T}(t2, t3, t4, default, size, min_val)
end

"""
    vertex_to_coordinate(arr, v)
Compute the coordinate from a vertex index `v` based on the grid size stored in `arr`
"""
function vertex_to_coordinate(arr::DynamicDimensionGridArray, v::Int)
    return (v - 1) ÷ arr.grid_size[1] + 1, (v - 1) % arr.grid_size[1] + 1
end

"""
    euclidean_distance(arr, v1, v2)
Compute the euclidean distance between `v1` and `v2` on the grid of `arr`, measured by number of blocks
"""
function euclidean_distance(arr::DynamicDimensionGridArray, v1::Int, v2::Int)
    coord1 = vertex_to_coordinate(arr, v1)
    coord2 = vertex_to_coordinate(arr, v2)
    return √sum((coord1 .- coord2) .^ 2)
end

function Base.getindex(arr::DynamicDimensionGridArray{T}, index::Vararg{Int}) where {T}
    if length(index) > 4
        index = index[(end - 3):end]
    end
    if length(index) == 4
        data = find_data(arr.d4, index)
        if !isnothing(data)
            return data
        end
        index = degenerate_tuple(index)
    end
    if length(index) == 3
        data = find_data(arr.d3, index)
        if !isnothing(data)
            return data
        end
        index = degenerate_tuple(index)
    end
    if length(index) == 2
        data = find_data(arr.d2, index)
        if !isnothing(data)
            return data
        end
    end

    return max(arr.min_val, arr.default * euclidean_distance(arr, index[1], index[2]))
end
function Base.setindex!(
    arr::DynamicDimensionGridArray{T}, value::T, index::Vararg{Int}
) where {T}
    if length(index) == 4
        set_data!(arr.d4, value, index)
    elseif length(index) == 3
        set_data!(arr.d3, value, index)
    elseif length(index) == 2
        set_data!(arr.d2, value, index)
    end
    return arr
end

function Base.iterate(arr::DynamicDimensionGridArray{T}, i=1) where {T}
    if i > length(arr)
        return nothing
    end
    if i <= length(arr.d2)
        return Pair(arr.d2[i].index, arr.d2[i].data), i + 1
    elseif i <= length(arr.d2) + length(arr.d3)
        return Pair(arr.d3[i - length(arr.d2)].index, arr.d3[i - length(arr.d2)].data),
        i + 1
    else
        return Pair(
            arr.d4[i - length(arr.d2) - length(arr.d3)].index,
            arr.d4[i - length(arr.d2) - length(arr.d3)].data,
        ),
        i + 1
    end
end

function empty(arr::DynamicDimensionGridArray{T}) where {T}
    return DynamicDimensionGridArray(arr.grid_size, arr.default)
end

"""
    delete!(arr, index)
Delete the node with `index` in `arr`
"""
function delete!(arr::DynamicDimensionGridArray{T}, index::NTuple{N,Int}) where {T,N}
    if length(index) == 4
        delete!(arr.d4, DimensionFreeData{T}(index))
    elseif length(index) == 3
        delete!(arr.d3, DimensionFreeData{T}(index))
    elseif length(index) == 2
        delete!(arr.d2, DimensionFreeData{T}(index))
    end
    return arr
end
