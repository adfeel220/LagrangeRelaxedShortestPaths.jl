
"""
    DynamicDimensionGridArray{T}
A dimension free array in which overindexing is allowed. No out of bound error for this array.
An special optimized version that is based on a grid and has index only 2-4.
The default value between two vertices are the euclidean distance between two vertices multiplied
by default value. The value is no smaller than the minimal value.

Implemented for fast searching and single entity modification, not suitable for vectorized computation
The indexing of this array follows the rules below:
1. If exact match of index exists, return the index
2. No exact match exists, try to find a shorter index. e.g. if right alignment, try to index `arr[1,2,3]`
while `arr[1,2,3]` doesn't exist but `arr[2,3]` exists, return `arr[2,3]`
3. If non of the degenerated indices exists, return the default euclidean distance.
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
function DynamicDimensionGridArray(
    size::NTuple{2,Int}; default::T=one(Float64), min_val::T=default
) where {T}
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
function euclidean_distance(
    arr::DynamicDimensionGridArray{T}, v1::Int, v2::Int
)::T where {T}
    coord1 = vertex_to_coordinate(arr, v1)
    coord2 = vertex_to_coordinate(arr, v2)
    return √sum((coord1 .- coord2) .^ 2)
end

"""
    getindex(DynamicDimensionGridArray, index...)
Efficient implementation of dynamic indexing by explicitly giving the size of index
utilizing multiple dispatching
"""
function Base.getindex(arr::DynamicDimensionGridArray{T}, index::NTuple{4,Int}) where {T}
    data = find_data(arr.d4, index)
    if !isnothing(data)
        return data
    end
    return arr[degenerate_tuple(index)]
end
function Base.getindex(arr::DynamicDimensionGridArray{T}, index::Vararg{Int,4}) where {T}
    return Base.getindex(arr, index)
end

function Base.getindex(arr::DynamicDimensionGridArray{T}, index::NTuple{3,Int}) where {T}
    data = find_data(arr.d3, index)
    if !isnothing(data)
        return data
    end
    return arr[degenerate_tuple(index)]
end
function Base.getindex(arr::DynamicDimensionGridArray{T}, index::Vararg{Int,3}) where {T}
    return Base.getindex(arr, index)
end

function Base.getindex(arr::DynamicDimensionGridArray{T}, index::NTuple{2,Int}) where {T}
    data = find_data(arr.d2, index)
    if !isnothing(data)
        return data
    end
    return max(arr.min_val, arr.default * euclidean_distance(arr, index[1], index[2]))
end
function Base.getindex(arr::DynamicDimensionGridArray{T}, index::Vararg{Int,2}) where {T}
    return Base.getindex(arr, index)
end

function Base.getindex(arr::DynamicDimensionGridArray{T}, index...) where {T}
    return max(
        arr.min_val, arr.default * euclidean_distance(arr, index[end - 1], index[end - 2])
    )
end

function Base.setindex!(
    arr::DynamicDimensionGridArray{T}, value::T, index::NTuple{N,Int}
) where {T,N}
    if N == 4
        set_data!(arr.d4, value, index)
    elseif N == 3
        set_data!(arr.d3, value, index)
    elseif N == 2
        set_data!(arr.d2, value, index)
    end
    return arr
end
function Base.setindex!(
    arr::DynamicDimensionGridArray{T}, value::T, index::Vararg{Int,N}
) where {T,N}
    return Base.setindex!(arr, value, index)
end

"""
    iterate(::DynamicDimensionGridArray)
Traverse the underlying data storage trees
"""
function Base.iterate(arr::DynamicDimensionGridArray{T}, i::K=1) where {T,K<:Integer}
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

"""
    empty(::DynamicDimensionGridArray; default, min_val)
Create an empty array with the same default and minimum value as the input array
"""
function empty(
    arr::DynamicDimensionGridArray{T}; default::T=arr.default, min_val::T=arr.min_val
)::DynamicDimensionGridArray{T} where {T}
    return DynamicDimensionGridArray(arr.grid_size; default=default, min_val=min_val)
end

"""
    haskey(arr::DynamicDimensionGridArray, key::NTuple)
Check if a key (index) is registered in the array
"""
function Base.haskey(arr::DynamicDimensionGridArray{T}, key::NTuple{N,T}) where {N,T}
    if N == 4
        return !isnothing(find_node(arr.d4, key))
    elseif N == 3
        return !isnothing(find_node(arr.d3, key))
    elseif N == 2
        return !isnothing(find_node(arr.d2, key))
    end
    return false
end

"""
    delete!(arr, index)
Delete the node with `index` in `arr`
"""
function delete!(arr::DynamicDimensionGridArray{T}, index::NTuple{N,Int}) where {T,N}
    if N == 4
        delete!(arr.d4, DimensionFreeData{T}(index))
    elseif N == 3
        delete!(arr.d3, DimensionFreeData{T}(index))
    elseif N == 2
        delete!(arr.d2, DimensionFreeData{T}(index))
    end
    return arr
end
