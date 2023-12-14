
"""
    DynamicDimensionArray2to4{T}
A dimension free array in which overindexing is allowed. No out of bound error for this array.
An special optimized version that only allows index to have length 2-4, as generic implementation generates
instable type that slows down computation.

Implemented for fast searching and single entity modification, not suitable for vectorized computation
The indexing of this array follows the rules below:
1. If exact match of index exists, return the index
2. No exact match exists, try to find a shorter index. e.g. if right alignment, try to index `arr[1,2,3]`
while `arr[1,2,3]` doesn't exist but `arr[2,3]` exists, return `arr[2,3]`
3. If non of the degenerated indices exists, return the default value.
"""
mutable struct DynamicDimensionArray2to4{T} <: AbstractDynamicDimensionArray{T}
    d2::AVLTree{DimensionFreeData{T,2}}
    d3::AVLTree{DimensionFreeData{T,3}}
    d4::AVLTree{DimensionFreeData{T,4}}

    default::T
end

function Base.length(arr::DynamicDimensionArray2to4)
    return length(arr.d2) + length(arr.d3) + length(arr.d4)
end

function Base.show(io::IO, arr::DynamicDimensionArray2to4)
    return show(
        io,
        """DynamicDimensionArray2to4{$(typeof(arr.default))}: $(length(arr)) registered entries with default = $(arr.default).""",
    )
end

"""
    DynamicDimensionArray2to4([default=0.0])
Create an empty `DynamicDimensionArray2to4` with a default value (`{Float64}(0.0)` if not specified).
"""
function DynamicDimensionArray2to4(default::T=zero(Float64)) where {T<:Number}
    t2 = AVLTree{DimensionFreeData{T,2}}()
    t3 = AVLTree{DimensionFreeData{T,3}}()
    t4 = AVLTree{DimensionFreeData{T,4}}()
    return DynamicDimensionArray2to4{T}(t2, t3, t4, default)
end

function Base.getindex(arr::DynamicDimensionArray2to4{T}, index::Vararg{Int,4}) where {T}
    data = find_data(arr.d4, index)
    if !isnothing(data)
        return data
    end
    return arr[degenerate_tuple(index)...]
end

function Base.getindex(arr::DynamicDimensionArray2to4{T}, index::Vararg{Int,3}) where {T}
    data = find_data(arr.d3, index)
    if !isnothing(data)
        return data
    end
    return arr[degenerate_tuple(index)...]
end

function Base.getindex(arr::DynamicDimensionArray2to4{T}, index::Vararg{Int,2}) where {T}
    data = find_data(arr.d2, index)
    if !isnothing(data)
        return data
    end
    return arr.default
end

function Base.getindex(arr::DynamicDimensionArray2to4{T}, index...) where {T}
    return arr.default
end

function Base.setindex!(
    arr::DynamicDimensionArray2to4{T}, value::T, index::Vararg{Int,N}
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

function Base.iterate(arr::DynamicDimensionArray2to4{T}, i=1) where {T}
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
    empty(arr::DynamicDimensionArray)
Create an empty array with the same default value as the input array
"""
function empty(arr::DynamicDimensionArray2to4{T}) where {T}
    return DynamicDimensionArray2to4(arr.default)
end

"""
    delete!(arr, index)
Delete the node with `index` in `arr`
"""
function delete!(arr::DynamicDimensionArray2to4{T}, index::NTuple{N,Int}) where {T,N}
    if length(index) == 4
        delete!(arr.d4, DimensionFreeData{T}(index))
    elseif length(index) == 3
        delete!(arr.d3, DimensionFreeData{T}(index))
    elseif length(index) == 2
        delete!(arr.d2, DimensionFreeData{T}(index))
    end
    return arr
end
