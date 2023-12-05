
"""
    DimensionFreeData{N,T}
A data of type `T` labeled by index as an `NTuple{N, Int}`.
Overloads `isless` and `==` for that the comparison is only the index.
Overloads `sorted_rank` for `AVLTree{DimensionFreeData}` to use `Tuple` or `Vararg` to index `DimensionFreeData`
"""
mutable struct DimensionFreeData{T,N}
    data::T
    index::NTuple{N,Int}
end
function DimensionFreeData{T}(index::NTuple{N,Int}) where {N,T}
    return DimensionFreeData{T,N}(zero(T), index)
end
function Base.show(io::IO, data::DimensionFreeData)
    return show(io, "DimensionFreeData: Value $(data.data) indexed by $(data.index)")
end
Base.isless(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index < d2.index
Base.:(==)(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index == d2.index
Base.:(==)(d::DimensionFreeData, idx::Vararg) = d.index == idx

"""
    find_node(tree, index)
Find node on an AVLTree storing DimensionFreeData with given index
"""
function find_node(tree::AVLTree{DimensionFreeData{T,N}}, index::NTuple{N,Int}) where {T,N}
    prev = nothing
    node = tree.root
    while !isnothing(node) && node.data.index != index
        prev = node
        if index < node.data.index
            node = node.leftChild
        else
            node = node.rightChild
        end
    end

    if !isnothing(node)
        return node
    end
    return nothing
end
function find_data(tree::AVLTree{DimensionFreeData{T,N}}, index::NTuple{N,Int}) where {T,N}
    node = find_node(tree, index)
    if isnothing(node)
        return nothing
    end
    return node.data.data
end
function set_data!(
    tree::AVLTree{DimensionFreeData{T,N}}, value::T, index::NTuple{N,Int}
) where {N,T}
    node = find_node(tree, index)
    if isnothing(node)
        insert!(tree, DimensionFreeData{T,N}(value, index))
    else
        node.data.data = value
    end
end

"""
    DynamicDimensionArray{T}
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
mutable struct DynamicDimensionArray{T}
    d2::AVLTree{DimensionFreeData{T,2}}
    d3::AVLTree{DimensionFreeData{T,3}}
    d4::AVLTree{DimensionFreeData{T,4}}

    default::T
end

Base.length(arr::DynamicDimensionArray) = length(arr.d2) + length(arr.d3) + length(arr.d4)

function Base.show(io::IO, arr::DynamicDimensionArray)
    return show(
        io,
        """DynamicDimensionArray{$(typeof(arr.default))}: $(length(arr)) registered entries with default = $(arr.default).""",
    )
end

"""
    DynamicDimensionArray([default=0.0])
Create an empty `DynamicDimensionArray` with a default value (`{Float64}(0.0)` if not specified).
"""
function DynamicDimensionArray(default::T=zero(Float64)) where {T}
    t2 = AVLTree{DimensionFreeData{T,2}}()
    t3 = AVLTree{DimensionFreeData{T,3}}()
    t4 = AVLTree{DimensionFreeData{T,4}}()
    return DynamicDimensionArray{T}(t2, t3, t4, default)
end

function Base.getindex(arr::DynamicDimensionArray{T}, index::Vararg{Int}) where {T}
    start_idx = 1
    if length(index) > 4
        start_idx = length(index) - 3
    end
    if length(index) == 4
        data = find_data(arr.d4, index)
        if !isnothing(data)
            return data
        end
        start_idx += 1
    end
    if length(index) == 3
        data = find_data(arr.d3, index)
        if !isnothing(data)
            return data
        end
        start_idx += 1
    end
    if length(index) == 2
        data = find_data(arr.d2, index)
        if !isnothing(data)
            return data
        end
    end
    return arr.default
end
function Base.setindex!(
    arr::DynamicDimensionArray{T}, value::T, index::Vararg{Int}
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

function Base.iterate(arr::DynamicDimensionArray{T}, i=1) where {T}
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

function delete!(arr::DynamicDimensionArray{T}, index::NTuple{N,Int}) where {T,N}
    if length(index) == 4
        delete!(arr.d4, DimensionFreeData{T}(index))
    elseif length(index) == 3
        delete!(arr.d3, DimensionFreeData{T}(index))
    elseif length(index) == 2
        delete!(arr.d2, DimensionFreeData{T}(index))
    end
    return arr
end
