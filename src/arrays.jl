
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
function find_node(tree::AVLTree{DimensionFreeData}, index::NTuple{N,Int}) where {N}
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
function find_data(tree::AVLTree{DimensionFreeData}, index::NTuple{N,Int}) where {N}
    node = find_node(tree, index)
    if isnothing(node)
        return nothing
    end
    return node.data.data
end
function set_data!(
    tree::AVLTree{DimensionFreeData}, value::T, index::NTuple{N,Int}
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

Implemented for fast searching and single entity modification, not suitable for vectorized computation
The indexing of this array follows the rules below:
1. If exact match of index exists, return the index
2. No exact match exists, try to find a shorter index. e.g. if right alignment, try to index `arr[1,2,3]`
while `arr[1,2,3]` doesn't exist but `arr[2,3]` exists, return `arr[2,3]`
3. If non of the degenerated indices exists, return the default value.
"""
mutable struct DynamicDimensionArray{T}
    data::AVLTree{DimensionFreeData}
    default::T
end
function Base.show(io::IO, arr::DynamicDimensionArray)
    return show(
        io,
        """DynamicDimensionArray{$(typeof(arr.default))}: $(length(arr.data)) registered entries with default = $(arr.default).""",
    )
end

"""
    DynamicDimensionArray([default=0.0])
Create an empty `DynamicDimensionArray` with a default value (`{Float64}(0.0)` if not specified).
"""
function DynamicDimensionArray(default::T=zero(Float64)) where {T}
    tree = AVLTree{DimensionFreeData}()
    return DynamicDimensionArray{T}(tree, default)
end

"""
    DynamicDimensionArray{T}(dimension_free_data...; default)
Create an array with the predefined `DimensionFreeData`, default value (=0) is passed by keyword arguments
"""
function DynamicDimensionArray{T}(
    data::Vararg{DimensionFreeData}; default::T=zero(T)
) where {T}
    tree = AVLTree{DimensionFreeData}()
    for d in data
        push!(tree, d)
    end
    return DynamicDimensionArray{T}(tree, default)
end

function Base.getindex(arr::DynamicDimensionArray{T}, index::Vararg{Int}) where {T}
    data = find_data(arr.data, index)
    if isnothing(data)
        for i in 1:length(index)
            degenerated_index = index[(begin + i):end]
            data = find_data(arr.data, degenerated_index)
            if !isnothing(data)
                return data
            end
        end
        return arr.default
    end
    return data
end
function Base.setindex!(
    arr::DynamicDimensionArray{T}, value::T, index::Vararg{Int}
) where {T}
    set_data!(arr.data, value, index)
    return arr
end
Base.length(arr::DynamicDimensionArray) = length(arr.data)

function Base.iterate(arr::DynamicDimensionArray, i=1)
    if i > length(arr)
        return nothing
    end
    return Pair(arr.data[i].index, arr.data[i].data), i + 1
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
