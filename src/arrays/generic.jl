
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
mutable struct DynamicDimensionArray{T} <: AbstractDynamicDimensionArray{T}
    data::AVLTree{DimensionFreeData}
    default::T
end
function Base.show(io::IO, arr::DynamicDimensionArray)
    return show(
        io,
        """DynamicDimensionArray{$(typeof(arr.default))}: $(length(arr.data)) registered entries with default = $(arr.default).""",
    )
end
Base.length(arr::DynamicDimensionArray) = length(arr.data)

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

"""
    find_node(tree, index)
Return node on an `AVLTree` with given index, return `nothing` if such index is not found

# Arguments
- `tree::AVLTree{DimensionFreeData}`: AVL tree storing data by index
- `index`: index of data to be retrieved
"""
function find_node(tree::AVLTree{DimensionFreeData}, index)
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

"""
    find_data(tree, index)
Return data value of a given index, return `nothing` if such index not found

# Arguments
- `tree::AVLTree{DimensionFreeData}`: AVL tree storing data by index
- `index`: index of data to be retrieved
"""
function find_data(tree::AVLTree{DimensionFreeData}, index)
    node = find_node(tree, index)
    if isnothing(node)
        return nothing
    end
    return node.data.data
end

# FIXME: unable to convert a concrete type to an non-concrete type
"""
    set_data!(tree, value, index)
Set `value` binded with `index` into the tree. Overwrite value for already existing index,
create a new node otherwise.

# Arguments
- `tree::AVLTree{DimensionFreeData}`: AVL tree storing data by index
- `value::T`: value to insert into the tree
- `index`: index of data to be retrieved
"""
function set_data!(tree::AVLTree{DimensionFreeData}, value, index)
    node = find_node(tree, index)
    if isnothing(node)
        push!(tree, DimensionFreeData(value, index))
    else
        node.data.data = value
    end
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
function Base.setindex!(arr::DynamicDimensionArray{T}, value, index...) where {T}
    set_data!(arr.data, value, index)
    return arr
end

function Base.iterate(arr::DynamicDimensionArray, i::K=1) where {K<:Integer}
    if i > length(arr)
        return nothing
    end
    return Pair(arr.data[i].index, arr.data[i].data), i + 1
end

function empty(arr::DynamicDimensionArray{T}) where {T}
    return DynamicDimensionArray(arr.default)
end

"""
    delete!(arr, index)
Delete the node with `index` in `arr`
"""
function delete!(arr::DynamicDimensionArray{T}, index::NTuple{N,Int}) where {T,N}
    if N == 4
        delete!(arr.d4, DimensionFreeData{T}(index))
    elseif N == 3
        delete!(arr.d3, DimensionFreeData{T}(index))
    elseif N == 2
        delete!(arr.d2, DimensionFreeData{T}(index))
    end
    return arr
end
