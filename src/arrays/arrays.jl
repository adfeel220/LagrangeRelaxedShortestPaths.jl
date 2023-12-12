
"""
    DimensionFreeData{N,T}
A data of type `T` labeled by index as an `NTuple{N, Int}`.
Overloads `isless` and `==` for that the comparison is only the index.
Overloads `sorted_rank` for `AVLTree{DimensionFreeData}` to use `Tuple` or `Vararg` to index `DimensionFreeData`
"""
mutable struct DimensionFreeData{T,N}
    data::T
    index::NTuple{N,Int}

    function DimensionFreeData(data::T, index::NTuple{N,Int}) where {N,T}
        return new{T,N}(data, index)
    end
end
function Base.show(io::IO, data::DimensionFreeData)
    return show(io, "DimensionFreeData: Value $(data.data) indexed by $(data.index)")
end
Base.isless(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index < d2.index
Base.:(==)(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index == d2.index
Base.:(==)(d::DimensionFreeData, idx::Vararg) = d.index == idx

"""
    find_node(tree, index)
Return node on an `AVLTree` with given index, return `nothing` if such index is not found

# Arguments
- `tree::AVLTree{DimensionFreeData{T,N}}`: AVL tree storing data by index
- `index::NTuple{N,Int}`: index of data to be retrieved
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

"""
    find_data(tree, index)
Return data value of a given index, return `nothing` if such index not found

# Arguments
- `tree::AVLTree{DimensionFreeData{T,N}}`: AVL tree storing data by index
- `index::NTuple{N,Int}`: index of data to be retrieved
"""
function find_data(tree::AVLTree{DimensionFreeData{T,N}}, index::NTuple{N,Int}) where {T,N}
    node = find_node(tree, index)
    if isnothing(node)
        return nothing
    end
    return node.data.data
end

"""
    set_data!(tree, value, index)
Set `value` binded with `index` into the tree. Overwrite value for already existing index,
create a new node otherwise.

# Arguments
- `tree::AVLTree{DimensionFreeData{T,N}}`: AVL tree storing data by index
- `value::T`: value to insert into the tree
- `index::NTuple{N,Int}`: index of data to be retrieved
"""
function set_data!(
    tree::AVLTree{DimensionFreeData{T,N}}, value::T, index::NTuple{N,Int}
) where {N,T}
    node = find_node(tree, index)
    if isnothing(node)
        push!(tree, DimensionFreeData(value, index))
    else
        node.data.data = value
    end
end

"""
    AbstractDynamicDimensionArray{T}
Supertype for dynamic dimension array with element type `T`
"""
abstract type AbstractDynamicDimensionArray{T} end

function Base.show(io::IO, arr::AbstractDynamicDimensionArray)
    return show(
        io, "DynamicDimensionArray{$(typeof(arr.default))} with default = $(arr.default)"
    )
end

"""
    degenerate_tuple(t)
Return a new tuple without the leftmost element of the original tuple in a type stable method
"""
function degenerate_tuple(t::NTuple{N,T})::NTuple{N - 1,T} where {N,T}
    return @inbounds ntuple(i -> t[i + 1], N - 1)
end