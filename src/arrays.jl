
"""
    DimensionFreeData{N,T}
A data of type `T` labeled by index as an `NTuple{N, Int}`.
Overloads `isless` and `==` for that the comparison is only the index.
Overloads `sorted_rank` for `AVLTree{DimensionFreeData}` to use `Tuple` or `Vararg` to index `DimensionFreeData`
"""
mutable struct DimensionFreeData{N,T}
    data::T
    index::NTuple{N,Int}

    function DimensionFreeData(data::T, index...) where {T}
        return new{length(index),T}(data, tuple(index...))
    end
end
function Base.show(io::IO, data::DimensionFreeData)
    return show(io, "DimensionFreeData: Value $(data.data) indexed by $(data.index)")
end
Base.isless(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index < d2.index
Base.:(==)(d1::DimensionFreeData, d2::DimensionFreeData) = d1.index == d2.index
Base.:(==)(d::DimensionFreeData, idx::Vararg) = d.index == idx
function sorted_rank(tree::AVLTree{DimensionFreeData}, key::NTuple{N,Int}) where {N}
    return sorted_rank(tree, DimensionFreeData(nothing, key...))
end
function sorted_rank(tree::AVLTree{DimensionFreeData}, key::Vararg{Int})
    return sorted_rank(tree, tuple(key))
end

function Base.haskey(tree::AVLTree{DimensionFreeData}, key::NTuple{N,Int}) where {N}
    return haskey(tree, DimensionFreeData(nothing, key...))
end
function delete!(tree::AVLTree{DimensionFreeData}, key::NTuple{N,Int}) where {N}
    return delete!(tree, DimensionFreeData(nothing, key...))
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
    data::AVLTree{DimensionFreeData}  # Union{DimensionFreeData{NTuple{2,T}, NTuple{3}...}
    default::T
    align_to_right::Bool
end
function Base.show(io::IO, arr::DynamicDimensionArray)
    align = arr.align_to_right ? "right" : "left"
    return show(
        io,
        """DynamicDimensionArray{$(typeof(arr.default))}: $(length(arr.data)) registered entries with default = $(arr.default). Index align to $(align).""",
    )
end

"""
    DynamicDimensionArray([default=0.0]; align_right=true)
Create an empty `DynamicDimensionArray` with a default value (`{Float64}(0.0)` if not specified) and align indices to the right.
"""
function DynamicDimensionArray(default::T=zero(Float64); align_right::Bool=true) where {T}
    tree = AVLTree{DimensionFreeData}()
    return DynamicDimensionArray{T}(tree, default, align_right)
end

"""
    DynamicDimensionArray{T}(dimension_free_data...; default, align_right)
Create an array with the predefined `DimensionFreeData`, default value (=0) and index alignment (=right) is passed by keyword arguments
"""
function DynamicDimensionArray{T}(
    data::Vararg{DimensionFreeData}; default::T=zero(T), align_right::Bool=true
) where {T}
    tree = AVLTree{DimensionFreeData}()
    for d in data
        push!(tree, d)
    end
    return DynamicDimensionArray{T}(tree, default, align_right)
end

function Base.getindex(arr::DynamicDimensionArray, index...)
    if !haskey(arr.data, index)
        for i in 1:length(index)
            degenerated_index =
                arr.align_to_right ? index[(begin + i):end] : index[begin:(end - i)]
            if haskey(arr.data, degenerated_index)
                elem_idx = sorted_rank(arr.data, degenerated_index)
                return arr.data[elem_idx].data
            end
        end
        return arr.default
    end

    elem_idx = sorted_rank(arr.data, index)
    return arr.data[elem_idx].data
end
function Base.setindex!(arr::DynamicDimensionArray, value, index...)
    if haskey(arr.data, index)
        elem_idx = sorted_rank(arr.data, index)
        arr.data[elem_idx].data = value
        return nothing
    end
    new_data = DimensionFreeData(value, index...)
    push!(arr.data, new_data)
    return nothing
end
Base.length(arr::DynamicDimensionArray) = length(arr.data)

function Base.iterate(arr::DynamicDimensionArray, i=1)
    if i > length(arr)
        return nothing
    end
    return Pair(arr.data[i].index, arr.data[i].data), i + 1
end

function delete!(arr::DynamicDimensionArray, index...)
    delete!(arr.data, index)
end