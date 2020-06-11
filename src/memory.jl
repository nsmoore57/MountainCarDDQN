using StatsBase
using DataStructures


struct Experience{T<:Real, V<:AbstractArray{T}, A, F}
    s::V
    a::A
    r::F
    s′::V
    done::Bool
end

mutable struct ReplayMemoryBuffer
    capacity::Int
    buffer::CircularBuffer{Experience}
end

# Constructor for Empty Memory
ReplayMemoryBuffer(n::Int) = ReplayMemoryBuffer(n, CircularBuffer{Experience}(n))

# Utility functions
Base.length(mem::ReplayMemoryBuffer) = length(mem.buffer)
Base.size(mem::ReplayMemoryBuffer) = length(mem)

# Memory Control
addexp!(mem::ReplayMemoryBuffer, Exp::Experience) = push!(mem.buffer, Exp)
function addexp!(mem::ReplayMemoryBuffer, s::AbstractArray{T}, a::A,
                 r::F, s′::AbstractArray{T}, d::Bool) where {T, A, F}
    addexp!(mem, Experience(s, a, r, s′, d))
end

function StatsBase.sample(mem::ReplayMemoryBuffer, num)
    ids = sample(1:length(mem), num; replace=false)
    s = hcat((mem.buffer[i].s for i in ids)...)
    r = hcat((mem.buffer[i].r for i in ids)...)
    s′ = hcat((mem.buffer[i].s′ for i in ids)...)
    d = hcat((mem.buffer[i].done for i in ids)...)

    # Actions need to be converted to Cartesian indices so that they address
    # into the correct place
    a = [CartesianIndex(0,0) for i in ids]
    for (i, idx) in enumerate(ids)
        @inbounds a[i] = CartesianIndex(mem.buffer[idx].a, i)
    end
    return (s, a, r, s′, d)
end
