using StatsBase
using DataStructures


struct Experience{T<:Real, V<:AbstractArray{T}, A, F}
    s::V
    a::A
    r::F
    s′::V
    done::Bool
end

mutable struct PriorityReplayMemoryBuffer{T}
    capacity::Int
    experience::CircularBuffer{Experience}
    priorities::CircularBuffer{T}
    α
    β
    ϵ
    β0
end

# Constructor for Empty Memory
PriorityReplayMemoryBuffer(n::Int; α=0.6, β=0.4, ϵ=1f-3) = PriorityReplayMemoryBuffer(n, CircularBuffer{Experience}(n), CircularBuffer{Float32}(n), α, β, ϵ, β)

# Utility functions
Base.length(mem::PriorityReplayMemoryBuffer) = length(mem.experience)
Base.size(mem::PriorityReplayMemoryBuffer) = length(mem)

# Memory Control
function addexp!(mem::PriorityReplayMemoryBuffer, Exp::Experience, reward=0.0)
    push!(mem.experience, Exp)
    push!(mem.priorities, abs.(reward))
end
function addexp!(mem::PriorityReplayMemoryBuffer, s::AbstractArray{T}, a::A,
                 r::F, s′::AbstractArray{T}, d::Bool, reward=0.0) where {T, A, F}
    addexp!(mem, Experience(s, a, convert(Float32, r), s′, d), abs.(reward))
end

# Update priorities for selected indicies - updated while training
function update_priorities!(mem::PriorityReplayMemoryBuffer, ids::Vector{T}, td_errs::V ) where {T<:Int, V <: AbstractArray}
    mem.priorities[ids] = (abs.(td_errs) .+ mem.ϵ).^mem.α
end

# Sample from the buffer
function StatsBase.sample(mem::PriorityReplayMemoryBuffer, num)
    ids = sample(1:length(mem), Weights(mem.priorities) , num, replace=false)
    s = hcat((mem.experience[i].s for i in ids)...)
    r = hcat((mem.experience[i].r for i in ids)...)
    s′ = hcat((mem.experience[i].s′ for i in ids)...)
    d = hcat((mem.experience[i].done for i in ids)...)
    weights = mem.priorities[ids] ./ sum(mem.priorities)
    weights = (length(mem)*weights).^(-mem.β)
    weights = weights ./ maximum(weights)

    # Actions need to be converted to Cartesian indices so that they address
    # into the correct place
    a = [CartesianIndex(0,0) for i in ids]
    for (i, idx) in enumerate(ids)
        @inbounds a[i] = CartesianIndex(mem.experience[idx].a, i)
    end
    return (s, a, r, s′, d, ids, weights)
end
