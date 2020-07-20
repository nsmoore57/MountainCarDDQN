# HandPolicy -------------------------------------------------------
mutable struct HandPolicy <: AbstractPolicy end

action(policy::HandPolicy, r, s::MountainCarState, A) = s.velocity < 0 ? 1 : 3
