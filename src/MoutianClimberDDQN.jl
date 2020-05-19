module MoutianClimberDDQN

using Reinforce
using Reinforce.MountainCarEnv: MountainCar

using Plots
gr()

mutable struct BasicCarPolicy <: Reinforce.AbstractPolicy end

Reinforce.action(policy::BasicCarPolicy, r, s, A) = s.velocity < 0 ? 1 : 3

env = MountainCar()

function episode!(env, π = RandomPolicy())
    ep = Episode(env, π)
    for (s, a, r, s′) in ep
        #gui(plot(env))
        print(r)
    end
    ep.total_reward, ep.niter
end

R, n = episode!(env, BasicCarPolicy())
println("reward: $R, iter: $n")
end # module
