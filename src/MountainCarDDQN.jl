module MountainCarDDQN

using Flux
using Reinforce
import Reinforce: action, reset!
using Reinforce.MountainCarEnv: MountainCar

using Plots
gr()

mutable struct ϵGreedyPolicy <: AbstractPolicy
    ϵ::AbstractFloat
    greedy::AbstractPolicy
end

function action(policy::ϵGreedyPolicy, r, s, A)
    rand(1) < policy.ϵ ? rand(A) : action(policy.greedy, r, s, A)
end

function reset!(policy::ϵGreedyPolicy)
    reset!(policy.greedy)
    # Leave the ϵ value alone since we want it to be preserved between episodes
end

mutable struct DeepQPolicy <: Reinforce.AbstractPolicy
    nn      # Neural Network for Deep Q function
end

function action(policy::DeepQPolicy, r, s, A)
    s.velocity < 0 ? 1 : 3
end

function build_DeepQPolicy(env, num_actions)
    # Create a neural network
    model = Dense(nfields(env.state), num_actions, σ)

    # build the Policy
    return DeepQPolicy(model)
end

function learn!(envir::E, qpolicy::DeepQPolicy, num_eps) where {E<:AbstractEnvironment}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(startep, qpolicy)

    for _ ∈ 1:num_eps
        ep = Episide(env, π)
        for (s, a, r, s′) ∈ ep
            # Update Q NN
        end

        # decrease ϵ to be more greedy as episodes increase
    end
end

env = MountainCar()

function episode!(env, π = RandomPolicy())
    ep = Episode(env, π)
    for (s, a, r, s′) ∈ ep
        #gui(plot(env))
        print(r)
    end
    ep.total_reward, ep.niter
end

R, n = episode!(env, DeepQPolicy())
println("reward: $R, iter: $n")
end # module
