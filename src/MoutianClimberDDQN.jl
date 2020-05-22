module MoutianClimberDDQN

using Reinforce
using Reinforce.MountainCarEnv: MountainCar

using Plots
gr()

mutable struct ϵGreedyPolicy <: Reinforce.AbstractPolicy
    ϵ::AbstractFloat
    greedy::Reinforce.AbstractPolicy
end

function Reinforce.action(policy::ϵGreedyPolicy, r, s, A)
    rand(1) < policy.ϵ ? rand(A) : action(policy.greedy, r, s, A)
end

function Reinforce.reset!(policy::ϵGreedyPolicy)
    reset!(policy.greedy)
    # Leave the ϵ value alone since we want it to be preserved between episodes
end

mutable struct QLearnPolicy <: Reinforce.AbstractPolicy
    ns::Int # Dimension of State
end

function Reinforce.action(policy::QLearnPolicy, r, s, A)
    s.velocity < 0 ? 1 : 3
end

function build_QLearnPolicy(state_dim::Int)
    # Create a neural network

    # build the Policy
end

function learn!(env::Reinforce.AbstractEnvironment, qpolicy::QLearnPolicy, num_eps)
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

R, n = episode!(env, QLearnPolicy())
println("reward: $R, iter: $n")
end # module
