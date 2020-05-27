module MountainCarDDQN

using Flux
using Reinforce
import Reinforce: action, reset!
using Reinforce.MountainCarEnv: MountainCar, MountainCarState

using Plots
gr()

mutable struct ϵGreedyPolicy <: AbstractPolicy
    ϵ::AbstractFloat
    greedy::AbstractPolicy
end


function action(policy::ϵGreedyPolicy, r, s, A)
    only(rand(1)) < policy.ϵ ? rand(A) : action(policy.greedy, r, s, A)
end

function reset!(policy::ϵGreedyPolicy)
    # Leave the ϵ value alone since we want it to be preserved between episodes
    reset!(policy.greedy)
end

mutable struct DeepQPolicy <: Reinforce.AbstractPolicy
    nn      # Neural Network for Deep Q function
end

function action(policy::DeepQPolicy, r, s, A)
    # s.velocity < 0 ? 1 : 3
    inputs = [s.position s.velocity]'
    argmax(policy.nn(inputs))[1]
end

function build_DeepQPolicy(env, num_actions)
    # Create a neural network
    model = Dense(nfields(env.state), num_actions, σ)

    # build the Policy
    DeepQPolicy(model)
end

_transformStateToInputs(state::MountainCarState) = [state.position; state.velocity]

function learn!(envir::E, qpolicy::DeepQPolicy, num_eps, discount_factor) where {E<:AbstractEnvironment}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(1.0, qpolicy)

    primaryNetwork = qpolicy.nn
    L(x,y) = Flux.mse(primaryNetwork(x), y)
    opt = ADAM()

    for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = 100000)
        for (s, a, r, s′) ∈ ep
            currentQ = primaryNetwork(_transformStateToInputs(s))
            target = copy(currentQ)
            target[a] = r + discount_factor*(1 - finished(env, s′))*(max(primaryNetwork(_transformStateToInputs(s′))...))

            Flux.train!(L, params(primaryNetwork), [(_transformStateToInputs(s), target)], opt)
        end
        # decrease ϵ to be more greedy as episodes increase
        π.ϵ *= .99
        println("Total Reward After Episode $i: $(ep.total_reward)")
    end
end

env = MountainCar()

function episode!(env, π = RandomPolicy())
    ep = Episode(env, π)
    for (s, a, r, s′) ∈ ep
        #gui(plot(env))
    end
    ep.total_reward, ep.niter
end

dQpolicy = build_DeepQPolicy(env, 3)

learn!(env, dQpolicy,10000,.9)

for _ ∈ 1:100
    R, n = episode!(env, dQpolicy)
    println("R: $R, iter: $n")
end
end # module
