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

mutable struct DeepQPolicy <: AbstractPolicy
    nn      # Neural Network for Deep Q function
end

mutable struct HandPolicy <: AbstractPolicy end

action(policy::HandPolicy, r, s, A) = s.velocity < 0 ? 1 : 3

function action(policy::DeepQPolicy, r, s, A)
    # s.velocity < 0 ? 1 : 3
    inputs = [s.position s.velocity]'
    argmax(policy.nn(inputs))[1]
end

function build_DeepQPolicy(env, num_actions)
    # Create a neural network
    # model = Dense(nfields(env.state), num_actions, σ)
    model = Chain(Dense(nfields(env.state), 100, σ),
                  Dense(100, num_actions, σ))

    # build the Policy
    DeepQPolicy(model)
end
