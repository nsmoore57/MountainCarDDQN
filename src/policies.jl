# Abstract QLearning Policy
abstract type QPolicy <: AbstractPolicy end
save_policy(policy::QPolicy, filename="model_checkpoint.bson") = BSON.bson(filename, Dict(:policy => policy))
load_policy(filename="model_checkpoint.bson") = BSON.load(filename)[:policy]

# \epsilonGreedyPolicy ----------------------------------------------
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


# DeepQPolicy ------------------------------------------------------
mutable struct DeepQPolicy <: QPolicy
    primaryNetwork      # Neural Network for Deep Q function
end

function action(policy::DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(policy.nn(inputs))[1]
end

get_QValues(policy::DeepQPolicy, inputs) = policy.primaryNetwork(inputs)
get_target(policy::DeepQPolicy, discount_factor, r, done, s′) = dropdims(r .+ discount_factor.*(1.0 .- done).*maximum(policy.primaryNetwork(s′); dims=1); dims=1)
get_params(policy::DeepQPolicy) = Flux.params(policy.primaryNetwork)


# Double_DeepQPolicy ------------------------------------------------------
mutable struct Double_DeepQPolicy <: QPolicy
    primaryNetwork      # Neural Network for Deep Q function
    targetNetwork       # Holds target values - call update_target to copy into this
end
Double_DeepQPolicy(primary) = Double_DeepQPolicy(primary, deepcopy(primary))

function action(policy::Double_DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(policy.primaryNetwork(inputs))[1]
end

get_QValues(policy::Double_DeepQPolicy, inputs) = policy.primaryNetwork(inputs)
get_target(policy::Double_DeepQPolicy, discount_factor, r, done, s′) = dropdims(r .+ discount_factor.*(1.0 .- done).*maximum(policy.targetNetwork(s′); dims=1); dims=1)
get_params(policy::Double_DeepQPolicy) = Flux.params(policy.primaryNetwork)

update_target(policy::Double_DeepQPolicy) = Flux.loadparams!(policy.targetNetwork, Flux.params(policy.primaryNetwork))


# DuelingDouble_DeepQPolicy ------------------------------------------------------
mutable struct DuelingDouble_DeepQPolicy <: QPolicy
    primaryVNetwork      # Neural Network for Value function
    primaryANetwork      # Neural Network for Advantage function
    targetVNetwork       # Holds target values - call update_target to copy into this
    targetANetwork       # Holds target values - call update_target to copy into this
end
DuelingDouble_DeepQPolicy(primaryV, primaryA) = DuelingDouble_DeepQPolicy(primaryV, primaryA, deepcopy(primaryV), deepcopy(primaryA))

function action(policy::DuelingDouble_DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(get_QValues(policy,inputs))[1]
end

get_QValues(policy::DuelingDouble_DeepQPolicy, inputs) = policy.primaryVNetwork(inputs) .+ policy.primaryANetwork(inputs) .- mean(policy.primaryANetwork(inputs), dims=1)
get_target(policy::DuelingDouble_DeepQPolicy, discount_factor, r, done, s′) = dropdims(r .+ discount_factor.*(1.0 .- done).*maximum(get_QValues(policy,s′); dims=1); dims=1)
get_params(policy::DuelingDouble_DeepQPolicy) = Flux.params(policy.primaryVNetwork, policy.primaryANetwork)

function update_target(policy::DuelingDouble_DeepQPolicy)
    Flux.loadparams!(policy.targetVNetwork, Flux.params(policy.primaryVNetwork))
    Flux.loadparams!(policy.targetANetwork, Flux.params(policy.primaryANetwork))
end

# HandPolicy -------------------------------------------------------
mutable struct HandPolicy <: AbstractPolicy end

action(policy::HandPolicy, r, s::MountainCarState, A) = s.velocity < 0 ? 1 : 3
