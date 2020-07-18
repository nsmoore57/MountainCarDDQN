function action(policy::MooreDDQN.DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(policy.nn(inputs))[1]
end

function action(policy::MooreDDQN.Double_DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(policy.primaryNetwork(inputs))[1]
end

function action(policy::MooreDDQN.DuelingDouble_DeepQPolicy, r, s::MountainCarState, A)
    inputs = [s.position s.velocity]'
    argmax(get_QValues(policy,inputs))[1]
end

# HandPolicy -------------------------------------------------------
mutable struct HandPolicy <: AbstractPolicy end

action(policy::HandPolicy, r, s::MountainCarState, A) = s.velocity < 0 ? 1 : 3
