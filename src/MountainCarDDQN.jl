module MountainCarDDQN

using Flux
using Juno
using Dates: now
using BSON: @save, @load
import BSON
using Random: seed!
import Reinforce
import Reinforce: reset!, actions, finished, step!, action, Episode
import Reinforce: AbstractPolicy, AbstractEnvironment
using LearnBase: DiscreteSet
using RecipesBase
import Distributions
using Plots
gr()

include("mountain_car.jl")
include("policies.jl")
include("learn.jl")
include("utils.jl")

# whether to load the saved model or start from scratch
const load_prior = true

# Mountain Car Envivornment
env = MountainCar()

# These are used for plotting the policy
const positions = 1.8*rand(10000)-1.2*ones(10000)
const velocitys = 0.14*rand(10000)-0.07*ones(10000)


# Either load previously saved policy or build a new one
if load_prior
    dQpolicy = DeepQPolicy(BSON.load("model_checkpoint.bson")[:primaryNetwork])
else
    dQpolicy = build_DeepQPolicy(env, 3)
end

# Other policies so we can compare them to the learned policy
handpolicy = HandPolicy()
randpolicy = Reinforce.RandomPolicy()

num_successes = learn!(env, dQpolicy, 2000, .99)
@show num_successes

handAvgReward = 0.0
randAvgReward = 0.0
learnAvgReward = 0.0

# Compare the learned policy to random policy and hand-made policy
Juno.@progress ["testing"] for i ∈ 1:100
    global handAvgReward
    global randAvgReward
    global learnAvgReward

    R, n = episode!(env, handpolicy)
    handAvgReward = i == 1 ? R : (handAvgReward*(i-1.0) + R)/float(i)

    R, n = episode!(env, randpolicy; maxn = 200)
    randAvgReward = i == 1 ? R : (randAvgReward*(i-1.0) + R)/float(i)

    R, n = episode!(env, dQpolicy; maxn = 200)
    learnAvgReward = i == 1 ? R : (learnAvgReward*(i-1.0) + R)/float(i)
end
println("hand  Avg: $handAvgReward")
println("rand  Avg: $randAvgReward")
println("learn Avg: $learnAvgReward")

# Plot the policy
PlotPolicy(handpolicy, 1000, 3)
PlotPolicy(dQpolicy, 1000)

end # module
