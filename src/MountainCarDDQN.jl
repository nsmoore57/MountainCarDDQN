using Flux
using Flux.NNlib
using BSON: @save, @load
import BSON
import Reinforce
import Reinforce: reset!, actions, finished, step!, action, Episode
import Reinforce: AbstractPolicy, AbstractEnvironment
using LearnBase: DiscreteSet
using Distributions
using RecipesBase
using ProgressMeter
using MooreDDQN
using Plots
gr()

# Right now there is a dependence in the order of loading for the first two files
# maybe re-structure to avoid this dependence
include("mountain_car.jl")
include("policies.jl")
include("utils.jl")

# whether to load the saved model or start from scratch
const load_prior = false

# Mountain Car Envivornment
env = MountainCar()

# These are used for plotting the policy
const positions = 1.8*rand(10000)-1.2*ones(10000)
const velocitys = 0.14*rand(10000)-0.07*ones(10000)


# Either load previously saved policy or build a new one
if load_prior
    dQpolicy = load_policy()
else
    dQpolicy = build_DeepQPolicy(env, 3, double=true, dueling=true)
end

# Other policies so we can compare them to the learned policy
handpolicy = HandPolicy()
randpolicy = Reinforce.RandomPolicy()

# Plot the policy before we get started
# PlotPolicy(dQpolicy, 1000, 3)

learnArgsList = Vector{Tuple{Symbol, Any}}()
push!(learnArgsList, (:maxn,               200))
push!(learnArgsList, (:update_freq,        100))
push!(learnArgsList, (:chkpt_freq,         0))
push!(learnArgsList, (:replay_buffer_size, 10000))
push!(learnArgsList, (:train_batch_size,   32))
push!(learnArgsList, (:render,             false))
push!(learnArgsList, (:plot_freq,          0))

learnArgs = (; learnArgsList...)

num_successes, losses = learn!(env, dQpolicy, 50, .99; learnArgs...)
# num_successes = learn!(env, dQpolicy, 100, .99)
@show num_successes
@show losses

handAvgReward = 0.0
handsuccesses = 0
randAvgReward = 0.0
randsuccesses = 0
learnAvgReward = 0.0
learnsuccesses = 0

# Compare the learned policy to random policy and hand-made policy
@showprogress 3 "Testing..." for i ∈ 1:100
    global handAvgReward
    global randAvgReward
    global learnAvgReward
    global handsuccesses
    global randsuccesses
    global learnsuccesses

    R, n = episode!(env, handpolicy)
    handAvgReward = i == 1 ? R : (handAvgReward*(i-1.0) + R)/float(i)
    (n < 200) && (handsuccesses += 1)

    R, n = episode!(env, randpolicy; maxn = 200)
    randAvgReward = i == 1 ? R : (randAvgReward*(i-1.0) + R)/float(i)
    (n < 200) && (randsuccesses += 1)

    R, n = episode!(env, dQpolicy; maxn = 200)
    learnAvgReward = i == 1 ? R : (learnAvgReward*(i-1.0) + R)/float(i)
    (n < 200) && (learnsuccesses += 1)
end
println("hand  Avg: $handAvgReward with $handsuccesses successes")
println("rand  Avg: $randAvgReward with $randsuccesses successes")
println("learn Avg: $learnAvgReward with $learnsuccesses successes")

# Plot the policy
# PlotPolicy(handpolicy, 1000, 3)
# PlotPolicy(dQpolicy, 10000)
