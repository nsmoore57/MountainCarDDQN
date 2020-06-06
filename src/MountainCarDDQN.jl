module MountainCarDDQN

using Flux
using Reinforce
using Juno
import Reinforce: action, reset!

include("mountain_car.jl")

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

mutable struct HandPolicy <: Reinforce.AbstractPolicy end

action(policy::HandPolicy, r, s, A) = s.velocity < 0 ? 1 : 3

function action(policy::DeepQPolicy, r, s, A)
    # s.velocity < 0 ? 1 : 3
    inputs = [s.position s.velocity]'
    argmax(policy.nn(inputs))[1]
end

function build_DeepQPolicy(env, num_actions)
    # Create a neural network
    # model = Dense(nfields(env.state), num_actions, σ)
    model = Chain(Dense(nfields(env.state), 10, σ),
                  Dense(10, num_actions, σ))

    # build the Policy
    DeepQPolicy(model)
end

_transformStateToInputs(state::MountainCarState) = [state.position; state.velocity]

function learn!(envir::E, qpolicy::DeepQPolicy, num_eps, discount_factor;
                update_freq=1000) where {E<:AbstractEnvironment}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(1.0, qpolicy)

    # Track number of sucessful attempts
    num_successes = 0

    primaryNetwork = qpolicy.nn
    if update_freq > 0
        targetNetwork = deepcopy(primaryNetwork)
    else
        targetNetwork = primaryNetwork
    end

    PlotPolicy(qpolicy, 1000, 5)
    # L(x,y) = Flux.mse(primaryNetwork(x), y)
    opt = Descent(0.01)

    Juno.@progress ["learn ep"] for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = 200)
        # ep = Episode(env, π)
        step = 1
        for (s, a, r, s′) ∈ ep
            inputs = _transformStateToInputs(s)
            target = float(r) .+ discount_factor*dropdims(maximum(targetNetwork(_transformStateToInputs(s′)); dims=1); dims=1)
            p = Flux.params(primaryNetwork)

            gs = Flux.gradient(p) do
                currentQ = primaryNetwork(inputs)
                currentQ_SA = currentQ[a]
                loss = 0.5*(currentQ_SA .- target)^2
            end
            Flux.Optimise.update!(opt, p, gs)
            # println(0.5*(primaryNetwork(inputs)[a] - target)^2)

            # If needed, update the target Network
            if update_freq > 0 && step % update_freq == 0
                Flux.loadparams!(targetNetwork, params(primaryNetwork))
            end

            # decrease ϵ to be more greedy as episodes increase
            π.ϵ -= 1.01/(num_eps*200)

            # Flux.train!(L, Flux.params(primaryNetwork), [(_transformStateToInputs(s), target)], opt)
            if finished(env, s′)
                num_successes += 1
            end

            step += 1
        end
        # PlotPolicy(qpolicy, 1000, 0.001)
        # println("Total Reward After Episode $i: $(ep.total_reward)")
    end
    num_successes
end

env = MountainCar()

function episode!(env, π = RandomPolicy(); maxn=0, render=false)
    ep = Episode(env, π; maxn=maxn)
    for (s, a, r, s′) ∈ ep
        if render
            gui(plot(env))
            sleep(0.01)
        end
    end
    ep.total_reward, ep.niter
end

positions = 1.8*rand(10000)-1.2*ones(10000)
velocitys = 0.14*rand(10000)-0.07*ones(10000)

function PlotPolicy(policy, N::Int, sleeptime=0.0)
    pos_back = Array{Float64, 1}()
    vel_back = Array{Float64, 1}()
    pos_forward = Array{Float64, 1}()
    vel_forward = Array{Float64, 1}()
    pos_none = Array{Float64, 1}()
    vel_none = Array{Float64, 1}()
    action_space = actions(MountainCar(), MountainCarState(0,0))
    for i ∈ 1:N
        a = action(policy, 0, MountainCarState(positions[i], velocitys[i]),action_space)
        if a == 1
            append!(pos_back, positions[i])
            append!(vel_back, velocitys[i])
        elseif a == 2
            append!(pos_none, positions[i])
            append!(vel_none, velocitys[i])
        else
            append!(pos_forward, positions[i])
            append!(vel_forward, velocitys[i])
        end
    end
    scatter(pos_back, vel_back);
    scatter!(pos_none, vel_none);
    display(scatter!(pos_forward, vel_forward))
    sleeptime != 0 && sleep(sleeptime)
end

dQpolicy = build_DeepQPolicy(env, 3)
handpolicy = HandPolicy()
randpolicy = RandomPolicy()

num_successes = learn!(env, dQpolicy, 20000, .9)
@show num_successes

handAvgReward = 0.0
randAvgReward = 0.0
learnAvgReward = 0.0

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
