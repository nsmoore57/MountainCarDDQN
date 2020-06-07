function learn!(envir::E, qpolicy::DeepQPolicy, num_eps, discount_factor;
                update_freq=1000, chkpt_freq=3000) where {E<:AbstractEnvironment}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(1.0, qpolicy)

    # Track number of sucessful attempts
    num_successes = 0

    # Set up the networks
    primaryNetwork = qpolicy.nn
    targetNetwork = update_freq > 0 ? deepcopy(primaryNetwork) : primaryNetwork

    # Buffer to hold the replay memory

    PlotPolicy(qpolicy, 1000, 5)
    # L(x,y) = Flux.mse(primaryNetwork(x), y)
    opt = Descent(0.01)

    # Track the number of training steps completed so far
    step = 1
    Juno.@progress ["learn ep"] for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = 200)
        # ep = Episode(env, π)
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
                Flux.loadparams!(targetNetwork, Flux.params(primaryNetwork))
            end

            # If desired, save the network
            if chkpt_freq > 0 && step % chkpt_freq == 0
                @save "model_checkpoint.bson" primaryNetwork
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
