function learn!(envir::E, qpolicy::QPolicy, num_eps, discount_factor;
                update_freq=1000, chkpt_freq=3000, replay_buffer_size=100,
                train_batch_size=64) where {E<:AbstractEnvironment}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(1.0, qpolicy)

    # Track number of sucessful attempts
    num_successes = 0

    # Replay Buffer
    # TODO Add keyword variable instead of hardcoded capacity
    mem = ReplayMemoryBuffer(replay_buffer_size)

    # L(x,y) = Flux.mse(primaryNetwork(x), y)
    opt = ADAM()

    # Params to optimize
    p = get_params(qpolicy)

    # Track the number of training steps completed so far
    step = 1
    @showprogress 3 "Learning..." for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = 300)
        # ep = Episode(env, π)
        for (s, a, r, s′) ∈ ep
            # Save the step into the replay buffer
            addexp!(mem, _transformStateToInputs(s), a, r, _transformStateToInputs(s′), finished(env, s′))

            # Fill the buffer before training or lowering ϵ
            length(mem) < replay_buffer_size && continue

            # Training..........
            (s_batch, a_batch, r_batch, s′_batch, done_batch) = sample(mem, train_batch_size)

            target = get_target(qpolicy, discount_factor, r_batch, done_batch, s′_batch)

            gs = Flux.gradient(p) do
                currentQ_SA = get_QValues(qpolicy, s_batch)[a_batch]
                loss = Flux.mse(currentQ_SA, target)
            end
            Flux.Optimise.update!(opt, p, gs)

            # If needed, update the target Network
            update_freq > 0 && step % update_freq == 0 && update_target(qpolicy)

            # If desired, save the network
            if chkpt_freq > 0 && step % chkpt_freq == 0
                save_policy(qpolicy)
                if step % chkpt_freq*2 == 0
                    PlotPolicy(qpolicy, 1000, 0)
                end
            end

            # decrease ϵ to be more greedy as episodes increase
            # TODO: Need a better way/place to control this
            #       Could make a function called decrease_ϵ(π) which is part of the ϵ greedy policy
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
    save_policy(qpolicy)
    num_successes
end
