function learn!(env::E, qpolicy::QPolicy, num_eps, discount_factor;
                maxn=200, update_freq=1000, chkpt_freq=3000, replay_buffer_size=100,
                train_batch_size=64, render=false) where {E<:AbstractEnvironment}
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
        ep = Episode(env, π; maxn = maxn)
        # ep = Episode(env, π)
        for (s, a, r, s′) ∈ ep
            # Save the step into the replay buffer
            addexp!(mem, _transformStateToInputs(s), a, r, _transformStateToInputs(s′), finished(env, s′))

            # Fill the buffer before training or lowering ϵ
            length(mem) < replay_buffer_size && continue

            # Training..........

            # decrease ϵ to be 10% exploration and 90% exploitation
            # TODO: Need a better way/place to control this
            #       Could make a function called decrease_ϵ(π) which is part of the ϵ greedy policy
            π.ϵ = 0.1

            # Sample a batch from the replay buffer
            (s_batch, a_batch, r_batch, s′_batch, done_batch) = sample(mem, train_batch_size)

            # Get the target values based on the next states
            target = get_target(qpolicy, discount_factor, r_batch, done_batch, s′_batch)

            # May need this to help set the data type for loss - not sure
            loss = 0.0

            # Track gradients while calculating the loss
            gs = Flux.gradient(p) do
                currentQ_SA = get_QValues(qpolicy, s_batch)[a_batch]
                loss = Flux.mse(currentQ_SA, target)
            end
            # Train the network(s)
            Flux.Optimise.update!(opt, p, gs)

            # If needed, update the target Network
            update_freq > 0 && step % update_freq == 0 && update_target(qpolicy)

            # If desired, save the network
            if chkpt_freq > 0 && step % chkpt_freq == 0
                save_policy(qpolicy)
                ## Plot the policy every other save
                render && step % (chkpt_freq*2) == 0 && PlotPolicy(qpolicy, 1000, 0)
            end

            # Record if this episode was successful
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
