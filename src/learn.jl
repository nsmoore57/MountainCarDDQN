# TODO: use a struct or named tuple to handle all these arguments

function learn!(env::E, qpolicy::Q, num_eps, γ;
                maxn=200, update_freq=3000, chkpt_freq=3000, replay_buffer_size=10000,
                train_batch_size=1200, render=false) where {E<:AbstractEnvironment, Q<:QPolicy}
    # Build an epsilon greedy policy for the learning
    π = ϵGreedyPolicy(1.0, qpolicy)

    # Track number of sucessful attempts
    num_successes = 0

    # Replay Buffer
    mem = PriorityReplayMemoryBuffer(replay_buffer_size)

    # ADAM Optimizer
    opt = ADAM()

    # Standard Gradient Descent Optimizer
    # opt = Descent(0.001)

    # Params to optimize
    p = get_params(qpolicy)

    losses = Float64[]

    # Track the number of training steps completed so far
    step = 1
    @showprogress 3 "Learning..." for i ∈ 1:num_eps
        ep = Episode(env, π; maxn = maxn)

        for (s, a, r, s′) ∈ ep
            # Save the step into the replay buffer
            addexp!(mem, _transformStateToInputs(s), a, r, _transformStateToInputs(s′), finished(env, s′), r)

            # Fill the buffer before training or lowering ϵ
            length(mem) < train_batch_size && continue

            # Training..........

            # decrease ϵ to be 10% exploration and 90% exploitation
            # TODO: Need a better way/place to control this
            #       Could make a function called decrease_ϵ(π) which is part of the ϵ greedy policy
            π.ϵ = 0.1

            # Sample a batch from the replay buffer
            (s_batch, a_batch, r_batch, s′_batch, done_batch, ids, weights) = sample(mem, train_batch_size)

            # Get the target values based on the next states
            target = get_target(qpolicy, γ, r_batch, done_batch, s′_batch)

            # May need this to help set the data type for loss - not sure
            loss = 0.0

            # Track the td errs
            td_errs = similar(target)

            # Track gradients while calculating the loss
            gs = Flux.gradient(p) do
                currentQ_SA = get_QValues(qpolicy, s_batch)[a_batch]
                td_errs = currentQ_SA .- target
                loss = Flux.huber_loss(td_errs.*weights)
            end

            # Train the network(s)
            Flux.Optimise.update!(opt, p, gs)

            # Update Priorities for selected memory elements
            update_priorities!(mem, ids, td_errs)

            # If needed, update the target Network
            update_freq > 0 && step % update_freq == 0 && update_target(qpolicy)
            update_freq > 0 && step % update_freq == 0 && push!(losses, Flux.huber_loss(td_errs))

            # If desired, save the network
            if chkpt_freq > 0 && step % chkpt_freq == 0
                save_policy(qpolicy)
                ## Plot the policy every other save
                ## TODO: Add a function argument to handle frequency of plotting
                render && step % (chkpt_freq*2) == 0 && PlotPolicy(qpolicy, 1000, 0)
            end

            # Record if this episode was successful
            if finished(env, s′)
                num_successes += 1
            end

            step += 1
        end # end of episode
        # PlotPolicy(qpolicy, 1000, 0.001)
        # println("Total Reward After Episode $i: $(ep.total_reward)")

        # Anneal β linearly toward 1.0
        mem.β -= (1.0 - mem.β0)/num_eps
    end
    chkpt_freq > 0 && save_policy(qpolicy)
    num_successes, losses
end

