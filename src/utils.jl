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
    png("/home/nicholas/Dropbox/Policy.png")
    sleeptime != 0 && sleep(sleeptime)
end

function episode!(env, π = RandomPolicy(); maxn=0, render=false)
    ep = Episode(env, π; maxn=maxn)
    for (s, a, r, s′) ∈ ep
        if render
            gui(plot(env))
            @show s
            @show a
            @show s′
            sleep(0.01)
        end
    end
    ep.total_reward, ep.niter
end

