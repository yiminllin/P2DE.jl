function SSP33!(state, solver, state_param)
    (; CFL, dt0, t0, T) = solver.param.timestepping_param
    (; output_interval) = solver.param.postprocessing_param
    (; theta_arr, Larr, rhsU, resW, resZ, Uq) = state.preallocation

    # TODO: very ugly hack... seems like I need to warm up the threads to avoid allocations?
    timer_dummy = TimerOutput()
    dt_dummy = dt0
    t_dummy = 0.0
    timer_dummy = TimerOutput()
    time_param = TimeParam(t=t_dummy, dt=dt_dummy, nstage=1, timer=timer_dummy)
    @benchmark rhs!($state, $solver, $state_param, $time_param)

    Nc = num_components(solver)
    Uhist = []
    Lhist = []
    thetahist = []
    thist = []
    dthist = []

    timer = TimerOutput()

    # Time integration
    t = t0

    i = 1
    dt = CFL * dt0
    @time while t < T
        @timeit_debug timer "SSP stages" begin
            dt = min(CFL * dt0, T - t)
            @. resW = Uq    # TODO: rename, resW is now the copy of previous time step Uq, and Uq is wi in paper
            dt = rhs!(state, solver, state_param, TimeParam(t=t, dt=dt, nstage=1, timer=timer))
            @. Uq = resW + dt * rhsU
            rhs!(state, solver, state_param, TimeParam(t=t, dt=dt, nstage=2, timer=timer))
            @. resZ = Uq + dt * rhsU
            @. Uq = 3 / 4 * resW + 1 / 4 * resZ
            rhs!(state, solver, state_param, TimeParam(t=t, dt=dt, nstage=3, timer=timer))
            @. resZ = Uq + dt * rhsU
            @. Uq = 1 / 3 * resW + 2 / 3 * resZ
        end

        t = t + dt
        i = i + 1

        push!(dthist, dt)
        if (mod(i, output_interval) == 0 || abs(t - T) < 1e-10)
            push!(thist, t)
            push!(Uhist, copy(Uq))
            push!(Lhist, copy(Larr))
            push!(thetahist, copy(theta_arr))
            println("Current time $t with time step size $dt, and final time $T, step $i")
            flush(stdout)
            total_conservation = check_conservation(state, solver)
            @show total_conservation
        end
    end

    println(timer)

    data_hist = DataHistory{Nc}(Uhist, Lhist, thetahist, thist, dthist)
    return data_hist
end
