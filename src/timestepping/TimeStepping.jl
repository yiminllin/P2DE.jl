function SSP33!(param, discrete_data, bcdata, prealloc, caches)
    @unpack CFL, dt0, t0, T = param.timestepping_param
    @unpack output_interval = param.postprocessing_param
    @unpack θ_arr, Larr, rhsU, resW, resZ = prealloc
    @unpack Uq = prealloc

    # TODO: very ugly hack... seems like I need to warm up the threads to avoid allocations?
    timer_dummy = TimerOutput()
    dt_dummy = dt0
    t_dummy = 0.0
    @benchmark rhs!($param, $discrete_data, $bcdata, $prealloc, $caches, $t_dummy, $dt_dummy, 1, $timer_dummy)

    Nc = get_num_components(param.equation)
    Uhist = []
    Lhist = []
    θhist = []
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
            dt = rhs!(param, discrete_data, bcdata, prealloc, caches, t, dt, 1, timer)
            @. Uq = resW + dt * rhsU
            rhs!(param, discrete_data, bcdata, prealloc, caches, t, dt, 2, timer)
            @. resZ = Uq + dt * rhsU
            @. Uq = 3 / 4 * resW + 1 / 4 * resZ
            rhs!(param, discrete_data, bcdata, prealloc, caches, t, dt, 3, timer)
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
            push!(θhist, copy(θ_arr))
            println("Current time $t with time step size $dt, and final time $T, step $i")
            flush(stdout)
            total_conservation = check_conservation(prealloc, param, discrete_data)
            @show total_conservation
        end
    end

    println(timer)

    data_hist = DataHistory{Nc}(Uhist, Lhist, θhist, thist, dthist)
    return data_hist
end
