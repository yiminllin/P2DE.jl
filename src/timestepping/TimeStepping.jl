function SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)
    @unpack CFL,dt0,t0,T    = param.timestepping_param
    @unpack output_interval = param.postprocessing_param
    @unpack Farr,Larr,αarr,LGLind,L_L2G_arr,L_G2L_arr,L_Vf_arr,rhsU,resW,resZ = prealloc
    @unpack Uq = prealloc

    Nc = get_num_components(param.equation)
    Uhist      = []
    Lhist      = []
    Fhist      = []
    thist      = []
    dthist     = []
    alphahist  = []
    LGLindhist = []
    L_L2G_hist = []
    L_G2L_hist = []
    L_Vf_hist  = []

    # Time integration
    t = t0 

    i = 1
    dt = CFL*dt0
    @time while t < T
        dt = min(CFL*dt0,T-t)
        @. resW = Uq    # TODO: rename, resW is now the copy of previous time step Uq, and Uq is wi in paper
        dt = rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,1)
        @. Uq = resW + dt*rhsU
        rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,2)
        @. resZ = Uq+dt*rhsU
        @. Uq = 3/4*resW+1/4*resZ
        rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,3)
        @. resZ = Uq+dt*rhsU
        @. Uq = 1/3*resW+2/3*resZ

        t = t+dt
        i = i+1

        push!(thist,t)
        push!(dthist,dt)
        if (mod(i,output_interval) == 0 || abs(t-T) < 1e-10)
            push!(Uhist,copy(Uq))
            push!(Lhist,copy(Larr))
            push!(Fhist,copy(Farr))
            push!(alphahist,copy(αarr))
            push!(LGLindhist,copy(LGLind))
            push!(L_L2G_hist,copy(L_L2G_arr))
            push!(L_G2L_hist,copy(L_G2L_arr))
            push!(L_Vf_hist,copy(Farr))
            println("Current time $t with time step size $dt, and final time $T, step $i")
            total_conservation = check_conservation(prealloc,param,discrete_data_gauss,discrete_data_LGL)
            @show total_conservation
        end
    end
    
    data_hist = DataHistory{Nc}(Uhist,Lhist,Fhist,alphahist,thist,dthist,LGLindhist,L_L2G_hist,L_G2L_hist,L_Vf_hist)
    return data_hist
end