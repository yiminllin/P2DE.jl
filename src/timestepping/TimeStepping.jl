function SSP33!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)
    @unpack CFL,dt0,t0,T    = param.timestepping_param
    @unpack output_interval = param.postprocessing_param
    @unpack Farr,Larr,αarr,LGLind,L_L2G_arr,L_G2L_arr,L_Vf_arr,rhsU,resW,resZ = prealloc
    @unpack Uq = prealloc

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
        if (dt > T-t)
            dt = T-t
        end
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
        end
    end
    
    data_hist = DataHistory(Uhist,Lhist,Fhist,alphahist,thist,dthist,LGLindhist,L_L2G_hist,L_G2L_hist,L_Vf_hist)
    return data_hist
end

function init_get_rhs!(param,entropyproj_limiter_type::NoEntropyProjectionLimiter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::AdaptiveFilter,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_modal_coefficients!(prealloc,param,discrete_data_gauss)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    apply_entropyproj_filtering!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function init_get_rhs!(param,entropyproj_limiter_type::ScaledExtrapolation,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    compute_entropyproj_limiting_param!(param,discrete_data_gauss,prealloc,nstage)
    if (nstage == 1)
        update_indicator!(prealloc,param.approximation_basis_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops)
    end
end

function get_rhs!(rhs_type::LowOrderPositivity,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsL,rhsU = prealloc
    dt = rhs_pos_Gauss(param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage)
    copyto!(rhsU,rhsL)
    return dt
end

function get_rhs!(rhs_type::EntropyStable,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsU = prealloc
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage)
    copyto!(rhsU,rhsH)
    return dt
end

function get_rhs!(rhs_type::ESLimitedLowOrderPos,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack rhsH,rhsL,rhsU = prealloc
    entropy_projection!(prealloc,param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,nstage)
    dt = rhs_pos_Gauss(param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage)
    rhs_modalESDG!(prealloc,param,discrete_data_gauss,discrete_data_LGL,bcdata,nstage)
    apply_zhang_shu_limiter!(prealloc,param,dt,nstage)
    return dt
end

function rhs!(param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    init_get_rhs!(param,param.entropyproj_limiter_type,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    dt = get_rhs!(param.rhs_type,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    return dt
end 

function check_positivity(U,param,discrete_data)
    @unpack K,EQN  = param
    @unpack Nq     = discrete_data.sizes
    rhomin  = Inf
    rhoemin = Inf
    rhomax  = -Inf
    rhoemax = -Inf
    for k = 1:K
        for i = 1:Nq
            rho = U[i,k][1]
            p    = pfun(EQN,U[i,k])
            rhoe = rhoe_ufun(EQN,U[i,k])
            rhomin = min(rhomin,rho)
            rhomax = max(rhomax,rho)
            rhoemin = min(rhoemin,rhoe)
            rhoemax = max(rhoemax,rhoe)
            if (rho < 0.0 || p < 0.0)
                @show i,k,rho,p
            end
        end
    end
    # @show rhomin,rhomax,rhoemin,rhoemax
end