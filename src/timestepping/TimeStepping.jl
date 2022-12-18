mutable struct DataHistory 
    Uhist     ::Vector{Array{SVector{3,Float64},2}}
    Lhist     ::Vector{Array{Float64,2}}
    Fhist     ::Vector{Array{Float64,2}}
    alphahist ::Vector{Array{Float64,2}}
    thist     ::Vector{Float64}
    dthist    ::Vector{Float64}
    LGLindhist::Vector{BitArray}
    L_L2G_hist::Vector{Array{Float64,2}}
    L_G2L_hist::Vector{Array{Float64,2}}
    L_Vf_hist ::Vector{Array{Float64,2}}
end

function get_rhs(U,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,nstage)
    @unpack RHSTYPE,K,MODIFYVF,FILTERTYPE = param
    @unpack Nfp                           = discrete_data_gauss.sizes
    @unpack rhsH,rhsL,rhsU                = prealloc
    if (MODIFYVF)
        limit_Vf!(U,param,discrete_data_gauss,prealloc,nstage)
    end
    if (FILTERTYPE != 0)
        apply_filter!(U,param,discrete_data_gauss,prealloc,nstage)
    end
    if (nstage == 1)
        update_indicator(U,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,prealloc)
    end
    if (RHSTYPE == 0)
        dt = rhs_pos_Gauss(U,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage)
        copyto!(rhsU,rhsL)
    elseif (RHSTYPE == 1)
        rhs_modalESDG(U,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,nstage)
        copyto!(rhsU,rhsH)
    else
        entropy_projection(U,param,discrete_data_gauss,discrete_data_LGL,prealloc,nstage)
        dt = rhs_pos_Gauss(U,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,t,dt,nstage,false)
        rhs_modalESDG(U,param,discrete_data_gauss,discrete_data_LGL,bcdata,prealloc,nstage,false)
        zhang_shu_limiter(U,param,discrete_data_gauss,prealloc,dt,nstage)
    end
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
            p    = pfun_1D(EQN,U[i,k])
            rhoe = rhoe_ufun_1D(EQN,U[i,k])
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

function SSP33!(U,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc)
    @unpack K,CFL,dt0,t0,T,OUTPUT_INTERVAL,FILTERTYPE,MODIFYVF                = param
    @unpack Nq,Nfp                                                            = discrete_data_gauss.sizes
    @unpack Farr,Larr,αarr,LGLind,L_L2G_arr,L_G2L_arr,L_Vf_arr,rhsU,resW,resZ = prealloc

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
        dt = get_rhs(U,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,1)
        @. resW = U + dt*rhsU
        get_rhs(resW,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,2)
        @. resZ = resW+dt*rhsU
        @. resW = 3/4*U+1/4*resZ
        get_rhs(resW,param,discrete_data_gauss,discrete_data_LGL,transfer_ops,bcdata,prealloc,t,dt,3)
        @. resZ = resW+dt*rhsU
        @. U = 1/3*U+2/3*resZ

        t = t+dt
        i = i+1
        
        push!(thist,t)
        push!(dthist,dt)
        if (mod(i,OUTPUT_INTERVAL) == 0 || abs(t-T) < 1e-10)
            push!(Uhist,copy(U))
            push!(Lhist,copy(Larr))
            push!(Fhist,copy(Farr))
            push!(alphahist,copy(αarr))
            push!(LGLindhist,copy(LGLind))
            push!(L_L2G_hist,copy(L_L2G_arr))
            push!(L_G2L_hist,copy(L_G2L_arr))
            push!(L_Vf_hist,copy(L_Vf_arr))
            println("Current time $t with time step size $dt, and final time $T, step $i")
        end
    end
    
    data_hist = DataHistory(Uhist,Lhist,Fhist,alphahist,thist,dthist,LGLindhist,L_L2G_hist,L_G2L_hist,L_Vf_hist)
    return data_hist
end

function dopri45_coeffs()
    rk4a = [0.0             0.0             0.0             0.0             0.0             0.0         0.0
            0.2             0.0             0.0             0.0             0.0             0.0         0.0
            3.0/40.0        9.0/40.0        0.0             0.0             0.0             0.0         0.0
            44.0/45.0      -56.0/15.0       32.0/9.0        0.0             0.0             0.0         0.0
            19372.0/6561.0 -25360.0/2187.0  64448.0/6561.0  -212.0/729.0    0.0             0.0         0.0
            9017.0/3168.0  -355.0/33.0      46732.0/5247.0  49.0/176.0      -5103.0/18656.0 0.0         0.0
            35.0/384.0      0.0             500.0/1113.0    125.0/192.0     -2187.0/6784.0  11.0/84.0   0.0 ]

    rk4c = vec([0.0 0.2 0.3 0.8 8.0/9.0 1.0 1.0 ])

    # coefficients to evolve error estimator = b1-b2
    rk4E = vec([71.0/57600.0  0.0 -71.0/16695.0 71.0/1920.0 -17253.0/339200.0 22.0/525.0 -1.0/40.0 ])

    return rk4a,rk4E,rk4c
end

function DOPRI45!(U,dt,ops,geom,param,bcdata,size_ops)
    ZETA,ETA,POSTOL,ZEROTOL,Nc,N,K,T,EQN,FILTERTYPE,CFL,CN,dt0 = param
    Np,Nq,Nfp,Nh = size_ops
    
    rka,rkE,rkc = dopri45_coeffs()
    # DOPRI storage
    Utmp = similar.(U)
    rhsUrk = ntuple(x->zero.(U),length(rkE))

    errEst = 0.0
    prevErrEst = 0.0

    t = 0.0
    i = 0
    interval = 5

    prevU = zero.(U)

    if (FILTERTYPE != 0)
        apply_filter!(U,ops,geom,param,size_ops,prealloc)
    end
    rhsU = rhs_modalESDG(U,md,rd,param,sizes,geom,bcdata,ops,prealloc)
    rhsUrk[1] .= rhsU # initialize DOPRI rhs (FSAL property)
    @time begin
    while t < T
        # DOPRI step and
        for INTRK = 2:7
            k = zero.(Utmp)
            for s = 1:INTRK-1
                k .= @. k + rka[INTRK,s]*rhsUrk[s]
            end
            Utmp .= U + dt*k
            if (FILTERTYPE != 0)
                apply_filter!(Utmp,ops,geom,param,size_ops,prealloc)
            end
            rhsU = rhs_modalESDG(U,md,rd,param,sizes,geom,bcdata,ops,prealloc)
            rhsUrk[INTRK] .= rhsU
        end
        errEstVec = zero.(Utmp)
        for s = 1:7
            errEstVec .= errEstVec + rkE[s]*rhsUrk[s]
        end

        errTol = 1e-5
        errEst = 0.0
        for field = 1:Nc
            # errEstScale = @. abs(errEstVec[field]) / (errTol*(1+abs(U[field])))
            # errEst += sum(errEstScale.^2) # hairer seminorm
            errEstScale = map(x->x[field], errEstVec) ./ (errTol*(1 .+ abs.(map(x->x[field], U))))
            errEst += sum(errEstScale.^2)
        end
        # errEst = sqrt(errEst/(length(U[1])*4))
        errEst = sqrt(errEst/(length(U)*4))
        if errEst < 1.0 # if err small, accept step and update
            U .= Utmp
            t += dt
            rhsUrk[1] .= rhsUrk[7] # use FSAL property
        end
        order = 5
        dtnew = .8*dt*(.9/errEst)^(.4/(order+1)) # P controller
        if i > 0 # use PI controller if prevErrEst available
                dtnew *= (prevErrEst/max(ZEROTOL,errEst))^(.3/(order+1))
        end
        dt = max(min(10*dt0,dtnew),1e-9) # max/min dt
        prevErrEst = errEst

        i = i + 1  # number of total steps attempted
        if i%interval==0
            preverr = norm(U - prevU)
            prevU .= U
            println("i = $i, t = $t, dt = $dtnew, errEst = $errEst, preverr = $preverr")
        end
    end
    end
end